# =============================================================================
# supervisor.py  –  Industrial-grade process supervisor  (v2 – patched)
# =============================================================================
# FIX #2:  Orphan SHM cleanup at startup via ipc.frame_store.cleanup_orphan_shm
# FIX #8:  Handles MSG_SETTINGS_SAVED from GUI → broadcasts CTRL_RELOAD_SETTINGS
#          to all child processes
# FIX #9:  GPU worker pool – spawns N detection workers (configurable)
#          Default = 1; increase for multi-camera throughput
# FIX #11: GUI is optional but restart limit removed (restarts forever)
# FIX #13: setup_process_logger("supervisor") → logs/supervisor.log
# =============================================================================

import os
import sys
import time
import signal
import multiprocessing
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.logger import setup_process_logger, get_logger
from utils.time_utils import uptime_str
from ipc.messages import (
    MSG_HEARTBEAT, MSG_ERROR, MSG_SETTINGS_SAVED,
    make_control, make_camera_restarted,
    CTRL_SHUTDOWN, CTRL_SOFT_RESET, CTRL_RELOAD_SETTINGS,
)

# ── tunables ────────────────────────────────────────────────────────────────
POLL_INTERVAL       = 3.0
HEARTBEAT_TIMEOUT   = 30.0
SOFT_RESTART_HRS    = 24.0
RESTART_BACKOFF     = [5, 10, 30, 60, 120]
GUI_OPTIONAL        = True      # system continues if GUI crashes
DETECTION_WORKERS   = 1         # FIX #9: set to 2 for heavier multi-camera loads
CONFIG_FILE         = "config.yaml"


# =============================================================================
# Config helpers
# =============================================================================

def _load_yaml(path: str = CONFIG_FILE) -> Dict[str, Any]:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[SUPERVISOR] Config load error: {e}")
        return {}


def _build_camera_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    from config.loader import ConfigManager, SETTINGS
    SETTINGS.load()

    mgr     = ConfigManager()
    app_cfg = mgr.load()

    yaml_cams = {c["id"]: c for c in cfg.get("cameras", [])}
    out = []

    for cam in app_cfg.cameras:
        entry = {
            "id":         cam.id,
            "rtsp_url":   yaml_cams.get(cam.id, {}).get("rtsp_url", cam.rtsp_url),
            "resolution": tuple(SETTINGS.processing_resolution),
            "zones":      [z.to_dict() for z in cam.zones],
        }
        out.append(entry)

    boundary_ids = {c.id for c in app_cfg.cameras}
    for cam_id, cam_data in yaml_cams.items():
        try:
            cam_id = int(cam_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid camera ID '{cam_id}' in config, skipping")
            continue
        if cam_id not in boundary_ids:
            entry = {
                "id":         cam_id,
                "rtsp_url":   cam_data.get("rtsp_url", ""),
                "resolution": tuple(SETTINGS.processing_resolution),
                "zones":      [],
            }
            out.append(entry)
            mgr.add_camera(cam_id, entry["rtsp_url"])

    if not out:
        out.append({
            "id": 1,
            "rtsp_url": cfg.get("default_rtsp",
                                "rtsp://admin:Pass_123@192.168.1.64:554/stream"),
            "resolution": tuple(SETTINGS.processing_resolution),
            "zones":      [],
        })
    if out:
        mgr.add_camera(1, out[0]["rtsp_url"])
    else:
        logger.error("No camera configurations available, cannot start camera manager")

    return out


# =============================================================================
# ProcessEntry
# =============================================================================

class ProcessEntry:
    def __init__(self, name: str, target: Callable[..., Any], args: tuple[Any, ...],
                 is_optional: bool = False):
        self.name        = name
        self.target      = target
        self.args        = args
        self.is_optional = is_optional
        self.process:    Optional[Process] = None
        self.restarts    = 0
        self.last_hb     = time.monotonic()
        self.started_at  = time.monotonic()
        self.backoff_i   = 0

    def spawn(self) -> None:
        self.process    = Process(target=self.target, args=self.args,
                                   name=self.name, daemon=False)
        self.process.start()
        self.started_at = time.monotonic()
        self.last_hb    = time.monotonic()

    def kill(self) -> None:
        if self.process and self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=5)

    def is_alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def next_backoff(self) -> float:
        delay = RESTART_BACKOFF[min(self.backoff_i, len(RESTART_BACKOFF) - 1)]
        self.backoff_i = min(self.backoff_i + 1, len(RESTART_BACKOFF) - 1)
        return delay

    def reset_backoff(self) -> None:
        self.backoff_i = 0


# =============================================================================
# Supervisor
# =============================================================================

class Supervisor:

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg          = cfg
        self.log          = get_logger("Supervisor")
        self._running     = True
        self._start_time  = time.time()
        self._last_soft_restart = time.monotonic()

        # IPC queues
        self.heartbeat_q    = Queue(maxsize=500)  # type: ignore
        self.relay_q        = Queue(maxsize=200)  # type: ignore
        self.relay_status_q = Queue(maxsize=200)  # type: ignore
        self.gui_control_q  = Queue(maxsize=50)   # type: ignore

        # FIX #9: one control queue per detection worker
        self.det_control_qs: List[Queue[Dict[str, Any]]] = [
            Queue(maxsize=50) for _ in range(DETECTION_WORKERS)
        ]

        # result_q is shared by all detection workers → GUI
        self.result_q = Queue(maxsize=200)  # type: ignore

        # Per-camera control queues
        self.cam_control_qs: Dict[int, Queue[Dict[str, Any]]] = {}

        self.camera_configs = _build_camera_configs(cfg)
        self.entries: Dict[str, ProcessEntry] = {}
        self._pending_restarts: Dict[str, float] = {}

    # ── setup ────────────────────────────────────────────────────────────────

    def _cleanup_orphan_shm(self) -> None:
        """FIX #2: remove stale shared memory from a previous crash."""
        try:
            from config.loader import SETTINGS
            from ipc.frame_store import cleanup_orphan_shm
            cam_ids  = [c["id"] for c in self.camera_configs]
            res      = self.camera_configs[0]["resolution"] if self.camera_configs \
                       else (1280, 720)
            if isinstance(res, (list, tuple)) and len(res) >= 2:
                w, h = int(res[0]), int(res[1])
            else:
                self.log.warning(f"Invalid resolution format: {res}, using default 1280x720")
                w, h = 1280, 720
            cleanup_orphan_shm(cam_ids, w, h)
        except Exception as e:
            self.log.warning(f"SHM orphan cleanup error: {e}")

    def _build_entries(self) -> None:
        from processes.camera_process    import run_camera_process
        from processes.detection_process import run_detection_process
        from processes.relay_process     import run_relay_process
        from processes.gui_process       import run_gui_process
        from config.loader import SETTINGS

        # Camera processes (one per camera)
        for cam in self.camera_configs:
            cid  = cam["id"]
            cq   = Queue(maxsize=10)
            self.cam_control_qs[cid] = cq
            name = f"camera_{cid}"
            self.entries[name] = ProcessEntry(
                name   = name,
                target = run_camera_process,
                args   = (
                    cid,
                    cam["rtsp_url"],
                    cam["resolution"],
                    self.heartbeat_q,
                    cq,
                ),
            )

        # FIX #9: spawn DETECTION_WORKERS GPU workers
        for wid in range(DETECTION_WORKERS):
            wname = f"detection_{wid}" if wid > 0 else "detection"
            self.entries[wname] = ProcessEntry(
                name   = wname,
                target = run_detection_process,
                args   = (
                    self.camera_configs,
                    self.heartbeat_q,
                    self.det_control_qs[wid],
                    self.result_q,
                    self.relay_q,
                    SETTINGS.violation_mode,
                    None,   # use module default RAM limit
                    None,   # use module default VRAM limit
                    wid,    # worker_id
                ),
            )

        # Relay
        self.entries["relay"] = ProcessEntry(
            name   = "relay",
            target = run_relay_process,
            args   = (
                self.heartbeat_q,
                Queue(maxsize=10),    # relay control queue (for CTRL_RELOAD_SETTINGS)
                self.relay_q,
                self.relay_status_q,
                SETTINGS.use_usb_relay,
                SETTINGS.usb_num_channels,
                SETTINGS.usb_serial,
                SETTINGS.relay_duration,
                SETTINGS.relay_cooldown,
            ),
        )
        # Keep a reference to relay's control queue
        self._relay_control_q = self.entries["relay"].args[1]

        # GUI  (FIX #11: is_optional=True but NO restart limit)
        self.entries["gui"] = ProcessEntry(
            name        = "gui",
            target      = run_gui_process,
            args        = (
                self.camera_configs,
                self.heartbeat_q,
                self.gui_control_q,
                self.result_q,
                self.relay_status_q,
                self.det_control_qs[0],   # GUI sends zone_updated to worker 0
            ),
            is_optional = True,
        )

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.log.info("=" * 72)
        self.log.info("  INDUSTRIAL VISION SAFETY SYSTEM  –  SUPERVISOR  v2")
        self.log.info("=" * 72)
        self.log.info(f"  Cameras         : {len(self.camera_configs)}")
        self.log.info(f"  Detection workers: {DETECTION_WORKERS}")
        self.log.info(f"  Config          : {CONFIG_FILE}")
        self.log.info("=" * 72)

        self._cleanup_orphan_shm()   # FIX #2
        self._build_entries()
        self._spawn_all()
        self._install_signal_handlers()

        try:
            while self._running:
                self._drain_heartbeat_queue()
                self._check_processes()
                self._apply_pending_restarts()
                self._check_scheduled_soft_restart()
                self._log_status()
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            self.log.info("KeyboardInterrupt – shutting down")
        finally:
            self.shutdown()

    # ── spawn ────────────────────────────────────────────────────────────────

    def _spawn_all(self) -> None:
        for entry in self.entries.values():
            self._spawn(entry)

    def _spawn(self, entry: ProcessEntry) -> None:
        try:
            entry.spawn()
            self.log.info(f"Spawned  [{entry.name}]  PID={entry.process.pid}")
        except Exception as e:
            self.log.error(f"Failed to spawn [{entry.name}]: {e}")

    # ── monitor ──────────────────────────────────────────────────────────────

    def _drain_heartbeat_queue(self) -> None:
        try:
            while True:
                msg    = self.heartbeat_q.get_nowait()
                source = msg.get("source", "unknown")
                mtype  = msg.get("type",   "")

                if mtype == MSG_HEARTBEAT:
                    for name, entry in self.entries.items():
                        if source.startswith(name.split("_")[0]) or source == name:
                            entry.last_hb = time.monotonic()
                            entry.reset_backoff()
                            break

                elif mtype == MSG_ERROR:
                    payload = msg.get("payload", {})
                    fatal   = payload.get("fatal", False)
                    err     = payload.get("error", "")
                    self.log.error(
                        f"[{source}] ERROR: {err}" + (" (FATAL)" if fatal else "")
                    )

                # FIX #8: GUI saved settings → broadcast reload to all workers
                elif mtype == MSG_SETTINGS_SAVED:
                    self.log.info("Settings saved – broadcasting CTRL_RELOAD_SETTINGS")
                    self._broadcast_settings_reload()

        except Exception:
            pass

    def _broadcast_settings_reload(self) -> None:
        """FIX #8: push CTRL_RELOAD_SETTINGS to every child process."""
        reload_msg = make_control("supervisor", CTRL_RELOAD_SETTINGS)
        # Detection workers
        for q in self.det_control_qs:
            try:
                q.put_nowait(reload_msg)
            except Exception:
                pass
        # Relay
        try:
            self._relay_control_q.put_nowait(reload_msg)
        except Exception:
            pass
        # GUI
        try:
            self.gui_control_q.put_nowait(reload_msg)
        except Exception:
            pass
        # Cameras (for resolution changes)
        for q in self.cam_control_qs.values():
            try:
                q.put_nowait(reload_msg)
            except Exception:
                pass

    def _check_processes(self) -> None:
        now = time.monotonic()
        for name, entry in list(self.entries.items()):
            if name in self._pending_restarts:
                continue

            if not entry.is_alive():
                code = entry.process.exitcode if entry.process else None
                # FIX #11: GUI optional but restarts forever – no restart cap
                self.log.warning(
                    f"[{name}] dead (exit={code}) – scheduling restart "
                    f"(total={entry.restarts})"
                )
                self._schedule_restart(entry)
                continue

            if now - entry.last_hb > HEARTBEAT_TIMEOUT:
                self.log.warning(
                    f"[{name}] heartbeat timeout "
                    f"({now - entry.last_hb:.0f}s) – killing"
                )
                entry.kill()
                self._schedule_restart(entry)

    def _schedule_restart(self, entry: ProcessEntry) -> None:
        delay = entry.next_backoff()
        self._pending_restarts[entry.name] = time.monotonic() + delay
        entry.restarts += 1
        self.log.info(f"[{entry.name}] restart in {delay}s (attempt {entry.restarts})")

    def _apply_pending_restarts(self) -> None:
        now  = time.monotonic()
        done = []
        for name, restart_at in list(self._pending_restarts.items()):
            if now >= restart_at:
                entry = self.entries.get(name)
                if entry:
                    self.log.info(f"[{name}] restarting now")
                    entry.kill()

                    # SHM LIFECYCLE FIX: for camera restarts, clean the old
                    # segment first so the new process always creates a fresh
                    # one, then notify detection workers to reattach.
                    if name.startswith("camera_"):
                        self._handle_camera_restart(name)

                    self._spawn(entry)
                done.append(name)
        for n in done:
            del self._pending_restarts[n]

    def _handle_camera_restart(self, entry_name: str) -> None:
        """
        SHM LIFECYCLE FIX – called before re-spawning a camera process.

        1. Determine camera_id from entry name (e.g. "camera_2" → 2).
        2. Clean up the old SHM segment so the new process creates a fresh one.
        3. Notify all detection workers via CTRL_CAMERA_RESTARTED so they
           call reader.reattach() exactly once, releasing the stale OS handle.
        """
        try:
            camera_id = int(entry_name.split("_")[-1])
        except (ValueError, IndexError):
            return

        # Step 1: cleanup old SHM segment (before new camera process writes to it)
        try:
            from ipc.frame_store import cleanup_shm_for_camera
            res = self.camera_configs[0]["resolution"] if self.camera_configs \
                  else (1280, 720)
            # Find the specific camera's resolution
            for cam in self.camera_configs:
                if cam["id"] == camera_id:
                    res = cam["resolution"]
                    break
            cleaned = cleanup_shm_for_camera(camera_id, res[0], res[1])
            if cleaned:
                self.log.info(
                    f"[{entry_name}] SHM segment cleaned before restart"
                )
        except Exception as e:
            self.log.warning(f"SHM cleanup for {entry_name} failed: {e}")

        # Step 2: notify detection workers to reattach their reader handle
        msg = make_camera_restarted("supervisor", camera_id)
        for q in self.det_control_qs:
            try:
                q.put_nowait(msg)
            except Exception:
                pass
        self.log.info(
            f"[{entry_name}] CTRL_CAMERA_RESTARTED sent to "
            f"{len(self.det_control_qs)} detection worker(s)"
        )

    # ── scheduled soft restart ───────────────────────────────────────────────

    def _check_scheduled_soft_restart(self) -> None:
        elapsed = time.monotonic() - self._last_soft_restart
        if elapsed < SOFT_RESTART_HRS * 3600:
            return

        self.log.info("=" * 60)
        self.log.info("  SCHEDULED SOFT RESTART  (24-hour maintenance)")
        self.log.info("=" * 60)
        self._last_soft_restart = time.monotonic()

        for q in self.det_control_qs:
            try:
                q.put_nowait(make_control("supervisor", CTRL_SOFT_RESET))
            except Exception:
                pass
        time.sleep(3)

        for name, entry in list(self.entries.items()):
            if name.startswith("camera_") or name.startswith("detection"):
                self.log.info(f"Soft-restarting [{name}]")
                entry.kill()
                time.sleep(1)
                self._spawn(entry)

    # ── status ───────────────────────────────────────────────────────────────

    def _log_status(self) -> None:
        uptime  = uptime_str(self._start_time)
        alive   = sum(1 for e in self.entries.values() if e.is_alive())
        total   = len(self.entries)
        pending = len(self._pending_restarts)
        self.log.debug(
            f"Status: up={uptime}  alive={alive}/{total}  "
            f"pending_restarts={pending}"
        )

    # ── shutdown ─────────────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame) -> None:
        self.log.info(f"Signal {signum} – initiating shutdown")
        self._running = False

    def shutdown(self) -> None:
        self.log.info("Supervisor shutting down all processes …")
        shutdown_msg = make_control("supervisor", CTRL_SHUTDOWN)

        for q in list(self.cam_control_qs.values()):
            try:
                q.put_nowait(shutdown_msg)
            except Exception:
                pass
        for q in self.det_control_qs:
            try:
                q.put_nowait(shutdown_msg)
            except Exception:
                pass
        for q in [self.gui_control_q, self._relay_control_q]:
            try:
                q.put_nowait(shutdown_msg)
            except Exception:
                pass

        time.sleep(2)
        for name, entry in list(self.entries.items()):
            if entry.is_alive():
                self.log.info(f"  Terminating [{name}]")
                entry.kill()

        self.log.info("Supervisor shutdown complete")


# =============================================================================
# Entry point
# =============================================================================

def main():
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    setup_process_logger("supervisor")   # FIX #13 → logs/supervisor.log
    cfg = _load_yaml()
    Supervisor(cfg).run()


if __name__ == "__main__":
    main()
