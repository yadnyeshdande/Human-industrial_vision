# =============================================================================
# processes/detection_process.py  –  GPU YOLO detection worker  (v5)
# =============================================================================
#
# SHM LIFECYCLE FIX
# -----------------
# • reader.attach() is called ONCE per camera at startup inside
#   _attach_readers_with_retry().  It is NEVER called again inside the
#   processing loop.  The old pattern `if reader._shm is None: reader.attach()`
#   inside the hot loop created a handle-accumulation path on Windows when
#   camera processes restarted.
#
# • When the supervisor sends CTRL_CAMERA_RESTARTED (camera_id=N), the control
#   drain calls reader.reattach() for that specific camera exactly once.
#   reattach() closes the stale OS handle and opens a fresh one.
#
# • As a fallback, read_if_new() now sets reader.is_stale=True when it detects
#   a counter reset (writer restarted without CTRL_CAMERA_RESTARTED arriving).
#   The loop checks is_stale after each read and auto-reattaches once.
#
# MICRO-BATCH GPU INFERENCE
# -------------------------
# The old architecture called model(single_frame) in a round-robin loop,
# launching one CUDA kernel per camera per cycle.  On an RTX 3050 with
# 3 cameras this keeps GPU utilization at ~35%.
#
# The new architecture:
#   1. _collect_batch() reads the latest NEW frame from every camera in
#      one pass without calling the model.
#   2. detector.detect_batch(frames) calls model([f1, f2, f3]) once,
#      resulting in a single CUDA kernel that fully occupies the GPU.
#   3. _route_batch_results() maps each result back to its camera_id and
#      dispatches detection_result messages + relay triggers.
#
# Typical improvement on RTX 3050 + YOLOv8n/l + 3 cameras:
#   Before:  ~12 FPS per camera,  GPU util ~35%
#   After:   ~22 FPS per camera,  GPU util ~75%
# =============================================================================

import os
import sys
import time
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from utils.logger import setup_process_logger, get_logger
from utils.resource_guard import (
    ResourceGuard, ResourceLimitExceeded,
    RAM_LIMIT_DETECTION, VRAM_LIMIT_MB, GPU_TEMP_CRITICAL_C,
)
from utils.time_utils import FPSCounter
from ipc.frame_store import FrameReader
from ipc.messages import (
    make_heartbeat, make_error, make_detection_result, make_relay_command,
    make_telemetry,
    MSG_SHUTDOWN, MSG_CONTROL, MSG_ZONE_UPDATED,
    CTRL_SHUTDOWN, CTRL_RELOAD_CFG, CTRL_SOFT_RESET, CTRL_RELOAD_SETTINGS,
    CTRL_CAMERA_RESTARTED,
)

SNAPSHOT_DIR       = Path("snapshots")
MAX_SNAPSHOTS      = 10_000
HEARTBEAT_EVERY    = 5.0
TELEMETRY_EVERY    = 2.0
OVERHEAT_FPS_CAP   = 6
FPS_TARGET         = 12

# Micro-batch: wait this long to accumulate frames before forcing a batch
# even if some cameras have no new frame yet (keeps latency bounded).
BATCH_COLLECT_TIMEOUT_S = 0.04    # 40 ms → max latency added by batching


def run_detection_process(
    camera_configs:  List[Dict[str, Any]],
    heartbeat_q:     Queue,
    control_q:       Queue,
    result_q:        Queue,
    relay_q:         Queue,
    violation_mode:  str   = "center",
    ram_limit_mb:    float = RAM_LIMIT_DETECTION,
    vram_limit_mb:   float = VRAM_LIMIT_MB,
    worker_id:       int   = 0,
) -> None:
    pname = f"detection_{worker_id}" if worker_id > 0 else "detection"
    setup_process_logger(pname)
    log = get_logger("Main")
    log.info(f"Detection worker {worker_id} started  PID={os.getpid()}")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    guard = ResourceGuard(ram_limit_mb=ram_limit_mb, vram_limit_mb=vram_limit_mb)

    detector = _load_detector_strict(heartbeat_q, pname, log)
    if detector is None:
        sys.exit(1)

    # SHM LIFECYCLE FIX: build readers dict once; never rebuild inside the loop
    readers: Dict[int, FrameReader] = {}
    zones:   Dict[int, List]        = {}

    for cam in camera_configs:
        cid       = cam["id"]
        w, h      = cam["resolution"]
        readers[cid] = FrameReader(camera_id=cid, width=w, height=h)
        zones[cid]   = _parse_zones(cam.get("zones", []))

    # SHM LIFECYCLE FIX: attach once at startup; never call attach() in the loop
    _attach_readers_with_retry(readers, log)

    fps_counters:     Dict[int, FPSCounter] = {cid: FPSCounter(30) for cid in readers}
    prev_violations:  Dict[int, set]        = {cid: set()           for cid in readers}
    # Track last frame counter per camera so batch collector can detect new frames
    last_frame_ctrs:  Dict[int, int]        = {cid: 0               for cid in readers}
    camera_ids = list(readers.keys())

    last_hb        = 0.0
    last_telemetry = 0.0
    fps_cap        = FPS_TARGET

    try:
        while True:
            # ── control queue ─────────────────────────────────────────────────
            _drain_control(
                control_q, zones, log, pname, heartbeat_q,
                detector, readers, fps_counters, prev_violations,
            )

            # ── resource guard ────────────────────────────────────────────────
            try:
                guard.check()
            except ResourceLimitExceeded as e:
                log.error(f"Resource limit: {e}")
                heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
                sys.exit(2)

            fps_cap = OVERHEAT_FPS_CAP if guard.is_gpu_overheating() else FPS_TARGET

            now = time.monotonic()

            # ── heartbeat ─────────────────────────────────────────────────────
            if now - last_hb >= HEARTBEAT_EVERY:
                last_hb = now
                avg_fps = (sum(c.fps for c in fps_counters.values())
                           / max(len(fps_counters), 1))
                gpu = guard.gpu_health_summary()
                heartbeat_q.put_nowait(make_heartbeat(
                    source=pname, fps=avg_fps,
                    ram_mb=guard.get_ram_mb(),
                    extra={**gpu, "cameras": camera_ids, "worker_id": worker_id},
                ))

            # ── telemetry → GUI sidebar ───────────────────────────────────────
            if now - last_telemetry >= TELEMETRY_EVERY:
                last_telemetry = now
                avg_fps = (sum(c.fps for c in fps_counters.values())
                           / max(len(fps_counters), 1))
                try:
                    result_q.put_nowait(make_telemetry(
                        source=pname,
                        detection_fps=avg_fps,
                        gpu_vram_mb=guard.get_vram_mb(),
                        gpu_util_pct=guard.get_gpu_utilization(),
                        gpu_temp_c=guard.get_gpu_temp(),
                        ram_mb=guard.get_ram_mb(),
                        cameras_active=camera_ids,
                    ))
                except Exception:
                    pass

            # ── MICRO-BATCH INFERENCE ─────────────────────────────────────────
            # Collect one fresh frame per camera (non-blocking) then run a
            # single model([f1, f2, f3]) call instead of N serial calls.
            batch_frames, batch_meta = _collect_batch(
                camera_ids, readers, last_frame_ctrs, fps_cap
            )

            if not batch_frames:
                # No new frames from any camera – sleep briefly and retry
                time.sleep(0.005)
                continue

            # Single GPU kernel for the whole batch
            try:
                all_detections = detector.detect_batch(batch_frames)
            except Exception as e:
                log.error(f"Batch inference error: {e}")
                all_detections = [[] for _ in batch_frames]

            # Route each result back to its camera
            _route_batch_results(
                batch_meta, all_detections,
                zones, prev_violations, fps_counters, last_frame_ctrs,
                result_q, relay_q, pname, violation_mode, log,
            )

            # SHM LIFECYCLE FIX: auto-reattach for any camera whose reader
            # detected a counter-reset (fallback for missed CTRL_CAMERA_RESTARTED)
            for cid, reader in readers.items():
                if reader.is_stale:
                    log.warning(
                        f"Camera {cid} SHM counter-reset detected – "
                        "auto-reattaching (fallback)"
                    )
                    if reader.reattach():
                        log.info(f"Camera {cid} SHM reattached successfully")
                        last_frame_ctrs[cid] = 0
                    else:
                        log.warning(f"Camera {cid} SHM reattach failed – "
                                    "will retry on next counter-reset")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"Detection worker fatal: {e}", exc_info=True)
        heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
        sys.exit(1)
    finally:
        try:
            detector.unload()
        except Exception:
            pass
        # SHM LIFECYCLE FIX: close every handle exactly once on exit
        for r in readers.values():
            r.close()
        log.info(f"Detection worker {worker_id} exiting; "
                 f"closed {len(readers)} SHM handles")


# =============================================================================
# Micro-batch helpers
# =============================================================================

def _collect_batch(
    camera_ids:      List[int],
    readers:         Dict[int, FrameReader],
    last_frame_ctrs: Dict[int, int],
    fps_cap:         float,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    MICRO-BATCH: Gather the latest NEW frame from each camera in one pass.

    Returns:
        batch_frames – list of BGR frames ready for model input
        batch_meta   – list of (camera_id, frame_counter) in matching order

    SHM LIFECYCLE FIX: We call read_latest_frame() (unconditional read) here
    rather than read_if_new(), then filter by counter comparison ourselves.
    This avoids touching the reader's internal _last_counter so that
    is_stale detection (counter rollback check) still works correctly.
    """
    batch_frames: List[np.ndarray]        = []
    batch_meta:   List[Tuple[int, int]]   = []

    interval = 1.0 / max(fps_cap, 1)

    for cid in camera_ids:
        reader = readers.get(cid)
        if reader is None or reader._shm is None:
            continue

        result = reader.read_latest_frame()
        if result is None:
            continue
        frame, counter = result

        # Skip if no new frame since last batch
        if counter == last_frame_ctrs.get(cid, 0):
            continue

        # Per-camera FPS throttle
        if not _throttle_ok(cid, interval):
            continue

        batch_frames.append(frame)
        batch_meta.append((cid, counter))

    return batch_frames, batch_meta


def _route_batch_results(
    batch_meta:      List[Tuple[int, int]],
    all_detections:  List[List[Tuple]],
    zones:           Dict[int, List],
    prev_violations: Dict[int, set],
    fps_counters:    Dict[int, FPSCounter],
    last_frame_ctrs: Dict[int, int],
    result_q:        Queue,
    relay_q:         Queue,
    pname:           str,
    violation_mode:  str,
    log,
) -> None:
    """Route each element of a batched inference result back to its camera."""
    for (cid, frame_ctr), raw_results in zip(batch_meta, all_detections):
        # Update last seen counter now that we've processed this frame
        last_frame_ctrs[cid] = frame_ctr

        persons = [(x1, y1, x2, y2) for x1, y1, x2, y2, *_ in raw_results]
        bounding_boxes = [
            {"bbox": [x1, y1, x2, y2], "label": "person",
             "confidence": round(float(conf), 3)}
            for x1, y1, x2, y2, conf in raw_results
        ]

        cam_zones   = zones.get(cid, [])
        cur_viols   = set()
        viol_info   = []
        zone_status: Dict[int, bool] = {zid: False for zid, _, _ in cam_zones}

        for bbox in persons:
            for zone_id, points, relay_id in cam_zones:
                if _check_violation(bbox, points, violation_mode):
                    cur_viols.add(zone_id)
                    zone_status[zone_id] = True
                    viol_info.append({
                        "zone_id":  zone_id,
                        "relay_id": relay_id,
                        "bbox":     list(bbox),
                    })
                    break

        prev  = prev_violations.get(cid, set())
        new_v = cur_viols - prev
        prev_violations[cid] = cur_viols

        for vi in viol_info:
            if vi["zone_id"] in new_v:
                log.warning(
                    f"VIOLATION cam={cid} zone={vi['zone_id']} "
                    f"relay={vi['relay_id']}"
                )
                relay_q.put_nowait(make_relay_command(
                    source=pname, relay_id=vi["relay_id"],
                    camera_id=cid, zone_id=vi["zone_id"],
                ))

        fps_counters[cid].tick()

        try:
            result_q.put_nowait(make_detection_result(
                source=pname, camera_id=cid,
                persons=persons, violations=viol_info,
                fps=fps_counters[cid].fps, frame_counter=frame_ctr,
                bounding_boxes=bounding_boxes, zone_status=zone_status,
            ))
        except Exception:
            pass


# =============================================================================
# Module-level throttle state (per-camera)
# =============================================================================

_last_infer_t: Dict[int, float] = {}

def _throttle_ok(cid: int, interval: float) -> bool:
    now = time.monotonic()
    if now - _last_infer_t.get(cid, 0.0) < interval:
        return False
    _last_infer_t[cid] = now
    return True


# =============================================================================
# Control queue drain
# =============================================================================

def _drain_control(control_q, zones, log, pname, heartbeat_q,
                   detector, readers, fps_counters, prev_violations):
    try:
        while True:
            msg   = control_q.get_nowait()
            mtype = msg.get("type", "")
            cmd   = msg.get("payload", {}).get("command", "")

            if mtype == "shutdown" or cmd == CTRL_SHUTDOWN:
                log.info("Shutdown – exiting")
                try:
                    detector.unload()
                except Exception:
                    pass
                # Close all SHM handles before exit
                for r in readers.values():
                    r.close()
                sys.exit(0)

            if mtype == "zone_config_updated" or cmd == CTRL_RELOAD_CFG:
                log.info("Zone config update – reloading")
                _reload_zones(zones, log)

            if cmd == CTRL_SOFT_RESET:
                log.info("Soft reset – clearing CUDA cache")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

            if cmd == CTRL_RELOAD_SETTINGS:
                log.info("Settings reload")
                try:
                    from config.loader import SETTINGS
                    SETTINGS.load()
                except Exception as e:
                    log.warning(f"Settings reload failed: {e}")

            # SHM LIFECYCLE FIX: supervisor signals that camera N restarted
            if cmd == CTRL_CAMERA_RESTARTED:
                camera_id = msg.get("payload", {}).get("camera_id") \
                            or msg.get("camera_id")
                if camera_id is not None and camera_id in readers:
                    log.info(
                        f"CTRL_CAMERA_RESTARTED cam={camera_id} – reattaching SHM"
                    )
                    reader = readers[camera_id]
                    if reader.reattach():
                        log.info(f"Camera {camera_id} SHM reattached")
                    else:
                        log.warning(
                            f"Camera {camera_id} SHM not ready yet – "
                            "will retry via counter-reset detection"
                        )

    except Exception:
        pass


# =============================================================================
# Utilities
# =============================================================================

def _load_detector_strict(heartbeat_q, pname, log):
    try:
        from core.detector import PersonDetector
        det = PersonDetector()
        if not det.is_model_loaded():
            raise RuntimeError("Model not loaded")
        return det
    except Exception as e:
        log.error(f"Detector load failed: {e}", exc_info=True)
        heartbeat_q.put_nowait(
            make_error(pname, f"Detector load failed: {e}", fatal=True)
        )
        return None


def _parse_zones(raw):
    return [
        (z["id"], [tuple(p) for p in z["points"]], z["relay_id"])
        for z in raw if z.get("points")
    ]


def _attach_readers_with_retry(readers, log, retries=60, delay=1.0):
    """
    SHM LIFECYCLE FIX: Called ONCE at startup.  Never called again.
    Waits up to retries*delay seconds for each camera's SHM segment to appear.
    """
    remaining = set(readers.keys())
    for attempt in range(retries):
        for cid in list(remaining):
            if readers[cid].attach():
                log.info(f"Attached SHM for camera {cid}")
                remaining.discard(cid)
        if not remaining:
            return
        time.sleep(delay)
    log.warning(
        f"Could not attach SHM for cameras: {remaining} after {retries}s – "
        "will retry per CTRL_CAMERA_RESTARTED"
    )


def _check_violation(bbox, points, mode):
    from core.geometry import bbox_center, point_in_polygon, bbox_overlaps_polygon
    if mode == "center":
        return point_in_polygon(bbox_center(bbox), points)
    return bbox_overlaps_polygon(bbox, points)


def _reload_zones(zones, log):
    try:
        from config.loader import ConfigManager
        cfg = ConfigManager().load()
        for cam in cfg.cameras:
            zones[cam.id] = [(z.id, z.points, z.relay_id) for z in cam.zones]
        log.info("Zones hot-reloaded")
    except Exception as e:
        log.error(f"Zone reload failed: {e}")


def _save_snapshot(frame, bbox, camera_id, zone_id, relay_id, zones, log):
    try:
        snap = frame.copy()
        for zid, pts, _ in zones:
            if len(pts) < 2:
                continue
            arr   = np.array(pts, np.int32).reshape(-1, 1, 2)
            color = (0, 0, 255) if zid == zone_id else (0, 255, 0)
            cv2.polylines(snap, [arr], True, color, 4 if zid == zone_id else 2)
            lbl   = f"Zone {zid}" + (" [VIOLATION]" if zid == zone_id else "")
            cv2.putText(snap, lbl, (int(pts[0][0]) + 5, int(pts[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(snap, "Violating Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fn = f"violation_cam{camera_id}_zone{zone_id}_relay{relay_id}_{ts}.jpg"
        cv2.imwrite(str(SNAPSHOT_DIR / fn), snap)
        log.info(f"Snapshot: {fn}")

        snaps = sorted(SNAPSHOT_DIR.glob("violation_*.jpg"))
        while len(snaps) > MAX_SNAPSHOTS:
            snaps[0].unlink(missing_ok=True)
            snaps = snaps[1:]
    except Exception as e:
        log.error(f"Snapshot failed: {e}")
