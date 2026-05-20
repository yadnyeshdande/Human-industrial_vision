# =============================================================================
# processes/detection_process.py  –  GPU YOLO detection worker  (v7)
# =============================================================================
#
# SHM LIFECYCLE FIX  (v6 – unchanged)
# MICRO-BATCH GPU INFERENCE / FIX LAG-3  (v6 – unchanged)
#
# v7 changes (Bug #1 + Feature #1):
# -----------------------------------
# BUG #1 – model not loading on change:
#   _drain_control() now hot-swaps the model on CTRL_RELOAD_SETTINGS /
#   CTRL_RELOAD_MODEL.  Uses detector_ref[0] (mutable single-element list)
#   so the inner function can replace the detector reference without a return.
#
#   CTRL_RELOAD_SETTINGS: swaps if model name OR target_classes changed.
#   CTRL_RELOAD_MODEL:    always swaps (force path, from settings_page).
#   violation_mode_ref[0] also updated so center↔overlap takes effect live.
#
# BUG #1 – startup model not loading:
#   _load_detector_strict() now calls SETTINGS.load() before constructing
#   PersonDetector, passing model_name/conf_threshold/target_classes from
#   SETTINGS.  Previously it used the class defaults, so the operator's
#   saved model was ignored on every startup.
#
# FEATURE #1 – class-labelled bounding boxes:
#   _route_batch_results() receives class_names dict and labels each bbox
#   with the real class name ("person", "helmet", etc.) + cls_id field.
#   Falls back to "class_N" if name not found in dict.
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
    CTRL_CAMERA_RESTARTED, CTRL_RELOAD_MODEL,
)

SNAPSHOT_DIR       = Path("snapshots")
MAX_SNAPSHOTS      = 10_000
HEARTBEAT_EVERY    = 5.0
TELEMETRY_EVERY    = 2.0
OVERHEAT_FPS_CAP   = 6
FPS_TARGET         = 12

# FIX LAG-3: this constant is now actually *used* inside _collect_batch().
BATCH_COLLECT_TIMEOUT_S = 0.12    # 120 ms


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

    # BUG #1 FIX: _load_detector_strict now reads SETTINGS first so the
    # operator's saved model (not the class default) is loaded at startup.
    detector = _load_detector_strict(heartbeat_q, pname, log)
    if detector is None:
        sys.exit(1)

    # Wrap in a mutable 1-element list so _drain_control can replace the
    # reference in-place without a return value.
    detector_ref: List = [detector]

    # Same pattern for violation_mode: updated live from SETTINGS on reload.
    violation_mode_ref: List[str] = [violation_mode]

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
    last_frame_ctrs:  Dict[int, int]        = {cid: 0               for cid in readers}
    camera_ids = list(readers.keys())

    last_hb        = 0.0
    last_telemetry = 0.0
    fps_cap        = FPS_TARGET
    _batch_count      = 0
    _batch_total_size = 0
    _infer_total_time = 0.0

    try:
        while True:
            # ── control queue ─────────────────────────────────────────────────
            _drain_control(
                control_q, zones, log, pname, heartbeat_q,
                detector_ref, violation_mode_ref,           # v7: ref lists
                readers, fps_counters, prev_violations,
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
                try:
                    heartbeat_q.put_nowait(
                        make_heartbeat(
                            source=pname, fps=avg_fps,
                            ram_mb=guard.get_ram_mb(),
                            extra={**gpu, "cameras": camera_ids, "worker_id": worker_id},
                        )
                    )
                except Exception:
                    pass

            # ── telemetry → GUI sidebar ───────────────────────────────────────
            if now - last_telemetry >= TELEMETRY_EVERY:
                last_telemetry = now
                avg_fps = (sum(c.fps for c in fps_counters.values())
                           / max(len(fps_counters), 1))
                avg_batch = (_batch_total_size / _batch_count) if _batch_count else 0.0
                avg_infer = (_infer_total_time / _batch_count) if _batch_count else 0.0
                extra = {
                    "avg_batch_size": round(avg_batch, 2),
                    "avg_infer_s":    round(avg_infer, 3),
                    # FEAT #1: report current model + active classes in telemetry
                    "model":          detector_ref[0]._model_name,
                    "target_classes": detector_ref[0].target_classes or "ALL",
                }
                try:
                    result_q.put_nowait(make_telemetry(
                        source=pname,
                        detection_fps=avg_fps,
                        gpu_vram_mb=guard.get_vram_mb(),
                        gpu_util_pct=guard.get_gpu_utilization(),
                        gpu_temp_c=guard.get_gpu_temp(),
                        ram_mb=guard.get_ram_mb(),
                        cameras_active=camera_ids,
                        extra=extra,
                    ))
                except Exception:
                    pass
                _batch_count      = 0
                _batch_total_size = 0
                _infer_total_time = 0.0

            # ── MICRO-BATCH INFERENCE ─────────────────────────────────────────
            batch_frames, batch_meta = _collect_batch(
                camera_ids, readers, last_frame_ctrs, fps_cap
            )

            if not batch_frames:
                time.sleep(0.005)
                continue

            t0 = time.monotonic()
            try:
                all_detections = detector_ref[0].detect_batch(batch_frames)
            except Exception as e:
                log.error(f"Batch inference error: {e}")
                all_detections = [[] for _ in batch_frames]
            infer_t = time.monotonic() - t0
            _batch_count      += 1
            _batch_total_size += len(batch_frames)
            _infer_total_time += infer_t

            _route_batch_results(
                batch_meta, all_detections,
                zones, prev_violations, fps_counters, last_frame_ctrs,
                result_q, relay_q, pname,
                violation_mode_ref[0], log,
                # FEAT #1: pass live class names for labelled bounding boxes
                class_names=detector_ref[0].get_class_names(),
            )

            # SHM LIFECYCLE FIX: auto-reattach on counter-reset fallback
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
        try:
            heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
        except Exception:
            pass
        sys.exit(1)
    finally:
        try:
            detector_ref[0].unload()
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
    MICRO-BATCH: Gather the latest NEW frame from each camera.

    FIX LAG-3: Loops until BATCH_COLLECT_TIMEOUT_S or all active cameras
    have contributed, whichever comes first.  Zero added latency in the
    common case (all frames already in SHM when called).
    """
    interval   = 1.0 / max(fps_cap, 1)
    start_time = time.monotonic()

    active_count = sum(
        1 for cid in camera_ids
        if readers.get(cid) is not None and readers[cid]._shm is not None
    )

    if active_count == 0:
        return [], []

    batch_frames: List[np.ndarray]      = []
    batch_meta:   List[Tuple[int, int]] = []

    while time.monotonic() - start_time < BATCH_COLLECT_TIMEOUT_S:
        batch_frames = []
        batch_meta   = []

        for cid in camera_ids:
            reader = readers.get(cid)
            if reader is None or reader._shm is None:
                continue

            result = reader.read_latest_frame()
            if result is None:
                continue

            frame, counter = result

            if counter == last_frame_ctrs.get(cid, 0):
                continue

            if not _throttle_ok(cid, interval):
                continue

            batch_frames.append(np.ascontiguousarray(frame))
            batch_meta.append((cid, counter))

        if len(batch_frames) == active_count:
            break

        time.sleep(0.002)

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
    class_names:     Optional[Dict[int, str]] = None,   # FEAT #1
) -> None:
    """Route each element of a batched inference result back to its camera."""
    if len(all_detections) != len(batch_meta):
        log.error(
            f"Batch size mismatch: expected {len(batch_meta)} detections, "
            f"got {len(all_detections)}"
        )
        min_len        = min(len(batch_meta), len(all_detections))
        batch_meta     = batch_meta[:min_len]
        all_detections = all_detections[:min_len]

    _names = class_names or {}

    for (cid, frame_ctr), raw_results in zip(batch_meta, all_detections):
        last_frame_ctrs[cid] = frame_ctr

        # detector v3 returns 6-tuples: (x1,y1,x2,y2,conf,cls_id)
        # detector v2 returned 5-tuples: (x1,y1,x2,y2,conf) – also handled
        persons = [(x1, y1, x2, y2) for x1, y1, x2, y2, *_ in raw_results]

        # FEAT #1: label each bbox with its real class name
        bounding_boxes = []
        for raw in raw_results:
            x1, y1, x2, y2 = raw[0], raw[1], raw[2], raw[3]
            conf   = raw[4] if len(raw) > 4 else 1.0
            cls_id = raw[5] if len(raw) > 5 else 0
            label  = _names.get(cls_id, f"class_{cls_id}")
            bounding_boxes.append({
                "bbox":       [x1, y1, x2, y2],
                "label":      label,
                "confidence": round(float(conf), 3),
                "cls_id":     cls_id,
            })

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
# Control queue drain  (v7 – hot model swap)
# =============================================================================

def _drain_control(
    control_q,
    zones,
    log,
    pname,
    heartbeat_q,
    detector_ref,        # List[PersonDetector] – mutable 1-element list (v7)
    violation_mode_ref,  # List[str]            – mutable 1-element list (v7)
    readers,
    fps_counters,
    prev_violations,
):
    """
    Drain all pending control messages.

    BUG #1 FIX:
        CTRL_RELOAD_SETTINGS: hot-swap if model or classes differ.
        CTRL_RELOAD_MODEL:    always hot-swap (force path).
        Both call detector_ref[0].reload() which:
          1. Calls unload() → frees GPU VRAM.
          2. Updates _model_name + target_classes.
          3. Calls _load() → loads new model.
        violation_mode_ref[0] updated immediately on CTRL_RELOAD_SETTINGS.
    """
    try:
        while True:
            msg   = control_q.get_nowait()
            mtype = msg.get("type", "")
            cmd   = msg.get("payload", {}).get("command", "")

            # ── shutdown ──────────────────────────────────────────────────────
            if mtype == "shutdown" or cmd == CTRL_SHUTDOWN:
                log.info("Shutdown – exiting")
                try:
                    detector_ref[0].unload()
                except Exception:
                    pass
                for r in readers.values():
                    r.close()
                sys.exit(0)

            # ── zone config reload ─────────────────────────────────────────────
            if mtype == "zone_config_updated" or cmd == CTRL_RELOAD_CFG:
                log.info("Zone config update – reloading")
                _reload_zones(zones, log)

            # ── soft reset ────────────────────────────────────────────────────
            if cmd == CTRL_SOFT_RESET:
                log.info("Soft reset – clearing CUDA cache")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass

            # ── settings reload + conditional model hot-swap (BUG #1 FIX) ────
            if cmd == CTRL_RELOAD_SETTINGS:
                log.info("CTRL_RELOAD_SETTINGS – reloading settings")
                try:
                    from config.loader import SETTINGS
                    SETTINGS.load()

                    # violation_mode takes effect immediately – no restart needed
                    new_mode = SETTINGS.violation_mode
                    if new_mode != violation_mode_ref[0]:
                        log.info(
                            f"violation_mode: {violation_mode_ref[0]!r} → {new_mode!r}"
                        )
                        violation_mode_ref[0] = new_mode

                    # Swap model only if something actually changed
                    new_model   = SETTINGS.yolo_model
                    new_classes = list(SETTINGS.target_classes)
                    cur_model   = detector_ref[0]._model_name
                    cur_classes = detector_ref[0].target_classes

                    if new_model != cur_model or new_classes != cur_classes:
                        log.info(
                            f"[HotReload] {cur_model!r} → {new_model!r}  "
                            f"classes: {cur_classes or 'ALL'} → {new_classes or 'ALL'}"
                        )
                        try:
                            detector_ref[0].reload(new_model, new_classes)
                            log.info(
                                f"[HotReload] SUCCESS  "
                                f"model={new_model!r}  "
                                f"classes={new_classes or 'ALL'}"
                            )
                        except Exception as e:
                            log.error(
                                f"[HotReload] FAILED: {e} – keeping previous model"
                            )
                    else:
                        log.info(
                            f"CTRL_RELOAD_SETTINGS: model unchanged ({cur_model!r}), "
                            "no reload needed"
                        )
                except Exception as e:
                    log.warning(f"Settings reload failed: {e}")

            # ── force model reload – always swap ──────────────────────────────
            if cmd == CTRL_RELOAD_MODEL:
                log.info("CTRL_RELOAD_MODEL – force-reloading model from SETTINGS")
                try:
                    from config.loader import SETTINGS
                    SETTINGS.load()
                    new_model   = SETTINGS.yolo_model
                    new_classes = list(SETTINGS.target_classes)
                    detector_ref[0].reload(new_model, new_classes)
                    violation_mode_ref[0] = SETTINGS.violation_mode
                    log.info(
                        f"[CTRL_RELOAD_MODEL] SUCCESS  "
                        f"model={new_model!r}  "
                        f"classes={new_classes or 'ALL'}"
                    )
                except Exception as e:
                    log.error(f"[CTRL_RELOAD_MODEL] FAILED: {e}")

            # ── SHM lifecycle: camera restarted ───────────────────────────────
            if cmd == CTRL_CAMERA_RESTARTED:
                camera_id = (msg.get("payload", {}).get("camera_id")
                             or msg.get("camera_id"))
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
        pass   # queue empty – normal exit


# =============================================================================
# Utilities
# =============================================================================

def _load_detector_strict(heartbeat_q, pname, log):
    """
    BUG #1 FIX: Call SETTINGS.load() before constructing PersonDetector so
    the operator's saved model / target_classes are used at startup instead
    of the class-level defaults.
    """
    try:
        from config.loader import SETTINGS
        SETTINGS.load()   # read app_settings.json into memory

        from core.detector import PersonDetector
        det = PersonDetector(
            model_name=SETTINGS.yolo_model,
            conf_threshold=SETTINGS.detection_confidence,
            target_classes=SETTINGS.target_classes,
        )
        if not det.is_model_loaded():
            raise RuntimeError("Model not loaded after PersonDetector()")
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
    """
    CHECK 1 – Both modes confirmed working:
      center  → point_in_polygon(bbox_center(bbox), points)
                Ray-casting on centroid. Fewer false positives.
      overlap → bbox_overlaps_polygon(bbox, points)
                All 4 bbox corners + centroid vs polygon, plus all polygon
                vertices vs bbox rectangle. More sensitive.
    Both use the same ray-casting implementation in core/geometry.py. ✅
    """
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