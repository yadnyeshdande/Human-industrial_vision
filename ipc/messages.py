# =============================================================================
# ipc/messages.py  –  Typed inter-process message protocol  (v4)
# =============================================================================
# NEW: MSG_TELEMETRY – detection → result_q → GUI sidebar (FIX #10)
# =============================================================================

import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
MSG_HEARTBEAT        = "heartbeat"
MSG_FRAME_READY      = "frame_ready"
MSG_DETECTION_RESULT = "detection_result"
MSG_RELAY_COMMAND    = "relay_command"
MSG_RELAY_STATUS     = "relay_status"
MSG_CONTROL          = "control"
MSG_ERROR            = "error"
MSG_CONFIG_RELOAD    = "config_reload"
MSG_SHUTDOWN         = "shutdown"
MSG_STATUS           = "status"
MSG_RESTART_ACK      = "restart_ack"
MSG_ZONE_UPDATED     = "zone_config_updated"
MSG_SETTINGS_SAVED   = "settings_saved"
MSG_SYSTEM_HEALTH    = "system_health"
MSG_TELEMETRY        = "telemetry"          # FIX #10: detection → GUI sidebar

# Control sub-commands
CTRL_SHUTDOWN        = "shutdown"
CTRL_RELOAD_CFG      = "reload_config"
CTRL_SOFT_RESET      = "soft_reset"
CTRL_PING            = "ping"
CTRL_RELOAD_SETTINGS = "reload_settings"
CTRL_CAMERA_RESTARTED = "camera_restarted"   # SHM LIFECYCLE: supervisor → detection


def _base(msg_type, source, camera_id=None):
    return {
        "type":      msg_type,
        "source":    source,
        "camera_id": camera_id,
        "timestamp": time.time(),
        "payload":   {},
    }


def make_heartbeat(source, camera_id=None, fps=0.0, ram_mb=0.0, extra=None):
    msg = _base(MSG_HEARTBEAT, source, camera_id)
    msg["payload"] = {"fps": fps, "ram_mb": ram_mb}
    if extra:
        msg["payload"].update(extra)
    return msg


def make_frame_ready(source, camera_id, shm_name, frame_counter):
    msg = _base(MSG_FRAME_READY, source, camera_id)
    msg["payload"] = {"shm_name": shm_name, "frame_counter": frame_counter}
    return msg


def make_detection_result(source, camera_id, persons, violations, fps,
                           frame_counter, bounding_boxes=None, zone_status=None):
    """Enriched detection result with bounding_boxes, zone_status."""
    msg = _base(MSG_DETECTION_RESULT, source, camera_id)
    msg["payload"] = {
        "persons":            persons,
        "violations":         violations,
        "fps":                fps,
        "frame_counter":      frame_counter,
        "bounding_boxes":     bounding_boxes or [
            {"bbox": list(p), "label": "person", "confidence": 1.0}
            for p in persons
        ],
        "zone_status":        zone_status or {v["zone_id"]: True for v in violations},
        "detection_timestamp": time.time(),
    }
    return msg


def make_telemetry(source, detection_fps, gpu_vram_mb, gpu_util_pct,
                   gpu_temp_c, ram_mb, cameras_active):
    """
    FIX #10: Published by detection worker → result_q → GUI sidebar.
    Decoupled from heartbeat so supervisor is not overloaded.
    """
    msg = _base(MSG_TELEMETRY, source)
    msg["payload"] = {
        "detection_fps":  round(float(detection_fps), 2),
        "gpu_vram_mb":    round(float(gpu_vram_mb),   1),
        "gpu_util":       round(float(gpu_util_pct),  1),
        "gpu_temp_c":     round(float(gpu_temp_c),    1),
        "ram_mb":         round(float(ram_mb),         1),
        "cameras_active": list(cameras_active),
    }
    return msg


def make_relay_command(source, relay_id, camera_id, zone_id, action="trigger"):
    msg = _base(MSG_RELAY_COMMAND, source, camera_id)
    msg["payload"] = {"relay_id": relay_id, "zone_id": zone_id, "action": action}
    return msg


def make_relay_status(source, relay_id, state, reason=""):
    msg = _base(MSG_RELAY_STATUS, source)
    msg["payload"] = {"relay_id": relay_id, "state": state, "reason": reason}
    return msg


def make_control(source, command, target=None, data=None):
    msg = _base(MSG_CONTROL, source)
    msg["payload"] = {"command": command, "target": target, "data": data or {}}
    return msg


def make_error(source, error, camera_id=None, fatal=False):
    msg = _base(MSG_ERROR, source, camera_id)
    msg["payload"] = {"error": error, "fatal": fatal}
    return msg


def make_status(source, data):
    msg = _base(MSG_STATUS, source)
    msg["payload"] = data
    return msg


def make_zone_updated(source):
    """FIX #12: notify detection workers that zone config changed."""
    return _base(MSG_ZONE_UPDATED, source)


def make_settings_saved(source):
    """GUI → supervisor: settings file updated."""
    return _base(MSG_SETTINGS_SAVED, source)


def make_system_health(source, data):
    msg = _base(MSG_SYSTEM_HEALTH, source)
    msg["payload"] = data
    return msg


def make_camera_restarted(source: str, camera_id: int) -> dict:
    """
    SHM LIFECYCLE FIX: Supervisor → detection workers.
    Tells detection to call reader.reattach(camera_id) once so the stale
    OS handle is released and a fresh one is opened.
    """
    msg = _base(MSG_CONTROL, source, camera_id)
    msg["payload"] = {
        "command":   CTRL_CAMERA_RESTARTED,
        "camera_id": camera_id,
    }
    return msg
