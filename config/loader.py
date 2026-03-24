# =============================================================================
# config/loader.py  –  Configuration management  (v4)
# =============================================================================
# FIX #9: AppSettings now persists last_page_index (0/1/2)
#         so GUI restores its last visible tab after restart.
# =============================================================================

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config.schema import AppConfig, Camera, Zone
from utils.logger import get_logger

logger = get_logger("ConfigLoader")

_BOUNDARIES_FILE = "human_boundaries.json"
_SETTINGS_FILE   = "app_settings.json"


def _atomic_json_write(path: Path, data: dict) -> None:
    """Atomic write: temp → validate → os.replace."""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    json.loads(json_str)   # validate
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json_str)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


# =============================================================================
# AppSettings
# =============================================================================

class AppSettings:
    def __init__(self):
        self.processing_resolution: Tuple[int, int] = (1280, 720)
        self.yolo_model:            str              = "yolov8n.pt"
        self.detection_confidence:  float            = 0.5
        self.violation_mode:        str              = "center"
        self.relay_cooldown:        float            = 5.0
        self.relay_duration:        float            = 1.0
        self.use_usb_relay:         bool             = False
        self.usb_num_channels:      int              = 8
        self.usb_serial:            Optional[str]    = None
        self.frame_queue_size:      int              = 30
        self.ui_update_fps:         int              = 30
        # FIX #9: restore last tab index on GUI restart
        self.last_page_index:       int              = 0

    def load(self, path: str = _SETTINGS_FILE) -> None:
        p = Path(path)
        if not p.exists():
            logger.info(f"No {path} – using defaults")
            return
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            self.processing_resolution = tuple(
                data.get("processing_resolution", list(self.processing_resolution))
            )
            self.yolo_model           = data.get("yolo_model",           self.yolo_model)
            self.detection_confidence = data.get("detection_confidence",  self.detection_confidence)
            self.violation_mode       = data.get("violation_mode",        self.violation_mode)
            self.relay_cooldown       = data.get("relay_cooldown",        self.relay_cooldown)
            self.relay_duration       = data.get("relay_duration",        self.relay_duration)
            self.use_usb_relay        = data.get("use_usb_relay",         self.use_usb_relay)
            self.usb_num_channels     = data.get("usb_num_channels",      self.usb_num_channels)
            self.usb_serial           = data.get("usb_serial",            self.usb_serial)
            self.frame_queue_size     = data.get("frame_queue_size",      self.frame_queue_size)
            self.ui_update_fps        = data.get("ui_update_fps",         self.ui_update_fps)
            # FIX #9
            try:
                self.last_page_index = int(data.get("last_page_index", self.last_page_index))
            except (ValueError, TypeError):
                logger.warning(f"Invalid last_page_index in {path}, using default: {self.last_page_index}")
            logger.info(f"Settings loaded from {path}")
        except Exception as e:
            logger.error(f"Settings load failed: {e} – using defaults")

    def save(self, path: str = _SETTINGS_FILE) -> None:
        from utils.time_utils import now_iso
        data = {
            "timestamp":             now_iso(),
            "processing_resolution": list(self.processing_resolution),
            "yolo_model":            self.yolo_model,
            "detection_confidence":  self.detection_confidence,
            "violation_mode":        self.violation_mode,
            "relay_cooldown":        self.relay_cooldown,
            "relay_duration":        self.relay_duration,
            "use_usb_relay":         self.use_usb_relay,
            "usb_num_channels":      self.usb_num_channels,
            "usb_serial":            self.usb_serial,
            "frame_queue_size":      self.frame_queue_size,
            "ui_update_fps":         self.ui_update_fps,
            "last_page_index":       self.last_page_index,  # FIX #9
        }
        try:
            _atomic_json_write(Path(path), data)
            logger.info(f"Settings saved atomically to {path}")
        except Exception as e:
            logger.error(f"Settings save failed: {e}")
            raise


SETTINGS = AppSettings()


# =============================================================================
# ConfigManager
# =============================================================================

class ConfigManager:
    def __init__(self, path: str = _BOUNDARIES_FILE):
        self._path = Path(path)
        self.config: Optional[AppConfig] = None
        self._next_zone_id = 1

    def load(self) -> AppConfig:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self.config = AppConfig.from_dict(data)
                self._sync_zone_id_counter()
                logger.info(
                    f"Config loaded: {len(self.config.cameras)} cameras, "
                    f"{sum(len(c.zones) for c in self.config.cameras)} zones"
                )
                return self.config
            except Exception as e:
                logger.error(f"Config load failed: {e} – starting empty")
        self.config = AppConfig()
        return self.config

    def save(self) -> bool:
        if self.config is None:
            return False
        try:
            from utils.time_utils import now_iso
            self.config.timestamp = now_iso()
            _atomic_json_write(self._path, self.config.to_dict())
            logger.info(f"Config saved ({self._path})")
            return True
        except Exception as e:
            logger.error(f"Config save failed: {e}")
            return False

    def _sync_zone_id_counter(self) -> None:
        for cam in self.config.cameras:
            for z in cam.zones:
                if z.id >= self._next_zone_id:
                    self._next_zone_id = z.id + 1

    def _used_relay_channels(self) -> Set[int]:
        if self.config is None:
            return set()
        return {z.relay_id for cam in self.config.cameras for z in cam.zones}

    def _next_free_relay_channel(self) -> int:
        used = self._used_relay_channels()
        ch = 1
        while ch in used:
            ch += 1
        return ch

    # ── cameras ──────────────────────────────────────────────────────────────

    def add_camera(self, camera_id: int, rtsp_url: str) -> bool:
        if self.config is None:
            self.load()
        if any(c.id == camera_id for c in self.config.cameras):
            return False
        self.config.cameras.append(Camera(id=camera_id, rtsp_url=rtsp_url))
        logger.info(f"Camera {camera_id} added")
        return True

    def remove_camera(self, camera_id: int) -> bool:
        if self.config is None:
            return False
        before = len(self.config.cameras)
        self.config.cameras = [c for c in self.config.cameras if c.id != camera_id]
        removed = len(self.config.cameras) < before
        if removed:
            logger.info(f"Camera {camera_id} removed")
            self.save()
        return removed

    def get_camera(self, camera_id: int) -> Optional[Camera]:
        if self.config is None:
            return None
        return next((c for c in self.config.cameras if c.id == camera_id), None)

    def get_all_cameras(self) -> List[Camera]:
        return self.config.cameras if self.config else []

    # ── zones ─────────────────────────────────────────────────────────────────

    def add_zone(self, camera_id: int,
                 points: List[Tuple[int, int]]) -> Optional[Zone]:
        camera = self.get_camera(camera_id)
        if camera is None:
            return None
        relay_id = self._next_free_relay_channel()
        zone = Zone(id=self._next_zone_id, points=points, relay_id=relay_id)
        self._next_zone_id += 1
        camera.zones.append(zone)
        logger.info(f"Zone {zone.id} → camera {camera_id} relay_id={relay_id}")
        return zone

    def remove_zone(self, camera_id: int, zone_id: int) -> bool:
        camera = self.get_camera(camera_id)
        if camera is None:
            return False
        before = len(camera.zones)
        camera.zones = [z for z in camera.zones if z.id != zone_id]
        return len(camera.zones) < before

    def update_zone(self, camera_id: int, zone_id: int,
                    points: List[Tuple[int, int]]) -> bool:
        camera = self.get_camera(camera_id)
        if camera is None:
            return False
        for z in camera.zones:
            if z.id == zone_id:
                z.points = points
                return True
        return False

    def update_processing_resolution(self, new_res: Tuple[int, int]) -> None:
        if self.config is None:
            return
        old_w, old_h = self.config.processing_resolution
        new_w, new_h = new_res
        sx, sy = new_w / old_w, new_h / old_h
        for cam in self.config.cameras:
            for z in cam.zones:
                z.points = [(int(x * sx), int(y * sy)) for x, y in z.points]
        self.config.processing_resolution = new_res
        logger.info(f"Resolution updated {(old_w, old_h)} → {new_res}")

    @property
    def processing_resolution(self) -> Tuple[int, int]:
        return self.config.processing_resolution if self.config else (1280, 720)
