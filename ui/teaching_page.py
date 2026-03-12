# =============================================================================
# ui/teaching_page.py  –  Zone teaching / editor tab  (v4)
# =============================================================================
# FIX #1: _load_initial_cameras() now merges BOTH:
#           a) camera_configs passed by supervisor (live processes)
#           b) cameras stored in human_boundaries.json (persistent config)
#         This ensures all cameras appear on GUI restart, even if supervisor
#         passes a stale or partial list.
# =============================================================================

from typing import Dict, List, Optional, Tuple
from multiprocessing import Queue

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QGridLayout, QScrollArea, QMessageBox, QComboBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal
from PyQt5.QtGui import QColor

from config.loader import ConfigManager
from ui.video_panel import VideoPanel
from ui.zone_editor import ZoneEditor
from ipc.frame_store import FrameReader
from ipc.messages import make_control, make_zone_updated, CTRL_RELOAD_CFG
from utils.logger import get_logger

logger = get_logger("TeachingPage")


class TeachingPage(QWidget):
    """Zone drawing and camera management tab."""

    zones_changed = pyqtSignal()

    def __init__(self, config_manager: ConfigManager,
                 camera_configs: List[Dict],
                 det_control_q: Queue,
                 parent=None):
        super().__init__(parent)
        self.config_manager    = config_manager
        self.camera_configs    = camera_configs
        self.det_control_q     = det_control_q
        self.video_panels:      Dict[int, VideoPanel]  = {}
        self.zone_editors:      Dict[int, ZoneEditor]  = {}
        self.panel_containers:  Dict[int, QWidget]     = {}
        self.frame_readers:     Dict[int, FrameReader] = {}

        self._setup_ui()
        self._load_initial_cameras()    # FIX #1: merged load
        self._start_frame_timer()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self.add_cam_btn = QPushButton("+ Add Camera")
        self.add_cam_btn.clicked.connect(self._add_camera_dialog)
        self.save_btn = QPushButton("💾 Save Configuration")
        self.save_btn.clicked.connect(self._save_configuration)
        self.clear_btn = QPushButton("🗑 Clear All Zones")
        self.clear_btn.clicked.connect(self._clear_all_zones)
        for btn in [self.add_cam_btn, self.save_btn, self.clear_btn]:
            toolbar.addWidget(btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.camera_grid = QGridLayout(content)
        self.camera_grid.setSpacing(8)
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        self.status_label = QLabel(
            "Ready – click 'Draw Zone' on a camera to begin"
        )
        self.status_label.setStyleSheet(
            "color: #aaaaaa; font-size: 11px; padding: 4px;"
        )
        layout.addWidget(self.status_label)

    # ── FIX #1 – camera loading ───────────────────────────────────────────────

    def _load_initial_cameras(self) -> None:
        """
        FIX #1: Load cameras from TWO sources and merge them:
          1. self.camera_configs  – what the supervisor passed (active processes)
          2. config_manager       – what human_boundaries.json contains (truth)

        This covers the restart scenario where supervisor may have a stale list
        while the JSON contains all cameras the operator configured.
        """
        from config.loader import SETTINGS
        res = tuple(SETTINGS.processing_resolution)
        loaded_ids: set = set()

        # Source 1: supervisor-provided camera_configs (live RTSP processes)
        for cam_cfg in self.camera_configs:
            cid = cam_cfg["id"]
            url = cam_cfg["rtsp_url"]
            r   = tuple(cam_cfg.get("resolution", res))
            self.config_manager.add_camera(cid, url)   # no-op if exists
            if cid not in loaded_ids:
                self._add_camera_panel(cid, r)
                reader = FrameReader(camera_id=cid, width=r[0], height=r[1])
                reader.attach()
                self.frame_readers[cid] = reader
                loaded_ids.add(cid)

        # Source 2: human_boundaries.json cameras not yet in the panel
        for cam in self.config_manager.get_all_cameras():
            cid = cam.id
            if cid in loaded_ids:
                continue   # already added above
            self._add_camera_panel(cid, res)
            reader = FrameReader(camera_id=cid, width=res[0], height=res[1])
            reader.attach()
            self.frame_readers[cid] = reader
            loaded_ids.add(cid)
            logger.info(f"TeachingPage – restored camera {cid} from config JSON")

        logger.info(
            f"TeachingPage loaded {len(loaded_ids)} cameras: {sorted(loaded_ids)}"
        )

    def _start_frame_timer(self) -> None:
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._update_frames)
        self._frame_timer.start(66)   # ~15 fps

    # ── camera panels ─────────────────────────────────────────────────────────

    def _add_camera_panel(self, camera_id: int,
                          resolution: Tuple[int, int]) -> None:
        from config.loader import SETTINGS

        container = QGroupBox(f"Camera {camera_id}")
        vbox = QVBoxLayout(container)

        panel = VideoPanel(camera_id=camera_id,
                           processing_resolution=resolution)
        panel.setMinimumHeight(280)
        vbox.addWidget(panel)

        editor = ZoneEditor(parent=panel.video_label)
        editor.zone_created.connect(
            lambda pts, cid=camera_id: self._on_zone_created(cid, pts)
        )
        editor.zone_modified.connect(
            lambda zid, pts, cid=camera_id: self._on_zone_modified(cid, zid, pts)
        )
        editor.setGeometry(QRect(0, 0,
                                 panel.video_label.width(),
                                 panel.video_label.height()))

        ctrl = QHBoxLayout()
        draw_btn = QPushButton("✏ Draw Zone")
        draw_btn.clicked.connect(lambda _, e=editor: e.start_drawing())
        del_btn  = QPushButton("✕ Delete Zone")
        del_btn.clicked.connect(
            lambda _, cid=camera_id: self._delete_selected_zone(cid)
        )
        rem_btn  = QPushButton("Remove Camera")
        rem_btn.clicked.connect(
            lambda _, cid=camera_id: self._remove_camera(cid)
        )
        for b in [draw_btn, del_btn, rem_btn]:
            ctrl.addWidget(b)
        ctrl.addStretch()
        vbox.addLayout(ctrl)

        self.video_panels[camera_id]     = panel
        self.zone_editors[camera_id]     = editor
        self.panel_containers[camera_id] = container

        cols = 2
        idx  = len(self.panel_containers) - 1
        self.camera_grid.addWidget(container, idx // cols, idx % cols)

        self._load_zones_for_camera(camera_id)
        logger.info(f"Camera panel {camera_id} added")

    def _load_zones_for_camera(self, camera_id: int) -> None:
        cam = self.config_manager.get_camera(camera_id)
        if cam is None or camera_id not in self.zone_editors:
            return
        editor = self.zone_editors[camera_id]
        panel  = self.video_panels.get(camera_id)

        zone_data = []
        for zone in cam.zones:
            color = self._zone_color(zone.relay_id)          # QColor
            qr, qg, qb = color.red(), color.green(), color.blue()
            zone_data.append((zone.id, zone.points, (qr, qg, qb)))
            editor.add_zone(zone.id, zone.points, color)     # pass color – was missing
        if panel:
            panel.set_zones(zone_data)

    def _sync_zone_display(self, camera_id: int) -> None:
        cam = self.config_manager.get_camera(camera_id)
        if cam is None:
            return
        panel = self.video_panels.get(camera_id)
        if panel is None:
            return
        zone_data = []
        for zone in cam.zones:
            color = self._zone_color(zone.relay_id)
            qr, qg, qb = color.red(), color.green(), color.blue()
            zone_data.append((zone.id, zone.points, (qr, qg, qb)))
        panel.set_zones(zone_data)

    # ── frame update ─────────────────────────────────────────────────────────

    def _update_frames(self) -> None:
        for cid, reader in self.frame_readers.items():
            if reader._shm is None:
                reader.attach()
                continue
            result = reader.read_if_new()
            if result:
                frame, _ = result
                panel = self.video_panels.get(cid)
                if panel:
                    panel.update_frame(frame)

            # Keep zone editor overlay sized to video label
            editor = self.zone_editors.get(cid)
            panel  = self.video_panels.get(cid)
            if editor and panel:
                label  = panel.video_label
                target = QRect(0, 0, label.width(), label.height())
                if editor.geometry() != target:
                    editor.setGeometry(target)

            # Info text
            cam = self.config_manager.get_camera(cid)
            zone_count = len(cam.zones) if cam else 0
            p = self.video_panels.get(cid)
            if p:
                p.update_info(
                    f"Camera {cid} | Teaching Mode | Zones: {zone_count}"
                )

    # ── zone events ───────────────────────────────────────────────────────────

    def _on_zone_created(self, camera_id: int,
                         points: List[Tuple[int, int]]) -> None:
        panel = self.video_panels.get(camera_id)
        if panel:
            proc_pts = [panel.widget_to_processing(x, y) for x, y in points]
        else:
            proc_pts = points

        zone = self.config_manager.add_zone(camera_id, proc_pts)
        if zone:
            self._sync_zone_display(camera_id)
            self._load_zones_for_camera(camera_id)
            self.status_label.setText(
                f"Zone {zone.id} created on Camera {camera_id} "
                f"→ Relay {zone.relay_id}"
            )
            self.zones_changed.emit()

    def _on_zone_modified(self, camera_id: int, zone_id: int,
                          points: List[Tuple[int, int]]) -> None:
        panel = self.video_panels.get(camera_id)
        if panel:
            proc_pts = [panel.widget_to_processing(x, y) for x, y in points]
        else:
            proc_pts = points
        if self.config_manager.update_zone(camera_id, zone_id, proc_pts):
            self._sync_zone_display(camera_id)
            self.zones_changed.emit()

    def _delete_selected_zone(self, camera_id: int) -> None:
        editor = self.zone_editors.get(camera_id)
        if not editor:
            return
        zid = editor.selected_zone_id
        if zid is not None:
            self.config_manager.remove_zone(camera_id, zid)
            editor.remove_zone(zid)
            self._sync_zone_display(camera_id)
            self.status_label.setText(f"Zone {zid} deleted")
            self.zones_changed.emit()

    def _clear_all_zones(self) -> None:
        reply = QMessageBox.question(
            self, "Clear All Zones",
            "Remove ALL zones from ALL cameras?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            for cam in self.config_manager.get_all_cameras():
                cam.zones.clear()
            for cid in self.zone_editors:
                self.zone_editors[cid].clear_zones()
                self._sync_zone_display(cid)
            self.status_label.setText("All zones cleared")
            self.zones_changed.emit()

    def _remove_camera(self, camera_id: int) -> None:
        reply = QMessageBox.question(
            self, "Remove Camera",
            f"Remove camera {camera_id} and all its zones?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.config_manager.remove_camera(camera_id)
            if camera_id in self.panel_containers:
                w = self.panel_containers.pop(camera_id)
                self.camera_grid.removeWidget(w)
                w.deleteLater()
            self.video_panels.pop(camera_id, None)
            self.zone_editors.pop(camera_id, None)
            if camera_id in self.frame_readers:
                self.frame_readers[camera_id].close()
                self.frame_readers.pop(camera_id)
            self.zones_changed.emit()

    def _add_camera_dialog(self) -> None:
        from PyQt5.QtWidgets import QInputDialog
        url, ok = QInputDialog.getText(
            self, "Add Camera",
            "Enter RTSP URL:\n(e.g. rtsp://admin:pass@192.168.1.x:554/stream)"
        )
        if ok and url.strip():
            cid = max((c.id for c in self.config_manager.get_all_cameras()),
                      default=0) + 1
            from config.loader import SETTINGS
            res = tuple(SETTINGS.processing_resolution)
            self.config_manager.add_camera(cid, url.strip())
            self._add_camera_panel(cid, res)
            reader = FrameReader(camera_id=cid, width=res[0], height=res[1])
            reader.attach()
            self.frame_readers[cid] = reader
            self.status_label.setText(
                f"Camera {cid} added.  Restart supervisor to activate it."
            )

    # ── save ─────────────────────────────────────────────────────────────────

    def _save_configuration(self) -> None:
        if self.config_manager.save():
            try:
                self.det_control_q.put_nowait(make_zone_updated("gui"))
            except Exception:
                pass
            self.status_label.setText(
                "✅ Configuration saved – detection workers reloading zones"
            )
            QMessageBox.information(self, "Saved", "Configuration saved successfully.")
        else:
            QMessageBox.warning(self, "Error", "Failed to save configuration.")

    # ── shutdown ─────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """FIX #12: graceful teardown – stop timer and release SHM."""
        if hasattr(self, "_frame_timer"):
            self._frame_timer.stop()
        for reader in self.frame_readers.values():
            try:
                reader.close()
            except Exception:
                pass

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _zone_color(relay_id: int) -> QColor:
        colors = [
            QColor(0, 255, 0), QColor(0, 255, 255), QColor(255, 0, 255),
            QColor(255, 255, 0), QColor(255, 128, 0), QColor(128, 0, 255),
        ]
        return colors[(relay_id - 1) % len(colors)]
