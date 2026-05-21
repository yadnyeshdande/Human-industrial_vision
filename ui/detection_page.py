# =============================================================================
# ui/detection_page.py  –  Live detection visualization tab  (v5)
# =============================================================================
# FIX #2:  Multi-camera selector panel (left sidebar list)
# FIX #3:  Frame update loop reads SHM, converts to QImage, draws overlays.
# FIX #10: Reads MSG_TELEMETRY from result_q → updates sidebar stats panel.
# FIX #11: All detection results routed per camera_id; only selected cam shown.
# FIX #12: shutdown() method stops timers and detaches shared memory.
#
# v5 changes:
# -----------
# SMOOTH-1 – Render loop 15 fps → 30 fps (67 ms → 33 ms timer).
#   Video smoothness comes from how often update_frame() is called, not from
#   the detector.  Halving the timer interval doubles perceived smoothness at
#   near-zero CPU cost (SHM read is a memcpy, not a GPU call).
#
# SMOOTH-2 – Stale bbox clearing.
#   _last_detection_time[cid] is recorded every time a detection result
#   arrives.  In _update_display(), if (now - last_detection_time) exceeds
#   BBOX_STALE_S (2.5 s), bboxes are cleared via panel.set_persons([]).
#   This stops frozen bounding boxes persisting on a moving scene when the
#   detector is running slowly.
#
# SMOOTH-3 – Real video FPS measurement.
#   _vfps_times[cid] keeps a 3-second rolling window of SHM read timestamps.
#   This gives the true video stream rate independently of the detector.
#   Both rates are shown in the panel overlay:
#       "Cam 1 | 🎥 15.0fps | 🔍 0.3fps  conf=82%"
#   and the sidebar gains a "Video FPS" row above "Detection FPS".
#
# FPS MEANING:
#   🎥 Video FPS  – how fast the camera process is writing frames to SHM
#                   (always fast; reflects network/capture speed).
#   🔍 Detection FPS – how fast the YOLO model processes batches.
#                   0.2-0.3 on slow CPU/GPU is correct and expected.
#   These are two independent pipelines.  Video can be smooth while
#   detection is slow; that is the intended architecture.
# =============================================================================

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from multiprocessing import Queue

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QGroupBox, QScrollArea, QFrame, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QColor, QFont

from config.loader import ConfigManager
from ipc.frame_store import FrameReader
from ipc.messages import (
    MSG_DETECTION_RESULT, MSG_RELAY_STATUS, MSG_RELAY_HEALTH,
    MSG_SYSTEM_HEALTH, MSG_HEARTBEAT, MSG_TELEMETRY,
)
from ui.video_panel import VideoPanel
from utils.logger import get_logger
from utils.time_utils import uptime_str

logger = get_logger("DetectionPage")

# SMOOTH-2: clear bounding boxes if no detection result received in this time
BBOX_STALE_S = 2.5


class DetectionPage(QWidget):
    """
    Live detection display.

    FIX #2:  Left camera-selector panel – click to switch feed.
    FIX #3:  QTimer update loop reads SHM, renders overlays at ~30 fps.
    FIX #10: MSG_TELEMETRY updates the right-side stats panel.
    FIX #11: Per-camera result routing.
    v5:      Video fps measurement, stale bbox clearing, 30 fps render loop.
    """

    def __init__(
        self,
        config_manager:    ConfigManager,
        camera_configs:    List[Dict],
        result_q:          Queue,
        relay_status_q:    Queue,
        system_start_time: float,
        parent=None,
    ):
        super().__init__(parent)
        self.config_manager    = config_manager
        self.camera_configs    = camera_configs
        self.result_q          = result_q
        self.relay_status_q    = relay_status_q
        self.system_start_time = system_start_time

        # FIX #2: currently shown camera
        self._selected_cam_id: Optional[int] = None

        # Keyed by camera_id
        self.frame_readers: Dict[int, FrameReader] = {}
        self.video_panels:  Dict[int, VideoPanel]  = {}

        # Per-camera detection state
        self.latest_bboxes:      Dict[int, list]  = {}
        self.latest_zone_status: Dict[int, dict]  = {}
        self.latest_violations:  Dict[int, list]  = {}
        self.det_fps:            Dict[int, float] = {}

        # SMOOTH-2: timestamp of last detection result per camera
        self._last_detection_time: Dict[int, float] = {}

        # SMOOTH-3: rolling window of SHM read timestamps for video fps
        self._vfps_times: Dict[int, List[float]] = {}
        self._video_fps:  Dict[int, float]        = {}

        # Relay state
        self.relay_states: Dict[int, bool] = {}

        # Stats (FIX #10 – from MSG_TELEMETRY)
        self._telem_fps    = 0.0
        self._telem_vram   = 0.0
        self._telem_util   = 0.0
        self._telem_temp   = 0.0

        # Violation log
        self.violation_log:    List[str] = []
        self._violations_today = 0
        self._MAX_LOG          = 50

        self._setup_ui()
        self._init_camera_readers()     # FIX #3: attach SHM for all cameras
        self._start_timers()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        # ── top bar ──────────────────────────────────────────────────────────
        top = QHBoxLayout()
        self.status_label = QLabel("🟢 Detection Running")
        self.status_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #00cc44;"
        )
        self.uptime_label = QLabel("Uptime: 00:00:00")
        self.uptime_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.ts_label = QLabel()
        self.ts_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        top.addWidget(self.status_label)
        top.addStretch()
        top.addWidget(self.uptime_label)
        top.addWidget(self.ts_label)
        root.addLayout(top)

        # ── splitter: cam selector | video panel | stats sidebar ─────────────
        splitter = QSplitter(Qt.Horizontal)

        # CENTRE built FIRST so _video_area / _video_layout exist before the
        # camera-list widget fires currentItemChanged on its first setCurrentRow(0)
        self._video_area = self._build_video_area()
        splitter.addWidget(self._video_area)

        # LEFT: camera list – may immediately fire _on_camera_selected
        self._cam_list = self._build_camera_list()
        splitter.addWidget(self._cam_list)

        # RIGHT: stats sidebar (FIX #10)
        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 6)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([160, 800, 280])
        root.addWidget(splitter, stretch=1)

        # ── bottom stats bar ─────────────────────────────────────────────────
        self.stats_label = QLabel("Monitoring …")
        self.stats_label.setStyleSheet("color: #666; font-size: 10px;")
        root.addWidget(self.stats_label)

    # FIX #2: left-side camera selector list
    def _build_camera_list(self) -> QWidget:
        box = QGroupBox("Cameras")
        box.setMaximumWidth(180)
        vbox = QVBoxLayout(box)

        self._cam_list_widget = QListWidget()
        self._cam_list_widget.setStyleSheet("""
            QListWidget {
                background: #1a1a1a;
                border: 1px solid #444;
                color: #ddd;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background: #2a6ead;
                color: #fff;
            }
            QListWidget::item:hover {
                background: #2a3a4a;
            }
        """)
        self._cam_list_widget.currentItemChanged.connect(
            self._on_camera_selected
        )
        vbox.addWidget(self._cam_list_widget)

        # Populate list from configs
        known_ids: set = set()
        for cam_cfg in self.camera_configs:
            cid = cam_cfg["id"]
            if cid not in known_ids:
                self._add_cam_list_item(cid)
                known_ids.add(cid)
        for cam in self.config_manager.get_all_cameras():
            if cam.id not in known_ids:
                self._add_cam_list_item(cam.id)
                known_ids.add(cam.id)

        return box

    def _add_cam_list_item(self, camera_id: int) -> None:
        item = QListWidgetItem(f"📷 Camera {camera_id}")
        item.setData(Qt.UserRole, camera_id)
        item.setSizeHint(QSize(160, 36))
        self._cam_list_widget.addItem(item)
        if self._cam_list_widget.count() == 1:
            self._cam_list_widget.setCurrentRow(0)

    # FIX #3: central video panel container
    def _build_video_area(self) -> QWidget:
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        self._video_layout = vbox

        self._main_cam_label = QLabel("Select a camera →")
        self._main_cam_label.setAlignment(Qt.AlignCenter)
        self._main_cam_label.setStyleSheet(
            "color: #888; font-size: 20px; font-weight: bold;"
        )

        self._main_panel: Optional[VideoPanel] = None

        self._panel_placeholder = QWidget()
        ph_layout = QVBoxLayout(self._panel_placeholder)
        ph_layout.addWidget(self._main_cam_label)

        vbox.addWidget(self._panel_placeholder)
        return container

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMaximumWidth(300)
        vbox = QVBoxLayout(sidebar)

        # Relay status
        relay_box = QGroupBox("Relay Status")
        self.relay_layout = QVBoxLayout(relay_box)
        self.relay_status_labels: Dict[int, QLabel] = {}

        self.relay_health_label = QLabel("Relay: initialising...")
        self.relay_health_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.relay_layout.addWidget(self.relay_health_label)

        vbox.addWidget(relay_box)

        # FIX #10 + v5: stats panel with both fps values
        health_box = QGroupBox("Detection Stats")
        hlay = QVBoxLayout(health_box)

        # v5: Video FPS row added above Detection FPS
        self.health_video_fps_lbl = QLabel("Video FPS: —")
        self.health_det_fps_lbl   = QLabel("Detection FPS: —")
        self.health_gpu_vram_lbl  = QLabel("GPU VRAM: —")
        self.health_gpu_util_lbl  = QLabel("GPU Util: —")
        self.health_gpu_temp_lbl  = QLabel("GPU Temp: —")
        self.health_viols_lbl     = QLabel("Violations Today: 0")

        # v5: add a small explanatory note below the fps rows
        fps_note = QLabel(
            "🎥 = camera stream  🔍 = YOLO inference\n"
            "These are independent pipelines."
        )
        fps_note.setStyleSheet("color: #666; font-size: 9px;")
        fps_note.setWordWrap(True)

        for lbl in [
            self.health_video_fps_lbl, self.health_det_fps_lbl,
            fps_note,
            self.health_gpu_vram_lbl, self.health_gpu_util_lbl,
            self.health_gpu_temp_lbl, self.health_viols_lbl,
        ]:
            lbl.setStyleSheet(
                lbl.styleSheet() or "color: #ccc; font-size: 11px;"
            )
            hlay.addWidget(lbl)

        vbox.addWidget(health_box)

        # Violation log
        log_box = QGroupBox("Violation Log")
        llay = QVBoxLayout(log_box)
        self.log_label = QLabel("No violations yet")
        self.log_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_label.setWordWrap(True)
        self.log_label.setStyleSheet("color: #ff8080; font-size: 10px;")
        llay.addWidget(self.log_label)
        vbox.addWidget(log_box)

        vbox.addStretch()
        return sidebar

    # ── SHM init ──────────────────────────────────────────────────────────────

    def _init_camera_readers(self) -> None:
        """FIX #3: attach shared memory for all known cameras."""
        known_ids: set = set()
        for cam_cfg in self.camera_configs:
            cid = cam_cfg["id"]
            res = tuple(cam_cfg.get("resolution", (1280, 720)))
            if cid not in known_ids:
                self._ensure_reader_and_panel(cid, res)
                known_ids.add(cid)
        for cam in self.config_manager.get_all_cameras():
            if cam.id not in known_ids:
                from config.loader import SETTINGS
                res = tuple(SETTINGS.processing_resolution)
                self._ensure_reader_and_panel(cam.id, res)
                known_ids.add(cam.id)

    def _ensure_reader_and_panel(self, cid: int,
                                  res: Tuple[int, int]) -> None:
        if cid not in self.frame_readers:
            reader = FrameReader(camera_id=cid, width=res[0], height=res[1])
            reader.attach()
            self.frame_readers[cid] = reader
        if cid not in self.video_panels:
            panel = VideoPanel(camera_id=cid, processing_resolution=res)
            self.video_panels[cid] = panel
            self._reload_zones_for_camera(cid)

    # ── timers ────────────────────────────────────────────────────────────────

    def _start_timers(self) -> None:
        # SMOOTH-1: 30 fps render loop (was 15 fps / 67 ms in v4)
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_display)
        self._update_timer.start(33)    # 33 ms ≈ 30 fps

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clocks)
        self._clock_timer.start(1000)

    # ── camera selector ───────────────────────────────────────────────────────

    def _on_camera_selected(self, current: Optional[QListWidgetItem],
                             previous: Optional[QListWidgetItem]) -> None:
        """FIX #2: switch main panel to selected camera."""
        if current is None:
            return
        cid = current.data(Qt.UserRole)
        if cid == self._selected_cam_id:
            return
        self._selected_cam_id = cid
        self._swap_main_panel(cid)

    def _swap_main_panel(self, cid: int) -> None:
        """Replace the centre panel widget with the target camera's panel."""
        if not hasattr(self, "_video_layout"):
            return

        from config.loader import SETTINGS
        res = tuple(SETTINGS.processing_resolution)
        self._ensure_reader_and_panel(cid, res)
        panel = self.video_panels[cid]

        layout = self._video_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        layout.addWidget(panel)
        panel.setMinimumHeight(420)
        panel.setVisible(True)
        self._main_panel = panel
        self._reload_zones_for_camera(cid)
        logger.info(f"Detection feed switched to camera {cid}")

    def select_camera(self, cid: int) -> None:
        """External entry point (called from MainWindow shortcut logic)."""
        for row in range(self._cam_list_widget.count()):
            item = self._cam_list_widget.item(row)
            if item.data(Qt.UserRole) == cid:
                self._cam_list_widget.setCurrentRow(row)
                return

    # ── display loop ──────────────────────────────────────────────────────────

    def _update_display(self) -> None:
        """
        FIX #3 + #10 + #11 + v5:
          • Drain result_q and relay_status_q.
          • Read latest SHM frame for selected camera (30 fps).
          • Measure real video fps from SHM read timestamps.
          • Clear stale bboxes if no detection result for > BBOX_STALE_S.
        """
        # ── drain result_q ────────────────────────────────────────────────────
        try:
            while True:
                msg   = self.result_q.get_nowait()
                mtype = msg.get("type", "")

                if mtype == MSG_DETECTION_RESULT:
                    self._handle_detection(msg)

                elif mtype == MSG_TELEMETRY:
                    self._handle_telemetry(msg)

                elif mtype == MSG_SYSTEM_HEALTH:
                    self._handle_system_health(msg)

                elif mtype == MSG_HEARTBEAT:
                    payload = msg.get("payload", {})
                    if "vram_mb" in payload:
                        self._telem_vram = payload.get("vram_mb", 0.0)
                        self._telem_util = payload.get("gpu_util", 0.0)
                        self._telem_temp = payload.get("gpu_temp_c", 0.0)
                        self._refresh_stats_labels()
        except Exception:
            pass

        # ── drain relay_status_q ──────────────────────────────────────────────
        try:
            while True:
                msg   = self.relay_status_q.get_nowait()
                mtype = msg.get("type")

                if mtype == MSG_RELAY_STATUS:
                    rid   = msg["payload"].get("relay_id", 1)
                    state = msg["payload"].get("state", False)
                    self._update_relay_ui(rid, state)

                elif mtype == MSG_RELAY_HEALTH:
                    self._update_relay_health_ui(msg["payload"])

        except Exception:
            pass

        # ── SHM frame read + video fps measurement (SMOOTH-1, SMOOTH-3) ──────
        cid = self._selected_cam_id
        if cid is not None and cid in self.frame_readers:
            reader = self.frame_readers[cid]
            if reader._shm is None:
                reader.attach()
            if reader._shm is not None:
                result = reader.read_if_new()
                if result:
                    frame, _ = result

                    # SMOOTH-3: rolling 3-second window for video fps
                    t_now = time.monotonic()
                    times = self._vfps_times.setdefault(cid, [])
                    times.append(t_now)
                    # drop timestamps older than 3 seconds
                    cutoff = t_now - 3.0
                    while len(times) > 1 and times[0] < cutoff:
                        times.pop(0)
                    vfps = (
                        (len(times) - 1) / max(t_now - times[0], 0.001)
                        if len(times) > 1 else 0.0
                    )
                    self._video_fps[cid] = vfps

                    panel = self.video_panels.get(cid)
                    if panel and panel is self._main_panel:
                        panel.update_frame(frame)

        # ── SMOOTH-2: stale bbox clearing ─────────────────────────────────────
        # If no detection result has arrived for BBOX_STALE_S seconds, clear
        # the bounding boxes so frozen boxes don't persist on a moving scene.
        if cid is not None and self.latest_bboxes.get(cid):
            last_det = self._last_detection_time.get(cid, 0.0)
            if time.time() - last_det > BBOX_STALE_S:
                self.latest_bboxes[cid] = []
                panel = self.video_panels.get(cid)
                if panel and panel is self._main_panel:
                    panel.set_persons([])
                    vfps = self._video_fps.get(cid, 0.0)
                    panel.update_info(
                        f"Cam {cid} | 🎥 {vfps:.1f}fps | 🔍 waiting for detection…"
                    )

        # ── stats bar ─────────────────────────────────────────────────────────
        total_z = sum(len(c.zones) for c in self.config_manager.get_all_cameras())
        total_v = sum(len(v) for v in self.latest_violations.values())
        cam_str = f"Cam {cid}" if cid else "—"
        self.stats_label.setText(
            f"Viewing: {cam_str}  |  "
            f"Cameras: {len(self.frame_readers)}  |  "
            f"Zones: {total_z}  |  "
            f"Active violations: {total_v}"
        )

    # ── message handlers ──────────────────────────────────────────────────────

    def _handle_detection(self, msg: Dict) -> None:
        """FIX #11 + v5: store per-camera; update panel + timestamp."""
        cid     = msg.get("camera_id")
        payload = msg.get("payload", {})

        bboxes      = payload.get("bounding_boxes", [])
        zone_status = payload.get("zone_status", {})
        violations  = payload.get("violations", [])
        fps         = payload.get("fps", 0.0)

        self.latest_bboxes[cid]      = bboxes
        self.latest_zone_status[cid] = zone_status
        self.latest_violations[cid]  = violations
        self.det_fps[cid]            = fps

        # SMOOTH-2: record arrival time so stale-clear logic has a reference
        self._last_detection_time[cid] = time.time()

        # FIX #11: update video panel ONLY for the selected camera
        if cid == self._selected_cam_id:
            panel = self.video_panels.get(cid)
            if panel:
                persons = [tuple(b["bbox"]) for b in bboxes]
                panel.set_persons(persons)
                panel.set_zone_violations(zone_status)

                # SMOOTH-3: show BOTH video fps and detection fps
                vfps     = self._video_fps.get(cid, 0.0)
                max_conf = max(
                    (b.get("confidence", 0.0) for b in bboxes), default=0.0
                )
                conf_str = f" conf={max_conf:.0%}" if bboxes else ""
                viol_str = (
                    f" ⚠ {len(violations)} violation(s)" if violations else ""
                )
                panel.update_info(
                    f"Cam {cid} | 🎥 {vfps:.1f}fps | 🔍 {fps:.1f}fps"
                    f"{conf_str}{viol_str}"
                )

        # Log violations (all cameras)
        for vi in violations:
            ts    = datetime.now().strftime("%H:%M:%S")
            entry = (
                f"[{ts}] Cam {cid} Zone {vi['zone_id']} "
                f"Relay {vi['relay_id']}"
            )
            if entry not in self.violation_log[-5:]:
                self.violation_log.append(entry)
                self._violations_today += 1
                if len(self.violation_log) > self._MAX_LOG:
                    self.violation_log.pop(0)
                self.log_label.setText("\n".join(self.violation_log[-12:]))
                self.health_viols_lbl.setText(
                    f"Violations Today: {self._violations_today}"
                )

        # Update camera list item colour to signal violation
        for row in range(self._cam_list_widget.count()):
            item = self._cam_list_widget.item(row)
            if item.data(Qt.UserRole) == cid:
                if violations:
                    item.setForeground(QColor("#ff4444"))
                    item.setText(f"⚠ Camera {cid}")
                else:
                    item.setForeground(QColor("#dddddd"))
                    item.setText(f"📷 Camera {cid}")
                break

    def _handle_telemetry(self, msg: Dict) -> None:
        """FIX #10 + v5: populate sidebar stats from MSG_TELEMETRY."""
        p = msg.get("payload", {})
        self._telem_fps  = p.get("detection_fps", self._telem_fps)
        self._telem_vram = p.get("gpu_vram_mb",   self._telem_vram)
        self._telem_util = p.get("gpu_util",       self._telem_util)
        self._telem_temp = p.get("gpu_temp_c",     self._telem_temp)
        self._refresh_stats_labels()

    def _handle_system_health(self, msg: Dict) -> None:
        p = msg.get("payload", {})
        self._telem_vram = p.get("vram_mb",    self._telem_vram)
        self._telem_util = p.get("gpu_util",   self._telem_util)
        self._telem_temp = p.get("gpu_temp_c", self._telem_temp)
        self._refresh_stats_labels()

    def _refresh_stats_labels(self) -> None:
        # SMOOTH-3: video fps from the selected camera's rolling window
        cid  = self._selected_cam_id
        vfps = self._video_fps.get(cid, 0.0) if cid is not None else 0.0

        self.health_video_fps_lbl.setText(f"🎥 Video FPS: {vfps:.1f}")
        self.health_det_fps_lbl.setText(
            f"🔍 Detection FPS: {self._telem_fps:.1f}"
        )
        self.health_gpu_vram_lbl.setText(f"GPU VRAM: {self._telem_vram:.0f} MB")
        self.health_gpu_util_lbl.setText(f"GPU Util: {self._telem_util:.0f}%")
        color = "#ff4444" if self._telem_temp >= 80 else "#cccccc"
        self.health_gpu_temp_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.health_gpu_temp_lbl.setText(f"GPU Temp: {self._telem_temp:.0f} °C")

    def _update_relay_ui(self, relay_id: int, state: bool) -> None:
        self.relay_states[relay_id] = state
        if relay_id not in self.relay_status_labels:
            lbl = QLabel()
            lbl.setStyleSheet("font-size: 11px;")
            self.relay_status_labels[relay_id] = lbl
            self.relay_layout.addWidget(lbl)
        lbl   = self.relay_status_labels[relay_id]
        color = "#ff3333" if state else "#33cc33"
        icon  = "🔴" if state else "🟢"
        lbl.setText(f"{icon} Relay {relay_id}: {'ON' if state else 'OFF'}")
        lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _update_relay_health_ui(self, payload: dict) -> None:
        """Update relay health label from MSG_RELAY_HEALTH payload."""
        hw_type   = payload.get("hw_type",   "none")
        connected = payload.get("connected", False)
        channels  = payload.get("channels",  0)
        error     = payload.get("error",     "")

        if hw_type == "none":
            label = "Relay: Simulation"
            color = "#888888"
        elif connected:
            label = f"Relay OK {hw_type.upper()} ({channels} ch)"
            color = "#33cc33"
        else:
            label = f"Relay DISCONNECTED {hw_type.upper()}"
            color = "#ff3333"

        if error:
            label += f" – {error}"

        if hasattr(self, "relay_health_label"):
            self.relay_health_label.setText(label)
            self.relay_health_label.setStyleSheet(
                f"color: {color}; font-size: 11px; font-weight: bold;"
            )

    # ── clocks ────────────────────────────────────────────────────────────────

    def _update_clocks(self) -> None:
        self.uptime_label.setText("Uptime: " + uptime_str(self.system_start_time))
        self.ts_label.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ── zone sync ────────────────────────────────────────────────────────────

    def _reload_zones_for_camera(self, camera_id: int) -> None:
        cam = self.config_manager.get_camera(camera_id)
        if cam is None:
            return
        panel = self.video_panels.get(camera_id)
        if panel is None:
            return
        zone_data = [
            (z.id, z.points, _zone_color(z.relay_id))
            for z in cam.zones
        ]
        panel.set_zones(zone_data)

    def reload_all_zones(self) -> None:
        for cid in list(self.video_panels.keys()):
            self._reload_zones_for_camera(cid)

    # ── graceful shutdown (FIX #12) ───────────────────────────────────────────

    def shutdown(self) -> None:
        """FIX #12: stop timers and detach shared memory cleanly."""
        for attr in ["_update_timer", "_clock_timer"]:
            t = getattr(self, attr, None)
            if t:
                t.stop()
        for reader in self.frame_readers.values():
            try:
                reader.close()
            except Exception:
                pass
        logger.info("DetectionPage shutdown complete")


def _zone_color(relay_id: int) -> Tuple[int, int, int]:
    palette = [
        (0, 255, 0), (0, 255, 255), (255, 0, 255),
        (255, 255, 0), (255, 128, 0), (128, 0, 255),
    ]
    return palette[(relay_id - 1) % len(palette)]