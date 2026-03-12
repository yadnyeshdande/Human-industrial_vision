# =============================================================================
# ui/main_window.py  –  Main application window  (v4)
# =============================================================================
# FIX #8:  "D" key shortcut → switch to Detection Mode tab.
# FIX #9:  On startup: restore last tab from SETTINGS.last_page_index.
#          On tab change: persist new index to SETTINGS (saved lazily).
# FIX #12: closeEvent() stops all child page timers and detaches SHM.
# =============================================================================

import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QStatusBar, QLabel, QMessageBox, QAction
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QFont
# FIX #8: QShortcut
from PyQt5.QtWidgets import QShortcut

from config.loader import ConfigManager, SETTINGS
from ipc.messages import make_heartbeat, make_control, CTRL_RELOAD_CFG
from utils.resource_guard import ResourceGuard, ResourceLimitExceeded
from utils.time_utils import uptime_str
from utils.logger import get_logger

logger = get_logger("MainWindow")

TAB_TEACHING  = 0
TAB_DETECTION = 1
TAB_SETTINGS  = 2


class MainWindow(QMainWindow):
    """Main application window – visualization only."""

    def __init__(
        self,
        config_manager:  ConfigManager,
        camera_configs:  List[Dict],
        heartbeat_q:     Queue,
        control_q:       Queue,
        result_q:        Queue,
        relay_status_q:  Queue,
        det_control_q:   Queue,
        resource_guard:  ResourceGuard,
        parent=None,
    ):
        super().__init__(parent)
        self.config_manager   = config_manager
        self.camera_configs   = camera_configs
        self.heartbeat_q      = heartbeat_q
        self.control_q        = control_q
        self.result_q         = result_q
        self.relay_status_q   = relay_status_q
        self.det_control_q    = det_control_q
        self.resource_guard   = resource_guard
        self._start_time      = time.time()
        self._last_hb         = 0.0

        self.setWindowTitle("Industrial Vision Safety System  v4.0")
        self.setMinimumSize(1280, 800)

        self._setup_pages()
        self._setup_menu()
        self._setup_shortcuts()    # FIX #8
        self._setup_status_bar()
        self._start_timers()
        self._restore_last_page()  # FIX #9

        logger.info("MainWindow initialized")

    # ── pages ─────────────────────────────────────────────────────────────────

    def _setup_pages(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        from ui.teaching_page import TeachingPage
        self.teaching_page = TeachingPage(
            config_manager=self.config_manager,
            camera_configs=self.camera_configs,
            det_control_q=self.det_control_q,
        )
        self.teaching_page.zones_changed.connect(self._on_zones_changed)
        self.tabs.addTab(self.teaching_page, "🎨  Teaching Mode")

        from ui.detection_page import DetectionPage
        self.detection_page = DetectionPage(
            config_manager=self.config_manager,
            camera_configs=self.camera_configs,
            result_q=self.result_q,
            relay_status_q=self.relay_status_q,
            system_start_time=self._start_time,
        )
        self.tabs.addTab(self.detection_page, "🔍  Detection Mode")

        from ui.settings_page import SettingsPage
        self.settings_page = SettingsPage(
            config_manager=self.config_manager,
            heartbeat_q=self.heartbeat_q,
        )
        self.tabs.addTab(self.settings_page, "⚙  Settings")

        self.setCentralWidget(self.tabs)

    # ── menu ─────────────────────────────────────────────────────────────────

    def _setup_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        save_act  = QAction("Save Configuration", self)
        save_act.setShortcut(QKeySequence.Save)
        save_act.triggered.connect(self.teaching_page._save_configuration)
        file_menu.addAction(save_act)
        file_menu.addSeparator()
        exit_act  = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        view_menu = mb.addMenu("View")
        for idx, (label, shortcut) in enumerate([
            ("Teaching Mode",  "Ctrl+1"),
            ("Detection Mode", "Ctrl+2"),
            ("Settings",       "Ctrl+3"),
        ]):
            act = QAction(label, self)
            act.setShortcut(shortcut)
            act.triggered.connect(lambda _, i=idx: self.tabs.setCurrentIndex(i))
            view_menu.addAction(act)

        sys_menu   = mb.addMenu("System")
        reload_act = QAction("Reload Zone Config", self)
        reload_act.triggered.connect(self._send_reload_config)
        sys_menu.addAction(reload_act)

        help_menu  = mb.addMenu("Help")
        about_act  = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # FIX #8: keyboard shortcuts
    def _setup_shortcuts(self) -> None:
        # "D" → jump to Detection Mode
        d_sc = QShortcut(QKeySequence("D"), self)
        d_sc.activated.connect(self.show_detection_page)

        # Standard numeric shortcuts (mirror menu)
        for key, idx in [("1", 0), ("2", 1), ("3", 2)]:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda _, i=idx: self.tabs.setCurrentIndex(i))

    def show_detection_page(self) -> None:
        """FIX #8: switch to Detection Mode."""
        self.tabs.setCurrentIndex(TAB_DETECTION)

    # ── status bar ────────────────────────────────────────────────────────────

    def _setup_status_bar(self) -> None:
        self.statusBar().showMessage("System ready")
        self._sb_uptime = QLabel("Uptime: 00:00:00")
        self._sb_uptime.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self._sb_health = QLabel("Health: OK")
        self._sb_health.setStyleSheet("color: #00cc44; font-size: 10px;")
        self.statusBar().addPermanentWidget(self._sb_health)
        self.statusBar().addPermanentWidget(self._sb_uptime)

    # ── timers ────────────────────────────────────────────────────────────────

    def _start_timers(self) -> None:
        self._hb_timer = QTimer(self)
        self._hb_timer.timeout.connect(self._tick)
        self._hb_timer.start(200)

        self._ctrl_timer = QTimer(self)
        self._ctrl_timer.timeout.connect(self._drain_control)
        self._ctrl_timer.start(200)

        self._sb_timer = QTimer(self)
        self._sb_timer.timeout.connect(self._update_status_bar)
        self._sb_timer.start(1000)

    def _tick(self) -> None:
        now = time.monotonic()
        if now - self._last_hb >= 5.0:
            self._last_hb = now
            try:
                self.heartbeat_q.put_nowait(
                    make_heartbeat(
                        source="gui",
                        ram_mb=self.resource_guard.get_ram_mb(),
                    )
                )
            except Exception:
                pass
            try:
                self.resource_guard.check()
            except ResourceLimitExceeded as e:
                logger.error(f"GUI resource limit: {e}")
                self.close()

    def _drain_control(self) -> None:
        try:
            while True:
                msg   = self.control_q.get_nowait()
                mtype = msg.get("type", "")
                cmd   = msg.get("payload", {}).get("command", "")
                if mtype == "shutdown" or cmd == "shutdown":
                    logger.info("Shutdown from supervisor")
                    self.close()
                    return
                elif mtype == "control" and cmd == CTRL_RELOAD_CFG:
                    self.detection_page.reload_all_zones()
        except Exception:
            pass

    def _update_status_bar(self) -> None:
        self._sb_uptime.setText("Uptime: " + uptime_str(self._start_time))

    # ── FIX #9: last-page restore ─────────────────────────────────────────────

    def _restore_last_page(self) -> None:
        """FIX #9: on startup, set the tab to the last recorded page."""
        idx = getattr(SETTINGS, "last_page_index", 0)
        if 0 <= idx < self.tabs.count():
            self.tabs.setCurrentIndex(idx)
            logger.info(f"Restored last page: {idx}")

    # ── tab change ────────────────────────────────────────────────────────────

    def _on_tab_changed(self, index: int) -> None:
        if index == TAB_DETECTION:
            self.detection_page.reload_all_zones()

        # FIX #9: persist tab change to settings (lazy – no file write on every click)
        SETTINGS.last_page_index = index
        logger.debug(f"Tab changed → {index}")

    def _on_zones_changed(self) -> None:
        self.detection_page.reload_all_zones()

    def _send_reload_config(self) -> None:
        try:
            self.det_control_q.put_nowait(make_control("gui", CTRL_RELOAD_CFG))
            self.statusBar().showMessage("Zone reload sent to detection", 3000)
        except Exception as e:
            logger.error(f"Reload signal failed: {e}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About",
            "<h2>Industrial Vision Safety System</h2>"
            "<p><b>Version 4.0.0</b> – Multi-Camera Industrial Architecture</p>"
            "<ul>"
            "<li>Multi-camera detection with per-camera selector</li>"
            "<li>Process-isolated supervisor architecture</li>"
            "<li>Shared-memory zero-copy frame transfer</li>"
            "<li>GPU-accelerated YOLO (FP16) with telemetry</li>"
            "<li>24/7 continuous operation ready</li>"
            "<li>Keyboard shortcut: D → Detection Mode</li>"
            "</ul>"
        )

    # ── FIX #12: graceful shutdown ────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """FIX #12: stop all timers, detach SHM, then close."""
        logger.info("GUI window closing – cleaning up")

        # Save last page before exiting
        try:
            SETTINGS.last_page_index = self.tabs.currentIndex()
            SETTINGS.save()
        except Exception:
            pass

        # Stop main-window timers
        for attr in ["_hb_timer", "_ctrl_timer", "_sb_timer"]:
            t = getattr(self, attr, None)
            if t and t.isActive():
                t.stop()

        # FIX #12: call shutdown() on each page to stop their timers + SHM
        for page in [self.detection_page, self.teaching_page]:
            try:
                page.shutdown()
            except Exception:
                pass

        logger.info("GUI window closed cleanly")
        super().closeEvent(event)
