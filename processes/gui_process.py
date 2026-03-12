# =============================================================================
# processes/gui_process.py  –  Isolated GUI process  (v2 – patched)
# =============================================================================
# FIX #1:  RAM limit raised to 1000 MB
# FIX #8:  GUI emits MSG_SETTINGS_SAVED → supervisor broadcasts CTRL_RELOAD_SETTINGS
#          gui_control_q now passed through to SettingsPage
# FIX #13: setup_process_logger("gui") → logs/gui.log
# =============================================================================

import os
import sys
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.logger import setup_process_logger, get_logger
from utils.resource_guard import ResourceGuard, ResourceLimitExceeded, RAM_LIMIT_GUI
from ipc.messages import (
    make_heartbeat, make_error, make_control,
    MSG_SHUTDOWN, MSG_CONTROL, CTRL_SHUTDOWN, CTRL_RELOAD_CFG,
)

HEARTBEAT_INTERVAL = 5.0


def run_gui_process(
    camera_configs:  List[Dict[str, Any]],
    heartbeat_q:     Queue,
    control_q:       Queue,        # supervisor → GUI
    result_q:        Queue,        # detection  → GUI
    relay_status_q:  Queue,        # relay      → GUI
    det_control_q:   Queue,        # GUI → detection (zone reload)
    ram_limit_mb:    float = RAM_LIMIT_GUI,   # FIX #1: 1000 MB
) -> None:
    pname = "gui"
    setup_process_logger(pname)   # FIX #13 → logs/gui.log
    log = get_logger("Main")
    log.info(f"GUI process started  PID={os.getpid()}")

    guard = ResourceGuard(ram_limit_mb=ram_limit_mb)

    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("Industrial Vision Safety System")
        app.setStyle("Fusion")
        _apply_dark_theme(app)

        from config.loader import ConfigManager, SETTINGS
        SETTINGS.load()

        config_manager = ConfigManager()
        config = config_manager.load()

        app_res = SETTINGS.processing_resolution
        cfg_res = config.processing_resolution
        if app_res != cfg_res:
            log.warning(f"Resolution mismatch {app_res} vs {cfg_res} – rescaling zones")
            config_manager.update_processing_resolution(app_res)
            config_manager.save()

        from ui.main_window import MainWindow
        window = MainWindow(
            config_manager=config_manager,
            camera_configs=camera_configs,
            heartbeat_q=heartbeat_q,
            control_q=control_q,
            result_q=result_q,
            relay_status_q=relay_status_q,
            det_control_q=det_control_q,
            resource_guard=guard,
        )
        window.show()
        sys.exit(app.exec_())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error(f"GUI process fatal: {e}", exc_info=True)
        try:
            heartbeat_q.put_nowait(make_error(pname, str(e), fatal=True))
        except Exception:
            pass
        sys.exit(1)
    finally:
        log.info("GUI process exiting")


def _apply_dark_theme(app) -> None:
    try:
        from PyQt5.QtGui import QPalette, QColor
        palette = QPalette()
        palette.setColor(QPalette.Window,          QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText,      QColor(220, 220, 220))
        palette.setColor(QPalette.Base,            QColor(20, 20, 20))
        palette.setColor(QPalette.AlternateBase,   QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase,     QColor(220, 220, 220))
        palette.setColor(QPalette.ToolTipText,     QColor(220, 220, 220))
        palette.setColor(QPalette.Text,            QColor(220, 220, 220))
        palette.setColor(QPalette.Button,          QColor(50, 50, 50))
        palette.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
        palette.setColor(QPalette.BrightText,      QColor(255, 80, 80))
        palette.setColor(QPalette.Link,            QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight,       QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        app.setPalette(palette)
    except Exception:
        pass
