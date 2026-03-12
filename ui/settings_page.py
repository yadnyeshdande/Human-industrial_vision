# =============================================================================
# ui/settings_page.py  –  Settings tab  (v4 – unchanged logic, updated refs)
# =============================================================================

import json
from pathlib import Path
from typing import Optional
from multiprocessing import Queue

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QMessageBox,
)
from PyQt5.QtCore import Qt

from config.loader import ConfigManager, SETTINGS
from ipc.messages import make_settings_saved
from utils.logger import get_logger

logger = get_logger("SettingsPage")


class SettingsPage(QWidget):
    def __init__(self, config_manager: ConfigManager,
                 heartbeat_q: Optional[Queue] = None,
                 parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.heartbeat_q    = heartbeat_q
        self._setup_ui()
        self._load_into_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        det_box  = QGroupBox("Detection Settings")
        det_form = QFormLayout(det_box)
        self.model_combo = QComboBox()
        self._refresh_local_models()
        det_form.addRow("YOLO Model:", self.model_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        det_form.addRow("Confidence Threshold:", self.conf_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["center", "overlap"])
        det_form.addRow("Violation Mode:", self.mode_combo)

        self.res_combo = QComboBox()
        for r in ["640x360", "960x540", "1280x720", "1920x1080"]:
            self.res_combo.addItem(r)
        det_form.addRow("Processing Resolution:", self.res_combo)
        layout.addWidget(det_box)

        relay_box  = QGroupBox("Relay Settings")
        relay_form = QFormLayout(relay_box)
        self.use_relay_chk = QCheckBox("Use USB Relay")
        relay_form.addRow(self.use_relay_chk)

        self.relay_channels_spin = QSpinBox()
        self.relay_channels_spin.setRange(1, 16)
        relay_form.addRow("Relay Channels:", self.relay_channels_spin)

        self.relay_serial_edit = QLineEdit()
        self.relay_serial_edit.setPlaceholderText("Serial (optional)")
        relay_form.addRow("USB Serial:", self.relay_serial_edit)

        self.relay_cooldown_spin = QDoubleSpinBox()
        self.relay_cooldown_spin.setRange(0.5, 60.0)
        self.relay_cooldown_spin.setSingleStep(0.5)
        self.relay_cooldown_spin.setDecimals(1)
        relay_form.addRow("Relay Cooldown (s):", self.relay_cooldown_spin)

        self.relay_duration_spin = QDoubleSpinBox()
        self.relay_duration_spin.setRange(0.1, 10.0)
        self.relay_duration_spin.setSingleStep(0.1)
        self.relay_duration_spin.setDecimals(1)
        relay_form.addRow("Activation Duration (s):", self.relay_duration_spin)
        layout.addWidget(relay_box)

        btn_row = QHBoxLayout()
        save_btn  = QPushButton("💾 Save Settings")
        reset_btn = QPushButton("↩ Reset Defaults")
        save_btn.clicked.connect(self._save_settings)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(reset_btn)
        layout.addLayout(btn_row)

        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "color: #66cc66; font-size: 11px; padding: 4px;"
        )
        layout.addWidget(self.info_label)

        model_box   = QGroupBox("Model Management")
        model_form  = QFormLayout(model_box)
        refresh_btn = QPushButton("🔄 Refresh model list")
        refresh_btn.clicked.connect(self._refresh_local_models)
        model_note  = QLabel(
            "⚠  Place .pt files in the models/ folder.\n"
            "Models are NEVER downloaded automatically."
        )
        model_note.setStyleSheet("color: #ffaa44; font-size: 10px;")
        model_form.addRow(refresh_btn)
        model_form.addRow(model_note)
        layout.addWidget(model_box)
        layout.addStretch()

    def _load_into_ui(self) -> None:
        idx = self.model_combo.findText(SETTINGS.yolo_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.conf_spin.setValue(SETTINGS.detection_confidence)
        idx = self.mode_combo.findText(SETTINGS.violation_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        w, h    = SETTINGS.processing_resolution
        res_str = f"{w}x{h}"
        idx     = self.res_combo.findText(res_str)
        if idx >= 0:
            self.res_combo.setCurrentIndex(idx)
        self.use_relay_chk.setChecked(SETTINGS.use_usb_relay)
        self.relay_channels_spin.setValue(SETTINGS.usb_num_channels)
        self.relay_serial_edit.setText(SETTINGS.usb_serial or "")
        self.relay_cooldown_spin.setValue(SETTINGS.relay_cooldown)
        self.relay_duration_spin.setValue(SETTINGS.relay_duration)

    def _save_settings(self) -> None:
        SETTINGS.yolo_model           = self.model_combo.currentText()
        SETTINGS.detection_confidence = self.conf_spin.value()
        SETTINGS.violation_mode       = self.mode_combo.currentText()
        w, h = map(int, self.res_combo.currentText().split("x"))
        old_res = SETTINGS.processing_resolution
        SETTINGS.processing_resolution = (w, h)
        SETTINGS.use_usb_relay    = self.use_relay_chk.isChecked()
        SETTINGS.usb_num_channels = self.relay_channels_spin.value()
        serial = self.relay_serial_edit.text().strip()
        SETTINGS.usb_serial     = serial if serial else None
        SETTINGS.relay_cooldown = self.relay_cooldown_spin.value()
        SETTINGS.relay_duration = self.relay_duration_spin.value()
        try:
            SETTINGS.save()
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            return
        if (w, h) != old_res:
            self.config_manager.update_processing_resolution((w, h))
            self.config_manager.save()
            self.info_label.setText(
                "✅ Saved. Resolution changed – zones rescaled. Restart required."
            )
        else:
            self.info_label.setText(
                "✅ Settings saved – broadcasting reload to processes."
            )
        if self.heartbeat_q is not None:
            try:
                self.heartbeat_q.put_nowait(make_settings_saved("gui"))
            except Exception:
                pass
        logger.info("Settings saved")

    def _reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            from config.loader import AppSettings
            defaults = AppSettings()
            SETTINGS.__dict__.update(defaults.__dict__)
            self._load_into_ui()
            self.info_label.setText("Settings reset to defaults (not saved)")

    def _refresh_local_models(self) -> None:
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        pts = sorted(p.name for p in models_dir.glob("*.pt"))
        current = self.model_combo.currentText() if self.model_combo.count() else ""
        self.model_combo.clear()
        if pts:
            self.model_combo.addItems(pts)
        else:
            self.model_combo.addItem("(no models found in models/)")
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
