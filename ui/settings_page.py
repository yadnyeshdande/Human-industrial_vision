# =============================================================================
# ui/settings_page.py  –  Settings tab  (v5 – dual relay backend selector)
# =============================================================================
#
# Changes from v4:
#   • Relay type selector: radio group  None | USB | Ethernet
#   • USB sub-panel visible only when USB selected
#   • Ethernet sub-panel (IP, Port, Device ID, Channels) for Ethernet only
#   • Saves SETTINGS.relay_type + eth_relay_* fields
#   • Existing CTRL_RELOAD_SETTINGS broadcast path unchanged
#   • GUI never touches relay hardware – only writes config
# =============================================================================

from pathlib import Path
from typing import Optional
from multiprocessing import Queue

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QMessageBox,
    QRadioButton, QButtonGroup,
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

    # -------------------------------------------------------------------------
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Detection settings
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

        # Relay settings
        relay_box = QGroupBox("Relay Settings")
        rb_layout = QVBoxLayout(relay_box)

        # Radio buttons – relay type selector
        type_label = QLabel("Relay Backend:")
        type_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        rb_layout.addWidget(type_label)

        radio_row = QHBoxLayout()
        self._relay_btn_group = QButtonGroup(self)

        self._rb_none = QRadioButton("None (Simulation)")
        self._rb_usb  = QRadioButton("USB Relay")
        self._rb_eth  = QRadioButton("Ethernet Relay (Modbus TCP)")

        for i, rb in enumerate([self._rb_none, self._rb_usb, self._rb_eth]):
            self._relay_btn_group.addButton(rb, i)
            radio_row.addWidget(rb)
        radio_row.addStretch()
        rb_layout.addLayout(radio_row)

        info_lbl = QLabel(
            "Only ONE backend is active at a time.  "
            "Switch here as contingency if one type fails."
        )
        info_lbl.setStyleSheet("color: #ffaa44; font-size: 10px;")
        rb_layout.addWidget(info_lbl)

        # Shared timing (always visible)
        shared_form = QFormLayout()

        self.relay_cooldown_spin = QDoubleSpinBox()
        self.relay_cooldown_spin.setRange(0.5, 60.0)
        self.relay_cooldown_spin.setSingleStep(0.5)
        self.relay_cooldown_spin.setDecimals(1)
        shared_form.addRow("Relay Cooldown (s):", self.relay_cooldown_spin)

        self.relay_duration_spin = QDoubleSpinBox()
        self.relay_duration_spin.setRange(0.1, 10.0)
        self.relay_duration_spin.setSingleStep(0.1)
        self.relay_duration_spin.setDecimals(1)
        shared_form.addRow("Activation Duration (s):", self.relay_duration_spin)

        rb_layout.addLayout(shared_form)

        # USB sub-panel
        self._usb_panel = QGroupBox("USB Relay Configuration")
        usb_form = QFormLayout(self._usb_panel)

        self.relay_channels_spin = QSpinBox()
        self.relay_channels_spin.setRange(1, 16)
        usb_form.addRow("Relay Channels:", self.relay_channels_spin)

        self.relay_serial_edit = QLineEdit()
        self.relay_serial_edit.setPlaceholderText(
            "Serial (optional – leave blank for any)"
        )
        usb_form.addRow("USB Serial:", self.relay_serial_edit)

        rb_layout.addWidget(self._usb_panel)

        # Ethernet sub-panel
        self._eth_panel = QGroupBox(
            "Ethernet Relay Configuration (Waveshare Modbus TCP)"
        )
        eth_form = QFormLayout(self._eth_panel)

        self.eth_ip_edit = QLineEdit()
        self.eth_ip_edit.setPlaceholderText("e.g. 192.168.1.200")
        eth_form.addRow("IP Address:", self.eth_ip_edit)

        self.eth_port_spin = QSpinBox()
        self.eth_port_spin.setRange(1, 65535)
        self.eth_port_spin.setValue(502)
        eth_form.addRow("Modbus Port:", self.eth_port_spin)

        self.eth_device_id_spin = QSpinBox()
        self.eth_device_id_spin.setRange(1, 255)
        self.eth_device_id_spin.setValue(1)
        eth_form.addRow("Device / Slave ID:", self.eth_device_id_spin)

        self.eth_channels_spin = QSpinBox()
        self.eth_channels_spin.setRange(1, 32)
        self.eth_channels_spin.setValue(16)
        eth_form.addRow("Relay Channels:", self.eth_channels_spin)

        eth_note = QLabel(
            "On connect/reconnect all relays cycle ON then OFF as self-test.\n"
            "Heartbeat checks every 10 s. Unplugged cable -> relay process\n"
            "marks hardware dead and stops sending false outputs."
        )
        eth_note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        eth_form.addRow(eth_note)

        rb_layout.addWidget(self._eth_panel)
        layout.addWidget(relay_box)

        # Wire radio buttons -> show/hide sub-panels
        self._relay_btn_group.buttonClicked.connect(self._on_relay_type_changed)

        # Save / Reset
        btn_row = QHBoxLayout()
        save_btn  = QPushButton("Save Settings")
        reset_btn = QPushButton("Reset Defaults")
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

        # Model management
        model_box   = QGroupBox("Model Management")
        model_form  = QFormLayout(model_box)
        refresh_btn = QPushButton("Refresh model list")
        refresh_btn.clicked.connect(self._refresh_local_models)
        model_note  = QLabel(
            "Place .pt files in the models/ folder.\n"
            "Models are NEVER downloaded automatically."
        )
        model_note.setStyleSheet("color: #ffaa44; font-size: 10px;")
        model_form.addRow(refresh_btn)
        model_form.addRow(model_note)
        layout.addWidget(model_box)
        layout.addStretch()

    # -------------------------------------------------------------------------
    def _on_relay_type_changed(self, _button=None) -> None:
        rb = self._relay_btn_group.checkedId()
        self._usb_panel.setVisible(rb == 1)   # 1 = USB
        self._eth_panel.setVisible(rb == 2)   # 2 = Ethernet

    def _relay_type_str(self) -> str:
        rb = self._relay_btn_group.checkedId()
        return ["none", "usb", "ethernet"][rb]

    # -------------------------------------------------------------------------
    def _load_into_ui(self) -> None:
        idx = self.model_combo.findText(SETTINGS.yolo_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.conf_spin.setValue(SETTINGS.detection_confidence)

        idx = self.mode_combo.findText(SETTINGS.violation_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

        w, h = SETTINGS.processing_resolution
        idx  = self.res_combo.findText(f"{w}x{h}")
        if idx >= 0:
            self.res_combo.setCurrentIndex(idx)

        # Relay type – derive from relay_type; fall back to use_usb_relay
        relay_type = getattr(SETTINGS, "relay_type", None)
        if not relay_type:
            relay_type = "usb" if SETTINGS.use_usb_relay else "none"
        relay_type = relay_type.lower()

        if relay_type == "usb":
            self._rb_usb.setChecked(True)
        elif relay_type == "ethernet":
            self._rb_eth.setChecked(True)
        else:
            self._rb_none.setChecked(True)

        # USB fields
        self.relay_channels_spin.setValue(SETTINGS.usb_num_channels)
        self.relay_serial_edit.setText(SETTINGS.usb_serial or "")

        # Ethernet fields
        # NOTE: QLineEdit uses setText(), NOT setValue() – using setValue() here
        # is a crash bug. setText() is the correct method for QLineEdit.
        self.eth_ip_edit.setText(
            getattr(SETTINGS, "eth_relay_ip", "192.168.1.200")
        )
        self.eth_port_spin.setValue(
            getattr(SETTINGS, "eth_relay_port", 502)
        )
        self.eth_device_id_spin.setValue(
            getattr(SETTINGS, "eth_relay_device_id", 1)
        )
        self.eth_channels_spin.setValue(
            getattr(SETTINGS, "eth_relay_num_channels", 16)
        )

        # Shared timing
        self.relay_cooldown_spin.setValue(SETTINGS.relay_cooldown)
        self.relay_duration_spin.setValue(SETTINGS.relay_duration)

        # Sync panel visibility to match loaded type
        self._on_relay_type_changed()

    # -------------------------------------------------------------------------
    def _save_settings(self) -> None:
        SETTINGS.yolo_model           = self.model_combo.currentText()
        SETTINGS.detection_confidence = self.conf_spin.value()
        SETTINGS.violation_mode       = self.mode_combo.currentText()

        w, h    = map(int, self.res_combo.currentText().split("x"))
        old_res = SETTINGS.processing_resolution
        SETTINGS.processing_resolution = (w, h)

        # Relay type
        SETTINGS.relay_type    = self._relay_type_str()
        SETTINGS.use_usb_relay = (SETTINGS.relay_type == "usb")  # backward compat

        # USB fields
        SETTINGS.usb_num_channels = self.relay_channels_spin.value()
        serial = self.relay_serial_edit.text().strip()
        SETTINGS.usb_serial = serial if serial else None

        # Ethernet fields
        SETTINGS.eth_relay_ip           = self.eth_ip_edit.text().strip()
        SETTINGS.eth_relay_port         = self.eth_port_spin.value()
        SETTINGS.eth_relay_device_id    = self.eth_device_id_spin.value()
        SETTINGS.eth_relay_num_channels = self.eth_channels_spin.value()

        # Shared timing
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
                "Saved. Resolution changed – zones rescaled. Restart required."
            )
        else:
            self.info_label.setText(
                f"Saved – relay backend: {SETTINGS.relay_type.upper()}. "
                "Broadcasting reload to relay process."
            )

        # Notify supervisor -> relay_process hot-swaps backend
        if self.heartbeat_q is not None:
            try:
                self.heartbeat_q.put_nowait(make_settings_saved("gui"))
            except Exception:
                pass

        logger.info(f"Settings saved – relay_type={SETTINGS.relay_type}")

    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
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