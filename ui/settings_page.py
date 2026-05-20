# =============================================================================
# ui/settings_page.py  –  Settings tab  (v6 – model hot-reload + class filter)
# =============================================================================
#
# Bug #1 fix – model not loading on change:
#   _save_settings() now sends CTRL_RELOAD_SETTINGS to BOTH the supervisor
#   (which broadcasts it to relay) AND directly to the detection control queue
#   (det_control_q).  Previously only the supervisor received the message,
#   so the detection process never reloaded the model.
#
#   Additionally, on model_combo change the class panel is refreshed by
#   probing the .pt file via YOLO().names so the operator can immediately
#   see which classes are available before saving.
#
# Feature #1 – class filter panel:
#   • After a model is selected the "Detection Classes" group box is
#     populated with one checkbox per class.
#   • Operator ticks which classes should trigger alarms.
#   • Selection is stored as SETTINGS.target_classes (list of int IDs).
#   • On save, target_classes is persisted and CTRL_RELOAD_SETTINGS is sent
#     to the detection process which calls detector.reload() hot-in-place.
#   • "Select All" / "Select None" convenience buttons are provided.
#   • If no classes are ticked → target_classes = [] → YOLO detects all.
#
# Feature #1 – hot model change:
#   Changing the model combo immediately triggers _on_model_changed() which:
#     1. Loads the model metadata (names) without running inference.
#     2. Rebuilds the checkbox list.
#     3. Pre-ticks whatever is in SETTINGS.target_classes if those IDs exist
#        in the new model.
# =============================================================================

from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Queue

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox,
    QPushButton, QDoubleSpinBox, QSpinBox, QMessageBox,
    QRadioButton, QButtonGroup, QCheckBox, QScrollArea,
    QSizePolicy, QFrame,
)
from PyQt5.QtCore import Qt, QTimer

from config.loader import ConfigManager, SETTINGS
from ipc.messages import make_settings_saved, make_control, CTRL_RELOAD_SETTINGS
from utils.logger import get_logger

logger = get_logger("SettingsPage")

MODELS_DIR = Path(__file__).parent.parent / "models"


class SettingsPage(QWidget):
    def __init__(
        self,
        config_manager: ConfigManager,
        heartbeat_q:    Optional[Queue] = None,
        det_control_q:  Optional[Queue] = None,   # NEW: direct line to detection
        parent=None,
    ):
        super().__init__(parent)
        self.config_manager = config_manager
        self.heartbeat_q    = heartbeat_q
        self.det_control_q  = det_control_q   # detection process control queue

        # {class_id: QCheckBox}  – rebuilt when model changes
        self._class_checkboxes: Dict[int, QCheckBox] = {}
        # {class_id: class_name} from the currently probed model
        self._probed_names: Dict[int, str] = {}

        self._setup_ui()
        self._load_into_ui()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)

        # ── Scroll area wraps everything so it works at any window size ───────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(8)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        # ── Detection settings ────────────────────────────────────────────────
        det_box  = QGroupBox("Detection Settings")
        det_form = QFormLayout(det_box)

        self.model_combo = QComboBox()
        self._refresh_local_models()
        # When model changes → rebuild class panel
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        det_form.addRow("YOLO Model:", self.model_combo)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        det_form.addRow("Confidence Threshold:", self.conf_spin)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["center", "overlap"])
        mode_note = QLabel(
            "center: person's centre point must enter zone (fewer false positives)\n"
            "overlap: bounding box must overlap zone (more sensitive)"
        )
        mode_note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        det_form.addRow("Violation Mode:", self.mode_combo)
        det_form.addRow(mode_note)

        self.res_combo = QComboBox()
        for r in ["640x360", "960x540", "1280x720", "1920x1080"]:
            self.res_combo.addItem(r)
        det_form.addRow("Processing Resolution:", self.res_combo)
        layout.addWidget(det_box)

        # ── Class filter panel ────────────────────────────────────────────────
        self._class_box = QGroupBox("Detection Classes  (which classes trigger the alarm)")
        class_outer = QVBoxLayout(self._class_box)

        # Convenience buttons
        btn_row = QHBoxLayout()
        self._sel_all_btn  = QPushButton("Select All")
        self._sel_none_btn = QPushButton("Select None")
        self._sel_all_btn.clicked.connect(self._select_all_classes)
        self._sel_none_btn.clicked.connect(self._select_no_classes)
        btn_row.addWidget(self._sel_all_btn)
        btn_row.addWidget(self._sel_none_btn)
        btn_row.addStretch()
        class_outer.addLayout(btn_row)

        # Scrollable checkbox area
        self._class_scroll = QScrollArea()
        self._class_scroll.setWidgetResizable(True)
        self._class_scroll.setMaximumHeight(220)
        self._class_scroll.setFrameShape(QFrame.StyledPanel)
        self._class_inner = QWidget()
        self._class_layout = QVBoxLayout(self._class_inner)
        self._class_layout.setSpacing(2)
        self._class_scroll.setWidget(self._class_inner)
        class_outer.addWidget(self._class_scroll)

        self._class_status = QLabel(
            "Select a model above – class list will appear here."
        )
        self._class_status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        class_outer.addWidget(self._class_status)

        layout.addWidget(self._class_box)

        # ── Relay settings ────────────────────────────────────────────────────
        relay_box = QGroupBox("Relay Settings")
        rb_layout = QVBoxLayout(relay_box)

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
            "Only ONE backend is active at a time. "
            "Switch here as contingency if one type fails."
        )
        info_lbl.setStyleSheet("color: #ffaa44; font-size: 10px;")
        rb_layout.addWidget(info_lbl)

        # Shared timing
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
            "Heartbeat checks every 10 s. Unplugged cable → relay process\n"
            "marks hardware dead and stops sending false outputs."
        )
        eth_note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        eth_form.addRow(eth_note)
        rb_layout.addWidget(self._eth_panel)

        layout.addWidget(relay_box)

        # Wire relay type radio buttons → show/hide sub-panels
        self._relay_btn_group.buttonClicked.connect(self._on_relay_type_changed)

        # ── Save / Reset ──────────────────────────────────────────────────────
        btn_row2 = QHBoxLayout()
        save_btn  = QPushButton("Save Settings")
        reset_btn = QPushButton("Reset Defaults")
        save_btn.setStyleSheet(
            "QPushButton { background-color: #2a7a2a; color: white; "
            "font-weight: bold; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #3a9a3a; }"
        )
        save_btn.clicked.connect(self._save_settings)
        reset_btn.clicked.connect(self._reset_defaults)
        btn_row2.addWidget(save_btn)
        btn_row2.addWidget(reset_btn)
        layout.addLayout(btn_row2)

        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "color: #66cc66; font-size: 11px; padding: 4px;"
        )
        layout.addWidget(self.info_label)

        # ── Model management ──────────────────────────────────────────────────
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

    # =========================================================================
    # Class panel helpers
    # =========================================================================

    def _probe_model_classes(self, model_name: str) -> Dict[int, str]:
        """
        Load YOLO model metadata only (no warm-up, no inference) to extract
        class names.  Returns {} on any error so the UI degrades gracefully.
        """
        local_path = MODELS_DIR / model_name
        if not local_path.exists():
            return {}
        try:
            from ultralytics import YOLO
            m = YOLO(str(local_path))
            if hasattr(m, "names") and m.names:
                return dict(m.names)
        except Exception as e:
            logger.warning(f"Could not probe model classes for {model_name}: {e}")
        return {}

    def _rebuild_class_checkboxes(
        self,
        names: Dict[int, str],
        preselected: List[int],
    ) -> None:
        """
        Clear the checkbox scroll area and rebuild it for the given class map.

        names       : {class_id: class_name}
        preselected : class IDs that should be checked initially
        """
        # Remove old widgets
        for cb in self._class_checkboxes.values():
            cb.setParent(None)
            cb.deleteLater()
        self._class_checkboxes = {}

        if not names:
            self._class_status.setText(
                "Could not load class names from this model file."
            )
            return

        pre_set = set(preselected)

        for cls_id in sorted(names.keys()):
            cls_name = names[cls_id]
            cb = QCheckBox(f"[{cls_id}]  {cls_name}")
            cb.setChecked(cls_id in pre_set)
            self._class_layout.addWidget(cb)
            self._class_checkboxes[cls_id] = cb

        n = len(names)
        self._class_status.setText(
            f"{n} class{'es' if n != 1 else ''} available.  "
            "Tick the ones that should trigger the alarm. "
            "Leave all unchecked to detect every class."
        )
        logger.info(
            f"Class panel rebuilt for model with {n} classes; "
            f"preselected={sorted(pre_set)}"
        )

    def _get_selected_classes(self) -> List[int]:
        """Return sorted list of ticked class IDs.  Empty = all classes."""
        return sorted(
            cls_id
            for cls_id, cb in self._class_checkboxes.items()
            if cb.isChecked()
        )

    def _select_all_classes(self) -> None:
        for cb in self._class_checkboxes.values():
            cb.setChecked(True)

    def _select_no_classes(self) -> None:
        for cb in self._class_checkboxes.values():
            cb.setChecked(False)

    # =========================================================================
    # Signal handlers
    # =========================================================================

    def _on_relay_type_changed(self, _button=None) -> None:
        rb = self._relay_btn_group.checkedId()
        self._usb_panel.setVisible(rb == 1)   # 1 = USB
        self._eth_panel.setVisible(rb == 2)   # 2 = Ethernet

    def _relay_type_str(self) -> str:
        rb = self._relay_btn_group.checkedId()
        return ["none", "usb", "ethernet"][rb]

    def _on_model_changed(self, model_name: str) -> None:
        """
        Called whenever the model combo selection changes.

        Probes the .pt file for class names and rebuilds the checkbox panel.
        Pre-ticks classes that are already in SETTINGS.target_classes if those
        IDs exist in the new model (best-effort preservation across model swap).
        """
        if not model_name or model_name.startswith("("):
            # Placeholder text – no real model selected
            for cb in self._class_checkboxes.values():
                cb.setParent(None)
                cb.deleteLater()
            self._class_checkboxes = {}
            self._probed_names = {}
            self._class_status.setText("No model selected.")
            return

        self._class_status.setText(
            f"Loading class names from  {model_name} … "
            "(may take a few seconds on first load)"
        )
        # Use a single-shot timer so the label update actually renders before
        # the blocking YOLO() call happens.
        QTimer.singleShot(50, lambda: self._do_probe_and_rebuild(model_name))

    def _do_probe_and_rebuild(self, model_name: str) -> None:
        names = self._probe_model_classes(model_name)
        self._probed_names = names

        # Preserve whatever was selected before, if the IDs still exist
        preselected = [c for c in SETTINGS.target_classes if c in names]

        self._rebuild_class_checkboxes(names, preselected)

    # =========================================================================
    # Load settings → UI
    # =========================================================================

    def _load_into_ui(self) -> None:
        # Model combo
        idx = self.model_combo.findText(SETTINGS.yolo_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        # _on_model_changed fires automatically via currentTextChanged signal

        self.conf_spin.setValue(SETTINGS.detection_confidence)

        idx = self.mode_combo.findText(SETTINGS.violation_mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

        w, h = SETTINGS.processing_resolution
        idx  = self.res_combo.findText(f"{w}x{h}")
        if idx >= 0:
            self.res_combo.setCurrentIndex(idx)

        # Relay type
        relay_type = getattr(SETTINGS, "relay_type", None) or (
            "usb" if SETTINGS.use_usb_relay else "none"
        )
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
        self.eth_ip_edit.setText(getattr(SETTINGS, "eth_relay_ip", "192.168.1.200"))
        self.eth_port_spin.setValue(getattr(SETTINGS, "eth_relay_port", 502))
        self.eth_device_id_spin.setValue(getattr(SETTINGS, "eth_relay_device_id", 1))
        self.eth_channels_spin.setValue(getattr(SETTINGS, "eth_relay_num_channels", 16))

        # Timing
        self.relay_cooldown_spin.setValue(SETTINGS.relay_cooldown)
        self.relay_duration_spin.setValue(SETTINGS.relay_duration)

        # Sync sub-panel visibility
        self._on_relay_type_changed()

    # =========================================================================
    # Save
    # =========================================================================

    def _save_settings(self) -> None:
        SETTINGS.yolo_model           = self.model_combo.currentText()
        SETTINGS.detection_confidence = self.conf_spin.value()
        SETTINGS.violation_mode       = self.mode_combo.currentText()

        w, h    = map(int, self.res_combo.currentText().split("x"))
        old_res = SETTINGS.processing_resolution
        SETTINGS.processing_resolution = (w, h)

        # Relay
        SETTINGS.relay_type    = self._relay_type_str()
        SETTINGS.use_usb_relay = (SETTINGS.relay_type == "usb")
        SETTINGS.usb_num_channels = self.relay_channels_spin.value()
        serial = self.relay_serial_edit.text().strip()
        SETTINGS.usb_serial = serial if serial else None
        SETTINGS.eth_relay_ip           = self.eth_ip_edit.text().strip()
        SETTINGS.eth_relay_port         = self.eth_port_spin.value()
        SETTINGS.eth_relay_device_id    = self.eth_device_id_spin.value()
        SETTINGS.eth_relay_num_channels = self.eth_channels_spin.value()
        SETTINGS.relay_cooldown = self.relay_cooldown_spin.value()
        SETTINGS.relay_duration = self.relay_duration_spin.value()

        # ── FEATURE #1: save selected class IDs ──────────────────────────────
        SETTINGS.target_classes = self._get_selected_classes()

        # Persist to disk
        try:
            SETTINGS.save()
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))
            return

        # Resolution change → rescale zone coords
        if (w, h) != old_res:
            self.config_manager.update_processing_resolution((w, h))
            self.config_manager.save()
            self.info_label.setText(
                "Saved. Resolution changed – zones rescaled. Restart required."
            )
        else:
            tc = SETTINGS.target_classes
            tc_str = (
                "ALL classes"
                if not tc
                else ", ".join(
                    self._probed_names.get(c, str(c)) for c in tc
                )
            )
            self.info_label.setText(
                f"Saved — model: {SETTINGS.yolo_model}  |  "
                f"relay: {SETTINGS.relay_type.upper()}  |  "
                f"detecting: {tc_str}"
            )

        # ── BUG #1 FIX: broadcast reload to BOTH supervisor AND detection ─────
        #
        # Previous code only sent MSG_SETTINGS_SAVED to heartbeat_q (supervisor).
        # The supervisor then broadcasts CTRL_RELOAD_SETTINGS – but that can
        # take several seconds, and on Windows the multiprocessing Queue routing
        # sometimes drops or delays the message to detection workers.
        #
        # Fix: send CTRL_RELOAD_SETTINGS directly to det_control_q as well.
        # This guarantees the detection process receives it regardless of
        # supervisor timing.

        reload_msg = make_control("gui", CTRL_RELOAD_SETTINGS)

        # 1. Tell supervisor (→ relay_process hot-swap)
        if self.heartbeat_q is not None:
            try:
                self.heartbeat_q.put_nowait(make_settings_saved("gui"))
            except Exception:
                pass

        # 2. Tell detection worker directly (→ model hot-swap)
        if self.det_control_q is not None:
            try:
                self.det_control_q.put_nowait(reload_msg)
                logger.info(
                    "CTRL_RELOAD_SETTINGS sent directly to detection worker "
                    f"(model={SETTINGS.yolo_model}, "
                    f"classes={SETTINGS.target_classes or 'ALL'})"
                )
            except Exception as e:
                logger.warning(f"Could not send reload to detection worker: {e}")

        logger.info(
            f"Settings saved — relay_type={SETTINGS.relay_type}  "
            f"model={SETTINGS.yolo_model}  "
            f"target_classes={SETTINGS.target_classes or 'ALL'}"
        )

    # =========================================================================
    # Reset / Refresh
    # =========================================================================

    def _reset_defaults(self) -> None:
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Reset all settings to defaults?\n"
            "(target classes will be cleared → detect ALL classes)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            from config.loader import AppSettings
            defaults = AppSettings()
            SETTINGS.__dict__.update(defaults.__dict__)
            self._load_into_ui()
            self.info_label.setText("Settings reset to defaults (not saved)")

    def _refresh_local_models(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pts     = sorted(p.name for p in MODELS_DIR.glob("*.pt"))
        current = self.model_combo.currentText() if self.model_combo.count() else ""
        self.model_combo.clear()
        if pts:
            self.model_combo.addItems(pts)
        else:
            self.model_combo.addItem("(no models found in models/)")
        idx = self.model_combo.findText(current)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        # currentTextChanged fires if selection actually changed,
        # rebuilding the class panel automatically.
