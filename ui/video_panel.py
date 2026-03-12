# =============================================================================
# ui/video_panel.py  –  Camera display widget (preserved from original)
# =============================================================================

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from utils.logger import get_logger

logger = get_logger("VideoPanel")


class VideoPanel(QWidget):
    """Live video display with zone overlays and detection bounding boxes."""

    frame_clicked = pyqtSignal(int, int)

    def __init__(self, camera_id: int,
                 processing_resolution: Tuple[int, int] = (1280, 720),
                 parent=None):
        super().__init__(parent)
        self.camera_id          = camera_id
        self.processing_resolution = processing_resolution
        self.processing_width, self.processing_height = processing_resolution

        self.current_frame: Optional[np.ndarray] = None
        self.zones:          List[Tuple]          = []   # (zone_id, points, color_rgb)
        self.persons:        List[Tuple[int,int,int,int]] = []
        self.zone_violations: Dict[int, bool]     = {}

        # Letterbox geometry
        self.offset_x = 0
        self.offset_y = 0
        self.scale    = 1.0

        self._setup_ui()

    # ----------------------------------------------------------------- UI
    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #0a0a0a; border: 1px solid #444;"
        )
        self.video_label.setMinimumSize(320, 180)
        self.video_label.setScaledContents(False)
        layout.addWidget(self.video_label)

        self.info_label = QLabel(f"Camera {self.camera_id}")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            "color: #cccccc; font-size: 10px; padding: 2px;"
        )
        layout.addWidget(self.info_label)

    # ----------------------------------------------------------------- public API
    def update_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        self.current_frame = frame
        self._render()

    def set_persons(self, persons: List[Tuple[int, int, int, int]]) -> None:
        self.persons = persons

    def set_zones(self, zones: List[Tuple]) -> None:
        self.zones = zones

    def set_zone_violations(self, violations: Dict[int, bool]) -> None:
        self.zone_violations = violations

    def update_info(self, text: str) -> None:
        self.info_label.setText(text)

    # ----------------------------------------------------------------- rendering
    def _render(self) -> None:
        if self.current_frame is None:
            return

        frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

        # Person bounding boxes
        for x1, y1, x2, y2 in self.persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Person", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Zone overlays
        for zone_id, points, color in self.zones:
            if len(points) < 2:
                continue
            violated = self.zone_violations.get(zone_id, False)
            draw_color  = (255, 0, 0) if violated else color
            border_w    = 4 if violated else 2
            fill_alpha  = 0.35 if violated else 0.15

            pts = np.array(points, np.int32).reshape(-1, 1, 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], draw_color)
            cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
            cv2.polylines(frame, [pts], True, draw_color, border_w)

            lbl = f"Zone {zone_id}" + (" [VIOLATION]" if violated else "")
            cv2.putText(frame, lbl,
                        (int(points[0][0]) + 5, int(points[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 if violated else 0.5,
                        draw_color,
                        3 if violated else 2)

        # Convert → QPixmap with aspect-ratio preservation
        h, w = frame.shape[:2]
        q_img  = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        widget_size = self.video_label.size()
        if widget_size.width() <= 0 or widget_size.height() <= 0:
            widget_size.setWidth(max(640, self.processing_width))
            widget_size.setHeight(max(360, self.processing_height))

        scaled = pixmap.scaled(widget_size, Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)

        self.offset_x = (widget_size.width()  - scaled.width())  // 2
        self.offset_y = (widget_size.height() - scaled.height()) // 2
        self.scale    = scaled.width() / w if w > 0 else 1.0

        self.video_label.setPixmap(scaled)
        self.update()

    # ----------------------------------------------------------------- coordinate mapping
    def widget_to_processing(self, wx: int, wy: int) -> Tuple[int, int]:
        px = int((wx - self.offset_x) / self.scale) if self.scale > 0 else 0
        py = int((wy - self.offset_y) / self.scale) if self.scale > 0 else 0
        px = max(0, min(px, self.processing_width  - 1))
        py = max(0, min(py, self.processing_height - 1))
        return px, py

    def processing_to_widget(self, px: int, py: int) -> Tuple[int, int]:
        wx = int(px * self.scale + self.offset_x)
        wy = int(py * self.scale + self.offset_y)
        return wx, wy
