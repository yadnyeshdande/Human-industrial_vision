# =============================================================================
# ui/zone_editor.py  –  Polygon zone drawing overlay (preserved from original)
# =============================================================================

from typing import Dict, List, Optional, Tuple

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPolygon

from utils.logger import get_logger

logger = get_logger("ZoneEditor")


class ZoneEditor(QWidget):
    """Transparent overlay for drawing and editing polygon zones."""

    zone_created  = pyqtSignal(list)           # [(x,y), ...]
    zone_modified = pyqtSignal(int, list)      # zone_id, [(x,y), ...]
    zone_selected = pyqtSignal(int)            # zone_id

    POINT_RADIUS = 6
    CLOSE_DIST   = 15   # pixels: close polygon if click near first point

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setMouseTracking(True)

        # Zones: {zone_id: [(x,y), ...]}
        self._zones:  Dict[int, List[Tuple[int,int]]] = {}
        self._colors: Dict[int, QColor]               = {}

        # Drawing state
        self._drawing       = False
        self._current_pts:  List[QPoint]  = []
        self._hover_pt:     Optional[QPoint] = None

        # Edit state
        self._selected_zone: Optional[int] = None
        self._drag_zone:     Optional[int] = None
        self._drag_pt_idx:   Optional[int] = None

        self._edit_enabled  = True

    # ----------------------------------------------------------------- properties
    @property
    def selected_zone_id(self) -> Optional[int]:
        return self._selected_zone

    # ----------------------------------------------------------------- mode
    def set_edit_enabled(self, enabled: bool) -> None:
        self._edit_enabled = enabled
        if not enabled:
            self._drawing      = False
            self._current_pts  = []
            self._hover_pt     = None
        self.update()

    def start_drawing(self) -> None:
        if self._edit_enabled:
            self._drawing     = True
            self._current_pts = []
            self.update()

    def cancel_drawing(self) -> None:
        self._drawing     = False
        self._current_pts = []
        self.update()

    # ----------------------------------------------------------------- zone management
    def add_zone(self, zone_id: int, points: List[Tuple[int,int]],
                 color=None) -> None:
        """Add or replace a zone.

        color accepts:
          • QColor            – used directly
          • (r, g, b) tuple   – converted to QColor
          • None              – defaults to green QColor(0, 255, 0)
        """
        if color is None:
            qcolor = QColor(0, 255, 0)
        elif isinstance(color, QColor):
            qcolor = color
        else:
            # RGB tuple (r, g, b)
            qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
        self._zones[zone_id]  = list(points)
        self._colors[zone_id] = qcolor
        self.update()

    def remove_zone(self, zone_id: int) -> None:
        self._zones.pop(zone_id, None)
        self._colors.pop(zone_id, None)
        if self._selected_zone == zone_id:
            self._selected_zone = None
        self.update()

    def clear_zones(self) -> None:
        self._zones.clear()
        self._colors.clear()
        self._selected_zone = None
        self.update()

    def get_zones(self) -> List[Tuple[int, List[Tuple[int,int]]]]:
        return [(zid, list(pts)) for zid, pts in self._zones.items()]

    def update_zone_points(self, zone_id: int,
                           points: List[Tuple[int,int]]) -> None:
        if zone_id in self._zones:
            self._zones[zone_id] = list(points)
            self.update()

    # ----------------------------------------------------------------- paint
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw existing zones
        for zone_id, pts in self._zones.items():
            if len(pts) < 2:
                continue
            color    = self._colors.get(zone_id, QColor(0, 255, 0))
            selected = (zone_id == self._selected_zone)
            alpha    = 80 if selected else 40
            pen_w    = 3  if selected else 2

            fill_color = QColor(color)
            fill_color.setAlpha(alpha)
            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(color, pen_w, Qt.SolidLine))

            poly = QPolygon([QPoint(int(x), int(y)) for x, y in pts])
            painter.drawPolygon(poly)

            # Vertex dots
            painter.setBrush(QBrush(color))
            for x, y in pts:
                painter.drawEllipse(QPoint(int(x), int(y)),
                                    self.POINT_RADIUS, self.POINT_RADIUS)

            # Label
            if pts:
                lbl_color = QColor(color)
                lbl_color.setAlpha(255)
                painter.setPen(QPen(lbl_color))
                painter.drawText(int(pts[0][0]) + 8, int(pts[0][1]) - 8,
                                 f"Zone {zone_id}")

        # Draw in-progress polygon
        if self._drawing and self._current_pts:
            painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            for i in range(len(self._current_pts) - 1):
                painter.drawLine(self._current_pts[i],
                                 self._current_pts[i + 1])
            if self._hover_pt:
                painter.drawLine(self._current_pts[-1], self._hover_pt)

            # Vertex dots
            painter.setBrush(QBrush(QColor(255, 200, 0)))
            for pt in self._current_pts:
                painter.drawEllipse(pt, self.POINT_RADIUS, self.POINT_RADIUS)

        painter.end()

    # ----------------------------------------------------------------- mouse
    def mousePressEvent(self, event) -> None:
        if not self._edit_enabled:
            return
        pos = event.pos()

        if self._drawing:
            if event.button() == Qt.LeftButton:
                # Close polygon?
                if (len(self._current_pts) >= 3
                        and self._near_first(pos)):
                    pts = [(p.x(), p.y()) for p in self._current_pts]
                    self._drawing     = False
                    self._current_pts = []
                    self.zone_created.emit(pts)
                else:
                    self._current_pts.append(QPoint(pos))
            elif event.button() == Qt.RightButton:
                self.cancel_drawing()
            self.update()
            return

        # Edit mode: select zone or drag point
        for zone_id, pts in self._zones.items():
            idx = self._point_near(pos, pts)
            if idx is not None:
                self._selected_zone = zone_id
                self._drag_zone     = zone_id
                self._drag_pt_idx   = idx
                self.zone_selected.emit(zone_id)
                self.update()
                return

        # Deselect
        self._selected_zone = None
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.pos()
        if self._drawing:
            self._hover_pt = pos
            self.update()
            return
        if self._drag_zone is not None and self._drag_pt_idx is not None:
            pts = self._zones[self._drag_zone]
            pts[self._drag_pt_idx] = (pos.x(), pos.y())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_zone is not None:
            zone_id = self._drag_zone
            self.zone_modified.emit(zone_id, list(self._zones[zone_id]))
        self._drag_zone   = None
        self._drag_pt_idx = None

    # ----------------------------------------------------------------- helpers
    def _near_first(self, pos: QPoint) -> bool:
        if not self._current_pts:
            return False
        fp = self._current_pts[0]
        return ((pos.x() - fp.x())**2 + (pos.y() - fp.y())**2) ** 0.5 < self.CLOSE_DIST

    def _point_near(self, pos: QPoint,
                    pts: List[Tuple[int,int]]) -> Optional[int]:
        for i, (x, y) in enumerate(pts):
            if ((pos.x() - x)**2 + (pos.y() - y)**2) ** 0.5 < self.POINT_RADIUS + 5:
                return i
        return None
