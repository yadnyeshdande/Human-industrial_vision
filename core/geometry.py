# =============================================================================
# core/geometry.py  –  Zone geometry (preserved from original)
# =============================================================================

from typing import List, Tuple


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_polygon(point: Tuple[float, float],
                     polygon: List[Tuple[int, int]]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    if len(polygon) < 3:
        return False
    x, y   = point
    n      = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def bbox_overlaps_polygon(bbox: Tuple[int, int, int, int],
                          polygon: List[Tuple[int, int]]) -> bool:
    """True if any corner of bbox is inside polygon, or polygon centre in bbox."""
    if not polygon or len(polygon) < 3:
        return False
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2),
               ((x1 + x2) // 2, (y1 + y2) // 2)]
    for pt in corners:
        if point_in_polygon(pt, polygon):
            return True
    # Also check if any polygon vertex is inside the bbox
    for vx, vy in polygon:
        if x1 <= vx <= x2 and y1 <= vy <= y2:
            return True
    return False


def point_in_rect(point: Tuple[float, float],
                  rect: Tuple[int, int, int, int]) -> bool:
    """Backward-compatible rectangle test (wraps polygon version)."""
    x1, y1, x2, y2 = rect
    return point_in_polygon(point, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
