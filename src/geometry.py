"""
src/geometry.py
Geometry primitives: polygon containment, line crossing, trajectory angles,
Time-To-Collision (TTC), vector alignment, and homography pixel-to-meter scaling.

All public functions operate on plain Python lists / NumPy arrays and are
intentionally free of any CV or ML imports so they can be unit-tested in
isolation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases ---------------------------------------------------------------
# ---------------------------------------------------------------------------

Point2D = tuple[float, float] | Sequence[float]
Polygon = Sequence[Point2D]
LineSegment = tuple[Point2D, Point2D]


# ---------------------------------------------------------------------------
# Point-in-Polygon -----------------------------------------------------------
# ---------------------------------------------------------------------------

def point_in_polygon(point: Point2D, polygon: Polygon) -> bool:
    """
    Ray-casting point-in-polygon test.

    Parameters
    ----------
    point   : (x, y)
    polygon : sequence of (x, y) vertices (open or closed)

    Returns
    -------
    bool
    """
    x, y = float(point[0]), float(point[1])
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[j][0]), float(polygon[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def polygon_contains_bbox_center(bbox: Sequence[float], polygon: Polygon) -> bool:
    """Return True if the centre of *bbox* lies inside *polygon*."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return point_in_polygon((cx, cy), polygon)


def bbox_overlaps_polygon(bbox: Sequence[float], polygon: Polygon) -> bool:
    """
    Conservative check: any corner or the centre of *bbox* is inside *polygon*.
    """
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2),
               ((x1+x2)/2, (y1+y2)/2)]
    return any(point_in_polygon(c, polygon) for c in corners)


# ---------------------------------------------------------------------------
# Line / Segment Intersection ------------------------------------------------
# ---------------------------------------------------------------------------

def _ccw(a: Point2D, b: Point2D, c: Point2D) -> bool:
    return (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])


def segments_intersect(seg1: LineSegment, seg2: LineSegment) -> bool:
    """
    Return True if segment *seg1* and *seg2* intersect.

    Uses the classic collinear / orientation algorithm.
    """
    a, b = seg1
    c, d = seg2
    return (_ccw(a, c, d) != _ccw(b, c, d)) and (_ccw(a, b, c) != _ccw(a, b, d))


def segment_intersection_point(
    seg1: LineSegment, seg2: LineSegment
) -> tuple[float, float] | None:
    """
    Return the exact (x, y) intersection of two segments, or None if parallel.
    """
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    if not (0.0 <= t <= 1.0):
        return None
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    if not (0.0 <= u <= 1.0):
        return None
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def bbox_crosses_line(
    bbox: Sequence[float],
    line_p1: Point2D,
    line_p2: Point2D,
    prev_bbox: Sequence[float] | None = None,
) -> bool:
    """
    Return True if the vehicle represented by *bbox* has crossed the line
    defined by *line_p1*→*line_p2* between *prev_bbox* and *bbox*.

    Falls back to a simple centre-side-of-line test when *prev_bbox* is None.
    """
    def _centre(b: Sequence[float]) -> tuple[float, float]:
        return ((b[0]+b[2])/2, (b[1]+b[3])/2)

    def _side(pt: Point2D) -> float:
        """Signed distance from point to the line (positive / negative = different sides)."""
        lx = float(line_p2[0]) - float(line_p1[0])
        ly = float(line_p2[1]) - float(line_p1[1])
        return lx * (float(pt[1]) - float(line_p1[1])) - ly * (float(pt[0]) - float(line_p1[0]))

    if prev_bbox is None:
        # Just check foot of bbox (bottom-centre) crossing
        cx, by = (bbox[0]+bbox[2])/2, bbox[3]
        return abs(_side((cx, by))) < 30.0  # tolerance band

    cx_now, cy_now = _centre(bbox)
    cx_prv, cy_prv = _centre(prev_bbox)
    traj_seg: LineSegment = ((cx_prv, cy_prv), (cx_now, cy_now))
    line_seg: LineSegment = (line_p1, line_p2)
    return segments_intersect(traj_seg, line_seg)


# ---------------------------------------------------------------------------
# Vector / Trajectory Maths --------------------------------------------------
# ---------------------------------------------------------------------------

def velocity_vector(
    positions: Sequence[Point2D], window: int = 5
) -> tuple[float, float]:
    """
    Estimate velocity vector (vx, vy) from the last *window* positions.

    Returns (0, 0) if insufficient data.
    """
    if len(positions) < 2:
        return (0.0, 0.0)
    pts = positions[-min(window, len(positions)):]
    vx = float(pts[-1][0]) - float(pts[0][0])
    vy = float(pts[-1][1]) - float(pts[0][1])
    n = max(len(pts) - 1, 1)
    return (vx / n, vy / n)


def speed_pixels_per_frame(positions: Sequence[Point2D], window: int = 5) -> float:
    """Return the magnitude of the velocity vector (px/frame)."""
    vx, vy = velocity_vector(positions, window)
    return math.hypot(vx, vy)


def speed_kmh(
    positions: Sequence[Point2D],
    pixels_per_meter: float,
    fps: float,
    window: int = 5,
) -> float:
    """
    Convert pixel displacement to km/h.

    speed_kmh = (px/frame  * fps  / px_per_m) * 3.6
    """
    px_per_frame = speed_pixels_per_frame(positions, window)
    if pixels_per_meter <= 0 or fps <= 0:
        return 0.0
    m_per_s = (px_per_frame * fps) / pixels_per_meter
    return m_per_s * 3.6


def vector_dot_normalized(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """
    Cosine similarity of two 2-D vectors.  Returns value in [-1, 1].
    """
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0
    return (v1[0]*v2[0] + v1[1]*v2[1]) / (mag1 * mag2)


def heading_angle_degrees(velocity: tuple[float, float]) -> float:
    """Return bearing in degrees (0° = East, CCW positive)."""
    return math.degrees(math.atan2(-velocity[1], velocity[0]))


def trajectory_total_angle_change(positions: Sequence[Point2D]) -> float:
    """
    Compute the total signed heading change (degrees) over the trajectory.

    Positive = counter-clockwise turn; used for U-turn detection.
    """
    if len(positions) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(positions) - 1):
        dx1 = float(positions[i][0]) - float(positions[i-1][0])
        dy1 = float(positions[i][1]) - float(positions[i-1][1])
        dx2 = float(positions[i+1][0]) - float(positions[i][0])
        dy2 = float(positions[i+1][1]) - float(positions[i][1])
        if math.hypot(dx1, dy1) < 1e-6 or math.hypot(dx2, dy2) < 1e-6:
            continue
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        diff = math.degrees(angle2 - angle1)
        # Normalise to [-180, 180]
        diff = (diff + 180) % 360 - 180
        total += diff
    return total


# ---------------------------------------------------------------------------
# Time-To-Collision (TTC) ----------------------------------------------------
# ---------------------------------------------------------------------------

def calculate_ttc(
    pos1: Point2D,
    vel1: tuple[float, float],
    pos2: Point2D,
    vel2: tuple[float, float],
    *,
    min_distance_px: float = 20.0,
) -> float:
    """
    Constant-velocity Time-To-Collision estimate (seconds at current fps=1).

    The caller must divide by fps to get physical seconds, or pass velocities
    in px/s.

    Returns ``float('inf')`` if objects are diverging or already overlapping
    below *min_distance_px*.

    Formula:
        d = ||pos2 - pos1||
        vrel = vel1 - vel2   (relative approach velocity)
        TTC  = d / ||vrel||  if vrel · (pos2 - pos1) > 0   (approaching)
    """
    dx = float(pos2[0]) - float(pos1[0])
    dy = float(pos2[1]) - float(pos1[1])
    dist = math.hypot(dx, dy)
    if dist < min_distance_px:
        return 0.0

    rvx = float(vel1[0]) - float(vel2[0])
    rvy = float(vel1[1]) - float(vel2[1])
    # Component of relative velocity along the line connecting centres
    closing = (rvx * dx + rvy * dy) / (dist + 1e-12)

    if closing <= 0:
        return float("inf")  # Diverging

    return dist / closing


def pairwise_min_ttc(
    tracks: list[dict],
    *,
    fps: float,
    ttc_threshold_s: float = 5.0,
) -> tuple[float, int, int]:
    """
    Compute the minimum TTC across all vehicle pairs.

    Parameters
    ----------
    tracks  : list of track dicts with keys 'tid', 'pos', 'vel'
    fps     : video fps (used to convert frame-based TTC to seconds)

    Returns
    -------
    (min_ttc_s, tid_a, tid_b)  — TTC in seconds and the two involved track IDs.
    """
    min_ttc = float("inf")
    tid_a, tid_b = -1, -1
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            t1, t2 = tracks[i], tracks[j]
            ttc_frames = calculate_ttc(t1["pos"], t1["vel"], t2["pos"], t2["vel"])
            ttc_s = ttc_frames / max(fps, 1.0)
            if ttc_s < min_ttc:
                min_ttc = ttc_s
                tid_a = t1["tid"]
                tid_b = t2["tid"]
    return min_ttc, tid_a, tid_b


# ---------------------------------------------------------------------------
# Fire / Smoke colour saliency ----------------------------------------------
# ---------------------------------------------------------------------------

def is_fire_pixel_region(
    roi_bgr: np.ndarray,
    min_contour_area: float = 400.0,
) -> tuple[bool, float]:
    """
    Detect orange/red fire or grey smoke regions using HSV thresholding.

    Returns (detected: bool, max_contour_area: float).
    """
    import cv2  # imported here to keep module testable without cv2

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Fire: orange-red hues (0-25 and 160-180) with high saturation & value
    mask_fire_low  = cv2.inRange(hsv, (0, 120, 100),  (25, 255, 255))
    mask_fire_high = cv2.inRange(hsv, (160, 120, 100), (180, 255, 255))
    # Smoke: low saturation, mid-high value
    mask_smoke = cv2.inRange(hsv, (0, 0, 60), (180, 50, 220))
    combined = cv2.bitwise_or(cv2.bitwise_or(mask_fire_low, mask_fire_high), mask_smoke)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    max_area = max(cv2.contourArea(c) for c in contours)
    return max_area >= min_contour_area, max_area


# ---------------------------------------------------------------------------
# Misc helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def bbox_area(bbox: Sequence[float]) -> float:
    """Return bounding-box area in pixels²."""
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_iou(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    """Intersection-over-Union of two axis-aligned bounding boxes."""
    if len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0
    ix1 = max(float(bbox_a[0]), float(bbox_b[0]))
    iy1 = max(float(bbox_a[1]), float(bbox_b[1]))
    ix2 = min(float(bbox_a[2]), float(bbox_b[2]))
    iy2 = min(float(bbox_a[3]), float(bbox_b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(bbox_a) + bbox_area(bbox_b) - inter
    return inter / (union + 1e-9)


def bbox_centre(bbox: Sequence[float]) -> tuple[float, float]:
    """Return (cx, cy) centre point for a 4-element bbox or 2-element point."""
    try:
        if len(bbox) >= 4:
            return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
        elif len(bbox) >= 2:
            return (float(bbox[0]), float(bbox[1]))
    except Exception:
        pass
    return (0.0, 0.0)


def distance_between_centres(
    bbox_a: Sequence[float], bbox_b: Sequence[float]
) -> float:
    ca = bbox_centre(bbox_a)
    cb = bbox_centre(bbox_b)
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
