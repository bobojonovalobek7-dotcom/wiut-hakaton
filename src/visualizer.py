"""
src/visualizer.py
Professional HUD overlay for the WIUT Traffic Analytics system.

Renders:
  - Colour-coded bounding boxes with class labels & track IDs
  - Trajectory tails (polyline of recent track positions)
  - Semi-transparent ROI zone overlays (zebra, stop lines, solid dividers, intersection)
  - Active event alert banners (top centre of frame)
  - Circular risk gauge (bottom-right corner)
  - Traffic light state indicators per approach
  - Frame timestamp, FPS counter, and legend
"""

from __future__ import annotations

import collections
import math
import time
from typing import Optional

import cv2
import numpy as np

from .config import Config
from .detector import TrackState
from .geometry import calculate_ttc, bbox_centre
from .risk_engine import RiskScore
from .rules import TrafficEvent

# ---------------------------------------------------------------------------
# Colour palette (BGR) -------------------------------------------------------
# ---------------------------------------------------------------------------

PALETTE = {
    "car":           (60,  180,  75),   # emerald green
    "truck":         (255, 130,   0),   # deep orange
    "bus":           (180,  60, 255),   # purple
    "motorcycle":    ( 50, 200, 255),   # cyan
    "person":        (255,  50, 150),   # hot pink
    "bicycle":       (255, 220,   0),   # yellow
    "traffic_light": (255, 255, 255),   # white
    "default":       (200, 200, 200),   # grey

    # Zones
    "zebra":         (255, 220,   0),   # yellow
    "intersection":  ( 80, 200, 255),   # sky blue
    "stop_line_red": (  0,   0, 255),   # red
    "stop_line_grn": (  0, 220,   0),   # green
    "solid_line":    (  0, 165, 255),   # orange
    "road":          ( 40, 100,  40),   # dark green

    # Events
    "accident":         (  0,   0, 255),
    "near_miss":        (  0, 100, 255),
    "red_light":        (  0,  30, 200),
    "wrong_way":        (  0,  50, 255),
    "illegal_u_turn":   (150,  50, 255),
    "stopped_vehicle":  (  0, 180, 255),
    "jaywalking":       (255,  50, 200),
    "failure_to_yield": (  0,  80, 255),
    "illegal_turn":     (100, 100, 255),
    "solid_line_crossing": (0, 140, 255),
    "stop_line":        (  0, 200, 200),
    "congestion":       (100, 200, 255),
    "road_obstacle":    (200, 200,   0),
    "fire_smoke":       (  0, 100, 255),
}

EVENT_ICONS = {
    "accident":         "💥",
    "near_miss":        "⚠️",
    "red_light":        "🔴",
    "wrong_way":        "⛔",
    "illegal_u_turn":   "↩️",
    "stopped_vehicle":  "🚗",
    "jaywalking":       "🚶",
    "failure_to_yield": "🚧",
    "illegal_turn":     "↪️",
    "solid_line_crossing": "❌",
    "stop_line":        "🛑",
    "congestion":       "🚦",
    "road_obstacle":    "⬛",
    "fire_smoke":       "🔥",
}

# ---------------------------------------------------------------------------
# Main Visualizer -----------------------------------------------------------
# ---------------------------------------------------------------------------

class TrafficVisualizer:
    """
    Renders all HUD elements onto a copy of the input BGR frame.

    Parameters
    ----------
    config      : Loaded project Config.
    show_zones  : Draw ROI zone overlays.
    show_trails : Draw trajectory tail polylines.
    """

    def __init__(
        self,
        config: Config,
        *,
        show_zones: bool = True,
        show_trails: bool = True,
    ) -> None:
        self.cfg = config
        self.show_zones  = show_zones
        self.show_trails = show_trails

        # Rolling active event banners (event_name → time_remaining_s)
        self._banners: dict[str, float] = {}
        self._banner_duration_s = 4.0

        # FPS meter
        self._frame_times: collections.deque = collections.deque(maxlen=30)

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        tracks: list[TrackState],
        risk: Optional[RiskScore],
        active_events: list[str],
        timestamp_s: float,
        tl_states: dict[str, str],
    ) -> np.ndarray:
        """
        Draw all HUD elements onto *frame* and return the annotated copy.

        Parameters
        ----------
        frame         : Original BGR frame.
        tracks        : Current active tracking states.
        risk          : Latest RiskScore (or None).
        active_events : Names of currently active events.
        timestamp_s   : Current video timestamp in seconds.
        tl_states     : {tl_id: 'RED'|'GREEN'|'YELLOW'} mapping.
        """
        canvas = frame.copy()
        t_now = time.perf_counter()
        self._frame_times.append(t_now)

        if self.show_zones:
            self._draw_zones(canvas, tl_states)

        if self.show_trails:
            self._draw_trails(canvas, tracks)

        self._draw_collision_vectors(canvas, tracks)
        self._draw_tracks(canvas, tracks)
        self._draw_traffic_light_widgets(canvas, tl_states)
        self._draw_event_banners(canvas, active_events, timestamp_s)
        self._draw_hud_bar(canvas, timestamp_s, tracks)
        if risk:
            self._draw_risk_gauge(canvas, risk)

        return canvas

    def notify_event(self, event_name: str) -> None:
        """Register an event so its banner is shown for _banner_duration_s seconds."""
        self._banners[event_name] = self._banner_duration_s

    # ------------------------------------------------------------------
    # Zone overlays -----------------------------------------------------

    def _draw_zones(self, canvas: np.ndarray, tl_states: dict[str, str]) -> None:
        h, w = canvas.shape[:2]
        overlay = canvas.copy()

        # Intersection box
        if self.cfg.intersection_box:
            pts = np.array(self.cfg.intersection_box, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (*PALETTE["intersection"], ))
        cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
        overlay = canvas.copy()

        # Zebra crossings
        for zc in self.cfg.zebra_crossings:
            pts = np.array(zc.polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], PALETTE["zebra"])
            cv2.polylines(canvas, [pts], True, PALETTE["zebra"], 2)
        cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)
        overlay = canvas.copy()

        # Stop lines
        for sl in self.cfg.stop_lines:
            tl_id  = sl.associated_traffic_light
            state  = tl_states.get(tl_id, "RED")
            colour = PALETTE["stop_line_grn"] if state == "GREEN" else PALETTE["stop_line_red"]
            p1 = (int(sl.points[0][0]), int(sl.points[0][1]))
            p2 = (int(sl.points[1][0]), int(sl.points[1][1]))
            cv2.line(canvas, p1, p2, colour, 3, cv2.LINE_AA)

        # Solid lane dividers
        for solid in self.cfg.solid_lines:
            p1 = (int(solid.points[0][0]), int(solid.points[0][1]))
            p2 = (int(solid.points[1][0]), int(solid.points[1][1]))
            cv2.line(canvas, p1, p2, PALETTE["solid_line"], 3, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Collision prediction vectors --------------------------------------

    def _draw_collision_vectors(self, canvas: np.ndarray, tracks: list[TrackState]) -> None:
        movers = [t for t in tracks if (t.is_vehicle or t.is_bicycle) and t.speed_kmh > 4.0 and len(t.bbox) >= 4]
        for i in range(len(movers)):
            for j in range(i + 1, len(movers)):
                v1, v2 = movers[i], movers[j]
                ttc_frames = calculate_ttc(v1.bbox, v2.bbox, v1.vel_px, v2.vel_px)
                ttc_s = ttc_frames / max(self.cfg.camera.fps, 1.0)
                if 0.1 <= ttc_s <= 2.2:
                    c1 = (int((v1.bbox[0] + v1.bbox[2]) / 2), int((v1.bbox[1] + v1.bbox[3]) / 2))
                    c2 = (int((v2.bbox[0] + v2.bbox[2]) / 2), int((v2.bbox[1] + v2.bbox[3]) / 2))
                    # Draw warning line connecting the two converging vehicles
                    cv2.line(canvas, c1, c2, (0, 0, 255), 2, cv2.LINE_AA)
                    mid = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                    cv2.circle(canvas, mid, 6, (0, 0, 255), -1)
                    cv2.putText(canvas, f"TTC {ttc_s:.1f}s", (mid[0] + 8, mid[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Traffic light widgets --------------------------------------------

    def _draw_traffic_light_widgets(self, canvas: np.ndarray, tl_states: dict[str, str]) -> None:
        if not tl_states:
            return
        w = canvas.shape[1]
        x_start = w - 210
        y = 12
        overlay = canvas.copy()
        box_w = 200
        box_h = 24 + len(tl_states) * 22
        cv2.rectangle(overlay, (x_start, y), (x_start + box_w, y + box_h), (12, 16, 28), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.putText(canvas, "SIGNALS STATUS", (x_start + 10, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 200), 1, cv2.LINE_AA)
        row_y = y + 34
        for tl_id, state in tl_states.items():
            colour = (0, 220, 0) if state == "GREEN" else ((0, 220, 220) if state == "YELLOW" else (0, 0, 240))
            cv2.circle(canvas, (x_start + 18, row_y - 4), 6, colour, -1)
            name_short = tl_id.replace("tl_", "").upper()
            cv2.putText(canvas, f"{name_short}: {state}", (x_start + 32, row_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
            row_y += 20

    # ------------------------------------------------------------------
    # Track bounding boxes ----------------------------------------------

    def _draw_tracks(self, canvas: np.ndarray, tracks: list[TrackState]) -> None:
        for t in tracks:
            if not t.bbox or len(t.bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            is_speeding = t.is_vehicle and t.speed_kmh > 65.0
            colour = (0, 0, 255) if is_speeding else PALETTE.get(t.class_name, PALETTE["default"])
            thickness = 3 if is_speeding else 2

            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)

            # Label
            if is_speeding:
                label = f"#{t.track_id} {t.class_name} SPEEDING {t.speed_kmh:.0f}km/h"
            else:
                label = f"#{t.track_id} {t.class_name} {t.speed_kmh:.0f}km/h"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            lx, ly = max(x1, 0), max(y1 - 6, th + 4)
            cv2.rectangle(canvas, (lx, ly - th - 4), (lx + tw + 4, ly), colour, -1)
            text_color = (255, 255, 255) if is_speeding else (10, 10, 10)
            cv2.putText(canvas, label, (lx + 2, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Trajectory trails -------------------------------------------------

    def _draw_trails(self, canvas: np.ndarray, tracks: list[TrackState]) -> None:
        for t in tracks:
            pts = list(t.positions)
            if len(pts) < 2:
                continue
            colour = PALETTE.get(t.class_name, PALETTE["default"])
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
                p2 = (int(pts[i][0]),   int(pts[i][1]))
                alpha = 0.3 + 0.7 * (i / len(pts))
                faded = tuple(int(c * alpha) for c in colour)
                cv2.line(canvas, p1, p2, faded, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Event banners (top-centre) ----------------------------------------

    def _draw_event_banners(
        self,
        canvas: np.ndarray,
        active_events: list[str],
        timestamp_s: float,
    ) -> None:
        h, w = canvas.shape[:2]

        # Decay banners
        for evt in list(self._banners):
            self._banners[evt] -= (1 / max(self.cfg.camera.fps, 1.0))
            if self._banners[evt] <= 0:
                del self._banners[evt]

        # Register new ones
        for evt in active_events:
            if evt not in self._banners:
                self._banners[evt] = self._banner_duration_s

        if not self._banners:
            return

        banner_h = 36
        y_offset = 10
        for evt in self._banners:
            colour = PALETTE.get(evt, (0, 80, 200))
            label  = f"⚠ {evt.upper().replace('_', ' ')}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = (w - tw) // 2 - 10
            by = y_offset

            overlay = canvas.copy()
            cv2.rectangle(overlay, (bx, by), (bx + tw + 20, by + banner_h), colour, -1)
            cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

            cv2.putText(canvas, label,
                        (bx + 10, by + banner_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += banner_h + 6

    # ------------------------------------------------------------------
    # HUD bar (top-left info strip) ------------------------------------

    def _draw_hud_bar(self, canvas: np.ndarray, timestamp_s: float, tracks: list[TrackState]) -> None:
        h, w = canvas.shape[:2]
        fps = self._compute_fps()

        cars = sum(1 for t in tracks if t.class_name in ("car", "truck"))
        buses = sum(1 for t in tracks if t.class_name == "bus")
        peds = sum(1 for t in tracks if t.class_name == "person")
        max_spd = max((t.speed_kmh for t in tracks if t.is_vehicle), default=0.0)

        # Background strip
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (450, 60), (10, 12, 20), -1)
        cv2.addWeighted(overlay, 0.76, canvas, 0.24, 0, canvas)

        ts_str = _fmt_timestamp(timestamp_s)
        cv2.putText(canvas, "WIUT Traffic AI Analytics", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"T {ts_str}  |  {fps:.1f} FPS  |  Max: {max_spd:.0f} km/h", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Flow: {cars} Cars | {buses} Buses | {peds} Peds", (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 240, 150), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Risk gauge (bottom-right) ----------------------------------------

    def _draw_risk_gauge(self, canvas: np.ndarray, risk: RiskScore) -> None:
        h, w = canvas.shape[:2]
        cx, cy, radius = w - 90, h - 90, 72

        # Background circle
        overlay = canvas.copy()
        cv2.circle(overlay, (cx, cy), radius + 6, (10, 12, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Arc gauge: -135° to +135°
        start_angle = -225   # OpenCV angles: clockwise from 3 o'clock
        total_arc   = 270
        filled_arc  = int(total_arc * risk.overall)
        colour      = risk.colour_bgr

        _draw_arc(canvas, cx, cy, radius, start_angle, total_arc, (50, 50, 50), 8)
        if filled_arc > 0:
            _draw_arc(canvas, cx, cy, radius, start_angle, filled_arc, colour, 8)

        # Score text
        pct_text = f"{risk.overall * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.80, 2)
        cv2.putText(canvas, pct_text, (cx - tw//2, cy + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, colour, 2, cv2.LINE_AA)

        level_text = risk.level
        (lw, lh), _ = cv2.getTextSize(level_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, level_text, (cx - lw//2, cy + th//2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)

        cv2.putText(canvas, "RISK", (cx - 16, cy - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Private helpers ---------------------------------------------------

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        return (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0] + 1e-9)


# ---------------------------------------------------------------------------
# Utility helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _draw_arc(
    canvas: np.ndarray,
    cx: int, cy: int, radius: int,
    start_angle: int, sweep_angle: int,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw an arc using cv2.ellipse (angles are clockwise from 3 o'clock)."""
    end_angle = start_angle + sweep_angle
    cv2.ellipse(canvas, (cx, cy), (radius, radius),
                0, start_angle, end_angle, colour, thickness, cv2.LINE_AA)


def _fmt_timestamp(ts: float) -> str:
    m = int(ts) // 60
    s = int(ts) % 60
    ms = int((ts - int(ts)) * 100)
    return f"{m:02d}:{s:02d}.{ms:02d}"
