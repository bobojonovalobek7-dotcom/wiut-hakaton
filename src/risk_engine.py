"""
src/risk_engine.py
Part B — Real-Time Collision Risk Score Predictor.

Computes a risk index R(t) in [0.00, 1.00] representing the probability of
a collision within a configurable lookahead window (default: 5 seconds).

Components
----------
R_ttc       : Minimum pairwise Time-To-Collision contribution.
R_decel     : Sudden harsh-braking / deceleration spike contribution.
R_density   : Vehicle count density inside the intersection box.
R_conflict  : Count of trajectory intersection events projected over 5 s.

Final score is an exponential moving average (EMA) of these weighted components
so the HUD gauge is smooth rather than erratic.
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config, RiskEngineConfig
from .detector import TrackState
from .geometry import (
    bbox_overlaps_polygon,
    calculate_ttc,
    clamp,
    distance_between_centres,
    segments_intersect,
    velocity_vector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container ----------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class RiskScore:
    """Instantaneous risk assessment output."""
    overall: float                   # 0.00 – 1.00
    component_ttc: float
    component_decel: float
    component_density: float
    component_conflict: float
    ttc_min_s: float
    timestamp_s: float
    high_risk_pairs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def level(self) -> str:
        if self.overall < 0.30:
            return "LOW"
        elif self.overall < 0.60:
            return "MEDIUM"
        elif self.overall < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    @property
    def colour_bgr(self) -> tuple[int, int, int]:
        if self.overall < 0.30:
            return (0, 200, 80)      # green
        elif self.overall < 0.60:
            return (0, 165, 255)     # orange
        elif self.overall < 0.80:
            return (0, 50, 255)      # red
        else:
            return (60, 0, 180)      # deep red-purple


# ---------------------------------------------------------------------------
# Risk Engine ---------------------------------------------------------------
# ---------------------------------------------------------------------------

class CollisionRiskEngine:
    """
    Computes real-time collision risk for a stream of YOLO tracking frames.

    Parameters
    ----------
    config : Loaded project Config (contains risk_engine sub-config).
    """

    def __init__(self, config: Config) -> None:
        self.cfg   = config
        self.rcfg  = config.risk_engine
        self._ema: float = 0.0
        self._speed_history: dict[int, collections.deque] = {}
        self._prev_score: Optional[RiskScore] = None

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------

    def compute(
        self,
        tracks: list[TrackState],
        timestamp_s: float,
    ) -> RiskScore:
        """
        Compute the risk score for the current frame.

        Parameters
        ----------
        tracks      : All active tracking states from the detector.
        timestamp_s : Current video timestamp in seconds.

        Returns
        -------
        RiskScore
        """
        vehicles = [t for t in tracks if t.is_vehicle or t.is_bicycle]

        # Update speed history for deceleration computation
        for v in vehicles:
            if v.track_id not in self._speed_history:
                self._speed_history[v.track_id] = collections.deque(maxlen=30)
            self._speed_history[v.track_id].append(v.speed_kmh)

        # ── Component scores ──────────────────────────────────────────
        r_ttc,    ttc_min_s, high_risk_pairs = self._score_ttc(vehicles)
        r_decel                               = self._score_deceleration(vehicles)
        r_density                             = self._score_density(vehicles)
        r_conflict                            = self._score_conflict(vehicles)

        # ── Weighted sum ──────────────────────────────────────────────
        w = self.rcfg.weights
        raw = (
            w.ttc          * r_ttc    +
            w.deceleration * r_decel  +
            w.density      * r_density +
            w.conflict_angle * r_conflict
        )
        raw = clamp(raw, 0.0, 1.0)

        # ── EMA smoothing ─────────────────────────────────────────────
        alpha = self.rcfg.smoothing_factor
        self._ema = alpha * raw + (1 - alpha) * self._ema

        score = RiskScore(
            overall=round(self._ema, 4),
            component_ttc=round(r_ttc, 4),
            component_decel=round(r_decel, 4),
            component_density=round(r_density, 4),
            component_conflict=round(r_conflict, 4),
            ttc_min_s=round(ttc_min_s, 3),
            timestamp_s=round(timestamp_s, 3),
            high_risk_pairs=high_risk_pairs,
        )
        self._prev_score = score
        return score

    # ------------------------------------------------------------------
    # Component computations --------------------------------------------

    def _score_ttc(
        self, vehicles: list[TrackState]
    ) -> tuple[float, float, list[tuple[int, int]]]:
        """
        Compute R_ttc from minimum pairwise TTC.

        TTC in seconds is clipped at the lookahead window.
        Score = 1 - (ttc_min / lookahead_s) for approaching pairs.
        """
        fps = self.cfg.camera.fps
        lookahead = self.rcfg.lookahead_s
        thr_near_miss = self.cfg.thresholds.ttc_near_miss_s

        min_ttc = float("inf")
        high_risk_pairs: list[tuple[int, int]] = []

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                ttc_frames = calculate_ttc(a.centre, a.vel_px, b.centre, b.vel_px)
                ttc_s = ttc_frames / max(fps, 1.0)

                if ttc_s < lookahead:
                    min_ttc = min(min_ttc, ttc_s)
                    if ttc_s <= thr_near_miss * 2:
                        high_risk_pairs.append((a.track_id, b.track_id))

        if min_ttc == float("inf"):
            return 0.0, float("inf"), high_risk_pairs

        # Sigmoid-like mapping: score peaks as TTC → 0
        score = clamp(1.0 - (min_ttc / lookahead) ** 0.6, 0.0, 1.0)
        return score, min_ttc, high_risk_pairs

    def _score_deceleration(self, vehicles: list[TrackState]) -> float:
        """
        Detect sudden hard braking events using speed history.

        If any vehicle drops > 15 km/h within 10 frames, that's a braking spike.
        """
        max_drop = 0.0
        for v in vehicles:
            hist = list(self._speed_history.get(v.track_id, []))
            if len(hist) < 10:
                continue
            recent = hist[-10:]
            drop = max(recent) - min(recent[-3:])
            max_drop = max(max_drop, drop)

        # Scale: 0 km/h drop → 0 risk;  30 km/h drop → 1.0 risk
        return clamp(max_drop / 30.0, 0.0, 1.0)

    def _score_density(self, vehicles: list[TrackState]) -> float:
        """
        Count vehicles inside or near the intersection box.

        Normalise against a maximum expected occupancy (default: 10).
        """
        int_poly = self.cfg.intersection_box
        if not int_poly:
            return 0.0

        count = sum(
            1 for v in vehicles
            if bbox_overlaps_polygon(v.bbox, int_poly)
        )
        max_occupancy = 10
        return clamp(count / max_occupancy, 0.0, 1.0)

    def _score_conflict(self, vehicles: list[TrackState]) -> float:
        """
        Estimate trajectory conflict by projecting each vehicle's path
        over the 5-second lookahead and checking for projected path crossings.
        """
        fps = self.cfg.camera.fps
        lookahead_frames = int(self.rcfg.lookahead_s * fps)
        conflicts = 0

        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                # Project future position
                a_end = (
                    a.centre[0] + a.vel_px[0] * lookahead_frames,
                    a.centre[1] + a.vel_px[1] * lookahead_frames,
                )
                b_end = (
                    b.centre[0] + b.vel_px[0] * lookahead_frames,
                    b.centre[1] + b.vel_px[1] * lookahead_frames,
                )
                if segments_intersect(
                    (a.centre, a_end),
                    (b.centre, b_end),
                ):
                    conflicts += 1

        # Normalise: 5+ conflicts = full risk
        return clamp(conflicts / 5.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convenience -------------------------------------------------------

    @property
    def current_score(self) -> RiskScore | None:
        return self._prev_score

    def reset(self) -> None:
        self._ema = 0.0
        self._speed_history.clear()
        self._prev_score = None
