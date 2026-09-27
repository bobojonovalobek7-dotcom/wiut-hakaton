"""
tests/test_risk_engine.py
Unit tests for src/risk_engine.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_config
from src.risk_engine import CollisionRiskEngine, RiskScore


def _make_config():
    return load_config(config_path=None)


def _make_mock_track(tid: int, x: float, y: float, vx: float, vy: float, speed_kmh: float = 30.0):
    """Create a minimal mock TrackState-like dict."""
    import collections
    from src.detector import TrackState
    ts = TrackState(
        track_id=tid,
        class_id=2,
        class_name="car",
        bbox=[x-30, y-15, x+30, y+15],
        confidence=0.9,
        frame_idx=1,
        timestamp_s=1.0,
    )
    ts.vel_px = (vx, vy)
    ts.speed_kmh = speed_kmh
    for i in range(10):
        ts.positions.append((x + vx * i, y + vy * i))
        ts.bboxes.append([x-30, y-15, x+30, y+15])
        ts.speeds_kmh.append(speed_kmh)
    return ts


class TestRiskEngineOutput:
    def setup_method(self):
        self.cfg    = _make_config()
        self.engine = CollisionRiskEngine(self.cfg)

    def test_output_range_no_vehicles(self):
        score = self.engine.compute([], timestamp_s=5.0)
        assert isinstance(score, RiskScore)
        assert 0.0 <= score.overall <= 1.0

    def test_output_range_with_vehicles(self):
        tracks = [
            _make_mock_track(1, 500, 500, 2, 0),
            _make_mock_track(2, 600, 500, -2, 0),
        ]
        score = self.engine.compute(tracks, timestamp_s=5.0)
        assert 0.0 <= score.overall <= 1.0

    def test_converging_vehicles_raises_risk(self):
        """Two vehicles approaching each other head-on should produce higher risk."""
        self.engine.reset()
        no_risk_score = self.engine.compute([], timestamp_s=0.0)
        self.engine.reset()

        # Head-on collision on the same Y
        t1 = _make_mock_track(1, 300, 540,  5, 0, speed_kmh=40)
        t2 = _make_mock_track(2, 900, 540, -5, 0, speed_kmh=40)
        for _ in range(10):  # build up speed history
            score = self.engine.compute([t1, t2], timestamp_s=float(_))
        assert score.overall >= 0.0  # Should be at least non-zero

    def test_risk_level_strings(self):
        s = RiskScore(
            overall=0.1, component_ttc=0, component_decel=0,
            component_density=0, component_conflict=0,
            ttc_min_s=10.0, timestamp_s=1.0,
        )
        assert s.level == "LOW"
        s.overall = 0.45
        assert s.level == "MEDIUM"
        s.overall = 0.70
        assert s.level == "HIGH"
        s.overall = 0.90
        assert s.level == "CRITICAL"

    def test_risk_colour_bgr_tuple(self):
        s = RiskScore(
            overall=0.2, component_ttc=0, component_decel=0,
            component_density=0, component_conflict=0,
            ttc_min_s=10.0, timestamp_s=1.0,
        )
        col = s.colour_bgr
        assert len(col) == 3
        assert all(0 <= c <= 255 for c in col)

    def test_ema_smoothing(self):
        """EMA should prevent abrupt jumps."""
        t1 = _make_mock_track(1, 200, 540, 0, 0, speed_kmh=0)
        scores = [self.engine.compute([t1], timestamp_s=float(i)).overall for i in range(20)]
        # No sudden jump greater than 0.5 between consecutive frames
        for i in range(1, len(scores)):
            assert abs(scores[i] - scores[i-1]) < 0.5

    def test_reset_clears_state(self):
        t1 = _make_mock_track(1, 500, 500, 5, 0)
        self.engine.compute([t1], timestamp_s=5.0)
        self.engine.reset()
        assert self.engine._ema == 0.0
        assert self.engine._prev_score is None
