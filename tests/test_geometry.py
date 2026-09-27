"""
tests/test_geometry.py
Unit tests for src/geometry.py
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.geometry import (
    bbox_crosses_line,
    bbox_iou,
    bbox_overlaps_polygon,
    calculate_ttc,
    clamp,
    heading_angle_degrees,
    point_in_polygon,
    segments_intersect,
    speed_kmh,
    trajectory_total_angle_change,
    vector_dot_normalized,
    velocity_vector,
)


class TestPointInPolygon:
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]

    def test_inside_centre(self):
        assert point_in_polygon((50, 50), self.square) is True

    def test_outside(self):
        assert point_in_polygon((200, 200), self.square) is False

    def test_on_edge_ish(self):
        # Just inside a corner area
        assert point_in_polygon((1, 1), self.square) is True


class TestSegmentsIntersect:
    def test_crossing_segments(self):
        assert segments_intersect(((0, 0), (10, 10)), ((0, 10), (10, 0))) is True

    def test_parallel_segments(self):
        assert segments_intersect(((0, 0), (10, 0)), ((0, 5), (10, 5))) is False

    def test_non_intersecting(self):
        assert segments_intersect(((0, 0), (5, 0)), ((6, 0), (10, 0))) is False


class TestBboxIou:
    def test_full_overlap(self):
        b = [0, 0, 100, 100]
        assert pytest.approx(bbox_iou(b, b), abs=1e-6) == 1.0

    def test_no_overlap(self):
        a = [0, 0, 50, 50]
        b = [100, 100, 200, 200]
        assert bbox_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        a = [0, 0, 100, 100]
        b = [50, 50, 150, 150]
        iou = bbox_iou(a, b)
        assert 0.1 < iou < 0.2  # known value ~1/7


class TestCalculateTtc:
    def test_approaching_objects(self):
        ttc = calculate_ttc((0, 0), (1, 0), (100, 0), (-1, 0))
        assert ttc == pytest.approx(50.0, rel=0.05)

    def test_diverging_returns_inf(self):
        ttc = calculate_ttc((0, 0), (-1, 0), (100, 0), (1, 0))
        assert math.isinf(ttc)

    def test_already_very_close_returns_zero(self):
        ttc = calculate_ttc((0, 0), (1, 0), (5, 0), (-1, 0), min_distance_px=10.0)
        assert ttc == 0.0


class TestVelocityVector:
    def test_simple_rightward(self):
        pts = [(i, 0) for i in range(10)]
        vx, vy = velocity_vector(pts)
        assert pytest.approx(vx, abs=0.5) == 1.0
        assert pytest.approx(vy, abs=0.1) == 0.0

    def test_single_point_returns_zero(self):
        vx, vy = velocity_vector([(5, 5)])
        assert vx == 0.0 and vy == 0.0


class TestSpeedKmh:
    def test_stationary(self):
        pts = [(50, 50)] * 10
        s = speed_kmh(pts, pixels_per_meter=22.0, fps=30.0)
        assert s == pytest.approx(0.0, abs=0.1)

    def test_moving(self):
        # 22 px/frame @ 30 fps = 1 m/s = 3.6 km/h → approx
        pts = [(i * 22, 0) for i in range(10)]
        s = speed_kmh(pts, pixels_per_meter=22.0, fps=30.0)
        assert s > 1.0  # at least moving


class TestVectorDot:
    def test_parallel_same_direction(self):
        assert vector_dot_normalized((1, 0), (1, 0)) == pytest.approx(1.0)

    def test_opposite_direction(self):
        assert vector_dot_normalized((1, 0), (-1, 0)) == pytest.approx(-1.0)

    def test_perpendicular(self):
        assert vector_dot_normalized((1, 0), (0, 1)) == pytest.approx(0.0, abs=1e-6)


class TestTrajectoryAngle:
    def test_straight_line_no_angle(self):
        pts = [(i * 10, 0) for i in range(10)]
        angle = trajectory_total_angle_change(pts)
        assert abs(angle) < 5.0

    def test_u_turn(self):
        # Semicircle: go right then reverse left
        pts = [(math.cos(a) * 50, math.sin(a) * 50) for a in [0, 0.5, 1.0, 1.57, 2.1, 2.8, 3.14]]
        angle = abs(trajectory_total_angle_change(pts))
        assert angle > 100.0  # Significant turn


class TestClamp:
    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_min(self):
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_max(self):
        assert clamp(5.0, 0.0, 1.0) == 1.0
