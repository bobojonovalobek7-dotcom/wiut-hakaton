"""
src/rules.py
Traffic Event Rule Engine — evaluates all 14 traffic incident categories.

Each event is a named rule function.  The TrafficRuleEngine orchestrates all
rules per frame and maintains temporal debouncing so that continuous triggers
collapse into single incidents with precise t_start / t_end timestamps.

Events Covered
--------------
 1  accident            — sudden velocity collapse post-overlap
 2  near_miss           — converging trajectories with TTC < 1.5 s
 3  red_light           — crosses stop line while signal is RED
 4  wrong_way           — trajectory opposes lane flow vector
 5  illegal_u_turn      — cumulative heading change > 135°
 6  stopped_vehicle     — stationary > 10 s outside waiting queues
 7  jaywalking          — pedestrian outside zebra on active road
 8  failure_to_yield    — vehicle crosses zebra while pedestrian present
 9  illegal_turn        — turns from wrong lane
10  solid_line_crossing — centroid crosses solid divider
11  stop_line           — crosses stop line on RED then stops before box
12  congestion          — average lane speed < 5 km/h for > 5 s
13  road_obstacle       — unknown static object in active driveway
14  fire_smoke          — HSV flame/smoke detection
"""

from __future__ import annotations

import collections
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import Config
from .detector import TrackState, TrafficLightStateDetector, COCO_CLASS_NAMES
from .geometry import (
    bbox_crosses_line,
    bbox_overlaps_polygon,
    bbox_iou,
    calculate_ttc,
    distance_between_centres,
    is_fire_pixel_region,
    point_in_polygon,
    speed_kmh as geo_speed_kmh,
    trajectory_total_angle_change,
    vector_dot_normalized,
    velocity_vector,
    clamp,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event record ---------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class TrafficEvent:
    event: str
    t_start: float
    t_end: float
    confidence: float
    track_ids: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event": self.event,
            "t_start": round(float(self.t_start), 3),
            "t_end": round(float(self.t_end), 3),
            "duration": round(float(self.t_end - self.t_start), 3),
            "confidence": round(float(self.confidence), 3),
            "track_ids": [int(x) for x in self.track_ids],
        }


# ---------------------------------------------------------------------------
# Debounce state  -----------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class _ActiveEvent:
    """Ongoing, not-yet-closed event."""
    event: str
    t_start: float
    t_last: float
    confidence: float
    track_ids: list[int]
    meta: dict = field(default_factory=dict)

    def update(self, t: float, conf: float) -> None:
        self.t_last = t
        self.confidence = max(self.confidence, conf)

    def close(self, gap_tolerance_s: float = 1.0, current_t: float = 0.0) -> bool:
        """Returns True if event should be closed (no trigger in > gap_tolerance_s)."""
        return (current_t - self.t_last) > gap_tolerance_s


# ---------------------------------------------------------------------------
# Rule Engine ---------------------------------------------------------------
# ---------------------------------------------------------------------------

class TrafficRuleEngine:
    """
    Evaluates all 14 traffic event rules per frame, maintains temporal state,
    and emits fully formed TrafficEvent objects when incidents end.

    Parameters
    ----------
    config          : Loaded project Config object.
    tl_detector     : Shared TrafficLightStateDetector instance.
    gap_tolerance_s : Seconds without a new trigger before closing an active event.
    """

    # Events that can fire simultaneously for multiple track-pairs
    PAIRWISE_EVENTS = {"near_miss", "failure_to_yield", "accident"}

    def __init__(
        self,
        config: Config,
        tl_detector: TrafficLightStateDetector,
        gap_tolerance_s: float = 1.5,
    ) -> None:
        self.cfg = config
        self.tl  = tl_detector
        self.gap = gap_tolerance_s
        self.t   = config

        # Active (ongoing) incidents keyed by (event, track_id) or (event,)
        self._active: dict[tuple, _ActiveEvent] = {}

        # Finalized event log
        self.events: list[TrafficEvent] = []

        # Per-track stop timer: track_id → timestamp when it first stopped
        self._stop_start: dict[int, float] = {}

        # Congestion tracking
        self._congestion_start: Optional[float] = None

        # Fire/smoke consecutive frame counter
        self._fire_frames: int = 0

        # Stop-line-then-stop tracking
        self._crossed_stop_line_on_red: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API --------------------------------------------------------

    def evaluate(
        self,
        frame: np.ndarray,
        tracks: list[TrackState],
        frame_idx: int,
        timestamp_s: float,
    ) -> list[TrafficEvent]:
        """
        Evaluate all rules for the current frame.

        Returns any newly CLOSED incidents that were finalized this frame.
        """
        newly_closed: list[TrafficEvent] = []

        # Update active event timers — close any that have timed out
        to_close = [
            k for k, ae in self._active.items()
            if ae.close(self.gap, timestamp_s)
        ]
        for k in to_close:
            ae = self._active.pop(k)
            evt = TrafficEvent(
                event=ae.event,
                t_start=ae.t_start,
                t_end=ae.t_last,
                confidence=ae.confidence,
                track_ids=ae.track_ids,
                meta=ae.meta,
            )
            self.events.append(evt)
            newly_closed.append(evt)
            logger.info("EVENT CLOSED  [%s]  %.2f-%.2fs  tracks=%s",
                        ae.event, ae.t_start, ae.t_last, ae.track_ids)

        thr = self.cfg.thresholds
        vehicles  = [t for t in tracks if t.is_vehicle]
        peds      = [t for t in tracks if t.is_pedestrian]
        all_movers = vehicles + [t for t in tracks if t.is_bicycle]

        # ── Rule evaluations ──────────────────────────────────────────
        self._rule_stopped_vehicle(all_movers, timestamp_s, thr)
        self._rule_wrong_way(vehicles, timestamp_s, thr)
        self._rule_illegal_u_turn(all_movers, timestamp_s, thr)
        self._rule_solid_line_crossing(vehicles, timestamp_s, thr)
        self._rule_red_light_and_stop_line(vehicles, timestamp_s, thr)
        self._rule_jaywalking(peds, timestamp_s, thr, frame)
        self._rule_failure_to_yield(vehicles, peds, timestamp_s, thr)
        self._rule_near_miss(all_movers, timestamp_s, thr)
        self._rule_accident(all_movers, timestamp_s, thr)
        self._rule_illegal_turn(vehicles, timestamp_s, thr)
        self._rule_congestion(vehicles, timestamp_s, thr)
        self._rule_road_obstacle(tracks, timestamp_s, thr)
        self._rule_fire_smoke(frame, timestamp_s, thr)
        self._rule_junction_blocking(vehicles, timestamp_s, thr)
        self._rule_yielding_to_pedestrian(vehicles, peds, timestamp_s, thr)
        self._rule_speeding(vehicles, timestamp_s, thr)

        return newly_closed

    def flush_open_events(self, final_timestamp_s: float) -> list[TrafficEvent]:
        """Close all still-open events at end of video."""
        flushed: list[TrafficEvent] = []
        for k, ae in list(self._active.items()):
            evt = TrafficEvent(
                event=ae.event,
                t_start=ae.t_start,
                t_end=final_timestamp_s,
                confidence=ae.confidence,
                track_ids=ae.track_ids,
                meta=ae.meta,
            )
            self.events.append(evt)
            flushed.append(evt)
        self._active.clear()
        return flushed

    # ------------------------------------------------------------------
    # Active event helpers ----------------------------------------------

    def _fire(
        self,
        key: tuple,
        event_name: str,
        timestamp_s: float,
        confidence: float,
        track_ids: list[int],
        meta: dict | None = None,
    ) -> None:
        if key in self._active:
            self._active[key].update(timestamp_s, confidence)
        else:
            self._active[key] = _ActiveEvent(
                event=event_name,
                t_start=timestamp_s,
                t_last=timestamp_s,
                confidence=confidence,
                track_ids=track_ids,
                meta=meta or {},
            )

    # ------------------------------------------------------------------
    # ── Rule 6: Stopped Vehicle ──────────────────────────────────────

    def _rule_stopped_vehicle(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        waiting_polygons = [sl.points for sl in self.cfg.stop_lines]
        for v in vehicles:
            tid = v.track_id
            if v.speed_kmh < thr.stopped_vehicle_speed_kmh:
                # Check not in stop-line waiting area
                in_wait_zone = any(
                    point_in_polygon(v.centre, z) for z in waiting_polygons
                )
                if in_wait_zone:
                    self._stop_start.pop(tid, None)
                    continue
                if tid not in self._stop_start:
                    self._stop_start[tid] = ts
                stopped_dur = ts - self._stop_start[tid]
                if stopped_dur >= thr.stopped_vehicle_duration_s:
                    self._fire(
                        ("stopped_vehicle", tid), "stopped_vehicle",
                        ts, 0.75 + clamp(stopped_dur/60, 0, 0.20),
                        [tid], {"duration_s": stopped_dur},
                    )
            else:
                self._stop_start.pop(tid, None)

    # ------------------------------------------------------------------
    # ── Rule 4: Wrong Way ────────────────────────────────────────────

    def _rule_wrong_way(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        for v in vehicles:
            if len(v.positions) < 15:
                continue
            if speed_kmh_from_track(v, self.cfg) < 3.0:
                continue
            for lane in self.cfg.lanes:
                if not bbox_overlaps_polygon(v.bbox, lane.polygon):
                    continue
                vel = v.vel_px
                flow = tuple(lane.flow_vector)
                dot = vector_dot_normalized(vel, flow)   # type: ignore[arg-type]
                if dot < thr.wrong_way_dot_thresh:
                    self._fire(
                        ("wrong_way", v.track_id), "wrong_way",
                        ts, clamp((-dot - 0.45) / 0.55 * 0.6 + 0.4, 0.4, 1.0),
                        [v.track_id], {"lane": lane.id, "dot": dot},
                    )

    # ------------------------------------------------------------------
    # ── Rule 5: Illegal U-Turn ───────────────────────────────────────

    def _rule_illegal_u_turn(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        int_poly = self.cfg.intersection_box
        for v in vehicles:
            if len(v.positions) < 20:
                continue
            if not bbox_overlaps_polygon(v.bbox, int_poly):
                continue
            angle = abs(trajectory_total_angle_change(list(v.positions)))
            if angle >= thr.u_turn_angle_deg:
                conf = clamp(0.5 + (angle - thr.u_turn_angle_deg) / 90, 0.5, 0.95)
                self._fire(
                    ("illegal_u_turn", v.track_id), "illegal_u_turn",
                    ts, conf, [v.track_id], {"angle_deg": angle},
                )

    # ------------------------------------------------------------------
    # ── Rule 10: Solid Line Crossing ─────────────────────────────────

    def _rule_solid_line_crossing(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        for v in vehicles:
            if len(v.bboxes) < 2:
                continue
            curr_bbox = v.bbox
            prev_bbox = list(v.bboxes)[-2]
            for solid in self.cfg.solid_lines:
                p1, p2 = solid.points[0], solid.points[1]
                if bbox_crosses_line(curr_bbox, p1, p2, prev_bbox):
                    self._fire(
                        ("solid_line_crossing", v.track_id), "solid_line_crossing",
                        ts, 0.80, [v.track_id], {"line": solid.id},
                    )

    # ------------------------------------------------------------------
    # ── Rule 3: Red Light | Rule 11: Stop Line ───────────────────────

    def _rule_red_light_and_stop_line(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        int_poly = self.cfg.intersection_box
        for v in vehicles:
            if len(v.bboxes) < 2:
                continue
            prev_bbox = list(v.bboxes)[-2]

            for sl in self.cfg.stop_lines:
                p1, p2 = sl.points[0], sl.points[1]
                crossed = bbox_crosses_line(v.bbox, p1, p2, prev_bbox)
                if not crossed:
                    continue

                tl_state = self.tl.get_state(sl.associated_traffic_light, "RED")
                if tl_state == "RED":
                    # Determine if vehicle enters intersection box or stops before it
                    enters_box = bbox_overlaps_polygon(v.bbox, int_poly)
                    if enters_box:
                        # Red light running
                        self._fire(
                            ("red_light", v.track_id), "red_light",
                            ts, 0.90, [v.track_id],
                            {"stop_line": sl.id, "tl_state": tl_state},
                        )
                    else:
                        # Stopped before intersection box on RED
                        self._crossed_stop_line_on_red[v.track_id] = ts

            # Check stop-line event: crossed on RED, now stopped before box
            if v.track_id in self._crossed_stop_line_on_red:
                in_box = bbox_overlaps_polygon(v.bbox, int_poly)
                if not in_box and v.speed_kmh < thr.stopped_vehicle_speed_kmh:
                    self._fire(
                        ("stop_line", v.track_id), "stop_line",
                        ts, 0.80, [v.track_id],
                    )
                elif in_box:
                    # Already entered box → this was a red_light violation, not stop_line
                    del self._crossed_stop_line_on_red[v.track_id]

    # ------------------------------------------------------------------
    # ── Rule 7: Jaywalking ───────────────────────────────────────────

    def _rule_jaywalking(
        self,
        peds: list[TrackState],
        ts: float,
        thr,
        frame: np.ndarray,
    ) -> None:
        road_poly  = self.cfg.road_polygon
        zebra_list = [z.polygon for z in self.cfg.zebra_crossings]

        for ped in peds:
            if not bbox_overlaps_polygon(ped.bbox, road_poly):
                continue
            in_zebra = any(bbox_overlaps_polygon(ped.bbox, zp) for zp in zebra_list)
            if in_zebra:
                continue
            self._fire(
                ("jaywalking", ped.track_id), "jaywalking",
                ts, 0.70, [ped.track_id],
            )

    # ------------------------------------------------------------------
    # ── Rule 8: Failure to Yield ─────────────────────────────────────

    def _rule_failure_to_yield(
        self,
        vehicles: list[TrackState],
        peds: list[TrackState],
        ts: float,
        thr,
    ) -> None:
        for zebra in self.cfg.zebra_crossings:
            z_poly = zebra.polygon
            active_peds = [p for p in peds if bbox_overlaps_polygon(p.bbox, z_poly)]
            if not active_peds:
                continue
            for v in vehicles:
                if not bbox_overlaps_polygon(v.bbox, z_poly):
                    continue
                if v.speed_kmh < 1.0:
                    continue  # Vehicle already stopped
                ped_ids = [p.track_id for p in active_peds]
                conf = clamp(0.6 + v.speed_kmh / 50, 0.6, 0.95)
                self._fire(
                    ("failure_to_yield", v.track_id), "failure_to_yield",
                    ts, conf, [v.track_id] + ped_ids,
                    {"zebra": zebra.id},
                )

    # ------------------------------------------------------------------
    # ── Rule 2: Near Miss ────────────────────────────────────────────

    def _rule_near_miss(
        self, movers: list[TrackState], ts: float, thr
    ) -> None:
        fps = self.cfg.camera.fps
        for i in range(len(movers)):
            for j in range(i + 1, len(movers)):
                a, b = movers[i], movers[j]
                ttc_frames = calculate_ttc(a.centre, a.vel_px, b.centre, b.vel_px)
                ttc_s = ttc_frames / max(fps, 1.0)
                if 0.01 < ttc_s <= thr.ttc_near_miss_s:
                    iou = bbox_iou(a.bbox, b.bbox)
                    if iou > 0.01:
                        continue  # Actual overlap → handled by accident rule
                    conf = clamp(1.0 - ttc_s / thr.ttc_near_miss_s, 0.50, 0.99)
                    key = ("near_miss", min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                    self._fire(key, "near_miss", ts, conf,
                               [a.track_id, b.track_id], {"ttc_s": ttc_s})

    # ------------------------------------------------------------------
    # ── Rule 1: Accident ─────────────────────────────────────────────

    def _rule_accident(
        self, movers: list[TrackState], ts: float, thr
    ) -> None:
        for i in range(len(movers)):
            for j in range(i + 1, len(movers)):
                a, b = movers[i], movers[j]
                iou = bbox_iou(a.bbox, b.bbox)
                if iou < 0.08:
                    continue
                # Speed collapse check
                speed_a = a.speed_kmh
                speed_b = b.speed_kmh

                hist_a = list(a.speeds_kmh)
                hist_b = list(b.speeds_kmh)
                prev_a = float(np.mean(hist_a[:-5])) if len(hist_a) > 5 else speed_a
                prev_b = float(np.mean(hist_b[:-5])) if len(hist_b) > 5 else speed_b

                if prev_a > 5.0 and (speed_a / max(prev_a, 0.1)) < (1 - thr.accident_speed_drop_ratio):
                    conf = clamp(0.70 + iou * 0.25, 0.70, 0.99)
                    key = ("accident", min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                    self._fire(key, "accident", ts, conf,
                               [a.track_id, b.track_id], {"iou": iou})

    # ------------------------------------------------------------------
    # ── Rule 9: Illegal Turn ─────────────────────────────────────────

    def _rule_illegal_turn(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        int_poly = self.cfg.intersection_box
        for v in vehicles:
            if not bbox_overlaps_polygon(v.bbox, int_poly):
                continue
            if len(v.positions) < 10:
                continue

            # Find which lane the vehicle came from (last 20 frames)
            approach_pts = list(v.positions)[:-5]  # trajectory before intersection
            origin_lane = None
            for lane in self.cfg.lanes:
                if any(point_in_polygon(p, lane.polygon) for p in approach_pts):
                    origin_lane = lane
                    break

            if origin_lane is None:
                continue

            # Current heading
            recent = list(v.positions)
            vel = velocity_vector(recent, 5)
            if math.hypot(*vel) < 1.0:
                continue

            heading = math.degrees(math.atan2(vel[1], vel[0]))
            flow_heading = math.degrees(math.atan2(origin_lane.flow_vector[1],
                                                    origin_lane.flow_vector[0]))
            turn_angle = abs((heading - flow_heading + 180) % 360 - 180)

            # Classify movement
            if turn_angle < 30:
                movement = "straight"
            elif turn_angle < 120:
                movement = "left" if heading > flow_heading else "right"
            else:
                movement = "u_turn"

            if movement not in origin_lane.allowed_movements and movement != "straight":
                conf = clamp(0.50 + turn_angle / 360, 0.50, 0.90)
                self._fire(
                    ("illegal_turn", v.track_id), "illegal_turn",
                    ts, conf, [v.track_id],
                    {"origin_lane": origin_lane.id, "movement": movement},
                )

    # ------------------------------------------------------------------
    # ── Rule 12: Congestion ──────────────────────────────────────────

    def _rule_congestion(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        if not vehicles:
            return

        slow_lane_count = 0
        for lane in self.cfg.lanes:
            lane_vehicles = [
                v for v in vehicles
                if bbox_overlaps_polygon(v.bbox, lane.polygon)
            ]
            if not lane_vehicles:
                continue
            avg_speed = float(np.mean([v.speed_kmh for v in lane_vehicles]))
            if avg_speed < thr.congestion_speed_threshold_kmh:
                slow_lane_count += 1

        if slow_lane_count >= thr.congestion_min_lanes:
            if self._congestion_start is None:
                self._congestion_start = ts
            dur = ts - self._congestion_start
            if dur >= thr.congestion_min_duration_s:
                conf = clamp(0.60 + dur / 60, 0.60, 0.95)
                self._fire(
                    ("congestion",), "congestion",
                    ts, conf, [],
                    {"slow_lanes": slow_lane_count, "duration_s": dur},
                )
        else:
            self._congestion_start = None

    # ------------------------------------------------------------------
    # ── Rule 13: Road Obstacle ───────────────────────────────────────

    def _rule_road_obstacle(
        self, tracks: list[TrackState], ts: float, thr
    ) -> None:
        road_poly = self.cfg.road_polygon
        for t in tracks:
            if t.is_vehicle or t.is_pedestrian or t.is_bicycle:
                continue
            if not bbox_overlaps_polygon(t.bbox, road_poly):
                continue
            if t.speed_kmh > 5.0:
                continue
            # Unknown class in road driveway
            if tid_in_stop_timer := self._stop_start.get(t.track_id):
                dur = ts - tid_in_stop_timer
                if dur >= thr.road_obstacle_stationary_s:
                    self._fire(
                        ("road_obstacle", t.track_id), "road_obstacle",
                        ts, 0.65, [t.track_id], {"duration_s": dur},
                    )
            else:
                self._stop_start[t.track_id] = ts

    # ------------------------------------------------------------------
    # ── Rule 14: Fire / Smoke ────────────────────────────────────────

    def _rule_fire_smoke(
        self, frame: np.ndarray, ts: float, thr
    ) -> None:
        detected, max_area = is_fire_pixel_region(
            frame, thr.fire_smoke_min_contour_area
        )
        if detected:
            self._fire_frames += 1
            if self._fire_frames >= thr.fire_smoke_min_frames:
                conf = clamp(0.55 + max_area / 50000, 0.55, 0.95)
                self._fire(
                    ("fire_smoke",), "fire_smoke",
                    ts, conf, [],
                    {"max_area": max_area},
                )
        else:
            self._fire_frames = max(0, self._fire_frames - 1)

    # ------------------------------------------------------------------
    # ── Rule 15: Junction / Intersection Blocking (Trapped Vehicle) ──

    def _rule_junction_blocking(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        """Detect vehicles trapped/stopped in the middle of the intersection box."""
        int_poly = self.cfg.intersection_box
        for v in vehicles:
            tid = v.track_id
            if point_in_polygon(v.centre, int_poly) or bbox_overlaps_polygon(v.bbox, int_poly):
                if v.speed_kmh < 4.0:
                    if tid not in self._stop_start:
                        self._stop_start[tid] = ts
                    dur = ts - self._stop_start[tid]
                    if dur >= 3.0:
                        self._fire(
                            ("junction_blocking", tid), "junction_blocking",
                            ts, clamp(0.70 + dur / 20.0, 0.70, 0.95),
                            [tid], {"duration_s": round(dur, 2)},
                        )

    # ------------------------------------------------------------------
    # ── Rule 16: Yielding to Pedestrians ──────────────────────────────

    def _rule_yielding_to_pedestrian(
        self, vehicles: list[TrackState], peds: list[TrackState], ts: float, thr
    ) -> None:
        """Detect vehicle slowing down or stopping to yield for pedestrian on zebra crossing."""
        if not peds or not vehicles:
            return
        for z in self.cfg.zebra_crossings:
            peds_on_z = [p for p in peds if p.bbox and len(p.bbox) >= 4 and bbox_overlaps_polygon(p.bbox, z.polygon)]
            if not peds_on_z:
                continue
            for v in vehicles:
                if not v.bbox or len(v.bbox) < 4:
                    continue
                dists = [distance_between_centres(v.bbox, p.bbox) for p in peds_on_z if p.bbox and len(p.bbox) >= 4]
                if not dists:
                    continue
                dist = min(dists)
                if dist < 180.0 and v.speed_kmh < 5.0 and len(v.positions) > 5:
                    self._fire(
                        ("yielding_to_pedestrian", v.track_id), "yielding_to_pedestrian",
                        ts, 0.85, [v.track_id], {"ped_ids": [p.track_id for p in peds_on_z]},
                    )

    # ------------------------------------------------------------------
    # ── Rule 17: Overspeeding ─────────────────────────────────────────

    def _rule_speeding(
        self, vehicles: list[TrackState], ts: float, thr
    ) -> None:
        """Detect vehicle exceeding urban intersection speed limit (> 65 km/h)."""
        for v in vehicles:
            if v.speed_kmh > 65.0:
                conf = clamp(0.70 + (v.speed_kmh - 65.0) / 40.0, 0.70, 0.98)
                self._fire(
                    ("speeding", v.track_id), "speeding",
                    ts, conf, [v.track_id],
                    {"speed_kmh": round(v.speed_kmh, 1), "limit_kmh": 60.0},
                )


# ---------------------------------------------------------------------------
# Utility -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def speed_kmh_from_track(track: TrackState, cfg: Config) -> float:
    """Return km/h speed for a track using project calibration settings."""
    return geo_speed_kmh(
        list(track.positions),
        cfg.camera.pixels_per_meter,
        cfg.camera.fps,
        window=6,
    )
