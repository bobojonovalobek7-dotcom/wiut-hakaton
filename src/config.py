"""
src/config.py
Configuration loader and validator for the WIUT Traffic Event Detection System.

Loads config.json, validates all geometry fields, and provides scaled coordinate
accessors when the actual video resolution differs from the configured base resolution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data-classes (typed schema) ------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    resolution: tuple[int, int]   # (width, height)
    fps: float
    pixels_per_meter: float


@dataclass
class LaneConfig:
    id: str
    name: str
    polygon: list[list[float]]
    flow_vector: list[float]       # unit vector [dx, dy] dominant flow
    allowed_movements: list[str]   # 'straight', 'left', 'right', 'u_turn'
    speed_limit_kmh: float


@dataclass
class StopLineConfig:
    id: str
    name: str
    points: list[list[float]]      # [[x1,y1],[x2,y2]]
    associated_traffic_light: str
    flow_direction: list[float]


@dataclass
class ZebraConfig:
    id: str
    name: str
    polygon: list[list[float]]


@dataclass
class SolidLineConfig:
    id: str
    name: str
    points: list[list[float]]      # [[x1,y1],[x2,y2]]


@dataclass
class TrafficLightConfig:
    id: str
    name: str
    bbox: list[int]                # [x1, y1, x2, y2] ROI in frame
    default_initial_state: str     # 'RED', 'GREEN', 'YELLOW'
    cycle_red_s: float
    cycle_green_s: float
    cycle_yellow_s: float


@dataclass
class ThresholdConfig:
    stopped_vehicle_duration_s: float = 10.0
    stopped_vehicle_speed_kmh: float = 2.5
    ttc_near_miss_s: float = 1.5
    accident_speed_drop_ratio: float = 0.70
    accident_min_decel_ms2: float = 6.0
    accident_post_stop_s: float = 2.0
    congestion_speed_threshold_kmh: float = 5.0
    congestion_min_lanes: int = 2
    congestion_min_duration_s: float = 5.0
    u_turn_angle_deg: float = 135.0
    wrong_way_dot_thresh: float = -0.45
    wrong_way_min_distance_px: float = 50.0
    failure_to_yield_distance_px: float = 120.0
    jaywalking_min_duration_s: float = 1.2
    solid_line_cross_dist_px: float = 12.0
    road_obstacle_stationary_s: float = 3.5
    fire_smoke_min_contour_area: float = 400.0
    fire_smoke_min_frames: int = 10


@dataclass
class RiskWeights:
    ttc: float = 0.40
    deceleration: float = 0.25
    density: float = 0.20
    conflict_angle: float = 0.15


@dataclass
class RiskEngineConfig:
    lookahead_s: float = 5.0
    weights: RiskWeights = field(default_factory=RiskWeights)
    smoothing_factor: float = 0.20


@dataclass
class Config:
    """Top-level configuration container."""
    camera: CameraConfig
    intersection_box: list[list[float]]
    road_polygon: list[list[float]]
    lanes: list[LaneConfig]
    stop_lines: list[StopLineConfig]
    zebra_crossings: list[ZebraConfig]
    solid_lines: list[SolidLineConfig]
    traffic_lights: list[TrafficLightConfig]
    thresholds: ThresholdConfig
    risk_engine: RiskEngineConfig

    # Computed at load time
    scale_x: float = 1.0
    scale_y: float = 1.0


# ---------------------------------------------------------------------------
# Default 1080p fallback (used when config.json is absent) ------------------
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.json"

# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    actual_resolution: tuple[int, int] | None = None,
) -> Config:
    """
    Load and validate the configuration file.

    Parameters
    ----------
    config_path:
        Path to config.json.  Defaults to ``config.json`` in the project root.
    actual_resolution:
        ``(width, height)`` of the video being processed.  If provided and
        different from ``camera.resolution``, all polygon / line coordinates
        are scaled to match.

    Returns
    -------
    Config
        Fully populated, validated configuration object.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.warning(
            "config.json not found at %s – using built-in 1080p defaults.", path
        )
        raw: dict[str, Any] = _builtin_defaults()
    else:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    cfg = _parse_raw(raw)

    # Rescale coordinates if video resolution differs from configured resolution
    if actual_resolution and actual_resolution != tuple(cfg.camera.resolution):
        aw, ah = actual_resolution
        bw, bh = cfg.camera.resolution
        cfg.scale_x = aw / bw
        cfg.scale_y = ah / bh
        logger.info(
            "Scaling ROI from %dx%d -> %dx%d  (sx=%.3f, sy=%.3f)",
            bw, bh, aw, ah, cfg.scale_x, cfg.scale_y,
        )
        cfg = _rescale_config(cfg)

    return cfg


def get_np_polygon(coords: list[list[float]]) -> np.ndarray:
    """Convert list-of-[x,y] to numpy int32 array suitable for cv2."""
    return np.array(coords, dtype=np.int32)


# ---------------------------------------------------------------------------
# Private helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_raw(raw: dict[str, Any]) -> Config:
    cam_raw = raw["camera"]
    camera = CameraConfig(
        resolution=tuple(cam_raw["resolution"]),  # type: ignore[arg-type]
        fps=float(cam_raw["fps"]),
        pixels_per_meter=float(cam_raw["pixels_per_meter"]),
    )

    lanes = [
        LaneConfig(
            id=l["id"],
            name=l["name"],
            polygon=l["polygon"],
            flow_vector=l["flow_vector"],
            allowed_movements=l["allowed_movements"],
            speed_limit_kmh=float(l["speed_limit_kmh"]),
        )
        for l in raw.get("lanes", [])
    ]

    stop_lines = [
        StopLineConfig(
            id=s["id"],
            name=s["name"],
            points=s["points"],
            associated_traffic_light=s["associated_traffic_light"],
            flow_direction=s["flow_direction"],
        )
        for s in raw.get("stop_lines", [])
    ]

    zebras = [
        ZebraConfig(id=z["id"], name=z["name"], polygon=z["polygon"])
        for z in raw.get("zebra_crossings", [])
    ]

    solid_lines = [
        SolidLineConfig(id=sl["id"], name=sl["name"], points=sl["points"])
        for sl in raw.get("solid_lines", [])
    ]

    traffic_lights = [
        TrafficLightConfig(
            id=tl["id"],
            name=tl["name"],
            bbox=tl["bbox"],
            default_initial_state=tl["default_initial_state"],
            cycle_red_s=float(tl["cycle_red_s"]),
            cycle_green_s=float(tl["cycle_green_s"]),
            cycle_yellow_s=float(tl["cycle_yellow_s"]),
        )
        for tl in raw.get("traffic_lights", [])
    ]

    thresh_raw = raw.get("thresholds", {})
    thresholds = ThresholdConfig(**{
        k: thresh_raw[k] for k in ThresholdConfig.__dataclass_fields__
        if k in thresh_raw
    })

    re_raw = raw.get("risk_engine", {})
    w_raw = re_raw.get("weights", {})
    risk_engine = RiskEngineConfig(
        lookahead_s=float(re_raw.get("lookahead_s", 5.0)),
        weights=RiskWeights(**{
            k: w_raw[k] for k in RiskWeights.__dataclass_fields__ if k in w_raw
        }),
        smoothing_factor=float(re_raw.get("smoothing_factor", 0.20)),
    )

    return Config(
        camera=camera,
        intersection_box=raw.get("intersection_box", []),
        road_polygon=raw.get("road_polygon", []),
        lanes=lanes,
        stop_lines=stop_lines,
        zebra_crossings=zebras,
        solid_lines=solid_lines,
        traffic_lights=traffic_lights,
        thresholds=thresholds,
        risk_engine=risk_engine,
    )


def _rescale_point(p: list[float], sx: float, sy: float) -> list[float]:
    return [p[0] * sx, p[1] * sy]


def _rescale_polygon(poly: list[list[float]], sx: float, sy: float) -> list[list[float]]:
    return [_rescale_point(p, sx, sy) for p in poly]


def _rescale_config(cfg: Config) -> Config:
    sx, sy = cfg.scale_x, cfg.scale_y

    cfg.intersection_box = _rescale_polygon(cfg.intersection_box, sx, sy)
    cfg.road_polygon = _rescale_polygon(cfg.road_polygon, sx, sy)

    for lane in cfg.lanes:
        lane.polygon = _rescale_polygon(lane.polygon, sx, sy)

    for sl in cfg.stop_lines:
        sl.points = [_rescale_point(p, sx, sy) for p in sl.points]

    for zc in cfg.zebra_crossings:
        zc.polygon = _rescale_polygon(zc.polygon, sx, sy)

    for solid in cfg.solid_lines:
        solid.points = [_rescale_point(p, sx, sy) for p in solid.points]

    for tl in cfg.traffic_lights:
        tl.bbox = [
            int(tl.bbox[0] * sx),
            int(tl.bbox[1] * sy),
            int(tl.bbox[2] * sx),
            int(tl.bbox[3] * sy),
        ]

    return cfg


def _builtin_defaults() -> dict[str, Any]:
    """Minimal 1080p fallback geometry when no config.json found."""
    return {
        "camera": {"resolution": [1920, 1080], "fps": 30.0, "pixels_per_meter": 22.0},
        "intersection_box": [[640,360],[1280,360],[1360,780],[560,780]],
        "road_polygon": [[450,0],[1470,0],[1920,280],[1920,880],[1480,1080],[440,1080],[0,880],[0,280]],
        "lanes": [
            {"id": "lane_sb", "name": "Southbound", "polygon": [[880,0],[1040,0],[1060,360],[920,360]],
             "flow_vector": [0.0, 1.0], "allowed_movements": ["straight"], "speed_limit_kmh": 50.0},
            {"id": "lane_nb", "name": "Northbound", "polygon": [[940,780],[1100,780],[1180,1080],[1000,1080]],
             "flow_vector": [0.0, -1.0], "allowed_movements": ["straight"], "speed_limit_kmh": 50.0},
            {"id": "lane_wb", "name": "Westbound", "polygon": [[1280,480],[1920,460],[1920,620],[1300,640]],
             "flow_vector": [-1.0, 0.0], "allowed_movements": ["straight"], "speed_limit_kmh": 50.0},
            {"id": "lane_eb", "name": "Eastbound", "polygon": [[0,520],[600,500],[580,660],[0,680]],
             "flow_vector": [1.0, 0.0], "allowed_movements": ["straight"], "speed_limit_kmh": 50.0},
        ],
        "stop_lines": [
            {"id": "sl_n", "name": "North Stop Line", "points": [[720,360],[1060,360]],
             "associated_traffic_light": "tl_n", "flow_direction": [0.0, 1.0]},
            {"id": "sl_s", "name": "South Stop Line", "points": [[940,780],[1260,780]],
             "associated_traffic_light": "tl_s", "flow_direction": [0.0, -1.0]},
        ],
        "zebra_crossings": [
            {"id": "zc_n", "name": "North Zebra", "polygon": [[680,310],[1100,310],[1080,355],[700,355]]},
            {"id": "zc_s", "name": "South Zebra", "polygon": [[910,785],[1300,785],[1320,835],[890,835]]},
        ],
        "solid_lines": [
            {"id": "solid_n", "name": "North Solid Divider", "points": [[880,60],[920,360]]},
        ],
        "traffic_lights": [
            {"id": "tl_n", "name": "North Signal", "bbox": [1120,260,1160,350],
             "default_initial_state": "RED", "cycle_red_s": 15.0, "cycle_green_s": 25.0, "cycle_yellow_s": 3.0},
            {"id": "tl_s", "name": "South Signal", "bbox": [860,780,900,870],
             "default_initial_state": "RED", "cycle_red_s": 15.0, "cycle_green_s": 25.0, "cycle_yellow_s": 3.0},
        ],
        "thresholds": {},
        "risk_engine": {},
    }
