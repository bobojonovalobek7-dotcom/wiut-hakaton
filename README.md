# WIUT Traffic Event Detection & Risk Analytics System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO-v8%2Fv11-purple?logo=ultralytics"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/Events-14%20Categories-orange"/>
</p>

> **WIUT Hackathon 2024** — End-to-end AI-powered CCTV traffic analytics system that detects 14 distinct traffic incidents in real-time, with a 5-second collision risk prediction engine.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Module Breakdown](#module-breakdown)
5. [14 Traffic Event Categories](#14-traffic-event-categories)
6. [Part B: Collision Risk Engine](#part-b-collision-risk-engine)
7. [CLI Usage](#cli-usage)
8. [Web Dashboard](#web-dashboard)
9. [Configuration Reference](#configuration-reference)
10. [Testing](#testing)
11. [Hackathon Presentation Notes](#hackathon-presentation-notes)

---

## Overview

The system analyzes stationary urban intersection CCTV footage and produces:

- **Part A**: Structured event logs with precise `t_start`, `t_end`, `event_label`, and `confidence` for 14 traffic incidents
- **Part B**: A real-time collision risk index R(t) ∈ [0.00, 1.00] forecasting danger within a 5-second window
- **Annotated Video**: Visual HUD overlay with bounding boxes, trajectory tails, event alerts, and risk gauge
- **Interactive Dashboard**: Streamlit web app for uploading video, inspecting events, and exporting results

### Architecture

```
CCTV Video MP4
     │
     ▼
┌─────────────────────────────────────────────┐
│  DetectorTracker (src/detector.py)           │
│  YOLOv8 + ByteTrack → TrackState objects     │
│  (class, bbox, vel_px, speed_kmh, history)   │
└──────────────┬──────────────────────────────┘
               │  tracks[]
     ┌─────────┴──────────┐
     ▼                    ▼
┌─────────────┐    ┌────────────────┐
│ Rule Engine │    │ Risk Engine    │
│ (rules.py)  │    │(risk_engine.py)│
│ 14 events   │    │ TTC + density  │
│ debounced   │    │ + decel + conf │
└──────┬──────┘    └────────┬───────┘
       │                    │
       ▼                    ▼
   events.json          RiskScore
   events.csv           R(t)∈[0,1]
       │                    │
       └────────┬───────────┘
                ▼
        Visualizer (visualizer.py)
        Annotated MP4 + Streamlit
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics shapely streamlit pandas altair tqdm opencv-python
```

Or install everything from requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate a Demo Video (no real footage needed)

```bash
python tools/mock_data_generator.py --output sample_traffic.mp4 --duration 60
```

### 3. Run the CLI Pipeline

```bash
python main.py --video sample_traffic.mp4 --output_video output.mp4 --output_json events.json
```

### 4. Launch the Web Dashboard

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure

```
wiut-traffic-ai/
├── config.json                    # ROI geometry & thresholds (1080p defaults)
├── requirements.txt               # Python dependencies
├── main.py                        # CLI video processing pipeline
├── app.py                         # Streamlit web dashboard
├── README.md                      # This file
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loader, dataclass schema, coord scaling
│   ├── detector.py                # YOLOv8 + ByteTrack wrapper, TrackState
│   ├── geometry.py                # Polygon, line, TTC, vector, speed math
│   ├── rules.py                   # 14 event rule engine + temporal debouncing
│   ├── risk_engine.py             # Part B: 5-second collision risk predictor
│   └── visualizer.py              # HUD overlay, risk gauge, event banners
│
├── tools/
│   ├── __init__.py
│   └── mock_data_generator.py    # Synthetic CCTV video generator for testing
│
└── tests/
    ├── __init__.py
    ├── test_geometry.py           # Unit tests for geometry module
    └── test_risk_engine.py        # Unit tests for risk engine
```

---

## Module Breakdown

### `src/config.py`
- Loads and validates `config.json` using Python dataclasses
- Auto-scales all polygon/line coordinates when actual video resolution differs from configured base resolution (1920×1080)
- Provides built-in 1080p fallback geometry if no config file is found

### `src/detector.py`
- **`DetectorTracker`**: Wraps Ultralytics YOLO with ByteTrack persistent tracking
  - Returns `TrackState` objects with `track_id`, `bbox`, `positions` history, `vel_px` (velocity in px/frame), `speed_kmh`
  - Handles stale track pruning (removed after 3 seconds of absence)
- **`TrafficLightStateDetector`**: HSV-based colour classification (RED/GREEN/YELLOW) with deterministic simulation fallback
- **`video_frame_generator()`**: Iterator yielding `(frame_index, timestamp_s, bgr_frame)` tuples

### `src/geometry.py`
Pure-Python geometry primitives (no ML imports):
| Function | Description |
|---|---|
| `point_in_polygon()` | Ray-casting PIP test |
| `segments_intersect()` | Orientation-based segment crossing |
| `bbox_crosses_line()` | Vehicle trajectory vs stop line / divider |
| `calculate_ttc()` | Constant-velocity Time-To-Collision (frames) |
| `velocity_vector()` | Rolling window velocity estimate |
| `speed_kmh()` | px/frame → km/h via calibration scale |
| `vector_dot_normalized()` | Cosine similarity for wrong-way detection |
| `trajectory_total_angle_change()` | Cumulative heading change for U-turn |
| `is_fire_pixel_region()` | HSV-based fire/smoke blob detection |

### `src/rules.py`
- **`TrafficRuleEngine`**: Evaluates all 14 rules per frame
- **Temporal Debouncing**: Each continuous trigger accumulates into a single `TrafficEvent` with `t_start`/`t_end` (new events open on first trigger, close after `gap_tolerance_s = 1.5s` without re-trigger)
- **`TrafficEvent`**: `{event, t_start, t_end, duration, confidence, track_ids}`

### `src/risk_engine.py`
- **`CollisionRiskEngine`**: Computes R(t) ∈ [0.00, 1.00] from four weighted components
- Uses Exponential Moving Average (EMA, α=0.20) for smooth HUD display
- **`RiskScore`**: Contains `overall`, per-component values, `ttc_min_s`, `level` string, and BGR colour

### `src/visualizer.py`
- **`TrafficVisualizer`**: Renders all HUD elements onto a copy of the input frame
- Zone overlays: intersection box, zebra crossings, stop lines (red=RED, green=GREEN), solid dividers
- Per-track bounding boxes, class labels, speed readout, trajectory tails
- Circular arc-style risk gauge (bottom-right corner)
- Sliding event alert banners (top-centre, auto-expire after 4 seconds)

---

## 14 Traffic Event Categories

| # | Event | Detection Method | Threshold |
|---|-------|-----------------|-----------|
| 1 | `accident` | BBox IoU > 0.08 + sudden speed collapse > 70% | IoU > 0.08 |
| 2 | `near_miss` | Converging TTC < 1.5 s without IoU contact | TTC < 1.5 s |
| 3 | `red_light` | Vehicle crosses stop line while signal is RED and enters intersection box | Signal = RED |
| 4 | `wrong_way` | Velocity vector dot-product with lane flow vector < -0.45 | dot < -0.45 |
| 5 | `illegal_u_turn` | Cumulative heading change > 135° inside intersection | angle > 135° |
| 6 | `stopped_vehicle` | Speed < 2.5 km/h for > 10 s outside waiting queues | 10 s |
| 7 | `jaywalking` | Pedestrian on active road polygon outside zebra zones | — |
| 8 | `failure_to_yield` | Vehicle traverses zebra polygon while pedestrian present | — |
| 9 | `illegal_turn` | Turn movement not in `allowed_movements` for origin lane | — |
| 10 | `solid_line_crossing` | Vehicle trajectory segment crosses solid divider line | — |
| 11 | `stop_line` | Vehicle crosses stop line on RED then stops before intersection | Speed < 2.5 km/h |
| 12 | `congestion` | ≥ 2 lanes with avg speed < 5 km/h for > 5 s | 5 km/h, 5 s |
| 13 | `road_obstacle` | Unknown-class static object in driveway > 3.5 s | 3.5 s |
| 14 | `fire_smoke` | HSV orange/red + grey mask contour area ≥ 400 px² for ≥ 10 frames | 10 frames |

### Output Format

```json
[
  {
    "event": "red_light",
    "t_start": 14.233,
    "t_end": 16.900,
    "duration": 2.667,
    "confidence": 0.904,
    "track_ids": [7]
  },
  {
    "event": "near_miss",
    "t_start": 22.100,
    "t_end": 23.467,
    "duration": 1.367,
    "confidence": 0.823,
    "track_ids": [3, 11]
  }
]
```

---

## Part B: Collision Risk Engine

### Mathematical Formulation

**R(t)** is a weighted linear combination of four independently computed sub-scores:

```
R_raw(t) = w_ttc · R_TTC + w_decel · R_decel + w_density · R_density + w_conflict · R_conflict

R(t) = α · R_raw(t) + (1-α) · R(t-1)     [EMA smoothing, α=0.20]
```

#### Component Scores

| Component | Formula | Description |
|-----------|---------|-------------|
| R_TTC | `1 - (TTC_min / lookahead)^0.6` | Sigmoid-like peak as TTC → 0 |
| R_decel | `max_speed_drop_kmh / 30` | Hard braking over 10 frames |
| R_density | `vehicles_in_intersection / 10` | Occupancy of intersection box |
| R_conflict | `trajectory_crossings / 5` | Projected path crossings in 5 s |

#### Default Weights

```json
{
  "ttc": 0.40,
  "deceleration": 0.25,
  "density": 0.20,
  "conflict_angle": 0.15
}
```

### Risk Level Mapping

| Score | Level | Colour |
|-------|-------|--------|
| 0.00 – 0.29 | LOW | 🟢 Green |
| 0.30 – 0.59 | MEDIUM | 🟠 Orange |
| 0.60 – 0.79 | HIGH | 🔴 Red |
| 0.80 – 1.00 | CRITICAL | 🟣 Deep Red |

---

## CLI Usage

```bash
python main.py \
  --video       footage.mp4 \
  --config      config.json \
  --output_video output.mp4 \
  --output_json  events.json \
  --output_csv   events.csv \
  --model       yolov8n.pt \
  --conf        0.35 \
  --frame_stride 1 \
  --show_live
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | — | Input video path (required) |
| `--config` | `config.json` | ROI configuration file |
| `--output_video` | `output.mp4` | Annotated output video |
| `--output_json` | `events.json` | Events log JSON |
| `--output_csv` | `events.csv` | Events log CSV |
| `--model` | `yolov8n.pt` | YOLO weights (auto-downloaded) |
| `--conf` | `0.35` | Detection confidence threshold |
| `--frame_stride` | `1` | Process every N-th frame |
| `--show_live` | off | Display real-time window |
| `--no_zones` | off | Disable ROI overlays |
| `--no_trails` | off | Disable trajectory trails |

---

## Web Dashboard

```bash
streamlit run app.py
```

Features:
- **Upload video** (MP4/AVI/MOV) or click **Generate Demo Video** for instant testing
- **Sidebar settings**: YOLO model selection, confidence, frame stride, toggles
- **Live progress bar** during processing
- **KPI cards**: total events, event types, severity sum, incident duration, peak risk
- **Breakdown chart**: horizontal bar chart by event type (Altair)
- **Timeline chart**: Gantt-style incident spans over the video timeline (Altair)
- **Event log table**: sortable, searchable, with confidence progress bar
- **Download**: `events.json` and `events.csv`

---

## Configuration Reference

`config.json` defines all scene geometry and thresholds. All coordinates are for a **1920×1080 base resolution** and are auto-scaled to match the actual video.

```json
{
  "camera": {
    "resolution": [1920, 1080],
    "fps": 30.0,
    "pixels_per_meter": 22.0
  },
  "thresholds": {
    "stopped_vehicle_duration_s": 10.0,
    "stopped_vehicle_speed_kmh": 2.5,
    "ttc_near_miss_s": 1.5,
    "congestion_speed_threshold_kmh": 5.0,
    "congestion_min_lanes": 2,
    "u_turn_angle_deg": 135.0,
    "wrong_way_dot_thresh": -0.45
  }
}
```

To customise for your intersection footage:
1. Use `tools/calibrate_roi.py` (or any image annotation tool) to map polygon coordinates
2. Update `config.json` with your intersection's stop lines, lane polygons, and zebra crossings
3. Tune `pixels_per_meter` using a known real-world distance (e.g., road lane width ≈ 3.5 m)

---

## Testing

Run the full unit test suite:

```bash
pytest tests/ -v
```

Test coverage:
- `tests/test_geometry.py`: 24 tests covering all geometric functions
- `tests/test_risk_engine.py`: 7 tests covering risk score range, EMA smoothing, level classification

---

## Hackathon Presentation Notes

### Key Technical Differentiators

1. **Zero External Dataset Required**: The `mock_data_generator.py` creates a test video on-the-fly, allowing immediate demonstration without relying on specific footage availability.

2. **Dual-Mode Traffic Light Detection**: The system first attempts HSV pixel analysis of the actual TL ROI in the frame. If inconclusive (small ROI, occlusion), it falls back to a deterministic phase simulation synchronized to the video timestamp — ensuring robust operation in all conditions.

3. **Temporal Debouncing Architecture**: Raw per-frame triggers are aggregated into cleanly bounded incidents. This eliminates duplicate events and produces exactly the `t_start`/`t_end` format required by the hackathon spec.

4. **CPU-Friendly Design**: Using `yolov8n.pt` (nano model, ~3.2M parameters) with configurable frame stride (e.g. `--frame_stride 2`) enables real-time processing on CPU-only machines while maintaining detection quality.

5. **Part B Risk Engine**: The 5-second lookahead collision risk score is computed from first principles (TTC, kinematics, density) without requiring any additional ML model — pure physics-based heuristics.

### Suggested Demo Flow

1. `python tools/mock_data_generator.py --output demo.mp4 --duration 60` → generates 60-second demo
2. `streamlit run app.py` → launch dashboard
3. Upload `demo.mp4` → click **Run Analysis**
4. Walk through the incident timeline, breakdown chart, and risk score
5. Show `events.json` download with exact timestamp format

### Performance Benchmarks (approximate)

| Hardware | Model | Stride | Speed |
|----------|-------|--------|-------|
| CPU (i7) | yolov8n.pt | 1 | ~4 fps |
| CPU (i7) | yolov8n.pt | 2 | ~8 fps |
| GPU (RTX 3080) | yolov8s.pt | 1 | ~45 fps |
