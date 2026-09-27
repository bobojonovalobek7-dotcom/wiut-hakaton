"""
main.py
Command-line video processing pipeline for the WIUT Traffic Event Detection System.

Usage
-----
  python main.py --video footage.mp4 [options]

Options
-------
  --video          Path to input MP4 / AVI video (required)
  --config         Path to config.json (default: config.json in project root)
  --output_video   Output annotated video path (default: output.mp4)
  --output_json    Output events JSON path (default: events.json)
  --output_csv     Output events CSV path (default: events.csv)
  --model          YOLO model weights (default: yolov8n.pt)
  --conf           Detection confidence threshold (default: 0.35)
  --frame_stride   Process every N-th frame (default: 1)
  --show_live      Display annotated video in real-time window
  --no_zones       Disable ROI zone overlays
  --no_trails      Disable trajectory trail rendering
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Local imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.config import load_config
from src.detector import DetectorTracker, TrafficLightStateDetector, video_frame_generator
from src.risk_engine import CollisionRiskEngine
from src.rules import TrafficRuleEngine
from src.visualizer import TrafficVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wiut.main")


# ---------------------------------------------------------------------------
# CLI argument parser -------------------------------------------------------
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WIUT Traffic Event Detection & Risk Analytics Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",         required=True,              help="Input video file")
    p.add_argument("--config",        default="config.json",      help="Configuration JSON")
    p.add_argument("--output_video",  default="output.mp4",       help="Output annotated video")
    p.add_argument("--output_json",   default="events.json",      help="Output events JSON")
    p.add_argument("--output_csv",    default="events.csv",       help="Output events CSV")
    p.add_argument("--snapshots_dir", default="snapshots",        help="Directory to save incident snapshot photos")
    p.add_argument("--model",         default="yolov8n.pt",       help="YOLO weights path or name")
    p.add_argument("--conf",          type=float, default=0.35,   help="Detection confidence threshold")
    p.add_argument("--frame_stride",  type=int,   default=1,      help="Process every N-th frame")
    p.add_argument("--show_live",     action="store_true",        help="Display live annotated window")
    p.add_argument("--no_zones",      action="store_true",        help="Disable ROI zone overlays")
    p.add_argument("--no_trails",     action="store_true",        help="Disable trajectory trail rendering")
    return p


# ---------------------------------------------------------------------------
# Main pipeline -------------------------------------------------------------
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> list[dict]:
    """
    Execute the full video processing pipeline.

    Returns
    -------
    List of event dicts suitable for JSON serialisation.
    """
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error("Input video not found: %s", video_path)
        sys.exit(1)

    # ── Read video metadata ───────────────────────────────────────────────
    probe = cv2.VideoCapture(str(video_path))
    video_fps    = probe.get(cv2.CAP_PROP_FPS)    or 30.0
    video_width  = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
    video_height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    logger.info(
        "Video: %s  |  %dx%d  @  %.1f fps  |  %d frames",
        video_path.name, video_width, video_height, video_fps, total_frames
    )

    # ── Load config ──────────────────────────────────────────────────────
    cfg = load_config(
        config_path=args.config,
        actual_resolution=(video_width, video_height),
    )
    # Override FPS from actual video
    cfg.camera.fps = video_fps

    # ── Initialise modules ───────────────────────────────────────────────
    detector  = DetectorTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        pixels_per_meter=cfg.camera.pixels_per_meter,
        fps=video_fps,
        frame_stride=args.frame_stride,
        tracker="bytetrack.yaml",
        device=None,   # auto
    )
    tl_detector  = TrafficLightStateDetector()
    rule_engine  = TrafficRuleEngine(cfg, tl_detector)
    risk_engine  = CollisionRiskEngine(cfg)
    visualizer   = TrafficVisualizer(
        cfg,
        show_zones=not args.no_zones,
        show_trails=not args.no_trails,
    )

    # ── Video writer ─────────────────────────────────────────────────────
    output_path = Path(args.output_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, video_fps, (video_width, video_height)
    )
    if not writer.isOpened():
        logger.warning("Could not open video writer for %s", output_path)

    # ── Frame processing loop ─────────────────────────────────────────────
    all_events:    list[dict]  = []
    active_events: list[str]   = []
    latest_risk    = None
    tl_states:     dict        = {}
    last_snap_time: dict[str, float] = {}
    snap_count:    int         = 1

    pbar = tqdm(total=total_frames, desc="Processing frames", unit="fr")

    for frame_idx, timestamp_s, frame in video_frame_generator(video_path):
        pbar.update(1)

        # ── Object detection & tracking ───────────────────────────────
        tracks = detector.process_frame(frame)

        # ── Traffic light state (per configured TL) ───────────────────
        for tl in cfg.traffic_lights:
            state = tl_detector.detect(
                frame, tl.id, tl.bbox, timestamp_s,
                initial_state=tl.default_initial_state,
                cycle_red_s=tl.cycle_red_s,
                cycle_green_s=tl.cycle_green_s,
                cycle_yellow_s=tl.cycle_yellow_s,
            )
            tl_states[tl.id] = state

        # ── Event rule evaluation ─────────────────────────────────────
        closed_this_frame = rule_engine.evaluate(
            frame, tracks, frame_idx, timestamp_s
        )
        for evt in closed_this_frame:
            all_events.append(evt.to_dict())
            logger.info("  → EVENT: %s  [%.2f-%.2fs]  conf=%.2f",
                        evt.event, evt.t_start, evt.t_end, evt.confidence)

        # Active events = events that were triggered in the last few seconds
        active_events = [
            k[0] if isinstance(k, tuple) else k
            for k, ae in rule_engine._active.items()
            if (timestamp_s - ae.t_last) < 2.0
        ]

        # ── Collision risk score ──────────────────────────────────────
        latest_risk = risk_engine.compute(tracks, timestamp_s)

        # ── Render frame ─────────────────────────────────────────────
        for evt_name in active_events:
            visualizer.notify_event(evt_name)

        annotated = visualizer.render(
            frame, tracks, latest_risk,
            active_events, timestamp_s, tl_states,
        )

        if writer.isOpened():
            writer.write(annotated)

        # ── Capture Incident Snapshots ─────────────────────────────────
        if args.snapshots_dir:
            snaps_path = Path(args.snapshots_dir)
            snaps_path.mkdir(parents=True, exist_ok=True)
            should_snap = False
            snap_tag = "event"
            if closed_this_frame:
                should_snap = True
                snap_tag = closed_this_frame[0].event
            elif active_events:
                snap_tag = active_events[0]
                if timestamp_s - last_snap_time.get(snap_tag, -99.0) >= 2.5:
                    should_snap = True
            elif latest_risk and latest_risk.overall >= 0.65:
                snap_tag = f"high_risk_{latest_risk.overall:.2f}"
                if timestamp_s - last_snap_time.get("high_risk", -99.0) >= 3.0:
                    should_snap = True

            if should_snap:
                last_snap_time[snap_tag] = timestamp_s
                snap_filename = f"snap_{snap_count:03d}_{snap_tag}_t{timestamp_s:.1f}s.jpg"
                cv2.imwrite(str(snaps_path / snap_filename), annotated)
                snap_count += 1

        if args.show_live:
            cv2.imshow("WIUT Traffic Analytics", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Live display closed by user.")
                break

    pbar.close()

    # ── Flush remaining open events ────────────────────────────────────────
    final_ts = total_frames / max(video_fps, 1.0)
    flushed  = rule_engine.flush_open_events(final_ts)
    for evt in flushed:
        all_events.append(evt.to_dict())

    writer.release()
    if args.show_live:
        cv2.destroyAllWindows()

    logger.info("Total events detected: %d", len(all_events))
    return all_events


# ---------------------------------------------------------------------------
# Output writers ------------------------------------------------------------
# ---------------------------------------------------------------------------

def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.number):
        return obj.item()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_json(events: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(events, fh, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("Events JSON → %s  (%d events)", path, len(events))


def write_csv(events: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        logger.warning("No events to write to CSV.")
        return
    fields = list(events[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    logger.info("Events CSV  → %s  (%d events)", path, len(events))


# ---------------------------------------------------------------------------
# Entry point ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logger.info("═" * 60)
    logger.info("  WIUT Traffic Event Detection & Risk Analytics System")
    logger.info("═" * 60)

    events = run_pipeline(args)

    write_json(events, args.output_json)
    write_csv(events,  args.output_csv)

    logger.info("Done. Annotated video → %s", args.output_video)
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
