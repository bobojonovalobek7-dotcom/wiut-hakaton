"""
src/detector.py
YOLOv8/v11 + ByteTrack wrapper for object detection and persistent multi-object tracking.

Provides:
  - DetectorTracker  — initialises YOLO + tracker, wraps per-frame inference
  - TrackState       — per-track data container (history, velocity, class)
  - process_frame()  — main inference call returning a list of TrackState objects
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs relevant to traffic analysis
_VEHICLE_CLASSES  = {2, 3, 5, 7}   # car, motorcycle, bus, truck
_PERSON_CLASS     = {0}             # person
_TRAFFIC_LIGHT    = {9}             # traffic light
_BICYCLE_CLASS    = {1}             # bicycle

ALL_TRACKED_CLASSES = _VEHICLE_CLASSES | _PERSON_CLASS | _TRAFFIC_LIGHT | _BICYCLE_CLASS

COCO_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic_light",
}

# ---------------------------------------------------------------------------
# TrackState ----------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """Per-object tracking state maintained across frames."""
    track_id: int
    class_id: int
    class_name: str
    bbox: list[float]                     # [x1, y1, x2, y2] current frame
    confidence: float
    frame_idx: int
    timestamp_s: float

    # History (capped to last N frames)
    positions: collections.deque = field(default_factory=lambda: collections.deque(maxlen=60))
    bboxes:    collections.deque = field(default_factory=lambda: collections.deque(maxlen=60))
    speeds_kmh: collections.deque = field(default_factory=lambda: collections.deque(maxlen=30))

    # Derived (updated each frame)
    vel_px: tuple[float, float] = (0.0, 0.0)   # pixels per frame
    speed_kmh: float = 0.0

    # Event state flags managed by rules engine
    flags: dict = field(default_factory=dict)

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.bbox[0]+self.bbox[2])/2, (self.bbox[1]+self.bbox[3])/2)

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in _VEHICLE_CLASSES

    @property
    def is_pedestrian(self) -> bool:
        return self.class_id in _PERSON_CLASS

    @property
    def is_bicycle(self) -> bool:
        return self.class_id in _BICYCLE_CLASS


# ---------------------------------------------------------------------------
# DetectorTracker -----------------------------------------------------------
# ---------------------------------------------------------------------------

class DetectorTracker:
    """
    Wraps Ultralytics YOLO + ByteTrack for persistent multi-object tracking.

    Parameters
    ----------
    model_path      : Path to YOLO weights (.pt). Defaults to 'yolov8n.pt'.
    conf_threshold  : Minimum detection confidence.
    iou_threshold   : NMS IoU threshold.
    device          : 'cpu', 'cuda', or 'mps'. Auto-detected if None.
    pixels_per_meter: Calibration constant for speed conversion.
    fps             : Video frame rate, used for velocity → speed conversion.
    frame_stride    : Process every N-th frame (1 = every frame).
    tracker         : Ultralytics tracker config name ('bytetrack.yaml' / 'botsort.yaml').
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        pixels_per_meter: float = 22.0,
        fps: float = 30.0,
        frame_stride: int = 1,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading YOLO model '%s' on device '%s' …", model_path, device)
        self.model  = YOLO(model_path)
        self.device = device
        self.conf   = conf_threshold
        self.iou    = iou_threshold
        self.ppm    = pixels_per_meter
        self.fps    = fps
        self.stride = max(1, frame_stride)
        self.tracker_cfg = tracker

        # Per-track state store
        self._tracks: dict[int, TrackState] = {}
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Public helpers ----------------------------------------------------

    def reset(self) -> None:
        """Clear all track history (call between video files)."""
        self._tracks.clear()
        self._frame_idx = 0

    @property
    def active_tracks(self) -> list[TrackState]:
        return list(self._tracks.values())

    # ------------------------------------------------------------------
    # Core inference ----------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        force: bool = False,
    ) -> list[TrackState]:
        """
        Run detection + tracking on *frame*.

        Parameters
        ----------
        frame : BGR numpy array.
        force : If True, bypass *frame_stride* check.

        Returns
        -------
        List of active TrackState objects (all currently tracked objects).
        """
        self._frame_idx += 1
        timestamp_s = self._frame_idx / max(self.fps, 1.0)

        if not force and (self._frame_idx % self.stride != 0):
            # Return cached tracks on skipped frames
            return self.active_tracks

        t0 = time.perf_counter()
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_cfg,
                classes=list(ALL_TRACKED_CLASSES),
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO inference error at frame %d: %s", self._frame_idx, exc)
            return self.active_tracks

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Frame %d  inference %.1f ms", self._frame_idx, elapsed_ms)

        result = results[0]
        current_tids: set[int] = set()

        if result.boxes is not None and result.boxes.id is not None:
            boxes  = result.boxes.xyxy.cpu().numpy()
            ids    = result.boxes.id.cpu().numpy().astype(int)
            clsids = result.boxes.cls.cpu().numpy().astype(int)
            confs  = result.boxes.conf.cpu().numpy()

            for i, tid in enumerate(ids):
                cls_id   = int(clsids[i])
                cls_name = COCO_CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
                bbox     = boxes[i].tolist()          # [x1, y1, x2, y2]
                conf     = float(confs[i])
                cx       = (bbox[0] + bbox[2]) / 2
                cy       = (bbox[1] + bbox[3]) / 2

                if tid not in self._tracks:
                    self._tracks[tid] = TrackState(
                        track_id=tid,
                        class_id=cls_id,
                        class_name=cls_name,
                        bbox=bbox,
                        confidence=conf,
                        frame_idx=self._frame_idx,
                        timestamp_s=timestamp_s,
                    )

                ts = self._tracks[tid]
                ts.bbox         = bbox
                ts.confidence   = conf
                ts.frame_idx    = self._frame_idx
                ts.timestamp_s  = timestamp_s
                ts.class_id     = cls_id
                ts.class_name   = cls_name

                ts.positions.append((cx, cy))
                ts.bboxes.append(bbox)

                # Compute velocity & speed
                ts.vel_px = self._estimate_velocity(ts.positions)
                ts.speed_kmh = self._to_kmh(ts.vel_px)
                ts.speeds_kmh.append(ts.speed_kmh)

                current_tids.add(tid)

        # Remove stale tracks (not seen for > 3 seconds)
        stale_tids = [
            tid for tid, ts in self._tracks.items()
            if tid not in current_tids
            and (self._frame_idx - ts.frame_idx) > int(self.fps * 3)
        ]
        for tid in stale_tids:
            del self._tracks[tid]

        return self.active_tracks

    # ------------------------------------------------------------------
    # Private helpers ---------------------------------------------------

    def _estimate_velocity(
        self,
        positions: collections.deque,
        window: int = 6,
    ) -> tuple[float, float]:
        pts = list(positions)
        if len(pts) < 2:
            return (0.0, 0.0)
        recent = pts[-min(window, len(pts)):]
        vx = (recent[-1][0] - recent[0][0]) / max(len(recent) - 1, 1)
        vy = (recent[-1][1] - recent[0][1]) / max(len(recent) - 1, 1)
        return (vx, vy)

    def _to_kmh(self, vel_px: tuple[float, float]) -> float:
        px_per_frame = (vel_px[0]**2 + vel_px[1]**2) ** 0.5
        m_per_s = (px_per_frame * self.fps) / max(self.ppm, 1.0)
        return m_per_s * 3.6


# ---------------------------------------------------------------------------
# Traffic light state (HSV-based) ------------------------------------------
# ---------------------------------------------------------------------------

class TrafficLightStateDetector:
    """
    Simple HSV-based traffic light colour classifier.

    Falls back to a simulated cycle if the ROI is not available or too small.
    """

    # HSV thresholds
    _RED_RANGES = [((0, 100, 80), (15, 255, 255)), ((165, 100, 80), (180, 255, 255))]
    _YELLOW_RANGE = ((15, 100, 80), (40, 255, 255))
    _GREEN_RANGE  = ((40, 60, 60),  (100, 255, 255))

    def __init__(self) -> None:
        self._states: dict[str, str] = {}      # tl_id → 'RED'|'GREEN'|'YELLOW'
        self._sim_timers: dict[str, float] = {}

    def detect(
        self,
        frame: np.ndarray,
        tl_id: str,
        bbox: list[int],
        timestamp_s: float,
        initial_state: str = "RED",
        cycle_red_s: float = 15.0,
        cycle_green_s: float = 25.0,
        cycle_yellow_s: float = 3.0,
    ) -> str:
        """
        Detect traffic light colour for *tl_id* in *frame*.

        Prefers HSV pixel analysis; falls back to deterministic simulation.
        """
        x1, y1, x2, y2 = bbox
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            roi = frame[y1:y2, x1:x2]
            state = self._classify_roi(roi)
            if state:
                self._states[tl_id] = state
                return state

        # Simulation fallback
        return self._simulate(
            tl_id, timestamp_s, initial_state,
            cycle_red_s, cycle_green_s, cycle_yellow_s,
        )

    def get_state(self, tl_id: str, default: str = "RED") -> str:
        return self._states.get(tl_id, default)

    # ------------------------------------------------------------------

    def _classify_roi(self, roi: np.ndarray) -> str | None:
        import cv2
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_px = sum(
            cv2.countNonZero(cv2.inRange(hsv, lo, hi))
            for lo, hi in self._RED_RANGES
        )
        yellow_px = cv2.countNonZero(cv2.inRange(hsv, *self._YELLOW_RANGE))
        green_px  = cv2.countNonZero(cv2.inRange(hsv, *self._GREEN_RANGE))
        total = roi.shape[0] * roi.shape[1]
        thresh = total * 0.10  # at least 10 % of ROI must be coloured

        if red_px > thresh and red_px >= max(yellow_px, green_px):
            return "RED"
        if yellow_px > thresh and yellow_px >= max(red_px, green_px):
            return "YELLOW"
        if green_px > thresh:
            return "GREEN"
        return None

    def _simulate(
        self,
        tl_id: str,
        ts: float,
        initial: str,
        red_s: float,
        green_s: float,
        yellow_s: float,
    ) -> str:
        cycle = red_s + green_s + yellow_s
        # Offset so that initial_state starts at t=0
        offset = {"RED": 0.0, "GREEN": red_s, "YELLOW": red_s + green_s}.get(initial, 0.0)
        t = (ts + offset) % cycle
        if t < red_s:
            state = "RED"
        elif t < red_s + green_s:
            state = "GREEN"
        else:
            state = "YELLOW"
        self._states[tl_id] = state
        return state


# ---------------------------------------------------------------------------
# Video frame generator -------------------------------------------------------
# ---------------------------------------------------------------------------

def video_frame_generator(
    video_path: str | Path,
    *,
    start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """
    Yield (frame_index, timestamp_s, bgr_frame) tuples from a video file.

    Parameters
    ----------
    video_path  : Path to MP4 / AVI / etc.
    start_frame : First frame index to yield.
    max_frames  : Maximum number of frames to yield (None = all).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Opened video '%s': %.1f fps, %d total frames", video_path, fps, total
    )

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    yielded = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = frame_idx / fps
            yield frame_idx, ts, frame
            frame_idx += 1
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        cap.release()
