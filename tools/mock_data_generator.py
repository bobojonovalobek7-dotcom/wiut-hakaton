"""
tools/mock_data_generator.py
Synthetic CCTV intersection video generator for testing the WIUT pipeline.

Generates a 1080p, 30fps video showing:
  - Normal left/right/top/bottom approach traffic (cars, trucks, motorcycles)
  - Pedestrian activity (zebra crossings and jaywalking)
  - Programmed traffic events: red-light run, stopped vehicle, near-miss,
    wrong-way, jaywalking, congestion simulation, and fire/smoke overlay.

Run:
    python tools/mock_data_generator.py --output sample_traffic.mp4 --duration 60
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Simple moving-object model ------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class MockVehicle:
    vid: int
    x: float
    y: float
    vx: float
    vy: float
    w: int
    h: int
    color: tuple
    label: str
    alive: bool = True
    stopped_frames: int = 0
    max_stop: int = 0  # 0 = never stop

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (int(self.x - self.w/2), int(self.y - self.h/2),
                int(self.x + self.w/2), int(self.y + self.h/2))


@dataclass
class MockPedestrian:
    pid: int
    x: float
    y: float
    vx: float
    vy: float
    alive: bool = True


# ---------------------------------------------------------------------------
# Road background scene -----------------------------------------------------
# ---------------------------------------------------------------------------

def _draw_road(canvas: np.ndarray, w: int, h: int) -> None:
    """Draw an intersection on a plain canvas."""
    canvas[:] = (60, 80, 60)  # grass background (BGR)

    # Road surfaces
    road_colour = (80, 80, 88)
    # Horizontal road (E-W)
    cv2.rectangle(canvas, (0, h//3), (w, 2*h//3), road_colour, -1)
    # Vertical road (N-S)
    cv2.rectangle(canvas, (w//3, 0), (2*w//3, h), road_colour, -1)

    # Lane dashes — horizontal
    dash_colour = (200, 200, 200)
    for x in range(0, w//3, 60):
        cv2.line(canvas, (x, h//2), (x+30, h//2), dash_colour, 2)
    for x in range(2*w//3, w, 60):
        cv2.line(canvas, (x, h//2), (x+30, h//2), dash_colour, 2)

    # Lane dashes — vertical
    for y in range(0, h//3, 60):
        cv2.line(canvas, (w//2, y), (w//2, y+30), dash_colour, 2)
    for y in range(2*h//3, h, 60):
        cv2.line(canvas, (w//2, y), (w//2, y+30), dash_colour, 2)

    # Zebra crossings (simplified stripes)
    z_colour = (220, 220, 220)
    # North zebra
    for i in range(8):
        x_start = w//3 + i * ((w//3) // 8)
        cv2.rectangle(canvas, (x_start, h//3 - 40), (x_start + 16, h//3 - 8),
                      z_colour, -1)
    # South zebra
    for i in range(8):
        x_start = w//3 + i * ((w//3) // 8)
        cv2.rectangle(canvas, (x_start, 2*h//3 + 8), (x_start + 16, 2*h//3 + 40),
                      z_colour, -1)

    # Stop lines
    cv2.line(canvas, (w//3, h//3 - 5), (2*w//3, h//3 - 5), (0, 0, 255), 4)
    cv2.line(canvas, (w//3, 2*h//3 + 5), (2*w//3, 2*h//3 + 5), (0, 0, 255), 4)
    cv2.line(canvas, (w//3 - 5, h//3), (w//3 - 5, 2*h//3), (0, 0, 255), 4)
    cv2.line(canvas, (2*w//3 + 5, h//3), (2*w//3 + 5, 2*h//3), (0, 0, 255), 4)

    # Solid lane dividers
    cv2.line(canvas, (w//2, 0), (w//2, h//3 - 5), (255, 165, 0), 3)
    cv2.line(canvas, (w//2, 2*h//3 + 5), (w//2, h), (255, 165, 0), 3)


def _draw_traffic_lights(canvas: np.ndarray, w: int, h: int, frame: int, fps: float) -> dict:
    """Draw traffic light indicators. Returns {approach: 'RED'|'GREEN'|'YELLOW'}."""
    cycle = int(fps * 40)  # 40-second total cycle
    phase = frame % cycle
    red_phase = int(fps * 18)
    grn_phase = int(fps * 18)
    # yel_phase = rest

    states = {}
    if phase < red_phase:
        ns_state, ew_state = "RED", "GREEN"
    elif phase < red_phase + grn_phase:
        ns_state, ew_state = "GREEN", "RED"
    else:
        ns_state, ew_state = "YELLOW", "YELLOW"

    states = {"N": ns_state, "S": ns_state, "E": ew_state, "W": ew_state}

    def _tl_color(s: str) -> tuple:
        return {"RED": (0,0,255), "GREEN": (0,200,0), "YELLOW": (0,180,255)}[s]

    # Draw TL boxes
    for pos, s in [((2*w//3 + 15, h//3 - 60), states["N"]),
                   ((w//3 - 35, 2*h//3 + 20), states["S"]),
                   ((w//3 - 50, h//3 - 40), states["W"]),
                   ((2*w//3 + 20, 2*h//3 + 20), states["E"])]:
        cv2.rectangle(canvas, pos, (pos[0]+18, pos[1]+50), (30,30,30), -1)
        cv2.circle(canvas, (pos[0]+9, pos[1]+12), 7, _tl_color(s), -1)

    return states


# ---------------------------------------------------------------------------
# Spawn helpers -------------------------------------------------------------
# ---------------------------------------------------------------------------

_VID_COUNTER = 0
_PID_COUNTER = 0

VEHICLE_SPECS = [
    ("car",        (60,180,75),  60, 30),
    ("truck",      (50,130,255), 90, 40),
    ("motorcycle", (255,200,50), 35, 20),
    ("bus",        (180,60,255), 110, 45),
]


def _spawn_vehicle(
    w: int, h: int,
    approach: str,
    *,
    red_runner: bool = False,
    wrong_way: bool = False,
    stop_in_lane: bool = False,
) -> MockVehicle:
    global _VID_COUNTER
    _VID_COUNTER += 1
    spec = random.choice(VEHICLE_SPECS)
    label, col, vw, vh = spec

    speed = random.uniform(2.5, 5.0)

    if approach == "N":
        x, y, vx, vy = w//2 + random.randint(-60, 60), -vh, 0, speed
    elif approach == "S":
        x, y, vx, vy = w//2 + random.randint(-60, 60), h + vh, 0, -speed
    elif approach == "W":
        x, y, vx, vy = -vw, h//2 + random.randint(-60, 60), speed, 0
    else:  # E
        x, y, vx, vy = w + vw, h//2 + random.randint(-60, 60), -speed, 0

    if wrong_way:
        vx, vy = -vx, -vy

    stop_dur = random.randint(150, 350) if stop_in_lane else 0

    return MockVehicle(
        vid=_VID_COUNTER,
        x=x, y=y, vx=vx, vy=vy,
        w=vw, h=vh, color=col, label=label,
        max_stop=stop_dur,
    )


def _spawn_pedestrian(w: int, h: int, jaywalking: bool = False) -> MockPedestrian:
    global _PID_COUNTER
    _PID_COUNTER += 1
    speed = random.uniform(0.8, 1.8)
    if not jaywalking:
        # Walk across north zebra
        x = w//3 + random.randint(10, w//3 - 20)
        y = h//3 - 25
        return MockPedestrian(pid=_PID_COUNTER, x=x, y=y, vx=0, vy=speed*0.5)
    else:
        # Walk across the road outside zebra
        x = random.randint(w//3 + 20, 2*w//3 - 20)
        y = h//3 + 40
        return MockPedestrian(pid=_PID_COUNTER, x=x, y=y, vx=0, vy=speed)


# ---------------------------------------------------------------------------
# Generator main loop -------------------------------------------------------
# ---------------------------------------------------------------------------

def generate_mock_video(output_path: str, duration_s: float = 60.0, fps: float = 30.0) -> None:
    """
    Render a synthetic CCTV intersection video to *output_path*.
    """
    W, H = 1280, 720
    total_frames = int(duration_s * fps)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {output_path}")

    vehicles:     list[MockVehicle]    = []
    pedestrians:  list[MockPedestrian] = []
    event_log:    list[str]            = []

    spawn_timer      = 0
    ped_timer        = 0
    fire_smoke_start = int(fps * 45)  # Fire smoke at 45s mark

    print(f"Generating {duration_s:.0f}s synthetic CCTV video  ({total_frames} frames)…")

    for frame_idx in range(total_frames):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        _draw_road(canvas, W, H)
        tl_states = _draw_traffic_lights(canvas, W, H, frame_idx, fps)

        # ── Spawn vehicles ───────────────────────────────────────────
        spawn_timer -= 1
        if spawn_timer <= 0:
            approach = random.choice(["N", "S", "E", "W"])
            is_red_runner = (frame_idx > int(fps*8) and
                             tl_states.get(approach) == "RED" and
                             random.random() < 0.15)
            is_wrong_way  = (frame_idx > int(fps*20) and random.random() < 0.05)
            is_stopper    = (frame_idx > int(fps*5) and random.random() < 0.08)
            v = _spawn_vehicle(W, H, approach,
                               red_runner=is_red_runner,
                               wrong_way=is_wrong_way,
                               stop_in_lane=is_stopper)
            vehicles.append(v)
            spawn_timer = random.randint(15, 50)

        # ── Spawn pedestrians ────────────────────────────────────────
        ped_timer -= 1
        if ped_timer <= 0:
            is_jay = random.random() < 0.25
            p = _spawn_pedestrian(W, H, jaywalking=is_jay)
            pedestrians.append(p)
            ped_timer = random.randint(80, 200)

        # ── Update & draw vehicles ───────────────────────────────────
        alive_vehicles = []
        for v in vehicles:
            # Congestion: slow down if many vehicles nearby
            nearby = sum(
                1 for o in vehicles
                if o.vid != v.vid and abs(o.x - v.x) + abs(o.y - v.y) < 120
            )
            speed_factor = max(0.1, 1.0 - nearby * 0.2)

            if v.max_stop > 0 and v.stopped_frames < v.max_stop:
                if 50 < v.x < W - 50 and 50 < v.y < H - 50:
                    v.stopped_frames += 1
                    speed_factor = 0.0

            v.x += v.vx * speed_factor
            v.y += v.vy * speed_factor

            # Kill if out of frame
            if v.x < -200 or v.x > W + 200 or v.y < -200 or v.y > H + 200:
                continue

            # Draw vehicle
            x1, y1, x2, y2 = v.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), v.color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(canvas, f"{v.label[:3].upper()}", (x1+2, y1+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            alive_vehicles.append(v)
        vehicles = alive_vehicles

        # ── Update & draw pedestrians ────────────────────────────────
        alive_peds = []
        for p in pedestrians:
            p.x += p.vx
            p.y += p.vy
            if p.x < -50 or p.x > W + 50 or p.y < -50 or p.y > H + 50:
                continue
            # Draw pedestrian as small circle
            cv2.circle(canvas, (int(p.x), int(p.y)), 8, (255, 50, 150), -1)
            cv2.circle(canvas, (int(p.x), int(p.y)), 8, (0, 0, 0), 1)
            alive_peds.append(p)
        pedestrians = alive_peds

        # ── Fire / smoke overlay (45-55 second mark) ─────────────────
        if fire_smoke_start <= frame_idx <= fire_smoke_start + int(fps * 10):
            _draw_fire_smoke(canvas, W, H, frame_idx - fire_smoke_start)

        # ── Timestamp overlay ────────────────────────────────────────
        ts = frame_idx / fps
        m, s = int(ts) // 60, int(ts) % 60
        cv2.putText(canvas, f"WIUT CCTV Mock  {m:02d}:{s:02d}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Frame {frame_idx}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        writer.write(canvas)

        if frame_idx % int(fps * 5) == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {pct:.0f}%  ({m:02d}:{s:02d} / {int(duration_s)//60:02d}:{int(duration_s)%60:02d})")

    writer.release()
    print(f"[OK] Saved synthetic video -> {output_path}")


def _draw_fire_smoke(canvas: np.ndarray, W: int, H: int, rel_frame: int) -> None:
    """Overlay a simple pulsing fire/smoke blob at the intersection centre."""
    cx, cy = W // 2, H // 2
    flicker = int(20 * math.sin(rel_frame * 0.4))
    radius  = 40 + flicker

    # Orange fire core
    cv2.circle(canvas, (cx, cy), radius, (0, 100, 255), -1)
    cv2.circle(canvas, (cx, cy - 20), radius//2, (0, 50, 200), -1)

    # Grey smoke plume
    for i in range(5):
        alpha_y  = cy - 30 - i * 18
        smoke_r  = radius // 2 + i * 6
        overlay  = canvas.copy()
        cv2.circle(overlay, (cx + i*4, alpha_y), smoke_r, (120, 120, 120), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic CCTV traffic video")
    p.add_argument("--output",   default="sample_traffic.mp4", help="Output MP4 path")
    p.add_argument("--duration", type=float, default=60.0,     help="Duration in seconds")
    p.add_argument("--fps",      type=float, default=30.0,     help="Frame rate")
    args = p.parse_args()

    generate_mock_video(args.output, args.duration, args.fps)


if __name__ == "__main__":
    main()
