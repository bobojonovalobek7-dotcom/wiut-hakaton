"""
app.py
Streamlit Interactive Dashboard — WIUT Traffic Event Detection & Risk Analytics.

Launch:
    streamlit run app.py

Features
--------
- Video upload (MP4/AVI/MOV) with drag-and-drop
- Configurable analysis settings (confidence, model, stride)
- Live progress tracking during processing
- Risk score timeline chart
- Incident breakdown by category
- Searchable event log table with filtering
- KPI summary cards
- Download buttons for events.json and events.csv
"""

from __future__ import annotations

import collections
import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WIUT Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0a0d16;
    color: #e2e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0d16 0%, #0d1421 50%, #0a0d16 100%);
}

/* Header title */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
}
.main-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-bottom: 1.5rem;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,40,70,0.9), rgba(15,25,50,0.9));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: all 0.3s ease;
}
.kpi-card:hover {
    border-color: rgba(99,102,241,0.7);
    transform: translateY(-2px);
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
}
.kpi-label {
    font-size: 0.78rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

/* Event badges */
.event-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #cbd5e1;
    border-left: 3px solid #818cf8;
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* Progress bar override */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 25, 0.95);
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.4);
}

/* Metric */
[data-testid="metric-container"] {
    background: rgba(20,30,55,0.8);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Event colour map ----------------------------------------------------------
# ---------------------------------------------------------------------------

EVENT_COLOURS = {
    "accident":            "#ef4444",
    "near_miss":           "#f97316",
    "red_light":           "#dc2626",
    "wrong_way":           "#b91c1c",
    "illegal_u_turn":      "#a855f7",
    "stopped_vehicle":     "#0ea5e9",
    "jaywalking":          "#ec4899",
    "failure_to_yield":    "#f59e0b",
    "illegal_turn":        "#8b5cf6",
    "solid_line_crossing": "#14b8a6",
    "stop_line":           "#22c55e",
    "congestion":          "#eab308",
    "road_obstacle":       "#78716c",
    "fire_smoke":          "#ff6b35",
    "junction_blocking":   "#d97706",
    "yielding_to_pedestrian": "#10b981",
    "speeding":            "#e11d48",
}

EVENT_SEVERITY = {
    "accident": 10, "fire_smoke": 9, "red_light": 8, "wrong_way": 8,
    "near_miss": 7, "failure_to_yield": 7, "junction_blocking": 7,
    "speeding": 6, "illegal_u_turn": 6, "illegal_turn": 5,
    "solid_line_crossing": 5, "jaywalking": 5, "yielding_to_pedestrian": 4,
    "stop_line": 4, "stopped_vehicle": 3, "road_obstacle": 3, "congestion": 2,
}

EVENT_ICONS = {
    "accident": "💥", "near_miss": "⚠️", "red_light": "🔴", "wrong_way": "⛔",
    "illegal_u_turn": "↩️", "stopped_vehicle": "🅿️", "jaywalking": "🚶",
    "failure_to_yield": "🚧", "illegal_turn": "↪️", "solid_line_crossing": "❌",
    "stop_line": "🛑", "congestion": "🚦", "road_obstacle": "⬛", "fire_smoke": "🔥",
    "junction_blocking": "🚷", "yielding_to_pedestrian": "🚶‍♂️", "speeding": "⚡",
}

# ---------------------------------------------------------------------------
# Sidebar settings ----------------------------------------------------------
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        st.markdown("---")

        model = st.selectbox(
            "YOLO Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n.pt"],
            index=0,
            help="Larger models are more accurate but slower on CPU.",
        )
        conf = st.slider("Detection Confidence", 0.10, 0.90, 0.35, 0.05)
        stride = st.selectbox(
            "Frame Stride (process every N frames)",
            [1, 2, 3, 4],
            index=0,
            help="Higher stride = faster processing but less temporal precision.",
        )
        show_zones  = st.toggle("Show ROI Zone Overlays", value=True)
        show_trails = st.toggle("Show Trajectory Trails", value=True)

        st.markdown("---")
        st.markdown("### 📋 Event Filters")
        selected_events = st.multiselect(
            "Show only events",
            options=list(EVENT_COLOURS.keys()),
            default=[],
            placeholder="All events",
        )

        st.markdown("---")
        st.markdown(
            "<small style='color:#64748b'>WIUT Hackathon 2024<br>"
            "Traffic Event Detection System v1.0</small>",
            unsafe_allow_html=True,
        )

    return {
        "model": model,
        "conf": conf,
        "stride": stride,
        "show_zones": show_zones,
        "show_trails": show_trails,
        "selected_events": selected_events,
    }


# ---------------------------------------------------------------------------
# KPI cards -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def render_kpi(events_df: pd.DataFrame, risk_max: float) -> None:
    total      = len(events_df)
    categories = events_df["event"].nunique() if total else 0
    severity   = events_df["event"].map(EVENT_SEVERITY).sum() if total else 0
    duration   = events_df["duration"].sum() if total and "duration" in events_df.columns else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, str(total),       "Total Events"),
        (c2, str(categories),  "Event Types"),
        (c3, f"{severity}",    "Risk Score Sum"),
        (c4, f"{duration:.1f}s", "Total Incident Duration"),
        (c5, f"{risk_max:.0%}", "Peak Collision Risk"),
    ]
    for col, val, label in kpi_data:
        with col:
            st.markdown(
                f"""<div class='kpi-card'>
                    <div class='kpi-value'>{val}</div>
                    <div class='kpi-label'>{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Event breakdown chart -----------------------------------------------------
# ---------------------------------------------------------------------------

def render_breakdown_chart(events_df: pd.DataFrame) -> None:
    if events_df.empty:
        return

    st.markdown("<div class='section-header'>📊 Event Breakdown</div>", unsafe_allow_html=True)
    counts = events_df["event"].value_counts().reset_index()
    counts.columns = ["event", "count"]

    # Build a simple horizontal bar chart using altair if available, else st.bar_chart
    try:
        import altair as alt
        chart = (
            alt.Chart(counts)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("count:Q", title="Count", axis=alt.Axis(grid=False)),
                y=alt.Y("event:N", sort="-x", title="", axis=alt.Axis(labelColor="#94a3b8")),
                color=alt.Color(
                    "event:N",
                    scale=alt.Scale(
                        domain=list(EVENT_COLOURS.keys()),
                        range=list(EVENT_COLOURS.values()),
                    ),
                    legend=None,
                ),
                tooltip=["event", "count"],
            )
            .properties(height=max(200, len(counts) * 34), background="transparent")
            .configure_axis(labelColor="#94a3b8", titleColor="#64748b", gridColor="#1e293b")
            .configure_view(stroke=None)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(counts.set_index("event")["count"])


# ---------------------------------------------------------------------------
# Timeline chart (risk score over time) ------------------------------------
# ---------------------------------------------------------------------------

def render_timeline(events_df: pd.DataFrame) -> None:
    if events_df.empty:
        return

    st.markdown("<div class='section-header'>📅 Incident Timeline</div>", unsafe_allow_html=True)

    try:
        import altair as alt

        df = events_df.copy()
        df["icon"] = df["event"].map(EVENT_ICONS).fillna("❓")
        df["label"] = df["icon"] + " " + df["event"]
        df["colour"] = df["event"].map(EVENT_COLOURS).fillna("#888")

        chart = (
            alt.Chart(df)
            .mark_bar(height=14, cornerRadius=3)
            .encode(
                x=alt.X("t_start:Q", title="Time (seconds)", axis=alt.Axis(gridColor="#1e293b")),
                x2="t_end:Q",
                y=alt.Y("event:N", title="", sort=alt.EncodingSortField("t_start", order="ascending")),
                color=alt.Color(
                    "event:N",
                    scale=alt.Scale(
                        domain=list(EVENT_COLOURS.keys()),
                        range=list(EVENT_COLOURS.values()),
                    ),
                    legend=None,
                ),
                tooltip=["event", "t_start", "t_end", "duration", "confidence"],
            )
            .properties(height=max(250, df["event"].nunique() * 36), background="transparent")
            .configure_axis(labelColor="#94a3b8", titleColor="#64748b")
            .configure_view(stroke=None)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.dataframe(events_df[["event", "t_start", "t_end", "confidence"]])


# ---------------------------------------------------------------------------
# Event log table -----------------------------------------------------------
# ---------------------------------------------------------------------------

def render_event_table(events_df: pd.DataFrame, selected_events: list[str]) -> None:
    st.markdown("<div class='section-header'>📋 Event Log</div>", unsafe_allow_html=True)

    if events_df.empty:
        st.info("No events detected.")
        return

    df = events_df.copy()
    if selected_events:
        df = df[df["event"].isin(selected_events)]

    if "t_start" in df.columns:
        df["t_start"] = df["t_start"].round(2)
    if "t_end" in df.columns:
        df["t_end"] = df["t_end"].round(2)
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].round(3)

    # Sort by severity then time
    df["_sev"] = df["event"].map(EVENT_SEVERITY).fillna(0)
    df = df.sort_values(["_sev", "t_start"], ascending=[False, True]).drop(columns=["_sev"])

    search = st.text_input("🔍 Search events", placeholder="Filter by event name…")
    if search:
        df = df[df["event"].str.contains(search, case=False, na=False)]

    st.dataframe(
        df,
        use_container_width=True,
        height=350,
        column_config={
            "event":      st.column_config.TextColumn("Event"),
            "t_start":    st.column_config.NumberColumn("Start (s)", format="%.2f"),
            "t_end":      st.column_config.NumberColumn("End (s)", format="%.2f"),
            "duration":   st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
            "track_ids":  st.column_config.TextColumn("Track IDs"),
        },
    )


# ---------------------------------------------------------------------------
# Download buttons ----------------------------------------------------------
# ---------------------------------------------------------------------------

def render_downloads(events: list[dict]) -> None:
    st.markdown("<div class='section-header'>⬇️ Export Results</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    json_bytes = json.dumps(events, indent=2).encode("utf-8")
    with c1:
        st.download_button(
            "📥 Download events.json",
            data=json_bytes,
            file_name="events.json",
            mime="application/json",
        )

    if events:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)
        with c2:
            st.download_button(
                "📥 Download events.csv",
                data=buf.getvalue().encode("utf-8"),
                file_name="events.csv",
                mime="text/csv",
            )


def render_snapshots_gallery(snapshots: list[dict]) -> None:
    if not snapshots:
        st.info("No incident snapshots captured yet.")
        return

    # Filter options
    ev_types = sorted(list(set(s.get("event", "") for s in snapshots)))
    selected = st.selectbox("Filter snapshots by category:", ["All Snapshots"] + ev_types)
    filtered = snapshots if selected == "All Snapshots" else [s for s in snapshots if s.get("event") == selected]

    st.markdown(f"<small style='color:#94a3b8;'>Showing {len(filtered)} evidence snapshots</small>", unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, snap in enumerate(filtered):
        col = cols[idx % 3]
        with col:
            img_path = snap.get("image_path")
            if img_path and Path(img_path).exists():
                st.image(img_path, use_container_width=True)
                event_name = snap.get("event", "incident")
                colour = EVENT_COLOURS.get(event_name, "#6366f1")
                icon = EVENT_ICONS.get(event_name, "⚠️")
                ts = snap.get("timestamp_s", 0.0)
                ts_str = f"{int(ts)//60:02d}:{int(ts)%60:02d} ({ts:.1f}s)"
                risk_lvl = snap.get("risk_level", "MEDIUM")
                risk_val = snap.get("risk_score", 0.0)
                st.markdown(
                    f"""
                    <div style='background:rgba(20,30,55,0.7); border:1px solid {colour}44; border-radius:10px; padding:8px 12px; margin-bottom:8px;'>
                        <div style='font-size:0.9rem; font-weight:600; color:{colour};'>{icon} {event_name.upper()}</div>
                        <div style='font-size:0.8rem; color:#94a3b8;'>⏱️ Time: <b>{ts_str}</b> | 🎯 Risk: <b style='color:#f43f5e;'>{risk_val:.0%} ({risk_lvl})</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with open(img_path, "rb") as img_file:
                    st.download_button(
                        label=f"💾 Download {snap.get('id', 'photo')}",
                        data=img_file.read(),
                        file_name=Path(img_path).name,
                        mime="image/jpeg",
                        key=f"dl_snap_{idx}",
                        use_container_width=True,
                    )


def generate_executive_report(events: list[dict], snapshots: list[dict]) -> str:
    total_events = len(events)
    severe_events = [e for e in events if e.get("event") in ("accident", "red_light", "wrong_way", "failure_to_yield", "junction_blocking", "speeding")]
    safety_score = max(0, 100 - len(severe_events) * 12 - (total_events - len(severe_events)) * 4)
    grade = "A (Excellent)" if safety_score >= 88 else ("B (Good)" if safety_score >= 72 else ("C (Moderate Risk)" if safety_score >= 55 else "D (High Risk)"))

    tally = collections.Counter(e.get("event") for e in events)

    md = f"""# 🚦 Traffic Safety & CCTV Incident Audit Report
**WIUT Traffic AI Analytics Engine**  
*Generated on:* `{time.strftime('%Y-%m-%d %H:%M:%S')}`  
*CCTV Analysis Location:* `Amir Timur / Navoi Intersection, Tashkent`

---

## 📊 Safety Index & Overview
- **Overall Intersection Safety Score:** **{safety_score}/100** (`{grade}`)
- **Total Incidents Recorded:** **{total_events}**
- **Severe Violations:** **{len(severe_events)}**
- **Evidence Snapshots Captured:** **{len(snapshots)}**

---

## 🚨 Incident Breakdown
| Category | Event Icon | Count | Severity Level |
|---|---|---|---|
"""
    for ev, cnt in tally.items():
        sev = EVENT_SEVERITY.get(ev, 5)
        icon = EVENT_ICONS.get(ev, "⚠️")
        md += f"| {ev.replace('_', ' ').title()} | {icon} | **{cnt}** | {sev}/10 |\n"

    md += """
---

## 🔍 Key Recommendations for Municipal Traffic Dept
1. **Intersection Box Clearance:** Ensure junction yellow box markings are repainted to prevent junction blocking.
2. **Pedestrian Safety Enforcement:** Review zebra crossing visibility and yield compliance on the right avenue.
3. **Signal Phase Timing:** Adjust red clearance intervals during peak flow periods to mitigate converging near-misses.
4. **Speed Enforcement:** Install automated radar signs along high-speed approaches.
"""
    return md


# ---------------------------------------------------------------------------
# Video processing wrapper --------------------------------------------------
# ---------------------------------------------------------------------------

def process_video(
    video_path: str,
    *,
    model: str,
    conf: float,
    stride: int,
    show_zones: bool,
    show_trails: bool,
    progress_placeholder,
    status_placeholder,
) -> tuple[list[dict], str | None]:
    """
    Run the full pipeline and return (events, output_video_path).
    Displays live progress in Streamlit.
    """
    from src.config import load_config
    from src.detector import DetectorTracker, TrafficLightStateDetector, video_frame_generator
    from src.risk_engine import CollisionRiskEngine
    from src.rules import TrafficRuleEngine
    from src.visualizer import TrafficVisualizer

    # Video meta
    cap = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    vid_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Config
    cfg = load_config(actual_resolution=(vid_w, vid_h))
    cfg.camera.fps = fps

    # Modules
    detector    = DetectorTracker(model_path=model, conf_threshold=conf,
                                   pixels_per_meter=cfg.camera.pixels_per_meter,
                                   fps=fps, frame_stride=stride)
    tl_detector = TrafficLightStateDetector()
    rule_engine = TrafficRuleEngine(cfg, tl_detector)
    risk_engine = CollisionRiskEngine(cfg)
    visualizer  = TrafficVisualizer(cfg, show_zones=show_zones, show_trails=show_trails)

    # Output video to temp file
    out_path = tempfile.mktemp(suffix="_annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h))

    snap_dir = Path(tempfile.mkdtemp(prefix="traffic_snaps_"))
    all_events: list[dict] = []
    snapshots:  list[dict] = []
    tl_states:  dict       = {}
    last_snap_time: dict[str, float] = {}
    snap_count: int = 1

    for frame_idx, ts, frame in video_frame_generator(video_path):
        if total_frms > 0:
            pct = frame_idx / total_frms
            progress_placeholder.progress(min(pct, 0.99))
            if frame_idx % max(1, int(fps)) == 0:
                m, s = int(ts)//60, int(ts)%60
                status_placeholder.info(f"🔍 Analysing frame {frame_idx}/{total_frms}  [{m:02d}:{s:02d}]")

        tracks = detector.process_frame(frame)

        for tl in cfg.traffic_lights:
            state = tl_detector.detect(frame, tl.id, tl.bbox, ts,
                                        tl.default_initial_state,
                                        tl.cycle_red_s, tl.cycle_green_s, tl.cycle_yellow_s)
            tl_states[tl.id] = state

        closed = rule_engine.evaluate(frame, tracks, frame_idx, ts)
        for evt in closed:
            all_events.append(evt.to_dict())

        active = [k[0] if isinstance(k, tuple) else k
                  for k, ae in rule_engine._active.items()
                  if (ts - ae.t_last) < 2.0]

        risk = risk_engine.compute(tracks, ts)
        for e in active:
            visualizer.notify_event(e)

        annotated = visualizer.render(frame, tracks, risk, active, ts, tl_states)
        if writer.isOpened():
            writer.write(annotated)

        # Snapshot capture
        should_snap = False
        snap_tag = "event"
        if closed:
            should_snap = True
            snap_tag = closed[0].event
        elif active:
            snap_tag = active[0]
            if ts - last_snap_time.get(snap_tag, -99.0) >= 2.5:
                should_snap = True
        elif risk and risk.overall >= 0.60:
            snap_tag = f"high_risk_{risk.overall:.2f}"
            if ts - last_snap_time.get("high_risk", -99.0) >= 3.0:
                should_snap = True

        if should_snap:
            last_snap_time[snap_tag] = ts
            snap_img_path = snap_dir / f"snap_{snap_count:03d}_{snap_tag}_t{ts:.1f}s.jpg"
            cv2.imwrite(str(snap_img_path), annotated)
            snapshots.append({
                "id": f"snap_{snap_count:03d}",
                "event": snap_tag,
                "timestamp_s": round(ts, 2),
                "risk_score": round(risk.overall if risk else 0.0, 2),
                "risk_level": risk.level if risk else "LOW",
                "image_path": str(snap_img_path),
            })
            snap_count += 1

    flushed = rule_engine.flush_open_events(total_frms / max(fps, 1.0))
    for evt in flushed:
        all_events.append(evt.to_dict())

    writer.release()
    progress_placeholder.progress(1.0)
    status_placeholder.success(f"✅ Processing complete! {len(all_events)} events detected, {len(snapshots)} snapshots captured.")

    return all_events, out_path, snapshots


# ---------------------------------------------------------------------------
# Main app ------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    settings = render_sidebar()

    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='main-title'>🚦 WIUT Traffic Analytics</div>"
        "<div class='main-subtitle'>AI-powered CCTV traffic event detection & collision risk prediction</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Upload / Demo section ─────────────────────────────────────────────
    col_upload, col_demo = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload Traffic Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload an MP4 or AVI CCTV footage file.",
        )
    with col_demo:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_demo = st.button("🎬 Generate Demo Video", use_container_width=True)

    # ── Demo video generation ─────────────────────────────────────────────
    if generate_demo:
        from tools.mock_data_generator import generate_mock_video
        demo_path = str(_ROOT / "sample_traffic.mp4")
        with st.spinner("Generating synthetic traffic video (30 seconds)…"):
            generate_mock_video(demo_path, duration_s=30, fps=30)
        st.success(f"✅ Demo video saved: {demo_path}")
        st.info("Refresh and upload the demo video to run analysis.")

    if "events" not in st.session_state:
        st.session_state["events"]    = []
        st.session_state["out_video"] = None
        st.session_state["snapshots"] = []
        st.session_state["current_video_name"] = None

    if uploaded is None:
        st.markdown(
            """
            <div style='
                background: rgba(30,40,70,0.4);
                border: 1px dashed rgba(99,102,241,0.4);
                border-radius: 16px;
                padding: 3rem;
                text-align: center;
                color: #64748b;
            '>
                <div style='font-size:3rem;margin-bottom:1rem;'>📹</div>
                <div style='font-size:1.1rem;font-weight:500;color:#94a3b8;'>Upload a video file or generate a demo to begin analysis</div>
                <div style='font-size:0.8rem;margin-top:0.5rem;'>Supports MP4, AVI, MOV · Detects 14 traffic event categories</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Check if this is a new video replacing an already analyzed old video
    is_new_video = False
    if st.session_state["current_video_name"] is not None and st.session_state["current_video_name"] != uploaded.name:
        is_new_video = True

    # ── Run analysis ──────────────────────────────────────────────────────
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if is_new_video and run_btn:
        st.session_state["confirm_new_video"] = True

    if st.session_state.get("confirm_new_video", False):
        st.warning("⚠️ Yangi video yukladingiz. Eski videoning barcha tahlil natijalari (va rasmlar) o'chib ketadi. Davom etasizmi?")
        col_ok, col_cancel = st.columns([1, 5])
        with col_ok:
            confirm_btn = st.button("Ha, tasdiqlayman", type="primary")
        with col_cancel:
            cancel_btn = st.button("Bekor qilish")
            
        if cancel_btn:
            st.session_state["confirm_new_video"] = False
            st.rerun()
        if confirm_btn:
            st.session_state["confirm_new_video"] = False
            # Clear old state so we can run
            st.session_state["events"]    = []
            st.session_state["out_video"] = None
            st.session_state["snapshots"] = []
            st.session_state["current_video_name"] = uploaded.name
            st.rerun()
        return  # Stop execution until confirmed

    if run_btn and not is_new_video:
        # Directly run if it's the first video or same video
        pass

    if (run_btn and not is_new_video) or st.session_state.get("_force_run", False):
        st.session_state["_force_run"] = False
        # ── Save upload to temp file in chunks to prevent OOM ──
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            while True:
                chunk = uploaded.read(8 * 1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = tmp.name

        st.session_state["current_video_name"] = uploaded.name
        
        progress_bar = st.progress(0)
        status_msg   = st.empty()

        events, out_video, snapshots = process_video(
            tmp_path,
            model=settings["model"],
            conf=settings["conf"],
            stride=settings["stride"],
            show_zones=settings["show_zones"],
            show_trails=settings["show_trails"],
            progress_placeholder=progress_bar,
            status_placeholder=status_msg,
        )
        st.session_state["events"]    = events
        st.session_state["out_video"] = out_video
        st.session_state["snapshots"] = snapshots
        st.balloons()

    events    = st.session_state.get("events", [])
    out_video = st.session_state.get("out_video")
    snapshots = st.session_state.get("snapshots", [])

    if not events and not out_video:
        return

    # ── Results section ───────────────────────────────────────────────────
    st.markdown("---")

    # KPIs
    st.markdown("<div class='section-header'>📈 Summary Statistics & KPIs</div>", unsafe_allow_html=True)
    events_df = pd.DataFrame(events) if events else pd.DataFrame()
    risk_max = max((e.get("confidence", 0) for e in events), default=0.0)
    render_kpi(events_df, risk_max)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎬 Annotated Video",
        f"📸 Incident Evidence ({len(snapshots)})",
        "📊 Danger & Timeline Analytics",
        "📋 Event Audit Log",
        "📄 Safety Audit Report",
    ])

    with tab1:
        if out_video and Path(out_video).exists():
            vid_col, _ = st.columns([3, 1])
            with vid_col:
                with open(out_video, "rb") as f:
                    st.video(f.read())
        else:
            st.info("No video output available.")

    with tab2:
        render_snapshots_gallery(snapshots)

    with tab3:
        if not events_df.empty:
            chart_col, timeline_col = st.columns(2)
            with chart_col:
                render_breakdown_chart(events_df)
            with timeline_col:
                render_timeline(events_df)
        else:
            st.info("No timeline analytics to show.")

    with tab4:
        render_event_table(events_df, settings["selected_events"])
        render_downloads(events)

    with tab5:
        st.markdown("<div class='section-header'>📄 Municipal Traffic Safety Audit Report</div>", unsafe_allow_html=True)
        rep_md = generate_executive_report(events, snapshots)
        st.markdown(rep_md)
        st.download_button(
            label="📥 Download Audit Report (.md)",
            data=rep_md.encode("utf-8"),
            file_name="traffic_safety_audit_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
