"""
solution.py
Official evaluation entry point for the WIUT Traffic AI Hackathon.
Public Git repository with solution.py at its root.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import main

if __name__ == "__main__":
    # If no --video argument was provided, check for a default sample video
    args = sys.argv[1:]
    if not any(arg.startswith("--video") for arg in args):
        default_video = PROJECT_ROOT / "sample_traffic.mp4"
        if default_video.exists():
            sys.argv.extend(["--video", str(default_video)])

    main.main()
