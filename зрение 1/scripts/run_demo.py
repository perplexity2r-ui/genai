#!/usr/bin/env python3
"""Демо на синтетической сцене: весь конвейер + веб-оверлей в реальном времени.

    python scripts/run_demo.py --port 8000

Откройте http://localhost:8000 — там кадр, боксы, HP, векторы скорости врагов.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genshinvision.detect import HeuristicDetector
from genshinvision.hud import HudReader, load_calibration
from genshinvision.pipeline import Pipeline
from genshinvision.stream import Hub, pump, start_server
from genshinvision.synthetic import SyntheticGameSource
from genshinvision.track import SortTracker


def main(argv=None):
    parser = argparse.ArgumentParser(description="GenshinVision demo (синтетическая сцена)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=0, help="0 = бесконечно")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--enemies", type=int, default=3)
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args(argv)

    calib = load_calibration(args.calibration)
    source = SyntheticGameSource(
        width=args.width, height=args.height,
        n_enemies=args.enemies, fps=args.fps,
        max_frames=args.frames, realtime=False,
    )
    pipeline = Pipeline(source, HeuristicDetector(calib), HudReader(calib), SortTracker())
    hub = Hub()
    start_server(hub, host=args.host, port=args.port)

    print(f"Оверлей: http://localhost:{args.port}/  (Ctrl+C чтобы остановить)")
    try:
        pump(pipeline, hub, target_fps=args.fps, max_frames=args.frames)
    except KeyboardInterrupt:
        print("\nостановлено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
