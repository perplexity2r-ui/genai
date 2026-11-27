#!/usr/bin/env python3
"""Захват реального экрана (игра в окне/полноэкранно) → трекинг → оверлей.

    python scripts/run_live.py --backend yolo --weights runs/detect/best.pt
    python scripts/run_live.py                       # эвристика, без обучения
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genshinvision.capture import ScreenSource
from genshinvision.detect import make_detector
from genshinvision.hud import HudReader, load_calibration
from genshinvision.pipeline import Pipeline
from genshinvision.stream import Hub, pump, start_server
from genshinvision.track import SortTracker


def main(argv=None):
    parser = argparse.ArgumentParser(description="GenshinVision live")
    parser.add_argument("--backend", choices=["heuristic", "yolo"], default="heuristic")
    parser.add_argument("--weights", default=None, help="путь к best.pt для YOLO")
    parser.add_argument("--calibration", default=None, help="JSON с ROI HUD")
    parser.add_argument("--monitor", type=int, default=1, help="номер монитора (mss)")
    parser.add_argument("--region", default=None, help="x,y,w,h — область экрана")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.30, help="порог YOLO")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None, help="cuda / cpu / 0")
    parser.add_argument("--display", action="store_true", help="показать окно OpenCV")
    args = parser.parse_args(argv)

    calib = load_calibration(args.calibration)
    region = tuple(map(int, args.region.split(","))) if args.region else None
    source = ScreenSource(monitor=args.monitor, region=region)
    detector = make_detector(
        args.backend, calib,
        weights=args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device,
    )
    pipeline = Pipeline(source, detector, HudReader(calib), SortTracker())

    hub = Hub()
    start_server(hub, host=args.host, port=args.port)
    print(f"Оверлей: http://localhost:{args.port}/  |  backend={args.backend}  (Ctrl+C — стоп)")

    try:
        if args.display:
            import cv2

            interval = 1.0 / max(args.fps, 1.0)
            n = 0
            while True:
                out = pipeline.step()
                if out is None:
                    break
                n += 1
                hub.update(out.state.to_dict(), out.frame)
                cv2.imshow("GenshinVision", out.frame)
                if cv2.waitKey(max(1, int(interval * 1000))) & 0xFF == ord("q"):
                    break
                if args.frames and n >= args.frames:
                    break
            cv2.destroyAllWindows()
        else:
            pump(pipeline, hub, target_fps=args.fps, max_frames=args.frames)
    except KeyboardInterrupt:
        print("\nостановлено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
