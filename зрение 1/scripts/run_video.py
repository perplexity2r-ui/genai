#!/usr/bin/env python3
"""Прогон конвейера по видеофайлу: аннотированное видео + JSONL состояний.

    python scripts/run_video.py --input fight.mp4 --output out/annotated.mp4 --states out/states.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from genshinvision.capture import VideoSource
from genshinvision.detect import make_detector
from genshinvision.hud import HudReader, load_calibration
from genshinvision.pipeline import Pipeline
from genshinvision.track import SortTracker


def main(argv=None):
    parser = argparse.ArgumentParser(description="GenshinVision по видео")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="out/annotated.mp4")
    parser.add_argument("--states", default="out/states.jsonl")
    parser.add_argument("--backend", choices=["heuristic", "yolo"], default="heuristic")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)

    calib = load_calibration(args.calibration)
    source = VideoSource(args.input)
    detector = make_detector(
        args.backend, calib, weights=args.weights,
        conf=args.conf, imgsz=args.imgsz, device=args.device,
    )
    pipeline = Pipeline(source, detector, HudReader(calib), SortTracker())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        source.fps, source.frame_size,
    )
    states_path = Path(args.states)
    states_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    n = 0
    with open(states_path, "w", encoding="utf-8") as fh:
        while True:
            out = pipeline.step()
            if out is None:
                break
            n += 1
            writer.write(out.frame)
            fh.write(json.dumps(out.state.to_dict(), ensure_ascii=False) + "\n")
            if args.max_frames and n >= args.max_frames:
                break
    writer.release()

    dt = time.perf_counter() - t0
    print(f"Кадров: {n} за {dt:.1f} s ({n / max(dt, 1e-6):.1f} fps)")
    print(f"Видео:  {out_path}")
    print(f"Стейты: {states_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
