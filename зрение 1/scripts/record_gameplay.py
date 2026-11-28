#!/usr/bin/env python3
"""Запись геймплея с экрана + сэмплирование кадров для разметки датасета.

    python scripts/record_gameplay.py --seconds 120 --out data/raw/fight1.mp4 \
        --sample-dir data/frames --sample-every 1.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from genshinvision.capture import ScreenSource


def main(argv=None):
    parser = argparse.ArgumentParser(description="Запись экрана для датасета")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--region", default=None, help="x,y,w,h")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--out", default="data/raw/gameplay.mp4")
    parser.add_argument("--sample-dir", default=None, help="куда складывать кадры для разметки")
    parser.add_argument("--sample-every", type=float, default=1.0, help="секунд между кадрами")
    args = parser.parse_args(argv)

    region = tuple(map(int, args.region.split(","))) if args.region else None
    source = ScreenSource(monitor=args.monitor, region=region)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, source.frame_size,
    )

    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    if sample_dir:
        (sample_dir / "images").mkdir(parents=True, exist_ok=True)

    print(f"Запись {args.seconds:.0f} s @ {args.fps} fps → {out_path}")
    t_end = time.time() + args.seconds
    last_sample = 0.0
    n = 0
    while time.time() < t_end:
        ts, frame = source.read()
        writer.write(frame)
        n += 1
        if sample_dir and ts - last_sample >= args.sample_every:
            last_sample = ts
            name = sample_dir / "images" / f"gameplay_{int(ts * 1000)}.png"
            cv2.imwrite(str(name), frame)
        time.sleep(1.0 / args.fps)

    writer.release()
    print(f"Готово: {n} кадров → {out_path}")
    if sample_dir:
        print(f"Кадры для разметки: {sample_dir / 'images'}")
        print("Разметьте их (Roboflow/CVAT/labelImg) и см. training/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
