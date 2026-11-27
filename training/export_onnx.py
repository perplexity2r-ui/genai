#!/usr/bin/env python3
"""Экспорт обученной модели в ONNX/TensorRT для ускорения инференса.

    python training/export_onnx.py --weights runs/detect/genshin/weights/best.pt --half
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Экспорт YOLO-модели GenshinVision")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--half", action="store_true", help="FP16")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "torchscript"])
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Нужен ultralytics: pip install -r requirements-ml.txt")

    model = YOLO(args.weights)
    path = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    out = Path(path)
    print(f"Экспортировано: {out}")
    print("Использование: python scripts/run_live.py --backend yolo --weights", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
