#!/usr/bin/env python3
"""Дообучение YOLOv8 на собственных скриншотах Genshin Impact.

    pip install -r requirements-ml.txt
    python training/train_yolo.py --data training/dataset.yaml \
        --model yolov8n.pt --epochs 80 --imgsz 960 --device 0
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 на датасете GenshinVision")
    parser.add_argument("--data", default=str(Path(__file__).parent / "dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n/s/m или свой .pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="0 / cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="genshin")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Нужен ultralytics: pip install -r requirements-ml.txt")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=15,
        verbose=True,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nЛучшие веса: {best}")
    print(f"Запуск: python scripts/run_live.py --backend yolo --weights {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
