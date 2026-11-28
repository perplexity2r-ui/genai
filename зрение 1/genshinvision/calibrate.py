"""Интерактивная калибровка HUD-регионов по скриншоту игры.

Запуск:
    python -m genshinvision.calibrate screenshot.png -o config/my_calib.json

Для каждого региона: зажмите ЛКМ, обведите прямоугольник, Enter — принять,
r — перерисовать, q — выйти. Координаты сохраняются в долях ширины/высоты
экрана, поэтому калибровка переносится на любое разрешение.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .hud import DEFAULT_CALIBRATION

REGIONS = [
    ("player_hp_bar", "Полоска HP активного персонажа (зелёная)"),
    ("player_hp_text", "Текст 'HP 1234/5678' над полоской"),
    ("stamina_bar", "Полоска стамины"),
    ("boss_hp_bar", "Полоска HP босса (сверху по центру)"),
    ("combat_region", "Зона боя (там, где ходят противники) — большой прямоугольник"),
]


class _Selector:
    def __init__(self, img, title):
        self.img = img.copy()
        self.canvas = img.copy()
        self.title = title
        self.start = None
        self.end = None
        cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("calibrate", self._mouse)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.end = (x, y)
            self._draw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.end = (x, y)
            self._draw()

    def _draw(self):
        self.canvas = self.img.copy()
        if self.start and self.end:
            cv2.rectangle(self.canvas, self.start, self.end, (66, 200, 244), 2)

    def run(self):
        print(f"\n→ {self.title}")
        while True:
            cv2.imshow("calibrate", self.canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # Enter
                if self.start and self.end:
                    x1, y1 = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
                    x2, y2 = max(self.start[0], self.end[0]), max(self.start[1], self.end[1])
                    if x2 - x1 > 4 and y2 - y1 > 4:
                        return (x1, y1, x2 - x1, y2 - y1)
                print("   нарисуйте прямоугольник ЛКМ и нажмите Enter")
            elif key == ord("r"):
                self.start = self.end = None
                self._draw()
            elif key == ord("s"):
                return None  # пропустить регион
            elif key == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Калибровка HUD-регионов GenshinVision")
    parser.add_argument("screenshot", help="скриншот игры (png/jpg)")
    parser.add_argument("-o", "--out", default="config/my_calib.json", help="файл для сохранения")
    parser.add_argument("--print-defaults", action="store_true", help="показать встроенную калибровку и выйти")
    args = parser.parse_args(argv)

    if args.print_defaults:
        print(json.dumps(DEFAULT_CALIBRATION, indent=2, ensure_ascii=False))
        return 0

    img = cv2.imread(args.screenshot)
    if img is None:
        print(f"Не удалось прочитать {args.screenshot}")
        return 1
    h, w = img.shape[:2]
    print(f"Скриншот {w}x{h}. Enter — принять, r — сброс, s — пропустить, q — выход.")

    calib = {"resolution": [w, h]}
    for key, desc in REGIONS:
        sel = _Selector(img, f"{key}: {desc}")
        rect = sel.run()
        if rect is None:
            print("   пропущено")
            continue
        x, y, rw, rh = rect
        calib[key] = {
            "x": round(x / w, 5), "y": round(y / h, 5),
            "w": round(rw / w, 5), "h": round(rh / h, 5),
        }
        print(f"   {key} = {calib[key]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(json.dumps(DEFAULT_CALIBRATION))
    merged.update(calib)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    cv2.destroyAllWindows()
    print(f"\nСохранено: {out}")
    print("Запуск: python scripts/run_live.py --calibration", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
