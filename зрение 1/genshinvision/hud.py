"""HUD reading: HP / stamina / boss bar extraction from calibrated screen regions.

Everything works on *relative* screen coordinates (fractions of frame width and
height), so one calibration works at any resolution once the anchors are right.
Run ``python -m genshinvision.calibrate`` on a real screenshot to set them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .types import HudReading

# --------------------------------------------------------------------------- #
# Default layout. These match the bundled synthetic scene and are a reasonable
# starting point for a 16:9 windowed Genshin client — calibrate for your setup.
# --------------------------------------------------------------------------- #
DEFAULT_CALIBRATION: dict[str, Any] = {
    "resolution": [1280, 720],
    "hp_max": 12000,
    "player_hp_bar": {"x": 0.030, "y": 0.845, "w": 0.170, "h": 0.022},
    "player_hp_text": {"x": 0.030, "y": 0.800, "w": 0.170, "h": 0.038},
    "stamina_bar": {"x": 0.030, "y": 0.885, "w": 0.130, "h": 0.014},
    "boss_hp_bar": {"x": 0.280, "y": 0.075, "w": 0.440, "h": 0.020},
    "party_bars": [
        {"x": 0.030, "y": 0.560, "w": 0.110, "h": 0.013},
        {"x": 0.030, "y": 0.625, "w": 0.110, "h": 0.013},
        {"x": 0.030, "y": 0.690, "w": 0.110, "h": 0.013},
    ],
    "burst_icons": [
        {"x": 0.930, "y": 0.860, "w": 0.032, "h": 0.032},
        {"x": 0.930, "y": 0.790, "w": 0.032, "h": 0.032},
        {"x": 0.930, "y": 0.720, "w": 0.032, "h": 0.032},
    ],
    "minimap": {"cx": 0.925, "cy": 0.110, "r": 0.065},
    "combat_region": {"x": 0.150, "y": 0.130, "w": 0.700, "h": 0.620},
    "burst_ready_luma": 140.0,
    # --- иконки на миникарте: HSV-правила (OpenCV: H 0-179, S/V 0-255) ---
    "minimap_icons": {
        "teleport": {"h": [95, 128], "s": [120, 255], "v": [150, 255], "min_area": 15, "max_area": 450},
        "statue": {"h": [95, 128], "s": [35, 119], "v": [175, 255], "min_area": 15, "max_area": 450},
        "quest": {"h": [18, 35], "s": [130, 255], "v": [150, 255], "min_area": 12, "max_area": 350},
        "party": {"h": [0, 179], "s": [0, 50], "v": [195, 255], "min_area": 8, "max_area": 250},
    },
    # --- объекты мира: сундуки, телепорты, кукулиусы (подношения семи
    #     архонтам), головоломки. Цвет + форма. Для боёв — YOLO (training/). ---
    "object_rules": [
        {"class": "teleport_active", "h": [98, 128], "s": [110, 255], "v": [130, 255],
         "min_area": 350, "max_area": 12000, "aspect": [0.15, 1.5], "min_fill": 0.2},
        # неактивный телепорт/статуя светится красным («подойди и нажми F»);
        # диапазон [0, 6] не захватывает оранжевый (головоломки, сундуки)
        {"class": "teleport_inactive", "h": [0, 6], "s": [130, 255], "v": [120, 255],
         "min_area": 800, "max_area": 12000, "aspect": [0.15, 1.5], "min_fill": 0.35},
        {"class": "teleport_inactive", "h": [170, 179], "s": [130, 255], "v": [120, 255],
         "min_area": 800, "max_area": 12000, "aspect": [0.15, 1.5], "min_fill": 0.35},
        {"class": "oculus", "element": "anemo", "h": [35, 85], "s": [110, 255], "v": [120, 255],
         "min_area": 250, "max_area": 3500, "aspect": [0.5, 1.8], "min_fill": 0.15},
        {"class": "oculus", "element": "geo", "h": [15, 35], "s": [120, 255], "v": [130, 255],
         "min_area": 250, "max_area": 3000, "aspect": [0.5, 1.8], "min_fill": 0.15},
        {"class": "oculus", "element": "electro", "h": [138, 162], "s": [100, 255], "v": [120, 255],
         "min_area": 250, "max_area": 3500, "aspect": [0.5, 1.8], "min_fill": 0.15},
        {"class": "oculus", "element": "hydro", "h": [100, 120], "s": [110, 255], "v": [130, 255],
         "min_area": 250, "max_area": 2500, "aspect": [0.5, 1.8], "min_fill": 0.15},
        {"class": "oculus", "element": "cryo", "h": [85, 100], "s": [60, 255], "v": [150, 255],
         "min_area": 250, "max_area": 2500, "aspect": [0.5, 1.8], "min_fill": 0.15},
        # сундук: открытый (без кольца) / заблокированный (красное кольцо) —
        # статус уточняется в classify_chest_status() по кольцу вокруг бокса
        {"class": "chest_unlocked", "h": [18, 32], "s": [120, 255], "v": [70, 235],
         "min_area": 500, "max_area": 22000, "aspect": [0.7, 2.6], "min_fill": 0.3},
        {"class": "puzzle", "h": [4, 16], "s": [130, 255], "v": [140, 245],
         "min_area": 350, "max_area": 8000, "aspect": [0.4, 2.2], "min_fill": 0.15},
    ],
    # --- враги без полосок HP: слизни (стихийные тела видны всегда) ---
    "enemy_rules": [
        {"class": "enemy", "enemy_type": "slime", "element": "pyro", "h": [0, 14],
         "s": [130, 255], "v": [130, 255], "min_area": 2500, "max_area": 25000,
         "aspect": [0.7, 1.7], "min_fill": 0.35},
        {"class": "enemy", "enemy_type": "slime", "element": "hydro", "h": [100, 120],
         "s": [130, 255], "v": [130, 255], "min_area": 2500, "max_area": 25000,
         "aspect": [0.7, 1.7], "min_fill": 0.35},
        {"class": "enemy", "enemy_type": "slime", "element": "electro", "h": [138, 162],
         "s": [130, 255], "v": [130, 255], "min_area": 2500, "max_area": 25000,
         "aspect": [0.7, 1.7], "min_fill": 0.35},
        {"class": "enemy", "enemy_type": "slime", "element": "anemo", "h": [35, 85],
         "s": [130, 255], "v": [130, 255], "min_area": 2500, "max_area": 25000,
         "aspect": [0.7, 1.7], "min_fill": 0.35},
        {"class": "enemy", "enemy_type": "slime", "element": "cryo", "h": [85, 100],
         "s": [60, 255], "v": [150, 255], "min_area": 2500, "max_area": 25000,
         "aspect": [0.7, 1.7], "min_fill": 0.35},
        {"class": "enemy", "enemy_type": "slime", "element": "geo", "h": [28, 40],
         "s": [130, 255], "v": [130, 255], "min_area": 2500, "max_area": 25000,
         "aspect": [0.7, 1.7], "min_fill": 0.35},
    ],
    # оценка расстояния до объекта: dist_m ≈ real_size_m * focal_px / высота_бокса
    "focal_px": 900.0,
    "object_real_size_m": {
        "chest_unlocked": 0.7, "chest_locked": 0.7,
        "teleport_active": 2.2, "teleport_inactive": 2.2,
        "oculus": 0.35, "puzzle": 1.6,
        "enemy": 1.8, "elite": 2.0, "boss": 4.0,
        "slime": 1.0, "hilichurl": 1.6, "samachurl": 1.5, "fungus": 0.8,
        "mitachurl": 2.6, "lawachurl": 3.2, "kairagi": 2.2, "nobushi": 1.9,
        "ruin_guard": 2.6, "ruin_machine": 2.5, "ruin_hunter": 2.4,
        "whopperflower": 1.6, "abyss_mage": 1.9, "specter": 0.8,
        "vishap": 2.6, "hypostasis": 3.0, "regisvine": 5.0,
    },
    # примерное положение игрока на экране (третье лицо) — для оценки угрозы
    "player_anchor": {"x": 0.5, "y": 0.6},
    # зона поиска иконки клавиши «F» (справа от персонажа, у нижнего края)
    "ui_interact_region": {"x": [0.42, 0.98], "y": [0.28, 0.98]},
}

# HSV ranges (OpenCV scale: H 0-179, S/V 0-255)
_GREEN = (np.array([35, 60, 60]), np.array([95, 255, 255]))
_YELLOW = (np.array([15, 70, 90]), np.array([40, 255, 255]))
_WHITE = (np.array([0, 0, 170]), np.array([179, 70, 255]))
_RED_HI = (np.array([170, 80, 70]), np.array([179, 255, 255]))
_RED_LO = (np.array([0, 80, 70]), np.array([10, 255, 255]))


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #
def load_calibration(path: str | Path | None) -> dict[str, Any]:
    calib = json.loads(json.dumps(DEFAULT_CALIBRATION))  # deep copy
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            user = json.load(fh)
        _deep_merge(calib, user)
    return calib


def _deep_merge(base: dict, override: dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def crop_rel(frame: np.ndarray, roi: dict[str, float], inset: float = 0.03) -> np.ndarray:
    """Crop a relative ROI from a frame. ``inset`` trims edges (bar frames, text)."""
    h, w = frame.shape[:2]
    x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
    dx, dy = rw * inset, rh * inset
    x1 = int(max(0, (x + dx) * w))
    y1 = int(max(0, (y + dy) * h))
    x2 = int(min(w, (x + rw - dx) * w))
    y2 = int(min(h, (y + rh - dy) * h))
    if x2 <= x1 or y2 <= y1:
        return frame[0:0, 0:0]
    return frame[y1:y2, x1:x2]


def rel_to_px(roi: dict[str, float], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = frame_size
    return (int(roi["x"] * w), int(roi["y"] * h), int(roi["w"] * w), int(roi["h"] * h))


# --------------------------------------------------------------------------- #
# Bar analysis
# --------------------------------------------------------------------------- #
def fill_fraction(roi: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Optional[float]:
    """Fraction of a horizontal bar filled with the given HSV colour.

    Bars fill left-to-right, so the metric is the right-most contiguous filled
    column (after light smoothing) over total width.
    """
    if roi.size == 0 or roi.shape[1] < 4:
        return None
    # подрезаем строки: рамки полоски по краям не должны попадать в маску
    trim = min(2, max(0, roi.shape[0] // 5))
    if roi.shape[0] - 2 * trim >= 2:
        roi = roi[trim:roi.shape[0] - trim, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cols = mask.mean(axis=0) / 255.0  # per-column coverage 0..1
    win = max(3, roi.shape[0] // 2)
    kernel = np.ones(win) / win
    smooth = np.convolve(cols, kernel, mode="same")
    filled = smooth > 0.35
    if not filled.any():
        return 0.0
    # Ignore stray hits: use the end of the first contiguous run starting at x=0
    end = 0
    for i, ok in enumerate(filled):
        if ok:
            end = i
        elif i > win and not filled[max(0, i - win):i].any():
            break
    # Полоска — равномерная заливка слева; текст/шум отбрасываем по неравномерности
    run = smooth[:end + 1]
    if len(run) < 2 or float(run.mean()) < 0.5 or float(run.std()) > 0.32:
        return None
    return float(min(1.0, (end + 1) / roi.shape[1]))


def green_fraction(roi: np.ndarray) -> Optional[float]:
    return fill_fraction(roi, *_GREEN)


def white_fraction(roi: np.ndarray) -> Optional[float]:
    return fill_fraction(roi, *_WHITE)


def yellow_fraction(roi: np.ndarray) -> Optional[float]:
    return fill_fraction(roi, *_YELLOW)


def red_fraction(roi: np.ndarray) -> Optional[float]:
    hi = fill_fraction(roi, *_RED_HI) or 0.0
    lo = fill_fraction(roi, *_RED_LO) or 0.0
    val = max(hi, lo)
    return val


def white_bar_fraction(roi: np.ndarray) -> Optional[float]:
    """White-fill fraction, falling back to 'any bright pixel' coverage."""
    frac = white_fraction(roi)
    if frac is not None and frac > 0:
        return frac
    return fill_fraction(roi, np.array([0, 0, 120]), np.array([179, 120, 255]))


# --------------------------------------------------------------------------- #
# Optional OCR for the "HP 4812/12000" text
# --------------------------------------------------------------------------- #
_HP_RE = re.compile(r"(\d{2,7})\s*/\s*(\d{2,7})")


def ocr_hp_text(roi: np.ndarray) -> Optional[tuple[int, int]]:
    """Best-effort OCR of ``cur/max`` HP text. Requires pytesseract + tesseract."""
    if roi.size == 0:
        return None
    try:
        import pytesseract  # type: ignore
    except Exception:
        return None
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(gray, config="--psm 7")
        match = _HP_RE.search(text.replace(" ", ""))
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
class HudReader:
    def __init__(self, calibration: dict[str, Any]):
        self.calib = calibration

    def read(self, frame: np.ndarray) -> HudReading:
        out = HudReading()

        # --- player HP -----------------------------------------------------
        hp_roi = crop_rel(frame, self.calib["player_hp_bar"])
        frac = green_fraction(hp_roi)
        if frac is None:
            frac = white_bar_fraction(hp_roi)
        out.player_hp_frac = frac

        # --- HP text (optional absolute values) ---------------------------
        text_roi = crop_rel(frame, self.calib.get("player_hp_text", self.calib["player_hp_bar"]))
        parsed = ocr_hp_text(text_roi)
        if parsed is not None:
            out.player_hp_cur, out.player_hp_max = parsed
        else:
            hp_max = self.calib.get("hp_max")
            if frac is not None and hp_max:
                out.player_hp_max = int(hp_max)
                out.player_hp_cur = int(round(frac * hp_max))

        # --- stamina ------------------------------------------------------
        out.stamina_frac = yellow_fraction(crop_rel(frame, self.calib["stamina_bar"]))

        # --- boss bar -----------------------------------------------------
        boss = self.calib.get("boss_hp_bar")
        if boss:
            out.boss_hp_frac = white_bar_fraction(crop_rel(frame, boss))

        # --- party bars ---------------------------------------------------
        for roi in self.calib.get("party_bars", []):
            out.party_hp_fracs.append(green_fraction(crop_rel(frame, roi)))

        # --- burst icons --------------------------------------------------
        luma_thr = float(self.calib.get("burst_ready_luma", 140.0))
        for roi in self.calib.get("burst_icons", []):
            icon = crop_rel(frame, roi, inset=0.1)
            if icon.size == 0:
                out.burst_ready.append(False)
                continue
            luma = float(cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY).mean())
            out.burst_ready.append(luma >= luma_thr)

        return out
