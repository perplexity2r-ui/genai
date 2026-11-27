"""Понимание миникарты Genshin Impact.

Извлекает круглую миникарту и вытаскивает из неё:
- **курс игрока** (угол белой стрелки в центре, 0° = север, по часовой);
- **иконки** на карте: телепорты, статуи семи архонтов, домены, квесты, группа —
  правила цвета задаются в калибровке (``minimap_icons``);
- **движение игрока**: сдвиг текстуры карты между кадрами (phase correlation)
  — вектор скорости в мировых пикселях карты.

В Genshin стрелка игрока всегда в центре миникарты (карта едет под ней,
север — вверх), поэтому мировые координаты из одной миникарты не получить —
но курс, объекты рядом и направление движения получить можно.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


@dataclass
class MinimapIcon:
    type: str                      # teleport / statue / domain / quest / party
    x: float                       # -1..1 внутри круга миникарты
    y: float
    area: float

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "x": round(self.x, 3), "y": round(self.y, 3)}


@dataclass
class MinimapState:
    heading_deg: Optional[float] = None      # 0 = север, по часовой
    icons: list[MinimapIcon] = field(default_factory=list)
    motion_px_s: tuple[float, float] = (0.0, 0.0)  # куда движется игрок (карта, px/с)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading_deg": None if self.heading_deg is None else round(self.heading_deg, 1),
            "icons": [i.to_dict() for i in self.icons],
            "motion_px_s": [round(self.motion_px_s[0], 1), round(self.motion_px_s[1], 1)],
            "speed_px_s": round(
                float((self.motion_px_s[0] ** 2 + self.motion_px_s[1] ** 2) ** 0.5), 1
            ),
            "confidence": round(self.confidence, 2),
        }


def _hsv_mask(hsv: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    lower = np.array([rule["h"][0], rule["s"][0], rule["v"][0]], dtype=np.uint8)
    upper = np.array([rule["h"][1], rule["s"][1], rule["v"][1]], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


class MinimapReader:
    def __init__(self, calibration: dict[str, Any]):
        self.calib = calibration
        self._prev_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ util
    def _circle(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Возвращает (BGR-кроп круга, (cx, cy, r)) в пикселях кадра."""
        mm = self.calib["minimap"]
        h, w = frame.shape[:2]
        cx, cy, r = mm["cx"] * w, mm["cy"] * h, mm["r"] * w
        x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
        x2, y2 = min(w, int(cx + r)), min(h, int(cy + r))
        crop = frame[y1:y2, x1:x2].copy()
        local = (int(cx - x1), int(cy - y1), int(r))
        return crop, local

    @staticmethod
    def _circle_mask(shape, local) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.circle(mask, (local[0], local[1]), local[2], 255, -1)
        return mask

    # ------------------------------------------------------------------ read
    def read(self, frame: np.ndarray, dt: float = 1.0 / 30.0) -> MinimapState:
        crop, local = self._circle(frame)
        if crop.size == 0:
            return MinimapState()
        r = local[2]
        out = MinimapState()

        # --- курс игрока: белая стрелка в центре ------------------------
        out.heading_deg = self._heading(crop, local)

        # --- иконки ------------------------------------------------------
        out.icons = self._icons(crop, local)

        # --- движение карты (фазовая корреляция) ------------------------
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # мягкое кольцевое окно: без резких краёв (они дают ложный пик на нуле
        # сдвига) и без белой рамки/стрелки по центру
        h_img, w_img = gray.shape
        yy, xx = np.ogrid[:h_img, :w_img]
        dist = np.sqrt((xx - local[0]) ** 2 + (yy - local[1]) ** 2).astype(np.float32)
        outer = np.clip((r * 0.97 - dist) / max(r * 0.22, 1.0), 0.0, 1.0)
        inner = np.clip((dist - r * 0.28) / max(r * 0.14, 1.0), 0.0, 1.0)
        window_soft = (outer * inner).astype(np.float32)
        gray_masked = (gray * window_soft).astype(np.float32)
        if self._prev_gray is not None and self._prev_gray.shape == gray_masked.shape:
            shift, response = cv2.phaseCorrelate(self._prev_gray, gray_masked)
            if response > 0.02:
                # карта едет в противоположную сторону от движения игрока
                out.motion_px_s = (-shift[0] / max(dt, 1e-4), -shift[1] / max(dt, 1e-4))
                out.confidence = float(min(1.0, response * 10))
        self._prev_gray = gray_masked

        return out

    # ------------------------------------------------------------- internals
    def _heading(self, crop: np.ndarray, local) -> Optional[float]:
        cx, cy, r = local
        zone = int(r * 0.42)
        y1, y2 = max(0, cy - zone), min(crop.shape[0], cy + zone)
        x1, x2 = max(0, cx - zone), min(crop.shape[1], cx + zone)
        roi = crop[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([179, 60, 255]))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 8:
            return None
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            return None
        ccx, ccy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # острие стрелки — самая дальняя от центра тяжести точка контура
        pts = cnt.reshape(-1, 2).astype(np.float64)
        d = (pts[:, 0] - ccx) ** 2 + (pts[:, 1] - ccy) ** 2
        tip = pts[int(np.argmax(d))]
        dx, dy = tip[0] - ccx, tip[1] - ccy
        if dx * dx + dy * dy < 4:
            return None
        heading = float(np.degrees(np.arctan2(dx, -dy))) % 360.0
        return heading

    def _icons(self, crop: np.ndarray, local) -> list[MinimapIcon]:
        cx, cy, r = local
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        circle = self._circle_mask(crop.shape, local)
        # зона стрелки игрока в центре — не иконка
        cv2.circle(circle, (cx, cy), int(r * 0.30), 0, -1)

        icons: list[MinimapIcon] = []
        for name, rule in (self.calib.get("minimap_icons") or {}).items():
            mask = _hsv_mask(hsv, rule)
            mask = cv2.bitwise_and(mask, mask, mask=circle)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (rule.get("min_area", 15) <= area <= rule.get("max_area", 600)):
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = bw / max(bh, 1)
                if not (rule.get("aspect", [0.3, 3.0])[0] <= aspect <= rule.get("aspect", [0.3, 3.0])[1]):
                    continue
                m = cv2.moments(cnt)
                if m["m00"] == 0:
                    continue
                ix = (m["m10"] / m["m00"] - cx) / max(r, 1)
                iy = (m["m01"] / m["m00"] - cy) / max(r, 1)
                icons.append(MinimapIcon(type=name, x=float(ix), y=float(iy), area=float(area)))
        return icons
