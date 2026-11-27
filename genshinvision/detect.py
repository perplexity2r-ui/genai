"""Detector backends.

``HeuristicDetector`` — zero-training colour/contour detector. Understands the
UI bars well enough to drive the tracker and produce a working live state.

``YoloDetector`` — real object detection with an Ultralytics YOLOv8 model. Train
your own on recorded gameplay (see ``training/``) and pass ``--weights best.pt``.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import cv2
import numpy as np

from .hud import crop_rel, red_fraction, rel_to_px, white_bar_fraction
from .types import Detection

CLASS_NAMES: list[str] = [
    "player", "enemy", "elite", "boss",
    "enemy_hp_bar", "player_hp_bar", "stamina_bar",
    "damage_number", "pickup", "chest", "dialogue", "minimap",
    "teleport_waypoint", "statue_of_seven", "oculus", "puzzle",
]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# Классы, которые попадают в state["interactables"] (объекты мира)
INTERACTABLE_CLASSES = (
    "chest", "teleport_waypoint", "statue_of_seven", "oculus", "puzzle", "pickup",
)


def _det(cls_name: str, bbox, score: float, **attrs) -> Detection:
    return Detection(
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        score=float(score),
        cls_id=CLASS_IDS.get(cls_name, -1),
        cls_name=cls_name,
        attrs=attrs,
    )


def attach_hp_to_entities(dets: list[Detection]) -> None:
    """Copy ``hp_frac`` from ``*_hp_bar`` detections onto nearby entity boxes."""
    bars = [d for d in dets if d.cls_name.endswith("hp_bar")]
    for det in dets:
        if det.cls_name not in ("enemy", "elite", "boss", "player"):
            continue
        best: Optional[Detection] = None
        best_dx = float("inf")
        for bar in bars:
            inside_x = det.bbox[0] - 8 <= bar.cx <= det.bbox[2] + 8
            above = det.bbox[1] - 140 <= bar.cy <= det.bbox[1] + 20
            dx = abs(bar.cx - det.cx)
            if inside_x and above and dx < best_dx:
                best, best_dx = bar, dx
        if best is not None and "hp_frac" in best.attrs:
            det.attrs.setdefault("hp_frac", best.attrs["hp_frac"])


def detect_object_rules(
    frame: np.ndarray,
    rules: Sequence[dict[str, Any]],
    exclude_rects: Sequence[tuple[int, int, int, int]] = (),
) -> list[Detection]:
    """Rule-driven поиск объектов мира: сундуки, телепорты, кукулиусы, головоломки.

    Каждое правило — HSV-диапазон + фильтры формы (area/aspect/fill).
    ``exclude_rects`` — зоны, где объектов быть не может (HUD, тела врагов).
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dets: list[Detection] = []
    for rule in rules:
        lower = np.array([rule["h"][0], rule["s"][0], rule["v"][0]], dtype=np.uint8)
        upper = np.array([rule["h"][1], rule["s"][1], rule["v"][1]], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        for x1, y1, x2, y2 in exclude_rects:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c > x1c and y2c > y1c:
                mask[y1c:y2c, x1c:x2c] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lo_a, hi_a = rule.get("min_area", 50), rule.get("max_area", 1e9)
        a_lo, a_hi = rule.get("aspect", [0.2, 6.0])
        min_fill = rule.get("min_fill", 0.15)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (lo_a <= area <= hi_a):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < 4 or bw < 4:
                continue
            aspect = bw / max(bh, 1)
            if not (a_lo <= aspect <= a_hi):
                continue
            if area / float(bw * bh) < min_fill:
                continue
            cls_name = rule["class"]
            attrs: dict[str, Any] = {}
            if "element" in rule:
                attrs["element"] = rule["element"]
            dets.append(_det(cls_name, (x, y, x + bw, y + bh), 0.6, **attrs))
    # подавление дублей между правилами: побеждает первый
    kept: list[Detection] = []
    for det in dets:
        dup = False
        dcx, dcy = det.cx, det.cy
        for prev in kept:
            ix = max(0.0, min(det.bbox[2], prev.bbox[2]) - max(det.bbox[0], prev.bbox[0]))
            iy = max(0.0, min(det.bbox[3], prev.bbox[3]) - max(det.bbox[1], prev.bbox[1]))
            inter = ix * iy
            a1 = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
            a2 = (prev.bbox[2] - prev.bbox[0]) * (prev.bbox[3] - prev.bbox[1])
            inside = (prev.bbox[0] <= dcx <= prev.bbox[2] and prev.bbox[1] <= dcy <= prev.bbox[3])
            if inter / max(a1 + a2 - inter, 1e-6) > 0.5 or inside:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


class HeuristicDetector:
    """Colour + morphology detector tuned to the Genshin HUD.

    Scans the whole frame for white HP bars but masks out known HUD regions
    (minimap, player bars, boss bar...) so UI elements don't pollute results.
    """

    name = "heuristic"

    def __init__(self, calibration: dict[str, Any]):
        self.calib = calibration

    def _hud_rects(self, w: int, h: int) -> list[tuple[int, int, int, int]]:
        """Pixel rects of HUD elements to exclude (with a small margin)."""
        rects = []
        keys = ["player_hp_bar", "player_hp_text", "stamina_bar", "boss_hp_bar"]
        keys += [f"party_bars.{i}" for i in range(len(self.calib.get("party_bars", [])))]
        keys += [f"burst_icons.{i}" for i in range(len(self.calib.get("burst_icons", [])))]
        for key in keys:
            if "." in key:
                base, idx = key.split(".")
                roi = self.calib.get(base, [])[int(idx)]
            else:
                roi = self.calib.get(key)
            if not roi:
                continue
            x, y, rw, rh = rel_to_px(roi, (w, h))
            pad = 6
            extra_top = 30 if key == "boss_hp_bar" else 0  # имя босса над полоской
            rects.append((x - pad, y - pad - extra_top, x + rw + pad, y + rh + pad))
        mm = self.calib.get("minimap")
        if mm:
            cx, cy, r = mm["cx"] * w, mm["cy"] * h, mm["r"] * w
            rects.append((int(cx - r - 6), int(cy - r - 6), int(cx + r + 6), int(cy + r + 6)))
        return rects

    def detect(self, frame: np.ndarray, hud=None, track_zones=None) -> list[Detection]:
        dets: list[Detection] = []
        h, w = frame.shape[:2]
        hud_rects = self._hud_rects(w, h)

        # --- enemy HP bars: near-white thin horizontal bars ---------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([179, 70, 255]))
        for x1, y1, x2, y2 in hud_rects:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c > x1c and y2c > y1c:
                white[y1c:y2c, x1c:x2c] = 0

        k_w = max(6, w // 240)  # не сливать соседние полоски HP разных врагов
        merged = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, k_w), np.uint8))
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        enemy_rects: list[tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if not (4 <= bh <= 30):
                continue
            if bw < 30 or bw / max(bh, 1) < 3.2:
                continue
            if bw > 0.16 * w:
                continue  # слишком широкая полоска = слипшиеся две; пропускаем
            area_fill = cv2.contourArea(cnt) / float(bw * bh)
            if area_fill < 0.50:
                continue
            bar_roi = frame[y:y + bh, x:x + bw]
            hp_frac = white_bar_fraction(bar_roi)
            if hp_frac is None:
                continue  # не похоже на полоску (текст/шум)
            bar_bbox = (x, y, x + bw, y + bh)
            dets.append(_det("enemy_hp_bar", bar_bbox, 0.62, hp_frac=hp_frac))
            # Body proxy below the bar (bar sits above the head).
            body_w = min(bw * 1.5, 0.22 * w)   # ограничим прокси на случай слияния
            body_h = min(bw * 2.2, 0.5 * h)
            cx = x + bw / 2.0
            top = (y + bh) + bw * 0.55
            body = (
                max(0.0, cx - body_w / 2.0), max(0.0, top),
                min(float(w), cx + body_w / 2.0), min(float(h), top + body_h),
            )
            dets.append(_det("enemy", body, 0.55, hp_frac=hp_frac))
            enemy_rects.append((int(body[0]) - 4, int(body[1]) - 4, int(body[2]) + 4, int(body[3]) + 4))
            enemy_rects.append((x - 4, y - 4, x + bw + 4, y + bh + 4))

        # --- boss bar ----------------------------------------------------
        boss_roi = self.calib.get("boss_hp_bar")
        if boss_roi:
            roi = crop_rel(frame, boss_roi)
            frac = white_bar_fraction(roi)
            if frac is None or frac <= 0.02:
                frac = red_fraction(roi)
            if frac is not None and frac > 0.02:
                x, y, bw, bh = (
                    int(boss_roi["x"] * w), int(boss_roi["y"] * h),
                    int(boss_roi["w"] * w), int(boss_roi["h"] * h),
                )
                dets.append(_det("boss", (x, y, x + bw, y + bh), 0.6, hp_frac=frac))
                enemy_rects.append((x - 34, y - 34, x + bw + 34, y + bh + 34))

        # --- объекты мира: сундуки, телепорты, кукулиусы, головоломки ------
        rules = self.calib.get("object_rules") or []
        if rules:
            zones = list(enemy_rects)
            # зоны живых треков врагов (с прошлого кадра): тело врага не должно
            # выглядеть как объект, даже если его полоска HP потерялась
            for zb in (track_zones or []):
                pad = int(0.15 * (zb[3] - zb[1])) + 10
                zones.append((int(zb[0]) - pad, int(zb[1]) - pad, int(zb[2]) + pad, int(zb[3]) + pad))
            dets.extend(
                detect_object_rules(frame, rules, exclude_rects=hud_rects + zones)
            )

        attach_hp_to_entities(dets)
        return dets


class YoloDetector:
    """Ultralytics YOLOv8 backend (requires ``pip install ultralytics``)."""

    name = "yolo"

    def __init__(
        self,
        weights: str,
        calibration: dict[str, Any],
        conf: float = 0.30,
        imgsz: int = 960,
        device: Optional[str] = None,
        half: bool = True,
    ):
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YOLO backend требует пакет ultralytics: pip install -r requirements-ml.txt"
            ) from exc
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.calib = calibration
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.names: dict[int, str] = dict(self.model.names)

    def detect(self, frame: np.ndarray, hud=None, track_zones=None) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        res = results[0]
        dets: list[Detection] = []
        if res.boxes is None:
            return dets
        for box in res.boxes:
            cls_id = int(box.cls.item())
            name = self.names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "object")
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            score = float(box.conf.item())
            det = _det(name, (x1, y1, x2, y2), score)
            if name == "enemy_hp_bar":
                from .hud import white_bar_fraction as wbf

                pad = 2
                roi = frame[int(y1) + pad:int(y2) - pad, int(x1) + pad:int(x2) - pad]
                det.attrs["hp_frac"] = wbf(roi)
            dets.append(det)
        attach_hp_to_entities(dets)
        return dets


def make_detector(backend: str, calibration: dict[str, Any], weights: Optional[str] = None, **kwargs):
    if backend == "heuristic":
        return HeuristicDetector(calibration)
    if backend == "yolo":
        if not weights:
            raise ValueError("YOLO backend: укажите --weights path/to/best.pt")
        return YoloDetector(weights, calibration, **kwargs)
    raise ValueError(f"Неизвестный backend: {backend}")
