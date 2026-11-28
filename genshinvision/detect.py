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
    "damage_number", "pickup",
    "chest_unlocked", "chest_locked", "dialogue", "minimap",
    "teleport_active", "teleport_inactive", "oculus", "puzzle",
    "ui_interact_f",
]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# Классы, которые попадают в state["interactables"] (объекты мира)
INTERACTABLE_CLASSES = (
    "chest_unlocked", "chest_locked",
    "teleport_active", "teleport_inactive",
    "oculus", "puzzle", "pickup",
)

# Имена старых моделей -> новые статусные классы
LEGACY_CLASS_MAP = {
    "chest": "chest_unlocked",
    "teleport_waypoint": "teleport_active",
    "statue_of_seven": "teleport_active",  # статуи: та же логика active/inactive
}

# --------------------------------------------------------------------------- #
# Таксономия врагов: тонкий тип -> грубый уровень (tier).
# YOLO может учиться на тонких типах (slime, hilichurl, ruin_guard, ...), а
# конвейер автоматически приводит их к уровням enemy / elite / boss.
# --------------------------------------------------------------------------- #
ENEMY_TAXONOMY: dict[str, str] = {
    # — обычные (fodder) —
    "slime": "enemy", "hilichurl": "enemy", "samachurl": "enemy",
    "shaman": "enemy", "fungus": "enemy", "treasure_hoarder": "enemy",
    "fatui_skirmisher": "enemy", "nobushi": "enemy", "hilichurl_rogue": "enemy",
    # — элитные —
    "mitachurl": "elite", "lawachurl": "elite", "kairagi": "elite",
    "ruin_guard": "elite", "ruin_cruiser": "elite", "ruin_hunter": "elite",
    "ruin_drake": "elite", "ruin_grader": "elite", "ruin_machine": "elite",
    "whopperflower": "elite", "abyss_mage": "elite", "abyss_herald": "elite",
    "fatui_agent": "elite", "fatui_mage": "elite", "fatui_cicin": "elite",
    "specter": "elite", "vishap": "elite", "bathysmal_vishap": "elite",
    "eonblight": "elite", "breacher_primus": "elite",
    # — боссы —
    "hypostasis": "boss", "regisvine": "boss", "overworld_boss": "boss",
    "weekly_boss": "boss", "world_boss": "boss",
}

ENEMY_TIERS = ("enemy", "elite", "boss")


def tier_of(name: str) -> str:
    """Грубый уровень врага по тонкому типу (или уже готовому уровню)."""
    if name in ENEMY_TIERS:
        return name
    return ENEMY_TAXONOMY.get(name, "enemy")


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
            if "enemy_type" in rule:
                attrs["enemy_type"] = rule["enemy_type"]
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


def classify_chest_status(frame: np.ndarray, bbox) -> str:
    """Сундук с красным кольцом/куполом вокруг — «заблокирован».

    Смотрим на кольцо вокруг бокса: доля красных пикселей (тон ~0..12 или
    ~168..179, насыщенность высокая) выше порога — сундук ``chest_locked``.
    Без кольца — ``chest_unlocked`` («подойди и нажми F»).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 2 or bh <= 2:
        return "chest_unlocked"
    # внешняя граница кольца: расширяем бокс
    ox1, oy1 = max(0, x1 - int(0.35 * bw)), max(0, y1 - int(0.45 * bh))
    ox2, oy2 = min(w, x2 + int(0.35 * bw)), min(h, y2 + int(0.45 * bh))
    # внутренняя граница: сам сундук с небольшим запасом
    ix1, iy1 = max(0, x1 - int(0.08 * bw)), max(0, y1 - int(0.08 * bh))
    ix2, iy2 = min(w, x2 + int(0.08 * bw)), min(h, y2 + int(0.08 * bh))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ring = cv2.inRange(hsv, np.array([0, 110, 80]), np.array([12, 255, 255]))
    ring2 = cv2.inRange(hsv, np.array([168, 110, 80]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(ring, ring2)
    mask = np.zeros_like(red)
    mask[oy1:oy2, ox1:ox2] = 255
    mask[iy1:iy2, ix1:ix2] = 0
    total = int(cv2.countNonZero(mask))
    if total < 50:
        return "chest_unlocked"
    frac = cv2.countNonZero(cv2.bitwise_and(red, mask)) / float(total)
    return "chest_locked" if frac > 0.06 else "chest_unlocked"


# --- шаблон клавиши «F» для сверки глифа ------------------------------- #
_F_TEMPLATE: Optional[np.ndarray] = None


def _f_template() -> np.ndarray:
    global _F_TEMPLATE
    if _F_TEMPLATE is None:
        t = np.zeros((44, 34), np.uint8)
        cv2.putText(t, "F", (3, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.15, 255, 2)
        _F_TEMPLATE = t
    return _F_TEMPLATE


def detect_ui_interact_f(
    frame: np.ndarray,
    region: Optional[dict[str, Any]] = None,
    exclude_rects: Sequence[tuple[int, int, int, int]] = (),
) -> list[Detection]:
    """Иконка клавиши взаимодействия «F» с текстом-подсказкой («Обычный сундук»...).

    Появление этого класса — сигнал боту: *остановиться и нажать F*. Ищем
    светлую компактную кнопку с тёмным глифом внутри и сверяем глиф с
    шаблоном «F», чтобы не реагировать на другие буквы/кнопки UI.
    """
    h, w = frame.shape[:2]
    region = region or {"x": [0.42, 0.98], "y": [0.28, 0.98]}
    xa, xb = region["x"]
    ya, yb = region["y"]
    rx1, ry1 = int(xa * w), int(ya * h)
    rx2, ry2 = int(xb * w), int(yb * h)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    button = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 75, 255]))
    for x1, y1, x2, y2 in exclude_rects:
        x1c, y1c = max(0, x1 - rx1), max(0, y1 - ry1)
        x2c, y2c = min(roi.shape[1], x2 - rx1), min(roi.shape[0], y2 - ry1)
        if x2c > x1c and y2c > y1c:
            button[y1c:y2c, x1c:x2c] = 0
    button = cv2.morphologyEx(button, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(button, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmpl = _f_template()
    dets: list[Detection] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if not (250 <= area <= 6000) or not (18 <= bh <= 90):
            continue
        aspect = bw / max(bh, 1)
        if not (0.55 <= aspect <= 1.8):
            continue
        if area / float(bw * bh) < 0.35:
            continue
        # внутри кнопки — тёмная буква
        gx1, gy1 = x + int(0.15 * bw), y + int(0.15 * bh)
        gx2, gy2 = x + int(0.85 * bw), y + int(0.85 * bh)
        glyph_roi = roi[gy1:gy2, gx1:gx2]
        if glyph_roi.size == 0:
            continue
        gray = cv2.cvtColor(glyph_roi, cv2.COLOR_BGR2GRAY)
        dark = (gray < 100).astype(np.uint8)
        ratio = float(dark.mean())
        if not (0.05 <= ratio <= 0.55):
            continue
        glyph = cv2.resize(dark * 255, (tmpl.shape[1], tmpl.shape[0]))
        score = float(cv2.matchTemplate(glyph, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score < 0.30:
            continue
        # у «F» нет нижней перекладины (в отличие от «E»): низ глифа — только стойка
        bottom = dark[int(dark.shape[0] * 0.70):, :]
        cols = np.where(bottom.any(axis=0))[0]
        if cols.size and (cols.max() - cols.min() + 1) > 0.45 * dark.shape[1]:
            continue
        # расширяем bbox вправо на текст-подсказку («Обычный сундук», «Подобрать»...)
        tx2 = x + bw
        y_lo, y_hi = y, y + bh
        while tx2 - (x + bw) < 0.22 * w:
            col = hsv[y_lo:y_hi, tx2 + 1:tx2 + 2]
            if col.size == 0:
                break
            bright = ((col[:, :, 2] > 140) & (col[:, :, 1] < 110)).mean()
            if bright < 0.02:
                break
            tx2 += 1
        bx = (float(rx1 + x), float(ry1 + y), float(rx1 + tx2), float(ry1 + y + bh))
        dets.append(_det("ui_interact_f", bx, round(0.5 + 0.5 * score, 3)))
    return dets


def postprocess_world_status(frame: np.ndarray, dets: list[Detection]) -> None:
    """Статусы сундуков (открыт/закрыт) и подавление ложных телепортов.

    Красный купол/кольцо вокруг сундука не должен находиться как
    ``teleport_inactive`` — он сам по себе признак ``chest_locked``.
    """
    chests = [d for d in dets if d.cls_name in ("chest_unlocked", "chest_locked")]
    for det in list(dets):
        if det.cls_name in ("chest_unlocked", "chest_locked"):
            status = classify_chest_status(frame, det.bbox)
            det.cls_name = status
            det.cls_id = CLASS_IDS.get(status, det.cls_id)
        elif det.cls_name == "teleport_inactive":
            for ch in chests:
                inside = (
                    ch.bbox[0] - 4 <= det.cx <= ch.bbox[2] + 4
                    and ch.bbox[1] - 4 <= det.cy <= ch.bbox[3] + 4
                ) or (
                    det.bbox[0] <= ch.cx <= det.bbox[2]
                    and det.bbox[1] <= ch.cy <= det.bbox[3]
                )
                if inside:
                    dets.remove(det)
                    # сундук в красном куполе — заблокирован
                    ch.cls_name = "chest_locked"
                    ch.cls_id = CLASS_IDS.get("chest_locked", ch.cls_id)
                    break


def _bar_left_runs(roi: np.ndarray) -> list[int]:
    """Длины горизонтальных белых серединной строки (полоска — сплошной сегмент)."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([179, 70, 255]))
    if white.size == 0:
        return []
    mid = white[white.shape[0] // 2]
    runs: list[int] = []
    run = 0
    for px in mid:
        if px:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return runs


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

    def detect(self, frame: np.ndarray, hud=None, track_zones=None, bar_enemy_zones=None) -> list[Detection]:
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
            if bw > 0.11 * w or bh > 0.028 * h:
                continue  # слишком широкая/высокая полоска = слипшиеся две; пропускаем
            area_fill = cv2.contourArea(cnt) / float(bw * bh)
            if area_fill < 0.50:
                continue
            bar_roi = frame[y:y + bh, x:x + bw]
            # текст («Обычный сундук», цифры урона) даёт много мелких белых
            # серединной строки — полоска HP это один сплошной сегмент
            runs = _bar_left_runs(bar_roi)
            if not runs or max(runs) < 6 or len(runs) > 3:
                continue
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

        # --- враги без полосок HP: слизни и другие стихийные/крупные цели ---
        # правила enemy_rules ищут цветные тела (например, слизни), которые
        # видны даже без HUD; тела врагов с полосками уже найдены выше
        enemy_rule_dets: list[Detection] = []
        rules_enemy = self.calib.get("enemy_rules") or []
        if rules_enemy:
            zones = list(enemy_rects)
            # зоны врагов с полосками HP (из треков): их тела не должны
            # находиться заново как «слизень»; треки самих слизней сюда не
            # попадают, иначе они загасят самих себя
            for zb in (bar_enemy_zones or []):
                pad = int(0.15 * (zb[3] - zb[1])) + 10
                zones.append((int(zb[0]) - pad, int(zb[1]) - pad, int(zb[2]) + pad, int(zb[3]) + pad))
            enemy_rule_dets = detect_object_rules(frame, rules_enemy, exclude_rects=hud_rects + zones)
            dets.extend(enemy_rule_dets)

        # --- объекты мира: сундуки, телепорты, кукулиусы, головоломки ------
        rules = self.calib.get("object_rules") or []
        if rules:
            zones = list(enemy_rects)
            # зоны живых треков врагов (с прошлого кадра): тело врага не должно
            # выглядеть как объект, даже если его полоска HP потерялась
            for zb in (track_zones or []):
                pad = int(0.15 * (zb[3] - zb[1])) + 10
                zones.append((int(zb[0]) - pad, int(zb[1]) - pad, int(zb[2]) + pad, int(zb[3]) + pad))
            # тела врагов без полосок (слизни) — тоже не объекты
            for ed in enemy_rule_dets:
                x1, y1, x2, y2 = (int(v) for v in ed.bbox)
                zones.append((x1 - 6, y1 - 6, x2 + 6, y2 + 6))
            dets.extend(
                detect_object_rules(frame, rules, exclude_rects=hud_rects + zones)
            )

        # --- статусы объектов и кнопка взаимодействия «F» ----------------
        postprocess_world_status(frame, dets)
        hud_rects_ui = hud_rects + [tuple(int(v) for v in d.bbox) for d in dets
                                    if d.cls_name in ("enemy", "elite", "boss")]
        dets.extend(detect_ui_interact_f(frame, self.calib.get("ui_interact_region"),
                                         exclude_rects=hud_rects_ui))

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

    def detect(self, frame: np.ndarray, hud=None, track_zones=None, bar_enemy_zones=None) -> list[Detection]:
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
            name = LEGACY_CLASS_MAP.get(name, name)  # старые модели -> новые классы
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            score = float(box.conf.item())
            attrs: dict[str, Any] = {}
            cls_name = name
            # тонкие типы врагов (slime, hilichurl, ruin_guard, ...) -> tier
            if name in ENEMY_TAXONOMY:
                attrs["enemy_type"] = name
                cls_name = tier_of(name)
            elif name in ENEMY_TIERS:
                cls_name = name
            det = _det(cls_name, (x1, y1, x2, y2), score, **attrs)
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
