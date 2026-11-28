"""Момент взаимодействия (клавиша F) и статусы сундуков/телепортов."""
import numpy as np

from genshinvision.detect import (
    HeuristicDetector,
    classify_chest_status,
    detect_ui_interact_f,
)
from genshinvision.hud import DEFAULT_CALIBRATION
from genshinvision.synthetic import SyntheticGameSource, hsv_color


def _blank(w=1280, h=720):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (46, 52, 40)  # тёмный «грунт»
    return img


def _button(img, x=655, y=645, w=50, h=50, letter="F"):
    cv2 = np.__dict__ or None  # noqa — просто импорт ниже
    import cv2 as _cv2
    _cv2.rectangle(img, (x, y), (x + w, y + h), (240, 242, 245), -1)
    _cv2.putText(img, letter, (x + 13, y + int(0.78 * h)),
                 _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 25), 2)
    _cv2.putText(img, "Сундук", (x + w + 8, y + int(0.62 * h)),
                 _cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 240, 245), 2)
    return (x, y, x + w, y + h)


def test_ui_interact_f_detected():
    img = _blank()
    truth = _button(img)
    dets = detect_ui_interact_f(img, DEFAULT_CALIBRATION["ui_interact_region"])
    assert len(dets) == 1, f"ожидалась 1 кнопка F, найдено {len(dets)}"
    d = dets[0]
    assert d.cls_name == "ui_interact_f"
    # центр близок к истинной кнопке (bbox может включать текст справа)
    cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
    tcx, tcy = (truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2
    assert abs(cx - tcx) < 80 and abs(cy - tcy) < 30
    assert d.score > 0.5


def test_ui_other_letters_rejected():
    """Кнопка с другой буквой — это НЕ момент взаимодействия."""
    for letter in ("H", "E", "X", "?"):
        img = _blank()
        _button(img, letter=letter)
        dets = detect_ui_interact_f(img, DEFAULT_CALIBRATION["ui_interact_region"])
        assert dets == [], f"буква {letter!r} дала ложное срабатывание"


def test_ui_absent_no_detection():
    img = _blank()
    assert detect_ui_interact_f(img, DEFAULT_CALIBRATION["ui_interact_region"]) == []


def test_chest_status_by_red_ring():
    """Красное кольцо вокруг сундука — chest_locked, без кольца — chest_unlocked."""
    import cv2
    # сундук без кольца
    img = _blank()
    cv2.rectangle(img, (420, 560), (530, 640), hsv_color(24, 210, 230), -1)
    assert classify_chest_status(img, (420, 560, 530, 640)) == "chest_unlocked"
    # сундук в красном кольце
    cv2.circle(img, (300, 615), 68, hsv_color(0, 205, 235), 12)
    cv2.rectangle(img, (245, 575), (355, 655), hsv_color(24, 210, 230), -1)
    assert classify_chest_status(img, (245, 575, 355, 655)) == "chest_locked"


def test_synthetic_scene_statuses_end_to_end():
    """Сцена целиком: закрытый/открытый сундук, активный/неактивный телепорт, F."""
    det = HeuristicDetector(DEFAULT_CALIBRATION)
    src = SyntheticGameSource(seed=7, fps=30.0, n_enemies=2, max_frames=6)
    for _ in range(6):
        ts, frame = src.read()
    dets = det.detect(frame)
    names = sorted(d.cls_name for d in dets)
    for cls in ("chest_unlocked", "chest_locked", "teleport_active",
                "teleport_inactive", "ui_interact_f"):
        assert cls in names, f"{cls} не найден: {names}"
