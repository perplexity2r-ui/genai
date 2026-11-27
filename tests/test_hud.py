"""Тесты чтения HUD: доли заполнения полосок."""
import numpy as np

from genshinvision.hud import (
    DEFAULT_CALIBRATION,
    HudReader,
    green_fraction,
    white_bar_fraction,
    yellow_fraction,
)
from genshinvision.synthetic import SyntheticGameSource


def bar_image(frac, fill_bgr, w=400, h=20, bg=(42, 42, 48)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = bg
    img[:, : int(w * frac)] = fill_bgr
    return img


def test_green_fraction():
    img = bar_image(0.60, (110, 215, 105))
    assert abs(green_fraction(img) - 0.60) < 0.06


def test_white_fraction():
    img = bar_image(0.35, (238, 238, 238))
    assert abs(white_bar_fraction(img) - 0.35) < 0.06


def test_yellow_fraction():
    img = bar_image(0.80, (80, 215, 235))
    assert abs(yellow_fraction(img) - 0.80) < 0.08


def test_empty_bar_is_zero():
    img = bar_image(0.0, (238, 238, 238))
    assert (white_bar_fraction(img) or 0.0) == 0.0


def test_hud_reader_on_synthetic_frame():
    """HudReader на синтетическом кадре совпадает с ground truth."""
    src = SyntheticGameSource(seed=3, max_frames=25)
    reader = HudReader(DEFAULT_CALIBRATION)
    for _ in range(25):
        ts, frame = src.read()
    truth = src.last_truth
    reading = reader.read(frame)

    assert reading.player_hp_frac is not None
    assert abs(reading.player_hp_frac - truth["player_hp_frac"]) < 0.05
    assert reading.player_hp_max == src.player_hp_max
    assert abs(reading.player_hp_cur - truth["player_hp_cur"]) <= src.player_hp_max * 0.05 + 2
    assert reading.stamina_frac is not None and 0.0 < reading.stamina_frac <= 1.0
    assert len(reading.party_hp_fracs) == 3
    assert len(reading.burst_ready) == 3
