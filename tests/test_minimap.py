"""Тесты чтения миникарты: курс, иконки, движение игрока."""
import numpy as np

from genshinvision.hud import DEFAULT_CALIBRATION
from genshinvision.minimap import MinimapReader
from genshinvision.synthetic import SyntheticGameSource


def circ_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def run(n_frames=150, seed=21):
    src = SyntheticGameSource(seed=seed, fps=30.0, n_enemies=3, max_frames=n_frames)
    reader = MinimapReader(DEFAULT_CALIBRATION)
    states, truths = [], []
    for _ in range(n_frames):
        ts, frame = src.read()
        states.append(reader.read(frame, dt=1.0 / 30.0))
        truths.append(src.last_truth["minimap"])
    return src, states, truths


def test_heading_accuracy():
    _, states, truths = run()
    errs = [circ_diff(s.heading_deg, t["heading_deg"])
            for s, t in zip(states[10:], truths[10:])
            if s.heading_deg is not None]
    assert len(errs) > 100, f"курс найден только в {len(errs)} кадрах"
    assert float(np.mean(errs)) < 8.0, f"средняя ошибка курса {np.mean(errs):.1f}°"
    assert float(np.max(errs)) < 20.0


def test_icons_detected():
    _, states, truths = run()
    # проверяем каждый кадр: найденные иконки совпадают с истинными, ложных нет
    seen_types = set()
    checked_frames = 0
    for st, t in zip(states, truths):
        truth_icons = t["icons"]
        if not truth_icons:
            continue
        checked_frames += 1
        for ti in truth_icons:
            seen_types.add(ti["type"])
            match = [d for d in st.icons
                     if d.type == ti["type"]
                     and (d.x - ti["x"]) ** 2 + (d.y - ti["y"]) ** 2 < 0.25 ** 2]
            assert match, f"иконка {ti} не найдена: det={[d.to_dict() for d in st.icons]}"
        for d in st.icons:
            ok = any((d.x - ti["x"]) ** 2 + (d.y - ti["y"]) ** 2 < 0.25 ** 2 for ti in truth_icons)
            assert ok, f"ложная иконка {d.to_dict()} при truth={truth_icons}"
    assert checked_frames >= 30
    # за путь игрока видны минимум квест и телепорт
    assert {"quest", "teleport"} <= seen_types, seen_types


def test_motion_matches_player_velocity():
    src, states, truths = run()
    v = src.player_vel
    errs = []
    for s in states[30:]:
        mx, my = s.motion_px_s
        errs.append(((mx - v[0]) ** 2 + (my - v[1]) ** 2) ** 0.5)
    med = float(np.median(errs))
    assert med < 30.0, f"медианная ошибка движения {med:.1f} px/s"
