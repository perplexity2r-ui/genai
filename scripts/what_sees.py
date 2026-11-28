#!/usr/bin/env python3
"""Что агент понимает из кадра: сводная таблица по типовым ситуациям.

Печатает признаки восприятия и решение логики для набора сцен: открытое
поле, стена (издалека / вплотную / сбоку), край обрыва, узкий карниз, полёт
над пропастью, высокая площадка. Затем — разбор реальных кадров Genshin из
``examples/screens/``.

Запуск::

    python scripts/what_sees.py                 # только синтетика
    python scripts/what_sees.py --real          # плюс реальные кадры

Сцены синтетические (аналитический ray-AABB), поэтому модель не нужна;
для ``--real`` требуется скачанная модель (``scripts/fetch_model.py``).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parkour.config import ParkourConfig  # noqa: E402
from parkour.depth import SlowNormalizer  # noqa: E402
from parkour.perception import Perception  # noqa: E402
from parkour.policy import ParkourBrain  # noqa: E402
from parkour.sim import SCENARIOS, CharState, Simulator, World, render_disparity  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 6


def look(world, pos, yaw=0.0, steps=STEPS, cfg=None):
    """Прогоняет агента `steps` шагов и возвращает (первый кадр, все решения).

    Первый кадр — это честный ответ «что агент понял, оказавшись здесь»;
    дальше он уже действует и сцена меняется (залез на стену — под ногами
    край плато, и это действительно край).
    """
    cfg = cfg or ParkourConfig()
    sim = Simulator(world, state=CharState(x=pos[0], z=pos[2], y=pos[1], yaw=yaw), dt=0.1)
    perception = Perception(cfg.perception)
    brain = ParkourBrain(cfg.policy)
    normalizer = SlowNormalizer(cfg.depth.p_low, cfg.depth.p_high, cfg.depth.slow_norm_ema)
    gx, gz = world.goal
    first, seen = None, []
    for step in range(steps):
        raw = render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160))
        scene = perception.analyze(prox=normalizer(raw.copy()),
                                   raw_scale=float(np.percentile(raw, 98)))
        bearing = math.atan2(gx - sim.s.x, gz - sim.s.z) - sim.s.yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        action = brain.decide(scene, stamina=sim.s.stamina,
                              goal_bearing=bearing, now=step * 0.1)
        if first is None:
            first = (scene, action)
        seen.append(action.state.value)
        sim.step(action.keys, action.yaw)
    # «CLIMB CLIMB WALK WALK» -> «CLIMB x2 → WALK x2»
    compact = []
    for state in seen:
        if compact and compact[-1][0] == state:
            compact[-1][1] += 1
        else:
            compact.append([state, 1])
    return first[0], first[1], " → ".join(f"{s} x{n}" for s, n in compact)


def synthetic_rows(cfg):
    """Список (подпись, сцена, решение) по типовым ситуациям."""
    chasm, ledge, fall = SCENARIOS["chasm"](), SCENARIOS["ledge"](), SCENARIOS["fall"]()
    cases = [
        ("Открытое поле", World(name="flat", goal=(0.0, 45.0)), (0.0, 0.0, 0.0), 0.0),
        ("Стена в 6 м", SCENARIOS["wall"](), (0.0, 0.0, 14.0), 0.0),
        ("Стена вплотную (лицом)", SCENARIOS["wall"](), (0.0, 0.0, 20.0), 0.0),
        ("Стена сбоку (взгляд вдоль)", SCENARIOS["wall"](), (0.0, 0.0, 19.4), np.pi / 2),
        ("Край обрыва в 4 м", chasm, (0.0, 0.0, 18.0), 0.0),
        ("Край обрыва вплотную", chasm, (0.0, 0.0, 21.2), 0.0),
        ("Узкий карниз (прямо)", ledge, (0.0, 0.0, 10.0), 0.0),
        ("Карниз, смотрим в пропасть", ledge, (0.0, 0.0, 10.0), np.pi / 2),
        ("Полёт над пропастью", fall, (0.0, 12.0, 12.0), 0.0),
        ("Высокая площадка (стоим)", fall, (0.0, 18.0, 0.0), 0.0),
    ]
    rows = []
    for label, world, pos, yaw in cases:
        scene, action, sequence = look(world, pos, yaw=yaw, cfg=cfg)
        rows.append((label, scene, action, sequence))
    return rows


def real_rows(cfg):
    """Разбор реальных кадров Genshin (нужна модель)."""
    from PIL import Image

    from parkour.depth import DepthEstimator

    screens = os.path.join(ROOT, "examples", "screens")
    files = sorted(f for f in os.listdir(screens) if f.lower().endswith((".jpg", ".png")))
    estimator = DepthEstimator(cfg.depth).load()
    rows = []
    for name in files:
        frame = np.asarray(Image.open(os.path.join(screens, name)).convert("RGB"))
        perception = Perception(cfg.perception)
        prox = estimator.predict(frame)
        scene = perception.analyze(prox=prox, raw_scale=perception.raw_scale_from(estimator))
        action = ParkourBrain(cfg.policy).decide(scene, stamina=0.8, goal_bearing=0.0, now=0.0)
        rows.append((f"{name}  рядом={estimator.last_raw_scale:.2f}", scene, action, "—"))
    return rows


def print_rows(title, rows):
    print(title)
    print(f"{'Ситуация':32s} {'стена':>5s} {'обрыв':>5s} {'полёт':>5s} {'своб':>5s}  "
          f"{'решение':10s} клавиши")
    print("-" * 104)
    for label, scene, action, sequence in rows:
        yn = lambda flag: "да" if flag else "—"  # noqa: E731
        print(f"{label:32s} {yn(scene.wall):>5s} {yn(scene.ledge):>5s} {yn(scene.falling):>5s} "
              f"{scene.free_forward:5.2f}  {action.state.value:10s} {','.join(action.pressed())}")
        print(f"{'':32s} далее: {sequence}")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", action="store_true",
                        help="добавить разбор реальных кадров Genshin (нужна модель)")
    args = parser.parse_args(argv)
    cfg = ParkourConfig()
    print_rows("Синтетические ситуации (6 шагов агента):", synthetic_rows(cfg))
    if args.real:
        print_rows("Реальные кадры Genshin:", real_rows(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
