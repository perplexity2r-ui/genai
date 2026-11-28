#!/usr/bin/env python3
"""Записывает демо-материалы в docs/: нарезки прогонов и разбор реальных кадров.

Материалы нужны README: видно, что агент действительно карабкается, обходит
провал и раскрывает планер. Требуется скачанная модель (scripts/fetch_model.py).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parkour.config import ParkourConfig  # noqa: E402
from parkour.depth import DepthEstimator, SlowNormalizer  # noqa: E402
from parkour.perception import Perception  # noqa: E402
from parkour.policy import ParkourBrain  # noqa: E402
from parkour.sim import SCENARIOS, CharState, Simulator, render_disparity  # noqa: E402
from parkour.viz import colorize, draw_telemetry, panel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SIZE = (300, 533)


def sheet(name: str, picks=(10, 40, 70, 100)) -> str:
    """Прогон сценария с нарезкой кадров в одну картинку."""
    cfg = ParkourConfig()
    world = SCENARIOS[name]()
    gh, _ = world.height(np.array([world.start[0]], np.float32),
                         np.array([world.start[1]], np.float32))
    sim = Simulator(world, state=CharState(x=world.start[0], z=world.start[1],
                                           y=float(gh[0]), yaw=world.start[2]), dt=0.1)
    perception, brain = Perception(cfg.perception), ParkourBrain(cfg.policy)
    norm = SlowNormalizer(cfg.depth.p_low, cfg.depth.p_high, cfg.depth.slow_norm_ema)
    gx, gz = world.goal
    frames = []
    for step in range(320):
        prox = norm(render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160)))
        scene = perception.analyze(prox=prox, raw_scale=float(np.percentile(
            render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160)), 98)))
        bearing = np.arctan2(gx - sim.s.x, gz - sim.s.z) - sim.s.yaw
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        action = brain.decide(scene, stamina=sim.s.stamina,
                              goal_bearing=float(bearing), now=step * 0.1)
        img = colorize(prox, size=SIZE)
        img = draw_telemetry(img, [
            f"{name}   t={step * 0.1:4.1f} s",
            f"{action.state.value}   {' '.join(action.pressed())}",
            f"wall={scene.wall}  ledge={scene.ledge}",
            f"fall={scene.falling}  free={scene.free_forward:.2f}",
            f"stamina={sim.s.stamina:.2f}",
            f"pos=({sim.s.x:.1f}, {sim.s.y:.1f}, {sim.s.z:.1f})",
            f"до цели {np.hypot(sim.s.x - gx, sim.s.z - gz):.0f} м",
        ])
        frames.append(img)
        sim.step(action.keys, action.yaw)
        if not sim.s.alive or np.hypot(sim.s.x - gx, sim.s.z - gz) < 4.0:
            break
    picks = [min(p, len(frames) - 1) for p in picks]
    h, w = frames[0].shape[:2]
    out = Image.new("RGB", (w * len(picks), h))
    for i, p in enumerate(picks):
        out.paste(Image.fromarray(frames[p]), (i * w, 0))
    path = os.path.join(DOCS, f"sim_{name}.png")
    out.save(path)
    print("  ", path, f"({len(frames)} кадров прогона)")
    return path


def real_frames() -> str:
    """Разбор реальных кадров Genshin: фото | карта глубины | решение."""
    from parkour.policy import ParkourBrain as Brain

    cfg = ParkourConfig()
    est = DepthEstimator(cfg.depth).load()
    files = sorted(os.path.join(ROOT, "examples", "screens", f)
                   for f in os.listdir(os.path.join(ROOT, "examples", "screens"))
                   if f.lower().endswith((".jpg", ".png")))
    tiles = []
    for path in files:
        frame = np.asarray(Image.open(path).convert("RGB"))
        prox = est.predict(frame)
        perc = Perception(cfg.perception)
        scene = perc.analyze(prox=prox, raw_scale=perc.raw_scale_from(est))
        action = Brain(cfg.policy).decide(scene, stamina=0.8, now=0.0)
        # «Что агент понял»: приоритет как в логике — полёт, стена, край.
        if scene.falling:
            key = "GLIDE"
        elif scene.wall:
            key = "CLIMB"
        elif scene.ledge:
            key = "AVOID"
        else:
            key = action.state.value
        tiles.append(panel(frame, scene, size=(300, 169), info={
            "решение": key,
            "клавиши": ",".join(action.pressed()),
            "wall": f"{scene.wall}/{scene.wall_prox:.2f}",
            "ledge": f"{scene.ledge}/{scene.ledge_prox:.2f}",
            "fall": scene.falling,
            "free": f"{scene.free_forward:.2f}",
            "рядом": f"{est.last_raw_scale:.2f}",
            "мс": f"{est.last_latency_ms:.0f}",
        }))
    out = Image.new("RGB", (tiles[0].shape[1], sum(t.shape[0] for t in tiles)))
    y = 0
    for t in tiles:
        out.paste(Image.fromarray(t), (0, y))
        y += t.shape[0]
    path = os.path.join(DOCS, "real_frames.png")
    out.save(path)
    print("  ", path)
    return path


if __name__ == "__main__":
    os.makedirs(DOCS, exist_ok=True)
    print("синтетические прогоны:")
    sheet("wall")
    sheet("chasm")
    sheet("fall")
    print("реальные кадры:")
    real_frames()
