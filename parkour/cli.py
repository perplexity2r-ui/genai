"""CLI: демо на кадрах, прогон по сценариям, запуск на живой игре, бенчмарк."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

from .config import ParkourConfig


# --------------------------------------------------------------------- helpers
def _load_image(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save(img: np.ndarray, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)


def _expand(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        out.extend(hits)
    return [p for p in out if os.path.isfile(p)]


# ------------------------------------------------------------------------ demo
def cmd_demo(args: argparse.Namespace) -> int:
    from .depth import DepthEstimator
    from .perception import Perception
    from .policy import ParkourBrain
    from .viz import panel

    cfg = ParkourConfig.load(args.config) if args.config else ParkourConfig()
    files = _expand(args.images)
    if not files:
        print("нет входных изображений", file=sys.stderr)
        return 2

    estimator = DepthEstimator(cfg.depth).load()
    print(estimator.describe())
    perception = Perception(cfg.perception)
    brain = ParkourBrain(cfg.policy)

    for path in files:
        frame = _load_image(path)
        t0 = time.perf_counter()
        prox = estimator.predict(frame)
        raw = perception.raw_scale_from(estimator)
        scene = perception.analyze(prox=prox, raw_scale=raw)
        action = brain.decide(scene, stamina=args.stamina, goal_bearing=args.bearing,
                              now=time.monotonic())
        ms = (time.perf_counter() - t0) * 1e3
        print(f"\n{os.path.basename(path)}  {frame.shape[1]}x{frame.shape[0]}  {ms:.0f} мс")
        print("  сцена:", json.dumps(scene.as_dict(), ensure_ascii=False))
        print("  профиль:", scene.profile)
        print("  решение:", action.state.value, action.pressed(), "|", action.reason)
        if action.note:
            print("  пояснение:", action.note)
        if args.out:
            base = os.path.splitext(os.path.basename(path))[0]
            vis = panel(frame, scene, size=(args.height, int(args.height * 9 / 16)),
                        info={"state": action.state.value,
                              "keys": ",".join(action.pressed()),
                              "wall": f"{scene.wall}/{scene.wall_prox:.2f}",
                              "ledge": f"{scene.ledge}/{scene.ledge_prox:.2f}",
                              "fall": scene.falling,
                              "free": f"{scene.free_forward:.2f}",
                              "ms": f"{ms:.0f}"})
            out_path = os.path.join(args.out, f"{base}_decision.png")
            _save(vis, out_path)
            print("  визуализация:", out_path)
    return 0


# ------------------------------------------------------------------- scenarios
def cmd_simulate(args: argparse.Namespace) -> int:
    from .runner import EpisodeRunner
    from .sim import SCENARIOS

    cfg = ParkourConfig.load(args.config) if args.config else ParkourConfig()
    names = args.worlds or list(SCENARIOS)
    if args.video:
        return _record_episode(names[0], args, cfg)

    ok = 0
    for name in names:
        world = SCENARIOS[name]()
        res = EpisodeRunner(cfg).run(world, max_steps=args.max_steps)
        print(res.summary())
        if args.verbose:
            for step in res.trace[: args.verbose]:
                print("   ", step)
        ok += int(res.success)
    print(f"\nитого: {ok}/{len(names)} сценариев пройдено")
    return 0 if ok == len(names) else 1


def _record_episode(name: str, args: argparse.Namespace, cfg: ParkourConfig) -> int:
    """Записывает видео (GIF) прогона: кадр | карта глубины | клавиши."""
    from PIL import Image
    from .depth import SlowNormalizer
    from .perception import Perception
    from .policy import ParkourBrain
    from .sim import SCENARIOS, CharState, Simulator, render_disparity
    from .viz import colorize, draw_telemetry

    world = SCENARIOS[name]()
    gh, _ = world.height(np.array([world.start[0]], np.float32),
                         np.array([world.start[1]], np.float32))
    sim = Simulator(world, state=CharState(x=world.start[0], z=world.start[1],
                                           y=float(gh[0]), yaw=world.start[2]), dt=0.1)
    perception = Perception(cfg.perception)
    brain = ParkourBrain(cfg.policy)
    norm = SlowNormalizer(cfg.depth.p_low, cfg.depth.p_high, cfg.depth.slow_norm_ema)
    frames = []
    gx, gz = world.goal
    for step in range(args.max_steps):
        prox = norm(render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160)))
        scene = perception.analyze(prox=prox, raw_scale=float(np.percentile(
            render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160)), 98)))
        bearing = np.arctan2(gx - sim.s.x, gz - sim.s.z) - sim.s.yaw
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        action = brain.decide(scene, stamina=sim.s.stamina, goal_bearing=float(bearing),
                              now=step * 0.1)
        img = colorize(prox, size=(288, 512))
        img = draw_telemetry(img, [
            f"{name}  t={step * 0.1:4.1f}s",
            f"{action.state.value}  {' '.join(action.pressed())}",
            f"wall={scene.wall} ledge={scene.ledge} fall={scene.falling}",
            f"free={scene.free_forward:.2f} stamina={sim.s.stamina:.2f}",
            f"pos=({sim.s.x:.1f},{sim.s.y:.1f},{sim.s.z:.1f})",
            f"goal={np.hypot(sim.s.x - gx, sim.s.z - gz):.1f} м",
        ])
        frames.append(Image.fromarray(img))
        sim.step(action.keys, action.yaw)
        if not sim.s.alive or np.hypot(sim.s.x - gx, sim.s.z - gz) < 4.0:
            break
    out = args.video
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"записано {len(frames)} кадров -> {out}")
    print(f"финал: pos=({sim.s.x:.1f},{sim.s.y:.1f},{sim.s.z:.1f}) "
          f"расстояние до цели={np.hypot(sim.s.x - gx, sim.s.z - gz):.1f} м "
          f"{'OK' if np.hypot(sim.s.x - gx, sim.s.z - gz) < 4.0 else 'не дошёл'}")
    return 0


# ------------------------------------------------------------------------ bench
def cmd_bench(args: argparse.Namespace) -> int:
    from .depth import DepthEstimator, PROFILES, DepthConfig

    files = _expand(args.images) or []
    if not files:
        frame = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    else:
        frame = _load_image(files[0])
    print(f"кадр {frame.shape[1]}x{frame.shape[0]}")
    print(f"{'профиль':10s} {'вход':10s} {'p50':>7s} {'p95':>7s} {'FPS':>6s}")
    for name, hw in PROFILES.items():
        cfg = DepthConfig(input_hw=hw, threads=args.threads)
        est = DepthEstimator(cfg).load()
        est.predict(frame)
        times = []
        for _ in range(args.repeat):
            est.predict(frame)
            times.append(est.last_latency_ms)
        p50, p95 = np.percentile(times, 50), np.percentile(times, 95)
        print(f"{name:10s} {hw[1]}x{hw[0]:<5d} {p50:7.1f} {p95:7.1f} {1000 / p50:6.1f}")
    return 0


# ------------------------------------------------------------------------ live
def cmd_play(args: argparse.Namespace) -> int:
    from .agent import ParkourAgent

    cfg = ParkourConfig.load(args.config) if args.config else ParkourConfig()
    cfg.dry_run = not args.act
    cfg.target_fps = args.fps
    if args.region:
        cfg.capture_region = tuple(int(v) for v in args.region)
    agent = ParkourAgent(cfg)
    print(agent.describe())
    if args.act:
        print("ВНИМАНИЕ: агент будет нажимать клавиши. Alt+Tab для выхода, Ctrl+C для остановки.")
    agent.run(max_seconds=args.seconds, countdown_s=args.countdown)
    return 0


# ------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parkour",
        description="Genshin Parkour — локальная навигация по карте глубины (Depth Anything V2)")
    p.add_argument("--config", help="JSON с настройками (см. parkour config --out)")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="разбор готовых кадров: сцена + решение + визуализация")
    d.add_argument("images", nargs="+", help="файлы или маски (например 'shots/*.png')")
    d.add_argument("--out", default="out", help="куда сохранять визуализации")
    d.add_argument("--height", type=int, default=360)
    d.add_argument("--stamina", type=float, default=0.8)
    d.add_argument("--bearing", type=float, default=0.0)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("simulate", help="прогон по синтетическим сценариям")
    s.add_argument("--worlds", nargs="*", help="имена сценариев (по умолчанию все)")
    s.add_argument("--max-steps", type=int, default=1200)
    s.add_argument("--verbose", type=int, default=0)
    s.add_argument("--video", help="записать GIF прогона первого сценария")
    s.set_defaults(func=cmd_simulate)

    b = sub.add_parser("bench", help="замер скорости depth-сети")
    b.add_argument("--images", nargs="*", default=[])
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--repeat", type=int, default=5)
    b.set_defaults(func=cmd_bench)

    pl = sub.add_parser("play", help="запуск на живой игре (по умолчанию dry-run)")
    pl.add_argument("--act", action="store_true", help="реально нажимать клавиши")
    pl.add_argument("--fps", type=float, default=12.0)
    pl.add_argument("--seconds", type=float, default=None)
    pl.add_argument("--countdown", type=float, default=3.0)
    pl.add_argument("--region", nargs=4, metavar=("X", "Y", "W", "H"))
    pl.set_defaults(func=cmd_play)

    c = sub.add_parser("config", help="сохранить конфиг по умолчанию")
    c.add_argument("--out", default="parkour.json")
    c.set_defaults(func=lambda a: (ParkourConfig().save(a.out), print("сохранено:", a.out))[0])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
