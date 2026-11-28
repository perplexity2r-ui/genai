"""Замкнутый цикл «сцена -> восприятие -> решение -> действие» без игры.

Используется тестами, бенчмарком и CLI-демо: позволяет проверить, что агент
реально проходит полосу препятствий, а не просто выдаёт правдоподобные клавиши.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import ParkourConfig
from .perception import Perception
from .policy import ParkourBrain
from .depth import SlowNormalizer
from .sim import Camera, CharState, Simulator, World, render_disparity


@dataclass
class EpisodeResult:
    world: str
    success: bool
    steps: int
    sim_time: float
    final_pos: tuple
    goal_dist: float
    falls: int
    bump_wall: int
    climbed_s: float
    glided_s: float
    sprint_s: float
    distance: float
    alive: bool
    state_counts: Dict[str, int] = field(default_factory=dict)
    death: str = ""
    trace: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.world:8s} success={self.success!s:5s} t={self.sim_time:5.1f}s "
                f"goal_dist={self.goal_dist:5.1f} falls={self.falls} "
                f"bump={self.bump_wall} climb={self.climbed_s:4.1f}s "
                f"glide={self.glided_s:4.1f}s jumps={self.state_counts.get('CLIMB', 0)} "
                f"death={self.death or '-'}")


class EpisodeRunner:
    """Гоняет агента по синтетическому миру с честным рендером глубины."""

    def __init__(self, cfg: Optional[ParkourConfig] = None, render_hw=(96, 160),
                 camera: Optional[Camera] = None, use_depth_net: bool = False,
                 depth_estimator=None):
        self.cfg = cfg or ParkourConfig()
        self.render_hw = render_hw
        self.camera = camera or Camera()
        self.brain = ParkourBrain(self.cfg.policy)
        self.use_depth_net = use_depth_net
        self.estimator = depth_estimator

    def run(self, world: World, max_steps: int = 900, dt: float = 0.1,
            goal_radius: float = 4.0, trace_every: int = 10,
            record_render: bool = False) -> EpisodeResult:
        start = CharState(x=world.start[0], z=world.start[1], yaw=world.start[2])
        gh, _ = world.height(np.array([start.x], np.float32), np.array([start.z], np.float32))
        start.y = float(gh[0])
        sim = Simulator(world, state=start, dt=dt)
        perc = Perception(self.cfg.perception, estimator=self.estimator)
        norm = SlowNormalizer(self.cfg.depth.p_low, self.cfg.depth.p_high,
                              self.cfg.depth.slow_norm_ema)
        per_frame_ms: List[float] = []
        counts: Dict[str, int] = {}
        trace: List[dict] = []

        for step in range(max_steps):
            pos = sim.s.pos()
            t0 = time.perf_counter()
            disp = render_disparity(world, pos, sim.s.yaw, hw=self.render_hw, cam=self.camera)
            prox = norm(disp)
            raw = float(np.percentile(disp, 98))
            scene = perc.analyze(prox=prox, raw_scale=raw)
            per_frame_ms.append((time.perf_counter() - t0) * 1e3)

            gx, gz = world.goal
            bearing = math.atan2(gx - sim.s.x, gz - sim.s.z) - sim.s.yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            action = self.brain.decide(scene, stamina=sim.s.stamina,
                                       goal_bearing=bearing, now=step * dt)
            counts[action.state.value] = counts.get(action.state.value, 0) + 1
            sim.step(action.keys, action.yaw)

            if step % trace_every == 0:
                trace.append({
                    "t": round(step * dt, 1),
                    "pos": (round(sim.s.x, 1), round(sim.s.y, 1), round(sim.s.z, 1)),
                    "yaw": round(sim.s.yaw, 2),
                    "state": action.state.value,
                    "keys": action.pressed(),
                    "stamina": round(sim.s.stamina, 2),
                    "wall": scene.wall,
                    "ledge": scene.ledge,
                    "airborne": scene.airborne,
                    "free": round(scene.free_forward, 2),
                })

            gx, gz = world.goal
            goal_dist = math.hypot(sim.s.x - gx, sim.s.z - gz)
            if goal_dist <= goal_radius and sim.s.alive:
                break
            if not sim.s.alive:
                break
            if sim.s.y < -20.0:
                break

        gx, gz = world.goal
        goal_dist = math.hypot(sim.s.x - gx, sim.s.z - gz)
        ev = sim.events
        return EpisodeResult(
            world=world.name,
            success=bool(sim.s.alive and goal_dist <= goal_radius),
            steps=step + 1,
            sim_time=round((step + 1) * dt, 2),
            final_pos=(round(sim.s.x, 2), round(sim.s.y, 2), round(sim.s.z, 2)),
            goal_dist=round(goal_dist, 2),
            falls=ev.falls,
            bump_wall=ev.bump_wall,
            climbed_s=round(ev.climbed_s, 2),
            glided_s=round(ev.glided_s, 2),
            sprint_s=round(ev.sprint_s, 2),
            distance=round(ev.distance, 2),
            alive=sim.s.alive,
            death=ev.death,
            state_counts=counts,
            trace=trace,
        )
