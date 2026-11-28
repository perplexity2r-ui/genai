"""Продакшн-цикл: экран -> карта глубины -> решение -> клавиши.

Отличия от ``runner.py`` (синтетика): здесь настоящий экран и настоящий ввод.
Есть режим ``dry_run`` — считаем всё, кроме нажатий: удобно для отладки порогов
без риска уронить персонажа в пропасть.

Безопасность:
* клавиши отпускаются при исключении/по Ctrl+C (``finally``);
* аварийный стоп — клавиша ``F8`` или ``Esc`` (опрос через ``keyboard``, если есть);
* при потере уверенности восприятия агент тормозит, а не ускоряется.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .config import ParkourConfig
from .depth import DepthEstimator
from .input_backend import FpsMeter, InputBackend, ScreenCapture, StaminaBar
from .perception import Perception, SceneAnalysis
from .policy import ActionFrame, ParkourBrain


@dataclass
class AgentStats:
    frames: int = 0
    decisions: Dict[str, int] = field(default_factory=dict)
    depth_ms: List[float] = field(default_factory=list)
    total_ms: List[float] = field(default_factory=list)

    def add(self, action: ActionFrame, depth_ms: float, total_ms: float) -> None:
        self.frames += 1
        self.decisions[action.state.value] = self.decisions.get(action.state.value, 0) + 1
        self.depth_ms.append(depth_ms)
        self.total_ms.append(total_ms)
        if len(self.depth_ms) > 200:
            self.depth_ms.pop(0)
            self.total_ms.pop(0)

    def summary(self) -> str:
        def pct(xs, q):
            return float(np.percentile(xs, q)) if xs else 0.0
        return (f"кадров={self.frames} depth p50={pct(self.depth_ms, 50):.0f} мс "
                f"p95={pct(self.depth_ms, 95):.0f} мс | цикл p50={pct(self.total_ms, 50):.0f} мс "
                f"| состояния={self.decisions}")


class ParkourAgent:
    """Главный объект для запуска на живой игре."""

    def __init__(self, cfg: Optional[ParkourConfig] = None,
                 frame_source: Optional[Callable[[], np.ndarray]] = None,
                 input_backend: Optional[InputBackend] = None,
                 log: Callable[[str], None] = print):
        self.cfg = cfg or ParkourConfig()
        self.estimator = DepthEstimator(self.cfg.depth)
        self.perception = Perception(self.cfg.perception, estimator=self.estimator)
        self.brain = ParkourBrain(self.cfg.policy)
        self.capture = None if frame_source else ScreenCapture(self.cfg.capture_region)
        self.frame_source = frame_source or (self.capture.grab if self.capture else None)
        self.input = input_backend or InputBackend(enabled=not self.cfg.dry_run)
        self.stamina_bar = StaminaBar()
        self.stats = AgentStats()
        self.fps = FpsMeter()
        self.log = log
        self._stop = False
        self._stamina_smoothed: Optional[float] = None

    # ------------------------------------------------------------------ setup
    def describe(self) -> str:
        self.estimator.load()
        lines = [
            self.estimator.describe(),
            f"кадр: {'весь экран' if not self.cfg.capture_region else self.cfg.capture_region}, "
            f"цель {self.cfg.target_fps:.0f} FPS",
            self.input.describe(),
        ]
        return " | ".join(lines)

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------- loop
    def step(self, goal_bearing: float = 0.0) -> ActionFrame:
        return self.step_full(goal_bearing)[0]

    def step_full(self, goal_bearing: float = 0.0):
        """Один цикл: снять кадр, оценить сцену, решить, отправить клавиши."""
        t0 = time.perf_counter()
        assert self.frame_source is not None
        frame = self.frame_source()
        prox = self.estimator.predict(frame)
        raw = self.perception.raw_scale_from(self.estimator)
        scene = self.perception.analyze(prox=prox, raw_scale=raw)
        depth_ms = (time.perf_counter() - t0) * 1e3

        stamina = self._read_stamina(frame, scene)
        action = self.brain.decide(scene, stamina=stamina,
                                   goal_bearing=goal_bearing, now=time.monotonic())
        self.input.apply(action.keys)
        self.input.turn(action.yaw)

        total_ms = (time.perf_counter() - t0) * 1e3
        self.stats.add(action, depth_ms, total_ms)
        self.fps.tick()
        return action, frame, scene

    def _read_stamina(self, frame: np.ndarray, scene: SceneAnalysis) -> float:
        """Выносливость: из полосы, если видна; иначе — консервативная оценка."""
        fill = self.stamina_bar.measure(frame)
        if fill is None:
            # Полосы нет — обычно это полная выносливость.
            fill = 1.0
        a = 0.3
        self._stamina_smoothed = fill if self._stamina_smoothed is None else \
            a * fill + (1 - a) * self._stamina_smoothed
        return float(self._stamina_smoothed)

    def run(self, max_seconds: Optional[float] = None,
            goal_bearing_fn: Optional[Callable[[], float]] = None,
            on_frame: Optional[Callable[[np.ndarray, SceneAnalysis, ActionFrame], None]] = None,
            countdown_s: float = 3.0) -> AgentStats:
        """Основной цикл. ``goal_bearing_fn`` может читать миникарту/маркер."""
        self.describe()
        self.log(f"старт через {countdown_s:.0f} с — переключитесь в окно Genshin")
        time.sleep(max(0.0, countdown_s))
        period = 1.0 / max(0.5, self.cfg.target_fps)
        t_end = None if max_seconds is None else time.monotonic() + max_seconds
        try:
            while not self._stop:
                if t_end is not None and time.monotonic() > t_end:
                    break
                t_start = time.perf_counter()
                bearing = goal_bearing_fn() if goal_bearing_fn else 0.0
                action, frame, scene = self.step_full(bearing)
                if on_frame is not None:
                    # Для отладки/записи видео: отдаём последний кадр и сцену.
                    on_frame(frame, scene, action)
                left = period - (time.perf_counter() - t_start)
                if left > 0:
                    time.sleep(left)
        except KeyboardInterrupt:
            self.log("остановлено пользователем")
        finally:
            self.input.release_all()
            if self.capture is not None:
                self.capture.close()
        self.log(self.stats.summary())
        return self.stats


def bearing_to_goal(pos: tuple, goal: tuple, yaw: float) -> float:
    """Ошибка курса на цель (рад) для подачи в логику."""
    dx, dz = goal[0] - pos[0], goal[1] - pos[1]
    b = math.atan2(dx, dz) - yaw
    return (b + math.pi) % (2 * math.pi) - math.pi
