"""Логика паркура: целенаправленное движение + безопасный обход.

Идея. Локальная навигация по карте глубины не знает, где сундук, поэтому
базовое правило простое: **идти туда, куда смотрит камера** (игрок уже навёл её
на цель / квест-маркер). Агент вмешивается только когда впереди препятствие:

* стена -> карабканье (`Space` импульсами + `W`), а если стена не по зубам
  (мало выносливости) — короткий обход с возвратом на курс;
* край обрыва -> не шагаем в пустоту: обход, а при близком крае — назад;
* уже падаем -> раскрываем планер.

Обход всегда *ограничен по углу и времени* (dodge), после чего агент
возвращает исходный курс. Именно это отличает навигацию от «случайного
блуждания», в которое легко скатиться, если просто гнаться за «свободным»
сектором кадра.

Все клавиши — в терминах Genshin: `Space` = прыжок / карабканье / планер,
`Shift` = спринт, `W/A/S/D` — движение, камера — доворот мышью.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .config import PolicyConfig
from .perception import SceneAnalysis


def wrap_pi(a: float) -> float:
    """Приводит угол к диапазону (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class State(str, Enum):
    WALK = "WALK"
    SPRINT = "SPRINT"
    CLIMB = "CLIMB"
    AVOID = "AVOID"
    GLIDE = "GLIDE"
    BACK_OFF = "BACK_OFF"


ACTIONS = ("forward", "back", "left", "right", "sprint", "jump", "glide", "yaw")


@dataclass
class ActionFrame:
    state: State
    keys: Dict[str, bool] = field(default_factory=lambda: {k: False for k in ACTIONS})
    yaw: float = 0.0            # команда доворота камеры, -1..1
    reason: str = ""
    note: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def pressed(self) -> List[str]:
        return [k for k, v in self.keys.items() if v]

    def as_dict(self) -> dict:
        return {
            "state": self.state.value,
            "keys": self.pressed(),
            "yaw": round(self.yaw, 3),
            "reason": self.reason,
            "note": self.note,
        }


@dataclass
class Dodge:
    """Ограниченный по углу и времени обход препятствия."""
    dir: int = 0            # -1 влево, +1 вправо
    until: float = 0.0      # до какого времени активен
    angle: float = 0.0      # накопленный угол поворота (рад), для возврата
    reason: str = ""

    @property
    def active(self) -> bool:
        return self.dir != 0


class ParkourBrain:
    """Правила движения. Чистая логика: без ввода-вывода и без сна."""

    def __init__(self, cfg: Optional[PolicyConfig] = None):
        self.cfg = cfg or PolicyConfig()
        self.state = State.WALK
        self.state_since = 0.0
        self._now = 0.0
        self._last_now: Optional[float] = None
        self._last_jump = -1e9
        self._last_glide = -10.0
        self.goal_bearing = 0.0      # азимут цели, поданный снаружи (рад)
        self.dodge = Dodge()
        self._edge_dir = 0           # залипающее направление обхода края
        self._edge_until = 0.0
        self._side = 0               # текущая сторона обхода
        self._side_until = 0.0
        self._block: Optional[str] = None
        self._probe = 0              # 0: обход влево, 1: обход вправо, 2: планер
        self._probe_since = 0.0
        self._probe_dir = 1
        self._glide_open = False     # планер уже раскрыт в этом эпизоде полёта
        self._flight_started = False  # прыжок с края уже сделан
        self._glide_due = True        # можно нажать Space (с учётом паузы)
        self._was_airborne = False     # были в полёте после прыжка
        self._floor_hits = 0          # подряд идущих кадров «под ногами опора»
        self.history: List[tuple] = []

    # ------------------------------------------------------------------ время
    def tick(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else float(now)
        self._now = now
        return now

    @property
    def heading(self) -> float:
        """Ошибка курса: насколько цель «в стороне» от направления камеры."""
        return self.goal_bearing

    def _dt(self) -> float:
        dt = 0.0 if self._last_now is None else max(0.0, self._now - self._last_now)
        self._last_now = self._now
        return min(dt, 0.5)

    def _switch(self, new: State, reason: str) -> None:
        if new == self.state:
            return
        if self._now - self.state_since < self.cfg.min_hold_s and \
                new not in (State.GLIDE, State.BACK_OFF):
            return
        self.state = new
        self.state_since = self._now
        self.history.append((round(self._now, 3), new.value, reason))
        self.history = self.history[-64:]

    # ------------------------------------------------------------------ курс
    def _steer(self, err: float, gain: float = 1.0) -> float:
        """П-регулятор доворота по ошибке курса (рад): -1..1."""
        if abs(err) < self.cfg.yaw_deadband:
            return 0.0
        # Сначала гасим мёртвую зону, потом пропорционально — без «дребезга» у нуля.
        sign = 1.0 if err > 0 else -1.0
        mag = min(1.0, (abs(err) - self.cfg.yaw_deadband) * self.cfg.yaw_kp * gain)
        return sign * mag

    def _steer_to_goal(self, goal_bearing: float, gain: float = 1.0) -> float:
        """Доворот к цели по свежему азимуту (цель может двигаться — не копим дрейф)."""
        return self._steer(goal_bearing, gain)

    def _steer_offset(self, offset: float, gain: float = 1.0) -> float:
        """Доворот на смещённый курс: цель + offset (для обхода и отхода)."""
        return self._steer(offset, gain)

    # -------------------------------------------------------------- решения
    def _front_block(self, scene: SceneAnalysis) -> Optional[str]:
        """Что мешает идти прямо: 'wall', 'ledge' или None (с гистерезисом)."""
        cfg = self.cfg
        engaged = self._block is not None
        rel = cfg.block_release
        wall_hi = cfg.wall_engage_prox if not engaged or self._block != "wall" else cfg.wall_engage_prox * rel
        ledge_hi = cfg.ledge_engage_prox if not engaged or self._block != "ledge" else cfg.ledge_engage_prox * rel
        # Стена приоритетнее: она требует не обхода, а карабканья.
        if scene.wall and scene.wall_prox >= wall_hi:
            return "wall"
        if scene.ledge and scene.ledge_prox >= ledge_hi:
            return "ledge"
        if scene.free_forward < cfg.front_blocked_free:
            return "ledge" if scene.ledge else None
        return None

    def decide(self, scene: SceneAnalysis, stamina: float = 1.0,
               goal_bearing: float = 0.0, now: Optional[float] = None) -> ActionFrame:
        self.tick(now)
        self._dt()
        self.goal_bearing = wrap_pi(float(goal_bearing))
        cfg = self.cfg
        keys = {k: False for k in ACTIONS}
        block = self._front_block(scene)
        self._block = block

        # --- 0) Уже падаем: раскрываем планер.
        #        В Genshin Space в полёте ОТКРЫВАЕТ планер, а повторный Space его
        #        закрывает. Значит жать надо один раз, но с паузой: если планер
        #        закрылся сам (кончилась выносливость) или мы ещё падаем после
        #        прыжка, нажатие повторяется не чаще glide_period_s.
        if scene.falling:
            self.dodge = Dodge()
            self._floor_hits = 0
            self._was_airborne = True
            self._switch(State.GLIDE, "нет опоры под ногами")
            if (stamina >= cfg.stamina_glide_min and self._glide_due
                    and self._now - self._last_glide >= cfg.glide_period_s):
                keys["glide"] = True
                self._glide_due = False
                self._last_glide = self._now
            keys["forward"] = True
            yaw = self._steer_to_goal(goal_bearing, 0.5)
            return ActionFrame(State.GLIDE, keys, yaw, "падение — раскрываем планер",
                               "Space: планер; держим курс, чтобы дотянуть")

        # Опора под ногами снова есть. Два-три таких кадра подряд — считаем, что
        # приземлились, и разрешаем следующее раскрытие планера.
        self._floor_hits += 1
        if self._floor_hits >= 2 and self._was_airborne:
            self._was_airborne = False
            self._glide_due = True
            self._glide_open = False
            self._flight_started = False

        # --- 1) Впереди стена: подходим вплотную и карабкаемся.
        if block == "wall":
            # Пока стена ещё далеко (видна, но не прижала) — просто подходим:
            # в Genshin карабканье включается вплотную, и «ползти» заранее вредно.
            if 0.0 < scene.wall_prox < cfg.walk_into_wall_prox and \
                    scene.free_forward >= cfg.too_close_ground:
                self._switch(State.WALK, "подходим к стене")
                keys["forward"] = True
                return ActionFrame(State.WALK, keys, self._steer_to_goal(goal_bearing, 1.0),
                                   "подходим к стене")
            if stamina >= cfg.stamina_climb_min:
                self._commit = None
                self._switch(State.CLIMB, "стена впереди")
                keys["forward"] = True
                period = cfg.climb_boost_period_s if stamina > 0.5 else cfg.pulse_period_s
                if self._now - self._last_jump >= period:
                    keys["jump"] = True
                    self._last_jump = self._now
                return ActionFrame(State.CLIMB, keys, 0.0, "стена — карабкаемся",
                                   "Space импульсами: подъём по скале")
            # Выносливости нет: обходим стену сбоку, возвращаясь на курс после.
            side = self._sticky_side(scene)
            self._switch(State.AVOID, "стена, мало выносливости")
            keys["forward"] = True
            return ActionFrame(State.AVOID, keys,
                               self._steer_offset(side * cfg.avoid_bearing, 1.0),
                               "обход стены", "на карабканье не хватает выносливости")

        # --- 2) Впереди край/провал. Лестница эскалации, как у живого игрока:
        #        обход влево -> обход вправо -> прыжок и планер через провал.
        if block == "ledge":
            elapsed = self._now - self._probe_since
            if elapsed > cfg.probe_time_s and self._probe < 2:
                self._probe += 1
                self._probe_since = self._now
                self._probe_dir = -self._probe_dir
            if self._probe == 0 or self._probe == 1:
                if elapsed > cfg.probe_time_s:
                    self._probe_since = self._now
                side = self._probe_dir
                # Край вплотную и сбоку тоже пусто: сначала шаг назад.
                if scene.ledge_prox >= cfg.ledge_critical_prox and \
                        scene.free_forward < cfg.edge_forward_min:
                    self._switch(State.BACK_OFF, "край вплотную")
                    keys["back"] = True
                    return ActionFrame(State.BACK_OFF, keys,
                                       self._steer_offset(side * cfg.backoff_bearing * 0.5, 1.0),
                                       "край обрыва — назад",
                                       "отходим от края перед обходом")
                self._switch(State.AVOID, "обход вдоль края")
                keys["forward"] = True
                keys["left" if side < 0 else "right"] = True
                note = (f"проба {self._probe + 1}/2: обход {'влево' if side < 0 else 'вправо'}, "
                        f"край {scene.ledge_prox:.2f}")
                return ActionFrame(State.AVOID, keys,
                                   self._steer_offset(side * cfg.avoid_bearing, 1.0),
                                   "обрыв — обходим вдоль края", note)
            # Оба обхода не помогли: единственный путь — перелёт на планере.
            # Здесь делаем только прыжок: как только под ногами пропадёт опора,
            # сработает ветка падения и раскроет планер ровно один раз.
            if stamina >= cfg.stamina_glide_min:
                self._switch(State.GLIDE, "перелёт через провал")
                keys["forward"] = True
                if not self._flight_started:
                    keys["jump"] = True
                    self._flight_started = True
                    self._last_jump = self._now
                return ActionFrame(State.GLIDE, keys, self._steer_to_goal(goal_bearing, 0.7),
                                   "провал — перелёт на планере",
                                   "Space: прыжок, затем планер")
            # Выносливости нет: шагаем назад и ждём восстановления.
            self._switch(State.BACK_OFF, "нет выносливости на перелёт")
            keys["back"] = True
            return ActionFrame(State.BACK_OFF, keys, 0.0,
                               "ждём выносливость", "на планер пока не хватает сил")

        # --- 3) Путь свободен: идём к цели, при запасе свободы — бежим.
        self._probe = 0
        self._probe_since = self._now
        yaw = self._steer_to_goal(goal_bearing, gain=1.0)
        keys["forward"] = True
        if (scene.free_forward >= cfg.free_good and stamina > cfg.stamina_low
                and abs(self.heading) < cfg.sprint_heading_max):
            self._switch(State.SPRINT, "свободно и есть запас")
            keys["sprint"] = True
            return ActionFrame(State.SPRINT, keys, yaw, "свободно — бежим к цели")
        self._switch(State.WALK, "свободный коридор")
        return ActionFrame(State.WALK, keys, yaw, "идём к цели")

    # ------------------------------------------------------------------ обход
    def _sticky_side(self, scene: SceneAnalysis) -> int:
        """Сторона обхода, залипающая на несколько секунд (против дребезга)."""
        if self._side != 0 and self._now < self._side_until:
            return self._side
        sd = scene.sector_dist
        n = len(sd) if sd else 0
        if n >= 3:
            c = n // 2
            left = max(sd[:c])
            right = max(sd[c + 1:])
            # При равенстве сохраняем прошлую сторону.
            if abs(left - right) < 0.05 and self._side != 0:
                side = self._side
            else:
                side = -1 if left >= right else 1
        else:
            side = self._side or 1
        self._side = side
        self._side_until = self._now + self.cfg.side_hold_s
        return side

    def _edge_side(self, scene: SceneAnalysis) -> int:
        """Сторона обхода края: залипает на время, чтобы не дёргаться."""
        if self._edge_dir != 0 and self._now < self._edge_until:
            return self._edge_dir
        sd = scene.sector_dist
        n = len(sd) if sd else 0
        if n >= 3:
            c = n // 2
            left = max(sd[:c])
            right = max(sd[c + 1:])
            side = -1 if left >= right else 1
        else:
            side = self.dodge.dir or 1
        self._edge_dir = side
        self._edge_until = self._now + self.cfg.edge_follow_s
        return side

    def _dodge_side(self, scene: SceneAnalysis, prefer: Optional[int]) -> int:
        """Сторона обхода: где свободнее. Гистерезис — против дёрганья."""
        if prefer in (-1, 1):
            return prefer
        sd = scene.sector_dist
        if not sd:
            return 1 if self.dodge.dir >= 0 else -1
        n = len(sd)
        c = n // 2
        left = max(sd[:c]) if c > 0 else 0.0
        right = max(sd[c + 1:]) if c + 1 < n else 0.0
        center = sd[c]
        if left - center < self.cfg.lateral_margin and right - center < self.cfg.lateral_margin:
            # Оба края не лучше центра: уходим в сторону, где уже были (стабильность).
            return self.dodge.dir or (1 if self.goal_bearing <= 0 else -1)
        return -1 if left >= right else 1

    def _start_dodge(self, side: int, reason: str) -> None:
        self.dodge = Dodge(dir=int(side) or 1, until=self._now + self.cfg.dodge_time_s,
                           angle=0.0, reason=reason)

    def reset(self) -> None:
        self._glide_open = False
        self._flight_started = False
        self._glide_due = True
        self._floor_hits = 0
        self._was_airborne = False
        self._probe = 0
        self._probe_since = 0.0
        self._side = 0
        self._side_until = 0.0
        self._block = None
        self._edge_dir = 0
        self._edge_until = 0.0
        self.state = State.WALK
        self.state_since = 0.0
        self.goal_bearing = 0.0
        self.dodge = Dodge()
        self._last_now = None
        self.history.clear()


def stamina_from_bar(bar_fill: float) -> float:
    """Выносливость 0..1 из заполнения полосы (если детектор её нашёл)."""
    return float(max(0.0, min(1.0, bar_fill)))
