"""Синтетический мир для проверки логики паркура в замкнутом цикле.

Зачем: настоящая игра недоступна в CI, а «на глаз» пороги не настроить. Здесь
есть heightfield-мир из блоков и пропастей, аналитический рендер обратной
глубины (луч против AABB, без маршевых шагов — быстро и точно) и простая
физика персонажа в духе Genshin: ходьба, спринт, карабканье, падение, планер,
выносливость.

Рендер выдаёт 1/z, как настоящая depth-сеть до нормировки, поэтому восприятие
проверяется ровно в той же шкале, что и в игре.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

FAR = 150.0


# --------------------------------------------------------------------------- мир
@dataclass
class Box:
    """Блок (плато/скала): AABB от земли до top."""
    x0: float
    x1: float
    z0: float
    z1: float
    top: float
    base: float = -100.0

    def slab(self):
        return np.array([self.x0, self.base, self.z0]), np.array([self.x1, self.top, self.z1])


@dataclass
class Chasm:
    """Пропасть: земли нет (если над ней нет блока)."""
    x0: float
    x1: float
    z0: float
    z1: float
    floor: float = -80.0


@dataclass
class World:
    boxes: List[Box] = field(default_factory=list)
    chasms: List[Chasm] = field(default_factory=list)
    name: str = "world"
    base: float = 0.0
    # Глубина «пустоты» вне блоков. В настоящей игре даль скрывается туманом,
    # поэтому по умолчанию это маленькая обратная глубина (≈ дальней туман).
    void_depth: float = 45.0
    goal: Tuple[float, float] = (0.0, 40.0)
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, z, yaw

    # --------------------------------------------------------------- геометрия
    def height(self, x: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)
        h = np.zeros(np.broadcast(x, z).shape, dtype=np.float32)
        solid = np.ones_like(h, dtype=bool)
        # Базовая земля простирается дальше блоков, но обычно меньше max_dist,
        # чтобы горизонт читался как реальное расстояние.
        h[...] = self.base
        for c in self.chasms:
            m = (x >= c.x0) & (x <= c.x1) & (z >= c.z0) & (z <= c.z1)
            h = np.where(m, np.float32(c.floor), h)
            solid = np.where(m, False, solid)
        for b in self.boxes:
            m = (x >= b.x0) & (x <= b.x1) & (z >= b.z0) & (z <= b.z1)
            h = np.where(m, np.float32(b.top), h)
            solid = np.where(m, True, solid)
        return h, solid

    def ground_boxes(self) -> List[Box]:
        """Блоки, у которых верх — ходибельная поверхность (top > floor)."""
        return [b for b in self.boxes if b.top > -50]


# ------------------------------------------------------------------------ камера
@dataclass
class Camera:
    fov_y_deg: float = 60.0
    dist: float = 4.2
    height: float = 2.2
    look_ahead: float = 3.0
    look_height: float = 1.2
    pitch: float = 0.0     # дополнительный наклон вверх (рад)


def camera_basis(pos: np.ndarray, yaw: float, cam: Camera):
    fwd = np.array([math.sin(yaw), 0.0, math.cos(yaw)], dtype=np.float32)
    eye = pos + np.array([-fwd[0] * cam.dist, cam.height, -fwd[2] * cam.dist], dtype=np.float32)
    target = pos + np.array([fwd[0] * cam.look_ahead, cam.look_height, fwd[2] * cam.look_ahead],
                            dtype=np.float32)
    if cam.pitch:
        target[1] = pos[1] + cam.look_height + math.tan(cam.pitch) * cam.look_ahead
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-6)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    r = np.cross(f, world_up)
    r /= max(np.linalg.norm(r), 1e-6)
    u = np.cross(r, f)
    return eye, r, f, u


def _ray_aabb(origin: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """t входа луча в AABB (inf, если промах). Векторизовано по лучам."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tnear = np.nanmax(tmin, axis=1)
    tfar = np.nanmin(tmax, axis=1)
    hit = (tfar >= np.maximum(tnear, 0.0)) & (tnear > 0.0)
    return np.where(hit, tnear, np.inf).astype(np.float32)


def render_disparity(world: World, pos: Sequence[float], yaw: float,
                     hw: Tuple[int, int] = (96, 160), cam: Optional[Camera] = None,
                     max_dist: float = FAR, ray_offset: Optional[Tuple[float, float]] = None,
                     ray_noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Карта 1/z (без нормировки). Пустота -> 1/max_dist.

    Смещение камеры относительно «игрока» эмулирует свободную камеру Genshin
    (мышь двигает взгляд, персонаж при этом может стоять), а шум — дрожание
    монокулярной сети. Всё опционально: в тестах логики удобно без этого.
    """
    cam = cam or Camera()
    pos = np.asarray(pos, dtype=np.float32)
    eye, r, f, u = camera_basis(pos, yaw, cam)
    h, w = hw
    tan_y = math.tan(math.radians(cam.fov_y_deg) / 2.0)
    tan_x = tan_y * (w / h)
    ys = (1.0 - 2.0 * (np.arange(h, dtype=np.float32) + 0.5) / h)[:, None]
    xs = (2.0 * (np.arange(w, dtype=np.float32) + 0.5) / w - 1.0)[None, :]
    dirs = (f[None, None, :] + ys[..., None] * tan_y * u[None, None, :]
            + xs[..., None] * tan_x * r[None, None, :])
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs.reshape(-1, 3).astype(np.float32)
    if ray_offset is not None:
        # Параллакс: камера сдвинута вбок/вверх относительно точки персонажа.
        side, up = float(ray_offset[0]), float(ray_offset[1])
        eye = eye + r * side + np.array([0.0, up, 0.0], dtype=np.float32)
    if ray_noise > 0:
        rng = np.random.default_rng(seed)
        dirs = dirs + rng.normal(0.0, ray_noise, dirs.shape).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    # 1) Блоки (AABB).
    best = np.full(dirs.shape[0], np.inf, dtype=np.float32)
    for b in world.boxes:
        lo, hi = b.slab()
        t = _ray_aabb(eye, dirs, lo, hi)
        best = np.minimum(best, t)
    # 2) Земля y=0 и дно пропастей — плоскости. Над пропастью земли нет:
    #    сначала считаем попадание в y=0, затем отменяем его внутри пропастей.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = np.where(dirs[:, 1] < -1e-6, -eye[1] / dirs[:, 1], np.inf)
    if world.chasms:
        px = eye[0] + dirs[:, 0] * np.nan_to_num(t_ground, posinf=0.0)
        pz = eye[2] + dirs[:, 2] * np.nan_to_num(t_ground, posinf=0.0)
        over = np.zeros(dirs.shape[0], dtype=bool)
        for c in world.chasms:
            over |= (px >= c.x0) & (px <= c.x1) & (pz >= c.z0) & (pz <= c.z1)
        t_ground = np.where(over, np.inf, t_ground)
        # Дно пропасти — тоже плоскость (обычно очень глубоко, можно не считать).
        for c in world.chasms:
            with np.errstate(divide="ignore", invalid="ignore"):
                tf = np.where(dirs[:, 1] < -1e-6, (c.floor - eye[1]) / dirs[:, 1], np.inf)
            fx = eye[0] + dirs[:, 0] * np.nan_to_num(tf, posinf=0.0)
            fz = eye[2] + dirs[:, 2] * np.nan_to_num(tf, posinf=0.0)
            inside = (fx >= c.x0) & (fx <= c.x1) & (fz >= c.z0) & (fz <= c.z1)
            t_ground = np.where(inside & over, np.minimum(t_ground, tf), t_ground)
    best = np.minimum(best, t_ground)

    best = np.minimum(np.nan_to_num(best, nan=np.inf, posinf=np.inf), max_dist)
    # Куда не попали — «туман»/даль: фиксированная маленькая обратная глубина.
    # Иначе горизонтальная плоскость даёт попадания в сотни метров, и «рядом
    # только даль» становится неотличимо от «под ногами земля».
    void_disp = float(1.0 / max(world.void_depth, 1.0))
    finite = np.isfinite(best)
    disp = np.where(finite, 1.0 / np.maximum(np.where(finite, best, 1.0), 0.3), void_disp)
    return disp.reshape(h, w).astype(np.float32)


def normalize_disparity(disp: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(disp, p_low), np.percentile(disp, p_high)
    if hi - lo < 1e-9:
        return np.zeros_like(disp)
    return np.clip((disp - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------- персонаж
@dataclass
class CharState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    vy: float = 0.0
    stamina: float = 1.0
    climbing: bool = False
    gliding: bool = False
    airborne: bool = False
    alive: bool = True

    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)


@dataclass
class SimEvents:
    steps: int = 0
    falls: int = 0
    bump_wall: int = 0
    climbed_s: float = 0.0
    glided_s: float = 0.0
    sprint_s: float = 0.0
    distance: float = 0.0
    jumps: int = 0
    death: str = ""


class Simulator:
    """Простая физика в духе Genshin."""

    SPEED = 3.0
    SPRINT = 5.6
    CLIMB = 2.6
    GLIDE_FWD = 5.0
    GLIDE_DOWN = -1.8
    GRAVITY = 14.0
    STEP_UP = 0.45
    TURN = 2.4
    VOID_Y = -25.0

    def __init__(self, world: World, state: Optional[CharState] = None, dt: float = 0.1):
        self.world = world
        self.dt = dt
        x, z, yaw = world.start
        gh, _ = world.height(np.array([x], np.float32), np.array([z], np.float32))
        self.s = state or CharState(x=x, z=z, y=float(gh[0]), yaw=yaw)
        self.events = SimEvents()
        self.t = 0.0

    PROBE = 0.7   # насколько вперёд «щупаем» стену при карабканье

    def _ground(self, x: float, z: float) -> Tuple[float, bool]:
        gh, solid = self.world.height(np.array([x], np.float32), np.array([z], np.float32))
        return float(gh[0]), bool(solid[0])

    def _wall_ahead(self, s: "CharState") -> Tuple[float, bool]:
        """Высота препятствия прямо перед персонажем (для карабканья)."""
        px = s.x + math.sin(s.yaw) * self.PROBE
        pz = s.z + math.cos(s.yaw) * self.PROBE
        return self._ground(px, pz)

    # ------------------------------------------------------------------ step
    def step(self, keys: dict, yaw_cmd: float = 0.0) -> None:
        s, dt = self.s, self.dt
        self.t += dt
        self.events.steps += 1
        if not s.alive:
            return
        if yaw_cmd:
            s.yaw += float(np.clip(yaw_cmd, -1, 1)) * self.TURN * dt
            s.yaw = (s.yaw + math.pi) % (2 * math.pi) - math.pi

        fwd = np.array([math.sin(s.yaw), math.cos(s.yaw)], dtype=np.float32)
        right = np.array([math.cos(s.yaw), -math.sin(s.yaw)], dtype=np.float32)
        move = np.zeros(2, dtype=np.float32)
        if keys.get("forward"):
            move += fwd
        if keys.get("back"):
            move -= fwd * 0.7
        if keys.get("left"):
            move -= right
        if keys.get("right"):
            move += right
        n = float(np.linalg.norm(move))
        if n > 1e-6:
            move /= n

        # --- полёт / планирование
        if s.airborne or s.gliding:
            if keys.get("glide") and not s.gliding and s.stamina > 0.02:
                s.gliding = True
                s.vy = self.GLIDE_DOWN
            if s.gliding:
                s.stamina = max(0.0, s.stamina - 0.05 * dt)
                self.events.glided_s += dt
                s.vy = self.GLIDE_DOWN
                speed = self.GLIDE_FWD
            else:
                s.vy -= self.GRAVITY * dt
                speed = 2.0
            s.x += move[0] * speed * dt
            s.z += move[1] * speed * dt
            s.y += s.vy * dt
            gh, solid = self._ground(s.x, s.z)
            if solid and s.y <= gh and s.vy <= 0:
                impact = -s.vy
                s.y = gh
                s.vy = 0.0
                s.airborne = s.gliding = False
                if impact > 22.0:
                    s.alive = False
                    self.events.death = "падение с высоты"
                    self.events.falls += 1
            if s.y <= self.VOID_Y:
                s.alive = False
                self.events.death = "провал в пропасть"
                self.events.falls += 1
            return

        # --- карабканье
        if s.climbing:
            probe_h, probe_solid = self._wall_ahead(s)
            if not keys.get("forward") or s.stamina <= 0.0 or not probe_solid:
                s.climbing = False
            else:
                s.stamina = max(0.0, s.stamina - 0.10 * dt)
                self.events.climbed_s += dt
                s.y += self.CLIMB * (1.5 if keys.get("jump") else 1.0) * dt
                if s.y >= probe_h - 0.05:
                    # Забрались на верх стены: выходим на плато.
                    s.y = max(s.y, probe_h)
                    s.climbing = False
                    s.x += math.sin(s.yaw) * 0.8
                    s.z += math.cos(s.yaw) * 0.8
                return

        # --- прыжок с земли
        if keys.get("jump") and not s.airborne:
            s.airborne = True
            s.vy = 5.0
            self.events.jumps += 1
            return

        # --- наземное движение
        speed = self.SPEED
        if keys.get("sprint") and s.stamina > 0.02:
            speed = self.SPRINT
            s.stamina = max(0.0, s.stamina - 0.05 * dt)
            self.events.sprint_s += dt
        else:
            s.stamina = min(1.0, s.stamina + 0.12 * dt)

        if n > 1e-6:
            # Движение по горизонтали с учётом рельефа: если под ногами есть
            # опора, персонаж идёт по склону (как в Genshin), а не падает.
            nx = s.x + move[0] * speed * dt
            nz = s.z + move[1] * speed * dt
            probe_h, probe_solid = self._wall_ahead(s)
            gh, solid = self._ground(nx, nz)
            facing_wall = probe_solid and probe_h > s.y + self.STEP_UP
            if solid and gh > s.y + self.STEP_UP or facing_wall:
                # Упёрлись в стену: в Genshin это автоматически включает карабканье.
                self.events.bump_wall += 1
                s.climbing = True
                return
            if not solid:
                # Шаг в пропасть: сначала прыжок, потом падение.
                if keys.get("jump"):
                    s.vy = 5.0
                    self.events.jumps += 1
                else:
                    s.vy = 0.0
                s.x, s.z = nx, nz
                s.airborne = True
                return
            moved = math.hypot(nx - s.x, nz - s.z)
            self.events.distance += moved
            s.x, s.z = nx, nz
            if gh > s.y - 0.6:
                # идём по склону вверх/вниз
                s.y = gh if gh > s.y else max(gh, s.y - 8.0 * dt)
            else:
                # отвесный обрыв впереди — шагаем с него и падаем
                s.airborne = True
                s.vy = 0.0


# --------------------------------------------------------------------- сценарии
def _ground_height(world: World, x: float, z: float) -> float:
    gh, _ = world.height(np.array([x], np.float32), np.array([z], np.float32))
    return float(gh[0])


def world_flat() -> World:
    return World(name="flat", base=0.0, goal=(0.0, 45.0), start=(0.0, 0.0, 0.0))


def world_wall() -> World:
    # Цель — на плато за стеной: успех = залезть и дойти.
    return World(name="wall", goal=(0.0, 27.0), start=(0.0, 0.0, 0.0),
                 boxes=[Box(-25, 25, 20, 30, top=7.0)])


def world_chasm() -> World:
    """Провал с дном-рекой: можно перелететь на планере и вылезти на другой берег."""
    return World(name="chasm", goal=(0.0, 55.0), start=(0.0, 0.0, 0.0),
                 chasms=[Chasm(-25, 25, 22, 34, floor=-6.0)])


def world_ledge() -> World:
    """Карниз с двух сторон: слева и справа пропасть, идти можно только вперёд."""
    return World(
        name="ledge",
        goal=(0.0, 70.0),
        start=(0.0, 0.0, 0.0),
        boxes=[Box(-2.2, 2.2, -5, 70, top=0.0)],
        chasms=[Chasm(-40, -2.2, -5, 75), Chasm(2.2, 40, -5, 75)],
    )


def world_course() -> World:
    """Полоса препятствий: стена -> пропасть -> плато (залезть) -> цель."""
    w = World(name="course", start=(0.0, 0.0, 0.0), goal=(0.0, 100.0),
              boxes=[Box(-3, 3, 20, 26, top=6.0), Box(-30, 30, 55, 100, top=5.0)],
              chasms=[Chasm(-40, 40, 38, 48)])
    return w


def world_glide() -> World:
    """Высокий утёс, внизу долина и цель: единственный путь — планер.

    Это проверка сценария «под ногами пропасть -> раскрываем планер»: прыжок с
    края и планирование на 25 м вперёд при перепаде 12 м.
    """
    return World(name="glide", goal=(0.0, 40.0), start=(0.0, 0.0, 0.0),
                 boxes=[Box(-14, 14, -3, 2, top=12.0), Box(-30, 30, 26, 90, top=0.0)],
                 chasms=[Chasm(-40, 40, 2, 26, floor=-80.0)])


def world_fall() -> World:
    """Полный провал под ногами: проверяем, что агент спасается планером."""
    return World(name="fall", goal=(0.0, 74.0), start=(0.0, 0.0, 0.0),
                 boxes=[Box(-6, 6, -3, 3, top=18.0), Box(-30, 30, 46, 90, top=0.0)],
                 chasms=[Chasm(-40, 40, 3, 46, floor=-80.0)])


SCENARIOS = {
    "flat": world_flat,
    "wall": world_wall,
    "chasm": world_chasm,
    "ledge": world_ledge,
    "course": world_course,
    "glide": world_glide,
    "fall": world_fall,
}
