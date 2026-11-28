"""Тесты восприятия: признаки должны отделять стену, обрыв и открытое поле.

Сцены рендерятся синтетическим миром (аналитический ray-AABB), поэтому тесты
детерминированы и не требуют ни игры, ни ONNX-модели.
"""
import numpy as np
import pytest

from parkour.config import PerceptionConfig
from parkour.depth import SlowNormalizer
from parkour.perception import Perception
from parkour.sim import (Box, Chasm, World, render_disparity, world_chasm, world_wall)


def prox_of(world, pos, yaw=0.0, hw=(96, 160), normalizer=None):
    d = render_disparity(world, np.array(pos, dtype=np.float32), yaw, hw=hw)
    return (normalizer or SlowNormalizer())(d)


def analyze(world, pos, yaw=0.0, cfg=None):
    return Perception(cfg or PerceptionConfig()).analyze(prox=prox_of(world, pos, yaw))


class TestWall:
    def test_no_wall_in_open_field(self):
        sc = analyze(World(name="flat"), (0, 0, 0))
        assert not sc.wall
        assert not sc.ledge

    def test_wall_detected_up_close(self):
        sc = analyze(world_wall(), (0, 0, 19.4))
        assert sc.wall, f"стена в 0.6 м не распознана: {sc.as_dict()}"
        assert sc.wall_prox > 0.3

    @pytest.mark.parametrize("z", [16.0, 18.0])
    def test_wall_not_yet_close(self, z):
        """Издалека стена не должна включать карабканье."""
        sc = analyze(world_wall(), (0, 0, z))
        assert not (sc.wall and sc.wall_prox > 0.5)

    @pytest.mark.parametrize("z", [20.0, 22.0])
    def test_wall_right_in_face(self, z):
        """Уперевшись в стену, кадр целиком занят ею — это по-прежнему стена.

        Регресс: раньше такой кадр отбраковывался как «слишком близко сверху»
        и превращался в обрыв.
        """
        sc = analyze(world_wall(), (0, 0, z))
        assert sc.wall and not sc.ledge, f"стена вплотную: {sc.as_dict()}"
        assert sc.wall_prox > 0.8

    def test_high_plateau_is_not_wall(self):
        """Площадка под ногами занимает кадр целиком, но это пол, а не стена."""
        world = World(name="plateau", boxes=[Box(-60, 60, -60, 60, top=18.0)])
        sc = analyze(world, (0, 18.0, 0.0))
        assert not sc.wall, f"плато принято за стену: {sc.as_dict()}"
        assert not sc.ledge

    def test_floor_seen_from_above_is_not_wall(self):
        """Смотрим вниз на склон под собой: профиль падает вниз — не стена."""
        world = World(name="slope", boxes=[Box(-60, 60, 40, 90, top=0.0)],
                      chasms=[Chasm(-60, 60, -10, 40, floor=-80.0)])
        sc = analyze(world, (0, 30.0, 10.0))
        assert not sc.wall, f"пол под ногами принят за стену: {sc.as_dict()}"


class TestLedge:
    def test_ledge_detected_approaching(self):
        sc = analyze(world_chasm(), (0, 0, 16.0))
        assert sc.ledge, f"край обрыва не найден: {sc.as_dict()}"

    def test_ledge_prox_grows_when_closer(self):
        far = analyze(world_chasm(), (0, 0, 17.0))
        near = analyze(world_chasm(), (0, 0, 21.0))
        assert near.ledge_prox > far.ledge_prox
        assert near.free_forward < far.free_forward

    def test_narrow_ledge_walkable_straight(self):
        """На узком карнизе прямо идти можно: высокий free_forward и руль по нулям."""
        from parkour.policy import ParkourBrain, State
        world = World(name="ledge", boxes=[Box(-2.2, 2.2, -5, 70, top=0.0)],
                      chasms=[Chasm(-40, -2.2, -5, 75), Chasm(2.2, 40, -5, 75)])
        sc = analyze(world, (0, 0, 10))
        assert sc.free_forward > 0.6, f"карниз считается перекрытым: {sc.as_dict()}"
        action = ParkourBrain().decide(sc, now=0.0)
        assert action.state in (State.WALK, State.SPRINT)
        assert not action.keys["left"] and not action.keys["right"]

    def test_void_right_next_to_ledge_is_visible(self):
        """Если обрыв справа в 1.5 м — поворот вправо должен это показать."""
        world = World(name="ledge", boxes=[Box(-2.2, 2.2, -5, 70, top=0.0)],
                      chasms=[Chasm(-40, -2.2, -5, 75), Chasm(2.2, 40, -5, 75)])
        sc = analyze(world, (0, 0, 10), yaw=np.pi / 2)
        assert sc.ledge


class TestHudAndCharacter:
    @pytest.mark.parametrize("hud_box", [
        (0.00, 0.00, 0.23, 0.17),   # миникарта
        (0.82, 0.00, 1.00, 0.23),   # отряд
        (0.72, 0.82, 1.00, 1.00),   # способности
    ])
    def test_hud_is_ignored(self, hud_box):
        """Яркий HUD в своих зонах не должен менять признаки сцены."""
        cfg = PerceptionConfig()
        base = np.full((96, 160), 0.05, dtype=np.float32)
        base[40:60, 40:120] = 0.7      # пол/склон впереди
        dirty = base.copy()
        h, w = dirty.shape
        x0, y0, x1, y1 = hud_box
        dirty[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1.0
        a = Perception(cfg).analyze(prox=base)
        b = Perception(cfg).analyze(prox=dirty)
        assert b.wall == a.wall
        assert b.ledge == a.ledge
        assert abs(b.free_forward - a.free_forward) < 1e-6


class TestSlidingWindowNormalizer:
    def test_window_does_not_collapse(self):
        n = SlowNormalizer(min_width_ratio=0.35)
        flat = np.full((32, 32), 0.01, dtype=np.float32)   # вся сцена — даль
        n(flat)
        w0 = n.hi - n.lo
        for _ in range(50):
            n(flat)
        assert (n.hi - n.lo) >= 0.3 * w0

    def test_global_far_shift_is_visible(self):
        """Главный смысл медленного окна: общий сдвиг «всё далеко» виден."""
        n = SlowNormalizer(ema=0.02)
        near = np.full((32, 32), 0.5, dtype=np.float32)
        for _ in range(30):
            n(near)
        far = n(np.full((32, 32), 0.01, dtype=np.float32))
        assert float(far.mean()) < 0.2


class TestAirborne:
    """Падение над пропастью: главный признак — сырая близость кадра."""

    def _frame(self, world, pos, normalizer=None):
        d = render_disparity(world, np.array(pos, dtype=np.float32), 0.0, hw=(96, 160))
        return (normalizer or SlowNormalizer())(d), float(np.percentile(d, 98))

    def test_falling_detected_from_first_frame(self):
        """В полёте нет ни земли, ни края — только даль: это падение."""
        from parkour.sim import world_fall
        prox, raw = self._frame(world_fall(), (0, 14.0, 10.0))
        sc = Perception(PerceptionConfig()).analyze(prox=prox, raw_scale=raw)
        assert sc.falling, f"полёт над пропастью не распознан: {sc.as_dict()}"

    def test_standing_on_ground_is_not_falling(self):
        for world, pos in ((World(name="flat"), (0, 0, 0)),):
            prox, raw = self._frame(world, pos)
            assert not Perception(PerceptionConfig()).analyze(prox=prox, raw_scale=raw).falling

    def test_standing_on_high_plateau_is_not_falling(self):
        """Высокая площадка: даль вокруг есть, но пол под ногами близко."""
        from parkour.sim import world_fall
        prox, raw = self._frame(world_fall(), (0, 18.0, 0.0))
        assert not Perception(PerceptionConfig()).analyze(prox=prox, raw_scale=raw).falling

    def test_falling_wins_over_wall(self):
        """В полёте склон внизу не должен считаться стеной для карабканья."""
        from parkour.sim import world_fall
        perc = Perception(PerceptionConfig())
        prox, raw = self._frame(world_fall(), (0, 12.0, 12.0))
        sc = perc.analyze(prox=prox, raw_scale=raw)
        assert sc.falling
        assert not sc.wall and sc.wall_prox == 0.0

    def test_air_threshold_scales_with_net(self):
        """Порог «воздуха» выражен в 1/z, а не в пикселях: проверяем обе стороны."""
        cfg = PerceptionConfig()
        perc = Perception(cfg)
        prox = np.zeros((96, 160), dtype=np.float32)
        assert perc.analyze(prox=prox, raw_scale=cfg.air_abs_raw * 0.5).falling
        perc.reset()
        assert not perc.analyze(prox=prox, raw_scale=cfg.air_abs_raw * 4.0).falling
