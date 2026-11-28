"""Сквозные тесты: агент проходит полосу препятствий в замкнутом цикле.

Это главная проверка качества логики: восприятие и решение работают вместе,
на рендере, который ведёт себя как depth-сеть (та же шкала 1/z).
"""
import numpy as np
import pytest

from parkour.config import ParkourConfig
from parkour.runner import EpisodeRunner
from parkour.sim import (SCENARIOS, CharState, Simulator, render_disparity, world_chasm,
                         world_wall)


@pytest.fixture
def runner():
    return EpisodeRunner(ParkourConfig())


def run(name, max_steps=900):
    r = EpisodeRunner(ParkourConfig())
    return r.run(SCENARIOS[name](), max_steps=max_steps)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_reaches_goal(name):
    res = run(name)
    assert res.success, f"{name}: не дошёл до цели ({res.summary()})"
    assert res.alive
    assert res.falls == 0, f"{name}: агент упал"


def test_wall_requires_climbing():
    res = run("wall")
    assert res.climbed_s > 0.5, "стену нельзя пройти без карабканья"


def test_chasm_or_glide_requires_gliding_or_avoidance():
    res = run("fall")
    assert res.glided_s > 1.0, "провал нельзя пройти без планера"


def test_no_episode_deaths():
    for name in SCENARIOS:
        res = run(name)
        assert res.alive, f"{name}: персонаж погиб ({res.death})"


def test_runner_is_deterministic():
    a = run("course")
    b = run("course")
    assert (a.final_pos, a.steps) == (b.final_pos, b.steps)


def test_simulator_void_kills():
    """Страховка: если упасть в пропасть, эпизод завершается смертью."""
    w = world_chasm()
    sim = Simulator(w, state=CharState(x=0.0, z=25.0, y=0.0), dt=0.1)
    for _ in range(200):
        sim.step({"forward": True}, 0.0)
        if not sim.s.alive:
            break
    assert not sim.s.alive
    assert "пропасть" in sim.events.death


def test_render_scales_with_distance():
    """Ближняя стена должна давать большую обратную глубину, чем дальняя."""
    near = render_disparity(world_wall(), [0, 0, 19.0], 0.0)
    far = render_disparity(world_wall(), [0, 0, 5.0], 0.0)
    assert float(near.mean()) > float(far.mean())


def test_falling_always_opens_glider():
    """Инвариант по всем сценариям: падаем -> всегда GLIDE и никаких прыжков."""
    from parkour.config import ParkourConfig
    from parkour.depth import SlowNormalizer
    from parkour.perception import Perception
    from parkour.policy import ParkourBrain, State
    from parkour.sim import SCENARIOS, CharState, Simulator, render_disparity
    import math

    cfg = ParkourConfig()
    violations = []
    for name, make in SCENARIOS.items():
        world = make()
        gh, _ = world.height(np.array([world.start[0]], np.float32),
                             np.array([world.start[1]], np.float32))
        sim = Simulator(world, state=CharState(x=world.start[0], z=world.start[1],
                                               y=float(gh[0])), dt=0.1)
        perception, brain = Perception(cfg.perception), ParkourBrain(cfg.policy)
        norm = SlowNormalizer()
        gx, gz = world.goal
        for step in range(600):
            prox = norm(render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160)))
            scene = perception.analyze(prox=prox)
            bearing = math.atan2(gx - sim.s.x, gz - sim.s.z) - sim.s.yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            action = brain.decide(scene, stamina=sim.s.stamina,
                                 goal_bearing=bearing, now=step * 0.1)
            if scene.falling:
                if action.state is not State.GLIDE:
                    violations.append((name, step, "не GLIDE в полёте"))
                if action.keys["jump"]:
                    violations.append((name, step, "прыжок в воздухе"))
            sim.step(action.keys, action.yaw)
            if not sim.s.alive or sim.s.y < -20:
                break
            if math.hypot(sim.s.x - gx, sim.s.z - gz) < 4:
                break
    assert not violations, f"нарушения безопасности: {violations[:5]}"


def test_dropped_in_air_survives_with_glider():
    """Страховочный сценарий: персонаж уже в воздухе над пропастью.

    Раньше агент этого не понимал и «шёл» вперёд до земли. Теперь должен
    распознать полёт по сырой глубине, раскрыть планер и остаться жив.
    """
    from parkour.config import ParkourConfig
    from parkour.depth import SlowNormalizer
    from parkour.perception import Perception
    from parkour.policy import ParkourBrain, State
    from parkour.sim import SCENARIOS, CharState, Simulator, render_disparity

    cfg = ParkourConfig()
    world = SCENARIOS["fall"]()
    sim = Simulator(world, state=CharState(x=0.0, z=10.0, y=14.0, vy=0.0, airborne=True), dt=0.1)
    perception, brain, norm = Perception(cfg.perception), ParkourBrain(cfg.policy), SlowNormalizer()
    saw_glide = False
    for step in range(40):
        d = render_disparity(world, sim.s.pos(), sim.s.yaw, hw=(96, 160))
        scene = perception.analyze(prox=norm(d), raw_scale=float(np.percentile(d, 98)))
        action = brain.decide(scene, stamina=sim.s.stamina, goal_bearing=0.0, now=step * 0.1)
        if scene.falling:
            assert action.state is State.GLIDE
            saw_glide = saw_glide or action.keys["glide"]
        sim.step(action.keys, action.yaw)
    assert saw_glide, "планер так и не раскрылся"
    assert sim.s.alive, f"персонаж разбился: {sim.events.death}"
