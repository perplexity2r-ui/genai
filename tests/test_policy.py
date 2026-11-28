"""Тесты логики: правила должны быть безопасными на краю и решительными на стене."""
from parkour.perception import SceneAnalysis
from parkour.policy import ACTIONS, ParkourBrain, State


def scene(wall=False, wall_prox=0.0, ledge=False, ledge_prox=0.0, falling=False,
          free=0.7, sectors=None) -> SceneAnalysis:
    sectors = sectors or [0.7] * 9
    return SceneAnalysis(
        wall=wall, ledge=ledge, falling=falling, target_dir=0.0,
        wall_prox=wall_prox, ledge_prox=ledge_prox, ground=0.8,
        free_forward=free, sector_dist=list(sectors), confidence=0.9,
    )


class TestSafety:
    def test_falling_opens_glider(self):
        b = ParkourBrain()
        a = b.decide(scene(falling=True), now=0.0)
        assert a.state is State.GLIDE
        assert a.keys["glide"]

    def test_glider_opened_once_per_flight(self):
        """В Genshin повторный Space в полёте ЗАКРЫВАЕТ планер: жмём один раз."""
        b = ParkourBrain()
        first = b.decide(scene(falling=True), now=0.0)
        assert first.keys["glide"], "планер должен раскрыться сразу"
        for t in range(1, 100):
            later = b.decide(scene(falling=True), now=t * 0.1)
            assert not later.keys["glide"], "повторный Space закрыл бы планер"

    def test_glider_reopens_after_landing(self):
        b = ParkourBrain()
        b.decide(scene(falling=True), now=0.0)
        b.decide(scene(), now=0.2)          # коснулись земли
        b.decide(scene(), now=0.3)          # ещё кадр — приземление подтверждено
        again = b.decide(scene(falling=True), now=2.0)
        assert again.keys["glide"], "после приземления планер снова доступен"

    def test_reopen_respects_pause(self):
        """Слишком частый повторный Space закрыл бы планер — держим паузу."""
        b = ParkourBrain()
        assert b.decide(scene(falling=True), now=0.0).keys["glide"]
        b.decide(scene(), now=0.1)
        b.decide(scene(), now=0.2)          # приземление подтверждено
        assert not b.decide(scene(falling=True), now=0.4).keys["glide"]

    def test_never_walks_into_void(self):
        b = ParkourBrain()
        for t in range(40):
            a = b.decide(scene(ledge=True, ledge_prox=0.95, free=0.02), now=t * 0.1)
            assert not a.keys["forward"] or not a.keys["sprint"], \
                "агент не должен бежать в пропасть"

    def test_backs_off_from_critical_edge(self):
        b = ParkourBrain()
        a = b.decide(scene(ledge=True, ledge_prox=0.95, free=0.02,
                           sectors=[0.1, 0.1, 0.05, 0.02, 0.02, 0.02, 0.05, 0.1, 0.1]), now=0.0)
        assert a.keys["back"], "у самого края — только назад"


class TestClimbing:
    def test_wall_triggers_climb_with_space_pulses(self):
        b = ParkourBrain()
        a = b.decide(scene(wall=True, wall_prox=0.9, free=0.15), now=0.0)
        assert a.state is State.CLIMB
        assert a.keys["forward"]
        assert a.keys["jump"], "карабканье — это Space импульсами"

    def test_climb_bounded_by_stamina(self):
        b = ParkourBrain()
        a = b.decide(scene(wall=True, wall_prox=0.9, free=0.15), stamina=0.05, now=0.0)
        assert a.state is not State.CLIMB, "без выносливости карабканье не начинаем"
        assert a.keys["forward"]

    def test_no_jump_when_not_climbing(self):
        b = ParkourBrain()
        a = b.decide(scene(), now=0.0)
        assert not a.keys["jump"]


class TestMovement:
    def test_walks_when_clear(self):
        b = ParkourBrain()
        a = b.decide(scene(free=0.6), now=0.0)
        assert a.state in (State.WALK, State.SPRINT)
        assert a.keys["forward"]

    def test_sprints_only_on_clear_straight(self):
        b = ParkourBrain()
        a = b.decide(scene(free=0.9), stamina=1.0, goal_bearing=0.0, now=0.0)
        assert a.state is State.SPRINT and a.keys["sprint"]

    def test_no_sprint_when_goal_is_sideways(self):
        b = ParkourBrain()
        a = b.decide(scene(free=0.9), stamina=1.0, goal_bearing=1.0, now=0.0)
        assert not a.keys["sprint"]
        assert a.yaw > 0, "должен доворачивать к цели"

    def test_steers_to_goal(self):
        b = ParkourBrain()
        right = b.decide(scene(), goal_bearing=0.5, now=0.0)
        b.reset()
        left = b.decide(scene(), goal_bearing=-0.5, now=0.0)
        assert right.yaw > 0 > left.yaw
        assert abs(right.yaw - -left.yaw) < 1e-6

    def test_deadband_does_not_twitch(self):
        b = ParkourBrain()
        a = b.decide(scene(), goal_bearing=0.01, now=0.0)
        assert a.yaw == 0.0


class TestAvoid:
    def test_ledge_triggers_side_step(self):
        b = ParkourBrain()
        a = b.decide(scene(ledge=True, ledge_prox=0.5, free=0.3,
                           sectors=[0.9, 0.8, 0.5, 0.3, 0.2, 0.3, 0.5, 0.4, 0.3]), now=0.0)
        assert a.state is State.AVOID
        assert a.keys["left"] or a.keys["right"]

    def test_avoid_keeps_heading_goalward(self):
        """Обход не должен превращаться в бегство от цели навсегда."""
        b = ParkourBrain()
        a = b.decide(scene(ledge=True, ledge_prox=0.4, free=0.3), goal_bearing=0.0, now=0.0)
        assert abs(a.yaw) <= 1.0

    def test_escalates_left_right_glide(self):
        """Упершись в край, агент пробует оба обхода и только потом планирует."""
        b = ParkourBrain()
        states, presses = [], []
        sectors = [0.5, 0.5, 0.5, 0.4, 0.35, 0.4, 0.5, 0.5, 0.5]
        for t in range(int(6 / 0.1)):
            a = b.decide(scene(ledge=True, ledge_prox=0.5, free=0.3, sectors=sectors),
                         stamina=1.0, now=t * 0.1)
            states.append(a.state)
            if a.keys["glide"] or a.keys["jump"]:
                presses.append((round(t * 0.1, 1), "glide" if a.keys["glide"] else "jump"))
        assert State.AVOID in states and State.GLIDE in states, \
            "должны быть и оба обхода, и попытка перелёта"
        # Перелёт — это один прыжок; планер раскроется, когда пропадёт опора.
        assert len(presses) == 1 and presses[0][1] == "jump", \
            f"на перелёте нужен один прыжок, без спама Space: {presses}"


class TestKeys:
    def test_only_known_keys(self):
        b = ParkourBrain()
        a = b.decide(scene(), now=0.0)
        assert set(a.keys) == set(ACTIONS)
