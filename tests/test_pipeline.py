"""Сквозной тест: синтетическая сцена → детектор → трекер → состояние."""
import json

import numpy as np

from genshinvision.detect import HeuristicDetector
from genshinvision.hud import DEFAULT_CALIBRATION, HudReader
from genshinvision.pipeline import Pipeline
from genshinvision.state import StateBuilder
from genshinvision.synthetic import SyntheticGameSource
from genshinvision.track import SortTracker


def build_pipeline(**kw):
    src = SyntheticGameSource(seed=11, fps=30.0, **kw)
    pipe = Pipeline(
        src,
        HeuristicDetector(DEFAULT_CALIBRATION),
        HudReader(DEFAULT_CALIBRATION),
        SortTracker(min_hits=2),
    )
    return src, pipe


def test_pipeline_end_to_end():
    src, pipe = build_pipeline(n_enemies=3, max_frames=100)
    states = []
    truths = []
    for _ in range(100):
        out = pipe.step()
        assert out is not None
        states.append(out.state)
        truths.append(src.last_truth)
        # аннотированный кадр того же размера
        assert out.frame.shape == (720, 1280, 3)

    # --- JSON сериализуем ---
    payload = json.dumps(states[-1].to_dict())
    assert "enemies" in payload

    # --- HUD: HP игрока совпадает с ground truth ---
    hp_errs = []
    for st, tr in zip(states[40:], truths[40:]):
        assert st.hud.player_hp_frac is not None
        hp_errs.append(abs(st.hud.player_hp_frac - tr["player_hp_frac"]))
    assert float(np.mean(hp_errs)) < 0.03, f"средняя ошибка HP {np.mean(hp_errs):.3f}"

    # --- босс появляется после frame 40 ---
    boss_states = [st for st, tr in zip(states, truths) if tr["boss"] is not None]
    assert len(boss_states) > 40
    for st, tr in zip(states[45:], truths[45:]):
        if tr["boss"] is None:
            continue
        assert st.hud.boss_hp_frac is not None
        assert abs(st.hud.boss_hp_frac - tr["boss"]["hp_frac"]) < 0.06

    # --- враги: все найдены и отслеживаются (гуманоиды + слизни) ---
    dist_errs, hp_track_errs, vel_errs = [], [], []
    for st, tr in zip(states[40:], truths[40:]):
        assert st.in_combat
        tracked = [e for e in st.enemies if e.cls_name in ("enemy", "elite")]
        assert len(tracked) >= 4, f"найдено {len(tracked)} из 5 врагов"
        for truth in tr["enemies"]:
            tb = truth["bbox"]
            tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            best = min(
                tracked,
                key=lambda e: ((e.bbox[0] + e.bbox[2]) / 2 - tcx) ** 2
                + ((e.bbox[1] + e.bbox[3]) / 2 - tcy) ** 2,
            )
            bcx, bcy = (best.bbox[0] + best.bbox[2]) / 2, (best.bbox[1] + best.bbox[3]) / 2
            dist = ((bcx - tcx) ** 2 + (bcy - tcy) ** 2) ** 0.5
            dist_errs.append(dist)
            if best.hp_frac is not None and truth["hp_frac"] is not None:
                hp_track_errs.append(abs(best.hp_frac - truth["hp_frac"]))
            tvx, tvy = truth["velocity"]
            vel_errs.append(
                ((best.velocity[0] - tvx) ** 2 + (best.velocity[1] - tvy) ** 2) ** 0.5
            )

    assert float(np.mean(dist_errs)) < 60, f"средняя ошибка позиции {np.mean(dist_errs):.1f} px"
    assert float(np.mean(hp_track_errs)) < 0.08, f"средняя ошибка HP врага {np.mean(hp_track_errs):.3f}"
    med_vel = float(np.median(vel_errs))
    assert med_vel < 70, f"медианная ошибка скорости {med_vel:.1f} px/s"

    # --- ID треков врагов стабильны (5 врагов + небольшой запас на шум) ---
    ids = {e.track_id for st in states for e in st.enemies if e.cls_name in ("enemy", "elite")}
    assert 4 <= len(ids) <= 9, ids

    # --- объекты мира: сундуки по статусу, телепорты по цвету ---
    last = states[-1]
    by_class = {}
    for o in last.interactables:
        by_class.setdefault(o.cls_name, []).append(o)
    assert "chest_unlocked" in by_class, "открытый сундук не найден"
    assert "chest_locked" in by_class, "закрытый сундук не найден"
    assert "teleport_active" in by_class, "активный телепорт не найден"
    assert "teleport_inactive" in by_class, "неактивный телепорт не найден"
    assert "oculus" in by_class and len(by_class["oculus"]) == 2, "ожидалось 2 кукулиуса"
    assert "puzzle" in by_class, "головоломка не найдена"

    # сигналы боту по статусам
    assert by_class["chest_unlocked"][0].action == "press_f"
    assert by_class["chest_locked"][0].action == "search_area"
    assert by_class["teleport_active"][0].action == "ignore"
    assert by_class["teleport_inactive"][0].action == "press_f"

    # элементы кукулиусов различаются
    elements = {o.element for o in by_class["oculus"]}
    assert elements == {"anemo", "electro"}, elements

    # позиции объектов близки к истинным
    truth_objs = {t["class"]: t for t in src.last_truth["objects"] if t["class"] != "oculus"}
    for cls in ("chest_unlocked", "chest_locked", "teleport_active", "teleport_inactive", "puzzle"):
        t = truth_objs[cls]
        tcx, tcy = (t["bbox"][0] + t["bbox"][2]) / 2, (t["bbox"][1] + t["bbox"][3]) / 2
        best = min(
            by_class[cls],
            key=lambda o: ((o.bbox[0] + o.bbox[2]) / 2 - tcx) ** 2
            + ((o.bbox[1] + o.bbox[3]) / 2 - tcy) ** 2,
        )
        dist = (((best.bbox[0] + best.bbox[2]) / 2 - tcx) ** 2
                + ((best.bbox[1] + best.bbox[3]) / 2 - tcy) ** 2) ** 0.5
        assert dist < 80, f"{cls}: центр смещён на {dist:.0f} px"
        assert best.distance_m is not None and best.distance_m > 0

    # кукулиусы: хотя бы один из двух ближе 50 м и оба стабильны
    for o in by_class["oculus"]:
        assert o.distance_m is not None and o.distance_m < 50

    # --- момент взаимодействия: иконка клавиши «F» ---
    assert last.ui_prompt is not None, "кнопка F не найдена"
    assert last.ui_prompt["bbox"] is not None
    t_ui = src.last_truth["ui_prompt"]["bbox"]
    u = last.ui_prompt["bbox"]
    ucx, ucy = (u[0] + u[2]) / 2, (u[1] + u[3]) / 2
    tcx, tcy = (t_ui[0] + t_ui[2]) / 2, (t_ui[1] + t_ui[3]) / 2
    assert ((ucx - tcx) ** 2 + (ucy - tcy) ** 2) ** 0.5 < 40
    assert last.action_hint == "stop_and_press_f"  # F-подсказка важнее боя
    d = last.to_dict()
    assert d["ui"]["interact_f"] is True

    # --- миникарта: курс и движение ---
    mm = last.minimap
    assert mm["heading_deg"] is not None
    v = src.player_vel
    assert abs(mm["motion_px_s"][0] - v[0]) < 40
    assert abs(mm["motion_px_s"][1] - v[1]) < 40

    # --- понимание всех врагов: типы, стихии, аггро, угроза, сводка ---
    slimes = [e for e in last.enemies if e.enemy_type == "slime"]
    humanoids = [
        e for e in last.enemies if e.enemy_type != "slime" and e.cls_name == "enemy"
    ]
    assert len(slimes) == 2, f"слизни: {[(e.enemy_type, e.element) for e in last.enemies]}"
    assert {s.element for s in slimes} == {"pyro", "electro"}
    assert all(s.aggro is False for s in slimes), "слизни без полосок HP — не в аггро"
    assert len(humanoids) == 3
    assert all(h.aggro for h in humanoids), "гуманоиды с полосками HP — в аггро"
    for e in last.enemies:
        assert e.threat in ("high", "medium", "low")
        assert e.closing_px_s is not None

    summary = last.enemies_summary
    assert summary["total"] == 6           # 5 врагов + босс
    assert summary["by_tier"] == {"enemy": 5, "elite": 0, "boss": 1}
    assert summary["by_type"]["slime"] == 2        # тонкий тип знаем для стихийных
    assert summary["by_element"] == {"pyro": 1, "electro": 1}
    assert summary["aggro_count"] == 4
    assert summary["nearest_m"] is not None
    assert summary["threat_level"] in ("high", "medium", "low")


def test_state_builder_serializable():
    builder = StateBuilder("heuristic")
    state = builder.build(
        timestamp=1.0,
        frame_size=(1280, 720),
        hud=states_hud_stub(),
        tracks=[],
    )
    d = state.to_dict()
    assert d["player"]["hp_frac"] is None
    assert d["enemies"] == []
    json.dumps(d)


def states_hud_stub():
    from genshinvision.types import HudReading

    return HudReading()
