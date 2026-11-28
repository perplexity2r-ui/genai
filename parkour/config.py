"""Конфигурация агента. Все пороги — в безразмерной шкале близости [0, 1].

Шкала: 1.0 = «вплотную к камере», 0.0 = «далеко / горизонт». Это единственная
шкала, которая корректна для монокулярной depth-сети без калибровки.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from .depth import DepthConfig


@dataclass
class PerceptionConfig:
    work_hw: Tuple[int, int] = (96, 160)
    feature_ema: float = 0.55           # сглаживание вероятностных признаков
    sectors: int = 9
    smooth_sectors: int = 3

    # --- стена: «плато» близости по всей высоте кадра (p/ref).
    wall_top_thr: float = 0.22          # верх кадра 5..25% высоты
    wall_mid_thr: float = 0.28          # середина кадра
    wall_knee_thr: float = 0.42        # полоса «перед ногами»: у стены близко
    wall_profile_tol: float = 0.10     # близость у стены растёт сверху вниз
    # Стена — это ВЫСОКАЯ поверхность: близость не обрывается к верху кадра.
    # Сверху может быть небо (top ~0, профиль круто растёт вниз — далёкая стена)
    # или сама стена вплотную (top > 1: плато). Плоскость пола, снятая сверху
    # (карниз, площадка), тоже даёт высокий top, но у неё близость ПАДАЕТ к
    # низу кадра; поэтому нужен wall_drop_ok, а не «top сверху не вплотную».
    wall_top_max: float = 1.35         # (устарело: не используется в решении о стене)
    wall_high_top: float = 1.10        # top выше этого = «стена упирается в кадр»
    wall_high_drop: float = 0.60       # ... и при этом knee не ниже этой доли top
    wall_plateau_tol: float = 0.05     # ... и середина кадра не дальше верха
    # --- обрыв: резкий скачок профиля при «пустой» середине кадра.
    ledge_jump_thr: float = 0.26        # max(diff(profile))
    ledge_mid_thr: float = 0.35         # в середине кадра пусто
    ledge_row_min: float = 0.50         # скачок не выше середины кадра
    ledge_knee_thr: float = 0.42        # перед ногами пусто
    ledge_near_min: float = 0.60        # под ногами ещё есть своя земля
    side_view_near_max: float = 0.75    # смотрим вдоль края: своей земли мало
    # --- «рядом нет поверхности» по СЫРОЙ близости (до нормировки).
    # Это единственный критерий падения: по нормированной карте оно неотличимо
    # от ходьбы, поэтому порогов «доля дали / сдвиг сцены» здесь больше нет.
    air_abs_raw: float = 0.10           # 1/z ближайшей поверхности < 0.10 -> воздух (~10 м+)
    near_fall_ratio: float = 0.25       # полёт: близость упала ниже 25% базовой
    near_fall_ratio_ledge: float = 0.08 # у края (своя земля в кадре) — строже
    near_base_ema: float = 0.05         # медленное обновление базовой линии
    near_base_ema_fall: float = 0.02    # в полёте — ещё медленнее
    fall_latch_frames: int = 3          # гистерезис: не мигаем планером на шуме

    # Маски HUD (x0, y0, x1, y1) -> значения игнорируются.
    hud_masks: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: [
            (0.00, 0.00, 0.23, 0.17),   # миникарта
            (0.82, 0.00, 1.00, 0.23),   # отряд
            (0.72, 0.82, 1.00, 1.00),   # способности
            (0.36, 0.90, 0.64, 1.00),   # выносливость / подсказка
            (0.00, 0.94, 1.00, 1.00),   # субтитры
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyConfig:
    # --- границы вмешательства (пороги в шкале вероятностей восприятия).
    wall_engage_prox: float = 0.35     # стена «вплотную» -> карабкаемся
    ledge_engage_prox: float = 0.42    # край «близко» -> обход вдоль края
    walk_into_wall_prox: float = 0.20  # стена дальше этой — подходим, не ползём
    block_release: float = 0.55        # гистерезис выхода из обхода
    ledge_critical_prox: float = 0.62  # край вплотную -> назад
    too_close_ground: float = 0.18     # упор non-стена -> притормозить

    # --- курс цели (камера уже наведена на цель).
    yaw_rate: float = 1.8              # рад/с при команде 1.0 (как в Genshin)
    yaw_kp: float = 1.6                # усиление по ошибке курса
    yaw_deadband: float = 0.06         # мёртвая зона, рад
    sprint_heading_max: float = 0.30   # бежим только по почти прямому курсу

    # --- обход препятствия.
    dodge_time_s: float = 1.1          # длительность манёвра
    edge_yaw: float = 0.45             # темп доворота при обходе края
    edge_follow_s: float = 1.6         # сколько держим выбранную сторону края
    edge_forward_min: float = 0.42     # идти вдоль края «с подшагом вперёд»
    ledge_recover_prox: float = 0.18   # ниже этого края считаем, что отошли
    avoid_bearing: float = 0.85       # на сколько уводим курс при обходе, рад
    backoff_bearing: float = 0.9       # разворот при отходе от края, рад
    side_hold_s: float = 2.0
    probe_time_s: float = 2.2          # сколько пробуем каждый боковой обход           # сколько держим выбранную сторону обхода
    front_blocked_free: float = 0.26   # свобода «прямо» ниже -> путь перекрыт
    dodge_yaw: float = 0.55            # темп доворота на обходе
    lateral_margin: float = 0.06

    # --- ритм клавиш.
    pulse_period_s: float = 0.45
    climb_boost_period_s: float = 0.9
    glide_period_s: float = 1.5        # пауза перед повторным раскрытием планера
    glide_jump_gap: float = 0.35       # между прыжком и раскрытием планера
    min_hold_s: float = 0.25

    # --- безопасность/выносливость.
    free_good: float = 0.55
    stamina_climb_min: float = 0.18
    stamina_glide_min: float = 0.10
    stamina_low: float = 0.25
    stamina_climb_hold: float = 0.10
    stamina_glide_hold: float = 0.06

    # --- планер через провал.
    allow_glide_across: bool = True


@dataclass
class ParkourConfig:
    depth: DepthConfig = field(default_factory=DepthConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    # Захват экрана: (left, top, width, height); None = весь основной экран.
    capture_region: Optional[Tuple[int, int, int, int]] = None
    target_fps: float = 12.0
    dry_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["capture_region"] = list(self.capture_region) if self.capture_region else None
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParkourConfig":
        data = dict(data or {})
        depth = DepthConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in (data.get("depth") or {}).items()})
        perc = dict(data.get("perception") or {})
        for key in ("work_hw",):
            if key in perc and isinstance(perc[key], list):
                perc[key] = tuple(perc[key])
        if "hud_masks" in perc:
            perc["hud_masks"] = [tuple(m) for m in perc["hud_masks"]]
        pol = PolicyConfig(**{k: v for k, v in (data.get("policy") or {}).items()})
        region = data.get("capture_region")
        return cls(
            depth=depth,
            perception=PerceptionConfig(**perc),
            policy=pol,
            capture_region=tuple(region) if region else None,
            target_fps=float(data.get("target_fps", 12.0)),
            dry_run=bool(data.get("dry_run", True)),
        )

    @classmethod
    def load(cls, path: str) -> "ParkourConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
