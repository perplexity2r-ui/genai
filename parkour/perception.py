"""Восприятие сцены по карте близости.

Монокулярная сеть (Depth Anything / MiDaS) даёт глубину с точностью до масштаба,
поэтому абсолютные пороги вроде «стена ближе 3 метров» невозможны. Зато внутри
кадра устойчивы *отношения*, и именно на них построены все признаки:

1. ``ref`` — опорный уровень: близость пола прямо под персонажем (перцентиль).
   Всё остальное считаем как ``p / ref``.
2. ``wall`` — стена. В кадре это «плато»: близость высокая и почти одинаковая
   по всей высоте кадра (top_ratio ~ mid_ratio ~ 0.4..0.8). В открытом мире
   профиль — плавный рост от горизонта к ногам, сверху почти 0.
3. ``ledge`` — обрыв. В кадре это *резкий скачок*: сначала «далеко», затем
   сразу «близко». Сила скачка отличает край пропасти от плавного подъёма,
   а положение скачка по вертикали кодирует дистанцию до края (ниже = ближе).
4. ``falling`` — опорный уровень резко упал относительно своего медленного EMA
   (под ногами больше нет земли), либо сцена «уезжает» по вертикали.
5. ``sector_free`` — то же самое по N колонкам кадра: где есть пол впереди и
   где нет стены. Так рождается решение «обойти слева/справа».

Раскладка кадра Genshin учтена масками: миникарта, отряд, способности,
выносливость, субтитры и сам персонаж в центре не участвуют в статистике.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import PerceptionConfig
from .depth import DepthEstimator

# Насколько признак «впереди стена/обрыв» уменьшает свободу движения прямо.
WALL_FREE_PENALTY = 0.55
LEDGE_FREE_PENALTY = 1.20


@dataclass
class SceneAnalysis:
    wall: bool
    ledge: bool
    falling: bool
    target_dir: float            # -1..+1, 0 = прямо
    wall_prox: float             # 0..1, насколько стена близко
    ledge_prox: float            # 0..1, насколько край обрыва близко
    ground: float                # пол впереди / пол под ногами (0..1)
    free_forward: float
    sector_dist: List[float]
    confidence: float
    trend: float = 0.0
    profile: Tuple[float, ...] = ()
    prox: Optional[np.ndarray] = None

    @property
    def airborne(self) -> bool:
        """Совместимость с логикой: «нет опоры» = падение."""
        return self.falling

    def as_dict(self) -> dict:
        return {
            "wall": self.wall,
            "ledge": self.ledge,
            "falling": self.falling,
            "wall_prox": round(self.wall_prox, 3),
            "ledge_prox": round(self.ledge_prox, 3),
            "ground": round(self.ground, 3),
            "free_forward": round(self.free_forward, 3),
            "target_dir": round(self.target_dir, 3),
            "trend": round(self.trend, 3),
            "confidence": round(self.confidence, 3),
            "sector_dist": [round(float(s), 3) for s in self.sector_dist],
        }


class Perception:
    """Превращает кадр (или готовую карту близости) в SceneAnalysis."""

    # Зоны в долях кадра (x0, y0, x1, y1).
    REF_ZONES = ((0.26, 0.86, 0.42, 0.95), (0.58, 0.86, 0.74, 0.95))
    # Узкая центральная колонка (~16% ширины): это реальный коридор движения
    # персонажа. Шире — и в кадр попадают бока, где за краем узкого карниза
    # уже пропасть, хотя идти прямо можно.
    CENTER_X = (0.40, 0.60)

    def __init__(self, cfg: Optional[PerceptionConfig] = None,
                 estimator: Optional[DepthEstimator] = None):
        self.cfg = cfg or PerceptionConfig()
        self.estimator = estimator
        self._prev_prof: Optional[np.ndarray] = None
        self._near_base: Optional[float] = None   # базовая «близость пола» на земле
        self._ema: Dict[str, float] = {}
        self._fall_latch = 0

    def reset(self) -> None:
        self._prev_prof = None
        self._near_base = None
        self._ema.clear()
        self._fall_latch = 0

    # ------------------------------------------------------------- geometry
    def _prepare(self, prox: np.ndarray) -> np.ndarray:
        h, w = self.cfg.work_hw
        p = np.array(
            Image.fromarray(np.clip(prox, 0, 1).astype(np.float32)).resize((w, h), Image.BILINEAR),
            dtype=np.float32, copy=True,
        )
        for (x0, y0, x1, y1) in self.cfg.hud_masks:
            p[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = np.nan
        # Сам персонаж: центр нижней половины кадра.
        p[int(0.44 * h):int(0.92 * h), int(0.42 * w):int(0.58 * w)] = np.nan
        return p

    @staticmethod
    def _nan(v: float, default: float = 0.0) -> float:
        return default if not np.isfinite(v) else float(v)

    def _ref(self, p: np.ndarray) -> float:
        h, w = p.shape
        vals = []
        for (x0, y0, x1, y1) in self.REF_ZONES:
            z = p[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].ravel()
            vals.append(z[np.isfinite(z)])
        vals = np.concatenate(vals) if vals else np.array([0.0], dtype=np.float32)
        return self._nan(np.percentile(vals, 60) if vals.size else 0.0)

    # Фланги на высоте груди: стена занимает края кадра, центр закрыт персонажем.
    FLANK_ZONES = ((0.03, 0.34, 0.22, 0.60), (0.78, 0.34, 0.97, 0.60))

    def _flank_level(self, p: np.ndarray, ref: float) -> float:
        h, w = p.shape
        vals = []
        for (x0, y0, x1, y1) in self.FLANK_ZONES:
            z = p[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].ravel()
            vals.append(z[np.isfinite(z)])
        vals = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
        if not vals.size:
            return 0.0
        return float(np.mean(vals)) / (ref + 1e-3)

    def _profile(self, p: np.ndarray, ref: float, bins: int = 12) -> np.ndarray:
        """Профиль отношения p/ref сверху вниз по центральной части кадра."""
        h, w = p.shape
        x0, x1 = int(self.CENTER_X[0] * w), int(self.CENTER_X[1] * w)
        rows = np.linspace(int(0.05 * h), int(0.98 * h), bins + 1).astype(int)
        prof = np.zeros(bins, dtype=np.float32)
        for i in range(bins):
            y0, y1 = rows[i], max(rows[i] + 1, rows[i + 1])
            y1 = min(y1, h)
            band = p[y0:y1, x0:x1]
            v = band[np.isfinite(band)] if band.size else np.array([], dtype=np.float32)
            if not v.size:
                # Последняя полоса может попасть целиком в маски HUD (субтитры,
                # выносливость). Тогда переиспользуем предыдущую оценку, а не 0.
                prof[i] = prof[i - 1] if i > 0 else 0.0
            else:
                prof[i] = float(np.mean(v)) / (ref + 1e-3)
        return prof

    # ---------------------------------------------------------------- main
    def exclusion_mask(self, h: int, w: int) -> np.ndarray:
        """Маска HUD и персонажа в долях кадра — для оценки «что рядом»."""
        m = np.zeros((h, w), dtype=bool)
        for (x0, y0, x1, y1) in self.cfg.hud_masks:
            m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
        m[int(0.42 * h):int(0.99 * h), int(0.38 * w):int(0.62 * w)] = True
        return m

    def raw_scale_from(self, estimator: DepthEstimator) -> float:
        """Сырая близость без персонажа/HUD (см. DepthEstimator.raw_scale_masked)."""
        return estimator.raw_scale_masked((0, 0), self.exclusion_mask)

    def analyze(self, frame_rgb: Optional[np.ndarray] = None,
                prox: Optional[np.ndarray] = None,
                raw_scale: Optional[float] = None) -> SceneAnalysis:
        if prox is None:
            if frame_rgb is None:
                raise ValueError("нужен либо frame_rgb, либо prox")
            if self.estimator is None:
                self.estimator = DepthEstimator()
            prox = self.estimator.predict(frame_rgb)
            if raw_scale is None:
                raw_scale = self.estimator.raw_depth_scale()
        elif raw_scale is None and self.estimator is not None:
            raw_scale = self.estimator.raw_depth_scale()
        cfg = self.cfg
        p = self._prepare(prox)
        ref = self._ref(p)
        prof = self._profile(p, ref)

        # Профиль сверху вниз по 12 полосам: 0 — горизонт, 11 — под ногами.
        top = float(np.mean(prof[:4]))      # верх кадра: 5..25% высоты
        mid = float(np.mean(prof[4:8]))     # середина: 35..65%
        knee = float(np.mean(prof[-4:-2]))  # «перед ногами»: 65..83% высоты
        near = float(np.mean(prof[-2:]))    # своя земля под ногами
        gradient = float(np.max(np.diff(prof))) if prof.size > 1 else 0.0
        jump_bin = int(np.argmax(np.diff(prof)) + 1) if prof.size > 1 else 0
        jump_row = float(jump_bin / prof.size)

        # --- стена: «плато» близости и сверху, и в середине кадра.
        # Стена = «плато» близости: близко и у горизонта, и на высоте груди, и
        # прямо перед ногами (мы стоим вплотную). В открытом поле верх далёк.
        # У обрыва перед ногами пусто (knee низкий) — это не стена.
        # Стена — это вертикальная поверхность: близость нарастает сверху вниз
        # (горизонт -> грудь -> ноги) и сверху не «вплотную». Если картина
        # обратная (верх ближе низа), это поверхность под нами — например,
        # склон, который видно сверху при планировании, а не стена.
        grows_downward = prof[-1] >= prof[0] - cfg.wall_profile_tol
        # Стена бывает двух видов, и признаки у них разные:
        #  * в отдалении — сверху небо (top мал), близость нарастает сверху вниз
        #    (grows_downward). Это же отсекает обрыв: у него в середине пусто.
        #  * вплотную — стена занимает кадр целиком, профиль почти «плато»
        #    (top > 1) и расти вниз уже некуда. Здесь достаточно того, что
        #    близость у ног не рушится относительно верха (knee_vs_top): у
        #    плоскости пола под нами (площадка, широкий карниз) она падает.
        # Плоскость пола под нами (планируем над склоном) тоже занимает кадр
        # целиком, но близость у неё монотонно ПАДАЕТ сверху вниз: середина
        # кадра дальше верха. У стены, в которую упираемся, наоборот — к
        # середине кадра близость не меньше.
        knee_vs_top = knee >= cfg.wall_high_drop * top
        mid_vs_top = mid >= top - cfg.wall_plateau_tol
        near_plateau = top >= cfg.wall_high_top and knee_vs_top and mid_vs_top
        far_slope = grows_downward and top <= cfg.wall_high_top
        wall = bool(top >= cfg.wall_top_thr and mid >= cfg.wall_mid_thr
                    and knee >= cfg.wall_knee_thr and (near_plateau or far_slope))
        wall_prox = float(np.clip((top - cfg.wall_top_thr) / 0.45, 0.0, 1.0)) if top >= cfg.wall_top_thr else 0.0

        # --- обрыв: резкий скачок снизу-вверх, при этом в середине кадра пусто.
        # Край обрыва: резкий скачок «далеко -> близко» снизу кадра.
        #   jump_row >= 0.55 — впереди край уступа (низ кадра: своя земля);
        #   jump_row < 0.55  — смотрим вдоль/через край: пустота впереди,
        #                      но своя земля может быть не видна. Это тоже опасно.
        front_edge = jump_row >= cfg.ledge_row_min
        side_view = jump_row < cfg.ledge_row_min
        ledge = bool(gradient >= cfg.ledge_jump_thr and mid <= cfg.ledge_mid_thr
                     and (front_edge or (side_view and near < cfg.side_view_near_max))
                     and not wall)
        if side_view and ledge:
            jump_row = cfg.ledge_row_min   # нормируем «близость» для бокового вида
        # Чем ниже скачок, тем ближе край: jump_row 0.55 -> далеко, 0.80 -> вплотную.
        ledge_prox = 0.0
        if ledge:
            ledge_prox = float(np.clip((jump_row - cfg.ledge_row_min) / max(1e-3, 0.85 - cfg.ledge_row_min), 0.0, 1.0))

        # --- падение: рядом нет поверхности.
        # Решается ТОЛЬКО по сырой близости кадра (до нормировки): стоя на
        # земле пол в паре метров и она велика, над пропастью рушится на
        # порядок. По нормированной карте падение неотличимо от ходьбы —
        # и доля «дали», и сдвиг сцены по вертикали дают те же значения,
        # что и на земле (проверено), поэтому они не участвуют в решении.
        near_ok = self._near_ok(raw_scale, ledge)
        trend = self._vertical_trend(p)
        falling = bool(not near_ok)
        self._update_near_line(raw_scale, near_ok)
        if falling:
            self._fall_latch = cfg.fall_latch_frames
        elif self._fall_latch > 0:
            # Гистерезис: не «мигаем» планером на шумном кадре.
            self._fall_latch -= 1
            falling = True
        if falling:
            # В полёте «стены» из кадра — это дальние скалы внизу, а не
            # препятствие: карабкаться в воздухе бессмысленно.
            wall = False
            wall_prox = 0.0

        # --- пол впереди (для торможения перед стеной/сундуком).
        ground = float(np.clip(float(np.mean(prof[8:11])), 0.0, 1.5))

        # --- сектора: пол впереди и стена по каждому направлению.
        sector_dist = self._sectors(p, ref)
        n = len(sector_dist)
        center_free = float(sector_dist[n // 2]) if n else float(np.clip(ground, 0, 1))
        target_dir = self._best_direction(sector_dist)
        # Свобода «прямо» — это не только отсутствие стены, но и отсутствие обрыва:
        # шаг за край стоит дороже, чем обход.
        free_forward = float(np.clip(center_free * (1.0 - LEDGE_FREE_PENALTY * ledge_prox)
                                     * (1.0 - WALL_FREE_PENALTY * wall_prox), 0.0, 1.0))

        # --- сглаживание вероятностных признаков между кадрами.
        a = cfg.feature_ema
        for k, v in (("top", top), ("mid", mid), ("led", ledge_prox), ("wal", wall_prox)):
            self._ema[k] = v if k not in self._ema else a * v + (1 - a) * self._ema[k]

        contrast = abs(mid - top) + abs(near - top)
        confidence = float(np.clip(0.35 + 0.9 * contrast, 0.0, 1.0))

        return SceneAnalysis(
            wall=wall,
            ledge=ledge,
            falling=falling,
            target_dir=target_dir,
            # Вероятность стены имеет смысл только когда стена действительно
            # распознана: иначе сглаженный EMA «протекает» в логику.
            wall_prox=float(self._ema["wal"] if wall else 0.0),
            ledge_prox=float(self._ema["led"] if ledge else 0.0),
            ground=float(np.clip(ground, 0.0, 1.0)),
            free_forward=free_forward,
            sector_dist=sector_dist,
            confidence=confidence,
            trend=trend,
            profile=tuple(round(float(v), 3) for v in prof),
            prox=p,
        )

    # --------------------------------------------------------------- signals
    # --------------------------------------------------- сырая шкала глубины
    def _near_ok(self, raw_scale: Optional[float], ledge: bool) -> bool:
        """Есть ли рядом поверхность (пол) — по сырой близости кадра.

        Стоя на земле пол в 2–4 м, сырая близость высокая; полете над
        пропастью максимум кадра — дальний берег/туман, она ниже базовой линии
        в несколько раз. Порог относительный: калибровки модели не требуется.
        """
        if raw_scale is None or raw_scale <= 0:
            return True          # нет данных — доверяем остальным признакам
        # (а) Абсолютная проверка: 1/z ближайшей поверхности. В небе даже
        #     дальний берег не ближе ~15–20 м, то есть 1/z <= 0.06 (в условных
        #     единицах сети). На земле пол в 2–4 м: 1/z ~ 0.3–0.7.
        if raw_scale < self.cfg.air_abs_raw:
            return False
        # (б) Относительная: близость в кадре упала к своей базовой линии.
        #     У края обрыва своя земля ещё в кадре, поэтому порог строже —
        #     иначе край путался бы с полётом.
        if self._near_base:
            ratio = self.cfg.near_fall_ratio if ledge else self.cfg.near_fall_ratio_ledge
            if raw_scale < self._near_base * ratio:
                return False
        return True

    def _update_near_line(self, raw_scale: Optional[float], near_ok: bool) -> None:
        """Медленная базовая линия: сколько «близи» обычно в кадре на земле."""
        if raw_scale is None or raw_scale <= 0:
            return
        if self._near_base is None:
            if self._ground_confirmed(raw_scale):
                self._near_base = raw_scale
            return
        # Обновляем базовую линию только когда уверенно стоим на земле
        # (близость явно выше абсолютного порога «воздуха»). Иначе первая же
        # инициализация в полёте навсегда испортила бы калибровку.
        if not self._ground_confirmed(raw_scale):
            return
        a = self.cfg.near_base_ema
        self._near_base = (1 - a) * self._near_base + a * raw_scale

    def _ground_confirmed(self, raw_scale: float) -> bool:
        return raw_scale >= self.cfg.air_abs_raw * 2.5

    def _vertical_trend(self, p: np.ndarray) -> float:
        with np.errstate(invalid="ignore"):
            prof = np.nan_to_num(np.nanmean(p, axis=1))
        prev = self._prev_prof
        self._prev_prof = prof
        if prev is None or prev.shape != prof.shape or prof.size < 8:
            return 0.0
        a = prof - prof.mean()
        b = prev - prev.mean()
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            return 0.0
        n = prof.size
        max_lag = max(2, int(0.30 * n))
        best, best_lag = -1.0, 0
        for lag in range(-max_lag, max_lag + 1):
            aa = a[max(0, lag):n + min(0, lag)]
            bb = b[max(0, -lag):n - max(0, lag)]
            if aa.size < 4:
                continue
            c = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-6))
            if c > best:
                best, best_lag = c, lag
        return float(best_lag / n)

    def _sectors(self, p: np.ndarray, ref: float) -> List[float]:
        """Свобода по секторам: пол впереди есть И стены нет."""
        cfg = self.cfg
        h, w = p.shape
        n = cfg.sectors
        edges = np.linspace(int(0.05 * w), int(0.95 * w), n + 1).astype(int)
        out: List[float] = []
        for i in range(n):
            x0, x1 = edges[i], edges[i + 1]
            ahead = p[int(0.66 * h):int(0.80 * h), x0:x1]
            chest = p[int(0.30 * h):int(0.60 * h), x0:x1]
            a_vals = ahead[np.isfinite(ahead)]
            c_vals = chest[np.isfinite(chest)]
            a = float(np.mean(a_vals)) / (ref + 1e-3) if a_vals.size else 0.0
            c = float(np.mean(c_vals)) / (ref + 1e-3) if c_vals.size else 0.0
            support = float(np.clip(a / max(cfg.ledge_mid_thr, 1e-3), 0.0, 1.0))
            wall_free = 1.0 - float(np.clip(c / max(cfg.wall_top_thr, 1e-3), 0.0, 1.0))
            out.append(float(np.clip(0.6 * support + 0.4 * wall_free, 0.0, 1.0)))
        arr = np.asarray(out, dtype=np.float32)
        if cfg.smooth_sectors > 1 and arr.size >= cfg.smooth_sectors:
            k = cfg.smooth_sectors
            arr = np.convolve(arr, np.ones(k, np.float32) / k, mode="same")
        return arr.tolist()

    @staticmethod
    def _best_direction(sector_dist: List[float]) -> float:
        """Азимут лучшего направления; 0 — если центр и так достаточно свободен."""
        if not sector_dist:
            return 0.0
        arr = np.asarray(sector_dist, dtype=np.float32)
        n = arr.size
        if n == 1:
            return 0.0
        centers = np.linspace(-1.0, 1.0, n)
        score = arr - 0.20 * np.abs(centers)
        k = int(np.argmax(score))
        # Не крутим камеру без нужды: центр свободен — идём прямо.
        if arr[n // 2] >= 0.55 and arr[k] - arr[n // 2] < 0.18:
            return 0.0
        if arr[k] - arr[n // 2] < 0.12:
            return 0.0
        return float(centers[k])
