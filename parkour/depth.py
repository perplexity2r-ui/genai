"""Монокулярная оценка глубины (Depth Anything V2 / MiDaS ONNX).

Сеть возвращает *относительную обратную глубину* (disparity): больше значение —
ближе объект. Масштаб и сдвиг у монокулярных сетей не определены, поэтому здесь
мы приводим карту к устойчивому виду:

1. робастная нормировка по перцентилям p_low..p_high -> [0, 1];
2. сглаживание по времени (EMA) — снижает дрожание решения «идти/не идти».

Все пороги логики навигации задаются в этой безразмерной шкале, а не в метрах:
это единственный способ работать с монокулярной сетью без калибровки.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PATCH = 14  # patch size ViT-бэкбона: вход обязан быть кратен 14

# Профили "скорость/качество" (замер на 2 ядрах CPU, ViT-S int8):
#   fast    266x154  ~119 мс  (8.4 FPS)
#   balanced 294x168 ~139 мс  (7.2 FPS)   <- по умолчанию
#   quality 336x196  ~183 мс  (5.5 FPS)
PROFILES = {
    "fast": (154, 266),
    "balanced": (168, 294),
    "quality": (196, 336),
    "max": (252, 448),
}
DEFAULT_INPUT_HW = PROFILES["balanced"]


def _round_to_patch(v: int, patch: int = PATCH) -> int:
    return max(patch, int(round(v / patch)) * patch)


@dataclass
class DepthConfig:
    model_path: str = "models/da_vits14_int8.onnx"
    input_hw: Tuple[int, int] = DEFAULT_INPUT_HW
    threads: int = 2
    p_low: float = 2.0          # перцентиль «далеко»
    p_high: float = 98.0        # перцентиль «близко»
    ema: float = 0.35           # сглаживание карты между кадрами (0 = выключено)
    slow_norm_ema: float = 0.35  # скорость обновления окна нормировки (масштаб)
    out_hw: Optional[Tuple[int, int]] = None  # размер итоговой карты (по умолчанию = вход)


class SlowNormalizer:
    """Нормировка карты по медленно обновляемому окну [lo, hi].

    Обычная перцентильная нормировка «в кадре» убивает масштаб: падение с
    обрыва и обычная ходьба дают одинаковый профиль. Если же окно обновлять
    медленно (EMA), глобальный сдвиг «всё стало далеко» остаётся видимым —
    именно по нему восприятие ловит падение.
    """

    def __init__(self, p_low: float = 2.0, p_high: float = 98.0, ema: float = 0.35,
                 min_width_ratio: float = 0.35):
        self.p_low, self.p_high, self.ema = p_low, p_high, ema
        self.min_width_ratio = min_width_ratio
        self.lo: Optional[float] = None
        self.hi: Optional[float] = None
        self._w0: Optional[float] = None

    def __call__(self, disp: np.ndarray) -> np.ndarray:
        disp = np.asarray(disp, dtype=np.float32)
        lo = float(np.percentile(disp, self.p_low))
        hi = float(np.percentile(disp, self.p_high))
        if hi - lo < 1e-9:
            hi = lo + 1e-6
        if self.lo is None:
            self.lo, self.hi = lo, hi
            self._w0 = max(hi - lo, 1e-6)
        else:
            a = self.ema
            self.lo = (1 - a) * self.lo + a * lo
            self.hi = (1 - a) * self.hi + a * hi
        # Защита от вырожденного окна: не сжимаем масштаб больше, чем в
        # min_width_ratio раз от исходного. Иначе при планировании над пустотой
        # вся карта «зашкаливает» и признаки перестают различать даль/близь.
        min_w = max(self._w0 * self.min_width_ratio, 1e-6) if self._w0 else 1e-6
        if self.hi - self.lo < min_w:
            mid = 0.5 * (self.hi + self.lo)
            self.lo, self.hi = mid - 0.5 * min_w, mid + 0.5 * min_w
        return np.clip((disp - self.lo) / (self.hi - self.lo), 0.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        self.lo = self.hi = None

    def scale(self) -> float:
        """Ширина окна: сжимается, когда всё вокруг далеко."""
        if self.lo is None:
            return 1.0
        return float(self.hi - self.lo)


class DepthEstimator:
    """Обёртка над ONNX-моделью монокулярной глубины."""

    def __init__(self, cfg: Optional[DepthConfig] = None, model_path: Optional[str] = None):
        self.cfg = cfg or DepthConfig()
        if model_path:
            self.cfg.model_path = model_path
        self.normalizer = SlowNormalizer(self.cfg.p_low, self.cfg.p_high, self.cfg.slow_norm_ema)
        self._session = None
        self._input_name = "image"
        # Сырая шкала глубины последнего кадра: p98 обратной глубины (1/z).
        # Нормировка её уничтожает, а вместе с ней и признак «всё вокруг далеко»
        # (полёт над пропастью). Поэтому запоминаем до нормировки.
        self.last_raw_scale = 0.0
        self.last_raw_p2 = 0.0
        # Сырая карта обратной глубины (до нормировки) — для оценки «что рядом»
        # с исключением персонажа и HUD: в кадре Genshin герой всегда в паре
        # метров от камеры и иначе перебивает любую оценку дали.
        self.last_raw_disp: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        self.last_latency_ms = 0.0
        self.frames = 0

    # ------------------------------------------------------------------ model
    def load(self) -> "DepthEstimator":
        if self._session is not None:
            return self
        import onnxruntime as ort

        path = self.cfg.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ONNX-модель не найдена: {path}\n"
                "Скачайте её: python scripts/fetch_model.py"
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(self.cfg.threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(path, so, providers=providers or None)
        self._input_name = self._session.get_inputs()[0].name
        return self

    @property
    def ready(self) -> bool:
        return self._session is not None

    def describe(self) -> str:
        if self._session is None:
            return "DepthEstimator(не загружен)"
        h, w = self.cfg.input_hw
        size_mb = os.path.getsize(self.cfg.model_path) / 1e6
        prov = self._session.get_providers()[0].replace("ExecutionProvider", "")
        return (f"DepthEstimator({os.path.basename(self.cfg.model_path)}, {size_mb:.0f} МБ, "
                f"вход {w}x{h}, {prov}, {self.last_latency_ms:.0f} мс/кадр)")

    # ------------------------------------------------------------- inference
    def _preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        th, tw = self.cfg.input_hw
        th, tw = _round_to_patch(th), _round_to_patch(tw)
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = Image.fromarray(np.ascontiguousarray(image_rgb)).resize((nw, nh), Image.BILINEAR)
        small = np.asarray(img, dtype=np.uint8)
        x = small.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        disp = np.asarray(raw[0, 0] if raw.ndim == 4 else raw[0], dtype=np.float32)
        self.last_raw_scale = float(np.percentile(disp, 98))
        self.last_raw_p2 = float(np.percentile(disp, 2))
        self.last_raw_disp = disp
        norm = self.normalizer(disp)
        out_hw = self.cfg.out_hw
        if out_hw and tuple(out_hw) != norm.shape:
            norm = np.asarray(Image.fromarray(norm).resize((out_hw[1], out_hw[0]), Image.BILINEAR),
                              dtype=np.float32)
        if self.cfg.ema > 0 and self._prev is not None and self._prev.shape == norm.shape:
            a = self.cfg.ema
            norm = a * norm + (1.0 - a) * self._prev
        self._prev = norm
        return norm

    def raw_scale_masked(self, frame_shape, mask_fn=None) -> float:
        """Сырая близость (p98) по кадру без персонажа и HUD.

        ``mask_fn(h, w)`` возвращает булеву маску исключаемых пикселей.
        """
        disp = self.last_raw_disp
        if disp is None:
            return float(self.last_raw_scale)
        if mask_fn is None:
            return float(np.percentile(disp, 98))
        m = mask_fn(*disp.shape)
        vals = disp[~m]
        if vals.size < disp.size * 0.05:
            return float(np.percentile(disp, 98))
        return float(np.percentile(vals, 98))

    def raw_depth_scale(self) -> float:
        """Сырая «близость» кадра (p98 обратной глубины) — без нормировки.

        Растёт, когда рядом поверхность (пол под ногами), и рушится на порядок,
        когда вокруг только даль: это и есть признак полёта над пропастью.
        """
        return float(self.last_raw_scale)

    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Карта близости: 1 = вплотную, 0 = далеко/горизонт."""
        if self._session is None:
            self.load()
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("нужен RGB-кадр HxWx3")
        t0 = time.perf_counter()
        x = self._preprocess(image_rgb)
        raw = self._session.run(None, {self._input_name: x})[0]
        prox = self._postprocess(raw)
        self.last_latency_ms = (time.perf_counter() - t0) * 1e3
        self.frames += 1
        return prox

    def reset(self) -> None:
        self._prev = None
        self.normalizer.reset()


def raw_scale_to_distance(raw_scale: float) -> float:
    """Переводит сырую шкалу в условную дистанцию до ближайшей поверхности."""
    return 1.0 / max(float(raw_scale), 1e-6)


def proximity_to_depth(prox: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Псевдо-глубина: монотонно растёт с расстоянием (условные единицы)."""
    return 1.0 / (np.clip(prox, 0.0, 1.0) + eps)
