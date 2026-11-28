"""Визуализация: карта глубины + разметка восприятия + телеметрия агента.

Только numpy/PIL — работает «из коробки», без OpenCV.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .perception import SceneAnalysis

# Шрифты с кириллицей: стандартный PIL-шрифт её не содержит (рисует квадраты).
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
)


_FONT_CACHE = {}


def load_font(size: int = 14):
    """Шрифт с кириллицей; если не нашли — стандартный (латиница)."""
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in _FONT_CANDIDATES:
        try:
            _FONT_CACHE[size] = (ImageFont.truetype(path, size), True)
            return _FONT_CACHE[size]
        except Exception:
            continue
    try:
        _FONT_CACHE[size] = (ImageFont.load_default(size=size), False)
    except TypeError:
        _FONT_CACHE[size] = (ImageFont.load_default(), False)
    return _FONT_CACHE[size]

_PALETTE = np.array([
    [4, 4, 18], [28, 20, 78], [71, 24, 120], [120, 32, 130],
    [163, 46, 124], [203, 66, 106], [233, 99, 84], [248, 160, 105], [252, 225, 175],
], dtype=np.uint8)


def colorize(prox: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Карта близости -> RGB (тёмно-синий далеко, светло-жёлтый вплотную)."""
    p = np.clip(np.nan_to_num(np.asarray(prox, dtype=np.float32), nan=0.0), 0.0, 1.0)
    if size:
        p = np.asarray(Image.fromarray(p).resize((size[1], size[0]), Image.BILINEAR), np.float32)
    idx = (p * (len(_PALETTE) - 1)).astype(np.uint8)
    return _PALETTE[idx]


def overlay(scene: SceneAnalysis, prox: np.ndarray, size: Tuple[int, int] = (360, 640),
            show_sectors: bool = True) -> np.ndarray:
    """Цветная карта глубины + сектора и подписи признаков."""
    h, w = size
    img = Image.fromarray(colorize(prox, size=(h, w)))
    d = ImageDraw.Draw(img)
    if show_sectors:
        n = max(1, len(scene.sector_dist))
        for i, s in enumerate(scene.sector_dist):
            x = int(w * i / n)
            shade = int(120 + 135 * min(1.0, float(s)))
            d.line([(x, 0), (x, h)], fill=(shade, shade, shade), width=1)
            d.text((x + 2, h - 14), f"{s:.1f}", fill=(255, 255, 255))
    # Указатель направления к цели.
    tx = int(w * (scene.target_dir + 1) / 2)
    d.polygon([(tx, 8), (tx - 7, 24), (tx + 7, 24)], fill=(60, 255, 120))
    flags = []
    if scene.wall:
        flags.append(f"WALL {scene.wall_prox:.2f}")
    if scene.ledge:
        flags.append(f"LEDGE {scene.ledge_prox:.2f}")
    if scene.falling:
        flags.append("FALLING")
    label = "  ".join(flags) or "clear"
    font, cyrillic = load_font(14)
    try:
        lw = int(d.textlength(label, font=font))
    except Exception:
        lw = 8 * len(label)
    d.rectangle([4, 3, lw + 14, 22], fill=(0, 0, 0))
    d.text((6, 5), label, fill=(255, 255, 0), font=font)
    return np.asarray(img)


def draw_telemetry(img: np.ndarray, lines: Sequence[str], color=(0, 255, 0)) -> np.ndarray:
    out = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(out)
    font, cyrillic = load_font(14)
    y = 26
    for line in lines:
        text = str(line)
        if not cyrillic:
            text = text.translate(_LATIN_FALLBACK)
        # Подложка по фактической ширине текста: читаемо и на светлом кадре,
        # и на светлой карте глубины.
        try:
            tw = int(d.textlength(text, font=font))
        except Exception:
            tw = 7 * len(text)
        d.rectangle([4, y - 2, min(tw + 14, out.width - 2), y + 17], fill=(0, 0, 0))
        d.text((7, y), text, fill=color, font=font)
        y += 19
    return np.asarray(out)


_LATIN_FALLBACK = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "E",
    "Ж": "Zh", "З": "Z", "И": "I", "Й": "Y", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Ch", "Ш": "Sh", "Щ": "Sch", "Ъ": "",
    "Ы": "Y", "Ь": "", "Э": "E", "Ю": "Yu", "Я": "Ya",
})


def strip(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Склеивает кадры горизонтально."""
    return np.concatenate([f if f.ndim == 3 else np.stack([f] * 3, -1) for f in frames], axis=1)


def _text_block_width(lines: Sequence[str]) -> int:
    """Ширина самой длинной подписи в пикселях (для запаса под телеметрию)."""
    font, cyrillic = load_font(14)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    widest = 0
    for line in lines:
        text = str(line) if cyrillic else str(line).translate(_LATIN_FALLBACK)
        try:
            widest = max(widest, int(probe.textlength(text, font=font)))
        except Exception:
            widest = max(widest, 7 * len(text))
    return widest + 18


def panel(image_rgb: np.ndarray, scene: SceneAnalysis,
          size: Tuple[int, int] = (360, 640), info: Optional[dict] = None) -> np.ndarray:
    """Сравнение «кадр | карта глубины» с телеметрией.

    Панель расширяется до ширины подписей, чтобы строки не вылезали
    за карту глубины соседнего кадра.
    """
    photo = np.asarray(Image.fromarray(image_rgb).resize((size[1], size[0]), Image.BILINEAR))
    vis = overlay(scene, scene.prox if scene.prox is not None else np.zeros((8, 8)), size=size)
    lines = [f"{k}: {v}" for k, v in (info or {}).items()]
    out = np.concatenate([photo, vis], axis=1)
    need = _text_block_width(lines)
    if need > out.shape[1]:
        pad = np.zeros((out.shape[0], need - out.shape[1], 3), np.uint8)
        out = np.concatenate([out, pad], axis=1)
    return draw_telemetry(out, lines)
