"""Захват экрана и эмуляция ввода (Windows / Linux / macOS).

Genshin Impact — игра с агрессивным античитом, поэтому здесь нет ни инъекций в
процесс, ни чтения памяти: только пиксели экрана и настоящие клавиши/мышь,
как у живого игрока. Это принципиальное ограничение, а не удобство.

Ввод (клавиши и мышь) реализован двумя путями:
* ``pydirectinput`` — DirectInput-скан-коды Windows, которые принимает Unity;
* ``pyautogui`` — кроссплатформенный фолбэк.

Захват кадра:
* ``mss`` — быстро (5–10 мс на 1080p), предпочтительно;
* ``Pillow.ImageGrab`` — медленнее, но без лишних зависимостей.
"""
from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Клавиши Genshin: W/A/S/D, Space (прыжок/карабканье/планер), Shift (спринт).
KEYMAP = {
    "forward": "w",
    "back": "s",
    "left": "a",
    "right": "d",
    "sprint": "shift",
    "jump": "space",
    "glide": "space",   # в полёте повторный Space раскрывает планер
}


class ScreenCapture:
    """Быстрый захват области экрана в RGB numpy-массив."""

    def __init__(self, region: Optional[Tuple[int, int, int, int]] = None):
        self.region = region
        self._sct = None
        self._backend = "mss"
        try:
            import importlib.util
            if importlib.util.find_spec("mss") is None:
                self._backend = "pillow"
        except Exception:
            self._backend = "pillow"

    def _grab_mss(self) -> np.ndarray:
        import mss
        if self._sct is None:
            self._sct = mss.mss()
        mon = self._sct.monitors[1] if self.region is None else {
            "left": self.region[0], "top": self.region[1],
            "width": self.region[2], "height": self.region[3],
        }
        shot = self._sct.grab(mon)
        return np.asarray(shot, dtype=np.uint8)[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

    def _grab_pillow(self) -> np.ndarray:
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=self.region, all_screens=True) if self.region else ImageGrab.grab()
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    def grab(self) -> np.ndarray:
        if self._backend == "mss":
            try:
                return self._grab_mss()
            except Exception:
                self._backend = "pillow"
        return self._grab_pillow()

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            finally:
                self._sct = None


class InputBackend:
    """Эмуляция клавиш и движения мыши. Состояние клавиш — дельтами."""

    def __init__(self, enabled: bool = True, mouse_scale: float = 12.0):
        self.enabled = enabled
        self.mouse_scale = mouse_scale
        self._down: Dict[str, bool] = {}
        self._backend = None
        if enabled:
            self._backend = self._pick_backend()

    @staticmethod
    def _pick_backend():
        try:
            import pydirectinput
            pydirectinput.PAUSE = 0
            pydirectinput.FAILSAFE = False
            return ("direct", pydirectinput)
        except Exception:
            pass
        try:
            import pyautogui
            pyautogui.PAUSE = 0
            pyautogui.FAILSAFE = False
            return ("pyautogui", pyautogui)
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._backend is not None

    def describe(self) -> str:
        if not self.enabled:
            return "ввод выключен (dry-run)"
        if self._backend is None:
            return "ввод недоступен: установите pydirectinput или pyautogui"
        return f"ввод: {self._backend[0]} на {platform.system()}"

    # ---------------------------------------------------------------- клавиши
    def _send(self, key: str, down: bool) -> None:
        assert self._backend is not None
        kind, mod = self._backend
        if kind == "direct":
            if down:
                mod.keyDown(key)
            else:
                mod.keyUp(key)
        else:
            if down:
                mod.keyDown(key)
            else:
                mod.keyUp(key)

    def apply(self, keys: Dict[str, bool]) -> None:
        """Приводит нажатые клавиши к желаемому набору."""
        if not self.enabled or self._backend is None:
            return
        want = {KEYMAP[k] for k, v in keys.items() if v and k in KEYMAP}
        for key in list(self._down):
            if key not in want and self._down[key]:
                self._send(key, False)
                self._down[key] = False
        for key in want:
            if not self._down.get(key, False):
                self._send(key, True)
                self._down[key] = True

    # ------------------------------------------------------------------ мышь
    def turn(self, yaw: float) -> None:
        """Доворот камеры мышью: yaw -1..1 -> смещение в пикселях."""
        if not self.enabled or self._backend is None or abs(yaw) < 1e-3:
            return
        dx = int(round(float(np.clip(yaw, -1, 1)) * self.mouse_scale))
        if dx == 0:
            return
        kind, mod = self._backend
        try:
            if kind == "direct":
                mod.moveRel(dx, 0)
            else:
                mod.moveRel(dx, 0, _pause=False) if "moveRel" in dir(mod) else None
        except Exception:
            pass

    def release_all(self) -> None:
        if self._backend is None:
            return
        for key, is_down in list(self._down.items()):
            if is_down:
                try:
                    self._send(key, False)
                except Exception:
                    pass
        self._down.clear()


@dataclass
class StaminaBar:
    """Детектор жёлтой полосы выносливости (появляется при беге/карабканье).

    В Genshin полоса — жёлто-белая, появляется над персонажем по центру снизу.
    Примитивная, но честная эвристика по цвету и горизонтальной протяжённости;
    если полосы нет (полная выносливость/покой) — возвращаем 1.0.
    """

    def measure(self, frame_rgb: np.ndarray) -> Optional[float]:
        h, w = frame_rgb.shape[:2]
        band = frame_rgb[int(0.86 * h):int(0.93 * h), int(0.33 * w):int(0.67 * w)]
        if band.size == 0:
            return None
        r = band[:, :, 0].astype(np.int16)
        g = band[:, :, 1].astype(np.int16)
        b = band[:, :, 2].astype(np.int16)
        # жёлтый/белый: красный и зелёный высокие, синий заметно ниже
        mask = (r > 170) & (g > 140) & (b < 150) & ((r + g) > 2 * b + 60)
        if mask.mean() < 0.004:
            return None
        cols = mask.any(axis=0)
        fill = float(cols.sum()) / max(1, cols.size)
        return float(np.clip(fill, 0.0, 1.0))


class FpsMeter:
    def __init__(self, window: int = 30):
        self.window = window
        self.times = []

    def tick(self) -> None:
        self.times.append(time.perf_counter())
        if len(self.times) > self.window:
            self.times.pop(0)

    @property
    def fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0
