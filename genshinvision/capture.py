"""Frame sources: live screen capture, video files, images."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


class ScreenSource:
    """Grabs frames from a monitor (or a screen region) via mss."""

    def __init__(self, monitor: int = 1, region: Optional[tuple[int, int, int, int]] = None):
        try:
            import mss  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pip install mss") from exc
        import mss

        self._sct = mss.mss()
        if region is not None:
            x, y, w, h = region
            self._monitor = {"left": x, "top": y, "width": w, "height": h}
        else:
            mons = self._sct.monitors
            if monitor >= len(mons):
                raise ValueError(f"Монитор {monitor} не найден (доступно: {len(mons) - 1})")
            self._monitor = mons[monitor]
        self.frame_size = (self._monitor["width"], self._monitor["height"])

    def read(self) -> tuple[float, np.ndarray]:
        shot = np.array(self._sct.grab(self._monitor))  # BGRA
        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        return time.time(), frame

    def close(self) -> None:
        self._sct.close()


class VideoSource:
    """Reads a video file, optionally paced to real time."""

    def __init__(self, path: str | Path, realtime: bool = False):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео: {path}")
        self.realtime = realtime
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (w, h)
        self._t = 0.0

    def read(self) -> Optional[tuple[float, np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._t += 1.0 / max(self.fps, 1.0)
        if self.realtime:
            time.sleep(1.0 / max(self.fps, 1.0))
        return self._t, frame

    def close(self) -> None:
        self.cap.release()


class ImageSource:
    """A single still image, repeated forever (useful for calibration/tests)."""

    def __init__(self, path: str | Path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Не удалось открыть изображение: {path}")
        self.frame = img
        self.frame_size = (img.shape[1], img.shape[0])
        self._t = 0.0

    def read(self) -> tuple[float, np.ndarray]:
        self._t += 1.0 / 30.0
        return self._t, self.frame.copy()


class IterSource:
    """Wraps any iterable of frames (used by the synthetic scene and tests)."""

    def __init__(self, frames: Iterator[np.ndarray], frame_size: tuple[int, int], fps: float = 30.0):
        self._frames = frames
        self.frame_size = frame_size
        self.fps = fps
        self._t = 0.0

    def read(self) -> Optional[tuple[float, np.ndarray]]:
        try:
            frame = next(self._frames)
        except StopIteration:
            return None
        self._t += 1.0 / max(self.fps, 1.0)
        return self._t, frame
