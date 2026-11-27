"""End-to-end pipeline: frame -> HUD + detections -> tracks -> state + overlay."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from .capture import ScreenSource, VideoSource  # noqa: F401  (re-export)
from .hud import HudReader
from .state import StateBuilder
from .track import SortTracker
from .types import GameState

_COLORS = [
    (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]


@dataclass
class PipelineOutput:
    state: GameState
    frame: np.ndarray  # annotated BGR frame


class Pipeline:
    def __init__(self, source, detector, hud_reader: HudReader, tracker: SortTracker | None = None):
        self.source = source
        self.detector = detector
        self.hud = hud_reader
        self.tracker = tracker or SortTracker()
        self.builder = StateBuilder(backend_name=getattr(detector, "name", ""))
        self._fps = 0.0
        self._last_t: Optional[float] = None

    def step(self) -> Optional[PipelineOutput]:
        got = self.source.read()
        if got is None:
            return None
        timestamp, frame = got

        if self._last_t is not None:
            dt = max(timestamp - self._last_t, 1e-6)
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else self._fps * 0.9 + inst * 0.1
        self._last_t = timestamp

        hud_reading = self.hud.read(frame)
        detections = self.detector.detect(frame, hud=hud_reading)
        tracks = self.tracker.update(detections, timestamp)
        state = self.builder.build(
            timestamp=timestamp,
            frame_size=(frame.shape[1], frame.shape[0]),
            hud=hud_reading,
            tracks=tracks,
            detections=detections,
            fps=self._fps,
        )
        annotated = annotate(frame, state, backend=self.builder.backend_name)
        return PipelineOutput(state=state, frame=annotated)

    def run(self, max_frames: int = 0, on_output=None) -> int:
        """Process frames until the source is exhausted (or ``max_frames``)."""
        n = 0
        while True:
            out = self.step()
            if out is None:
                break
            n += 1
            if on_output is not None:
                on_output(out)
            if max_frames and n >= max_frames:
                break
        return n


# --------------------------------------------------------------------------- #
# Overlay rendering
# --------------------------------------------------------------------------- #
def annotate(frame: np.ndarray, state: GameState, backend: str = "") -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    scale = w / 1280.0
    th = max(1, int(round(scale)))

    for i, enemy in enumerate(state.enemies):
        color = _COLORS[enemy.track_id % len(_COLORS)]
        if enemy.cls_name == "boss":
            color = (0, 0, 255)
        x1, y1, x2, y2 = (int(v) for v in enemy.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)

        # velocity arrow (1/2 s ahead)
        vx, vy = enemy.velocity
        tip = (int(enemy.bbox[0] + (enemy.bbox[2] - enemy.bbox[0]) / 2 + vx * 0.5),
               int(enemy.bbox[1] + (enemy.bbox[3] - enemy.bbox[1]) / 2 + vy * 0.5))
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.arrowedLine(img, center, tip, color, th, tipLength=0.25)

        # predicted ghost box
        if enemy.predicted_bbox is not None:
            px1, py1, px2, py2 = (int(v) for v in enemy.predicted_bbox)
            cv2.rectangle(img, (px1, py1), (px2, py2), color, 1, cv2.LINE_AA)

        hp = "—" if enemy.hp_frac is None else f"{enemy.hp_frac * 100:.0f}%"
        speed = (vx * vx + vy * vy) ** 0.5
        label = f"#{enemy.track_id} {enemy.cls_name} hp:{hp} v:{speed:.0f}px/s"
        _label(img, label, x1, y1 - 6, color, scale)

    # --- status panel ---
    panel = np.zeros((int(150 * scale), int(330 * scale), 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)
    y0 = int(16 * scale)
    hud = state.hud

    def _bar(canvas, y, frac, color, text):
        frac = 0.0 if frac is None else float(np.clip(frac, 0.0, 1.0))
        bw = int(canvas.shape[1] * 0.55)
        x = int(12 * scale)
        yy = int(y)
        bh = int(14 * scale)
        cv2.rectangle(canvas, (x, yy), (x + bw, yy + bh), (70, 70, 76), -1)
        cv2.rectangle(canvas, (x, yy), (x + int(bw * frac), yy + bh), color, -1)
        cv2.putText(canvas, text, (x + bw + int(10 * scale), yy + bh - int(2 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (235, 235, 235), 1, cv2.LINE_AA)

    hp_text = "HP --"
    if hud.player_hp_frac is not None:
        if hud.player_hp_cur is not None and hud.player_hp_max is not None:
            hp_text = f"HP {hud.player_hp_cur}/{hud.player_hp_max}"
        else:
            hp_text = f"HP {hud.player_hp_frac * 100:.1f}%"
    _bar(panel, y0, hud.player_hp_frac, (80, 200, 90), hp_text)
    _bar(panel, y0 + int(26 * scale), hud.stamina_frac, (60, 200, 235),
         f"STA {'—' if hud.stamina_frac is None else f'{hud.stamina_frac * 100:.0f}%'}")
    if hud.boss_hp_frac is not None:
        _bar(panel, y0 + int(52 * scale), hud.boss_hp_frac, (60, 70, 235),
             f"BOSS {hud.boss_hp_frac * 100:.0f}%")

    info = f"{state.fps:4.0f} fps | {backend or '?':9s} | enemies: {len(state.enemies)} | {'COMBAT' if state.in_combat else 'peace'}"
    cv2.putText(panel, info, (int(12 * scale), int(128 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42 * scale, (200, 200, 200), 1, cv2.LINE_AA)

    img[0:panel.shape[0], 0:panel.shape[1]] = panel
    return img


def _label(img, text, x, y, color, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45 * scale
    (tw, th), _ = cv2.getTextSize(text, font, fs, 1)
    x = max(0, x)
    y = max(int(th + 2), y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (20, 20, 24), -1)
    cv2.putText(img, text, (x + 2, y - 2), font, fs, color, 1, cv2.LINE_AA)
