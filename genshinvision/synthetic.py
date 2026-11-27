"""Procedural Genshin-like combat scene with ground truth.

Powers the live demo and the tests: every frame is accompanied by the true
enemy positions / velocities / HP so the whole pipeline can be validated
without any real gameplay footage.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from .hud import DEFAULT_CALIBRATION


class SyntheticGameSource:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        n_enemies: int = 3,
        fps: float = 30.0,
        seed: int = 7,
        max_frames: int = 0,
        realtime: bool = False,
    ):
        self.w, self.h = width, height
        self.frame_size = (width, height)
        self.fps = fps
        self.max_frames = max_frames
        self.realtime = realtime
        self.calib = DEFAULT_CALIBRATION
        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._frame_idx = 0

        s = width / 1280.0
        self._s = s
        self.player_hp_max = int(self.calib.get("hp_max", 12000))
        self.player_hp_frac = 0.82

        # combat-region pixel bounds
        cr = self.calib["combat_region"]
        self._zone = (
            cr["x"] * width, cr["y"] * height,
            (cr["x"] + cr["w"]) * width, (cr["y"] + cr["h"]) * height,
        )
        # верхняя граница центров врагов: HP-полоска должна оставаться на экране
        self._bar_lift = 170.0 * s

        self.enemies: list[dict[str, Any]] = []
        colors = [(58, 72, 210), (170, 68, 210), (60, 160, 220), (90, 60, 160)]
        for i in range(n_enemies):
            bar_w = 90 * s
            self.enemies.append(
                {
                    "id": i,
                    "cx": float(self._rng.uniform(self._zone[0] + 120 * s, self._zone[2] - 120 * s)),
                    "cy": float(self._rng.uniform(
                        max(self._zone[1] + self._bar_lift + 10, 250 * s),
                        self._zone[3] - 140 * s)),
                    "vx": float(self._rng.uniform(-1, 1) * 130 * s),
                    "vy": float(self._rng.uniform(-1, 1) * 70 * s),
                    "hp_frac": float(self._rng.uniform(0.55, 1.0)),
                    "body_w": bar_w * 1.45,
                    "body_h": bar_w * 2.0,
                    "bar_w": bar_w,
                    "bar_h": 8 * s,
                    "color": colors[i % len(colors)],
                }
            )
        self.boss_active_at = 40
        self.boss_hp_frac = 0.86

        self._bg = self._make_background()
        self.last_truth: dict[str, Any] = {}

    # ------------------------------------------------------------------ util
    def _make_background(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        horizon = int(0.46 * self.h)
        sky_top = np.array([195, 160, 110], dtype=np.float64)   # BGR warm haze (насыщенный, не белый)
        sky_bot = np.array([215, 180, 130], dtype=np.float64)
        for y in range(horizon):
            a = y / max(horizon, 1)
            img[y, :] = (sky_top * (1 - a) + sky_bot * a).astype(np.uint8)
        ground_top = np.array([95, 110, 85], dtype=np.float64)
        ground_bot = np.array([45, 55, 42], dtype=np.float64)
        for y in range(horizon, self.h):
            a = (y - horizon) / max(self.h - horizon, 1)
            img[y, :] = (ground_top * (1 - a) + ground_bot * a).astype(np.uint8)
        # texture noise
        noise = self._rng.normal(0, 4, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # static rocks
        for _ in range(7):
            rw = int(self._rng.uniform(40, 130) * self._s)
            rh = int(self._rng.uniform(24, 70) * self._s)
            x = int(self._rng.uniform(0, self.w - rw))
            y = int(self._rng.uniform(0.55 * self.h, 0.92 * self.h - rh))
            cv2.rectangle(img, (x, y), (x + rw, y + rh), (72, 78, 68), -1)
            cv2.rectangle(img, (x, y), (x + rw, y + int(rh * 0.25)), (95, 100, 88), -1)
        return img

    def _draw_bar(self, img, x1, y1, x2, y2, frac, fill_bgr, bg_bgr=(42, 42, 48)):
        """Горизонтальная полоска со светлой рамкой (как в игровом HUD)."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (175, 175, 178), -1)  # светлая рамка
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_bgr, -1)
        fx = int(x1 + max(0.0, min(1.0, frac)) * (x2 - x1))
        if fx > x1:
            cv2.rectangle(img, (x1, y1), (fx, y2), fill_bgr, -1)

    # ------------------------------------------------------------------ step
    def read(self) -> Optional[tuple[float, np.ndarray]]:
        if self.max_frames and self._frame_idx >= self.max_frames:
            return None
        dt = 1.0 / self.fps
        self._t += dt
        self._frame_idx += 1
        img = self._bg.copy()
        s = self._s

        # --- enemies -----------------------------------------------------
        truth_enemies = []
        for e in self.enemies:
            e["cx"] += e["vx"] * dt
            e["cy"] += e["vy"] * dt
            m = 40 * s
            lo_x = self._zone[0] + m
            hi_x = self._zone[2] - m
            lo_y = max(self._zone[1] + self._bar_lift + 10, 250 * s)
            hi_y = self._zone[3] - 140 * s
            if e["cx"] < lo_x or e["cx"] > hi_x:
                e["vx"] *= -1
                e["cx"] = float(np.clip(e["cx"], lo_x, hi_x))
            if e["cy"] < lo_y or e["cy"] > hi_y:
                e["vy"] *= -1
                e["cy"] = float(np.clip(e["cy"], lo_y, hi_y))
            e["hp_frac"] = float(np.clip(e["hp_frac"] - 0.010 * dt - self._rng.uniform(0, 0.002), 0.05, 1.0))

            bw, bh = e["body_w"], e["body_h"]
            cx, cy = e["cx"], e["cy"]
            body_top = cy - bh / 2
            # body: rectangle + head circle (no white in palette)
            body = (int(cx - bw / 2), int(body_top), int(cx + bw / 2), int(body_top + bh))
            cv2.rectangle(img, (body[0], body[1]), (body[2], body[3]), e["color"], -1)
            cv2.rectangle(img, (body[0], body[1]), (body[2], body[3]), (25, 25, 30), 2)
            head_r = int(bw * 0.28)
            head_c = (int(cx), int(body_top - head_r * 0.6))
            cv2.circle(img, head_c, head_r, (120, 150, 210), -1)

            # HP bar above head
            bar_w, bar_h = e["bar_w"], e["bar_h"]
            bx1 = cx - bar_w / 2
            by1 = head_c[1] - head_r - bar_h - 8 * s
            self._draw_bar(img, bx1, by1, bx1 + bar_w, by1 + bar_h, e["hp_frac"], (238, 238, 238))

            truth_enemies.append(
                {
                    "id": e["id"],
                    "bbox": [body[0], body[1], body[2], body[3]],
                    "bar_bbox": [bx1, by1, bx1 + bar_w, by1 + bar_h],
                    "velocity": [e["vx"], e["vy"]],
                    "hp_frac": e["hp_frac"],
                }
            )

        # --- boss --------------------------------------------------------
        boss_truth = None
        if self._frame_idx >= self.boss_active_at:
            self.boss_hp_frac = float(max(0.02, self.boss_hp_frac - 0.004 * dt))
            b = self.calib["boss_hp_bar"]
            x1, y1 = b["x"] * self.w, b["y"] * self.h
            x2, y2 = x1 + b["w"] * self.w, y1 + b["h"] * self.h
            self._draw_bar(img, x1, y1, x2, y2, self.boss_hp_frac, (235, 235, 240))
            cv2.putText(img, "BOSS: Ruin Guardian", (int(x1), int(y1 - 6 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (235, 235, 235), 1, cv2.LINE_AA)
            boss_truth = {"hp_frac": self.boss_hp_frac}

        # --- player HUD --------------------------------------------------
        self.player_hp_frac = float(np.clip(
            self.player_hp_frac
            + (0.004 if self._frame_idx % 210 > 150 else -0.0035) * dt * 10
            - self._rng.uniform(0, 0.0006),
            0.15, 1.0))
        hp = self.calib["player_hp_bar"]
        self._draw_bar(img, hp["x"] * self.w, hp["y"] * self.h,
                       (hp["x"] + hp["w"]) * self.w, (hp["y"] + hp["h"]) * self.h,
                       self.player_hp_frac, (110, 215, 105))
        cur = int(self.player_hp_frac * self.player_hp_max)
        txt = self.calib["player_hp_text"]
        cv2.putText(img, f"HP {cur}/{self.player_hp_max}",
                    (int(txt["x"] * self.w + 4 * s), int((txt["y"] + txt["h"] * 0.75) * self.h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (250, 250, 250), 2, cv2.LINE_AA)

        stamina_frac = 0.55 + 0.4 * abs(((self._t * 0.12) % 1.0) - 0.5) * 2
        st = self.calib["stamina_bar"]
        self._draw_bar(img, st["x"] * self.w, st["y"] * self.h,
                       (st["x"] + st["w"]) * self.w, (st["y"] + st["h"]) * self.h,
                       stamina_frac, (80, 215, 235))

        for i, roi in enumerate(self.calib["party_bars"]):
            frac = 0.6 + 0.35 * abs(((self._t * 0.07 + i * 0.3) % 1.0) - 0.5) * 2
            self._draw_bar(img, roi["x"] * self.w, roi["y"] * self.h,
                           (roi["x"] + roi["w"]) * self.w, (roi["y"] + roi["h"]) * self.h,
                           frac, (110, 215, 105))

        for i, roi in enumerate(self.calib["burst_icons"]):
            ready = ((self._t * 0.10 + i * 0.37) % 1.0) > 0.45
            color = (250, 250, 250) if ready else (95, 95, 105)
            x1, y1 = roi["x"] * self.w, roi["y"] * self.h
            x2, y2 = x1 + roi["w"] * self.w, y1 + roi["h"] * self.h
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (30, 30, 34), 2)

        # minimap
        mm = self.calib["minimap"]
        center = (int(mm["cx"] * self.w), int(mm["cy"] * self.h))
        radius = int(mm["r"] * self.w)
        cv2.circle(img, center, radius, (60, 62, 58), -1)
        cv2.circle(img, center, radius, (225, 225, 225), 2)
        dot = (int(center[0] + radius * 0.4 * np.cos(self._t * 0.9)),
               int(center[1] + radius * 0.4 * np.sin(self._t * 0.9)))
        cv2.circle(img, dot, max(3, int(5 * s)), (90, 230, 90), -1)

        self.last_truth = {
            "player_hp_frac": self.player_hp_frac,
            "player_hp_cur": cur,
            "enemies": truth_enemies,
            "boss": boss_truth,
        }

        if self.realtime:
            time.sleep(1.0 / self.fps)
        return self._t, img

    def close(self) -> None:  # source protocol
        pass
