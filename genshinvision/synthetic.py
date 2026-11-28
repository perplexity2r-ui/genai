"""Процедурная «геншиноподобная» боевая сцена с ground truth.

Питает демо и тесты: каждый кадр сопровождается истинными позициями/скоростями/HP
врагов, курсом игрока, иконками миникарты и объектами мира (сундук, телепорт,
кукулиусы, головоломка) — весь конвейер можно проверить без реального геймплея.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from .hud import DEFAULT_CALIBRATION


def hsv_color(h: float, s: float, v: float) -> tuple[int, int, int]:
    """HSV (OpenCV-шкала) → BGR кортеж."""
    arr = np.uint8([[[h, s, v]]])
    return tuple(int(c) for c in cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0, 0])


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
        # приглушённые тона: не должны совпадать с HSV-правилами слизней/объектов
        colors = [(85, 90, 150), (120, 105, 150), (105, 125, 155), (95, 80, 130)]
        for i in range(n_enemies):
            bar_w = 90 * s
            # каждый враг в своей вертикальной полосе: тела не пересекаются
            # (перекрытие врагов — случай для YOLO, эвристика его не разбирает)
            bands = [(200 * s, 460 * s), (500 * s, 760 * s), (800 * s, 1080 * s)]
            lo_x, hi_x = bands[i % len(bands)]
            self.enemies.append(
                {
                    "id": i,
                    "cx": float(self._rng.uniform(lo_x + 75 * s, hi_x - 75 * s)),
                    "cy": float(self._rng.uniform(
                        max(self._zone[1] + self._bar_lift + 10, 250 * s),
                        self._zone[3] - 140 * s)),
                    "vx": float(self._rng.uniform(-1, 1) * 130 * s),
                    "vy": float(self._rng.uniform(-1, 1) * 70 * s),
                    "x_range": (lo_x, hi_x),
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

        # --- слизни: враги БЕЗ полосок HP (видны по цветному телу) --------
        # гуляют в нижней полосе зоны боя, чтобы не перекрывать гуманоидов
        self.slimes: list[dict[str, Any]] = [
            {
                "id": 100, "cx": 315 * s, "cy": 538 * s,
                "vx": 55 * s, "vy": 12 * s, "r": 35 * s,
                "x_range": (250 * s, 385 * s),
                "element": "pyro",
                "color": hsv_color(8, 200, 235), "hi": hsv_color(8, 150, 255),
            },
            {
                "id": 101, "cx": 735 * s, "cy": 538 * s,
                "vx": -50 * s, "vy": -11 * s, "r": 35 * s,
                "x_range": (675 * s, 790 * s),
                "element": "electro",
                "color": hsv_color(150, 190, 230), "hi": hsv_color(150, 150, 255),
            },
        ]

        # ---------------- миникарта: игрок едет по текстуре ----------------
        mm = self.calib["minimap"]
        self._mm_r = int(mm["r"] * width)
        self._mm_cx = int(mm["cx"] * width)
        self._mm_cy = int(mm["cy"] * height)
        self._map_tex = self._make_map_texture(1024, 1024)
        self.player_pos = np.array([480.0, 520.0])          # в координатах текстуры
        self.player_vel = np.array([42.0, -16.0]) * s       # px/с (1:1 с текстурой)
        self._mm_icons_world = [                            # (x, y, тип) в текстуре
            (440.0, 545.0, "quest"),
            (560.0, 460.0, "teleport"),
            (620.0, 520.0, "statue"),
            (660.0, 420.0, "teleport"),
        ]

        # ---------------- объекты мира (координаты кадра) ----------------
        self.objects = [
            {"class": "chest", "element": None, "x": 420 * s, "y": 560 * s, "w": 110 * s, "h": 80 * s},
            {"class": "teleport_waypoint", "element": None, "x": 1120 * s, "y": 380 * s, "w": 44 * s, "h": 130 * s},
            {"class": "oculus", "element": "anemo", "x": 620 * s, "y": 575 * s, "w": 48 * s, "h": 48 * s},
            {"class": "oculus", "element": "electro", "x": 800 * s, "y": 570 * s, "w": 48 * s, "h": 48 * s},
            {"class": "puzzle", "element": "monument", "x": 945 * s, "y": 530 * s, "w": 60 * s, "h": 110 * s},
        ]

        self._bg = self._make_background()
        self.last_truth: dict[str, Any] = {}

    # ------------------------------------------------------------------ util
    def _make_background(self) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        horizon = int(0.46 * self.h)
        sky_top = np.array([195, 165, 125], dtype=np.float64)   # BGR: насыщенность в «безопасном» коридоре
        sky_bot = np.array([215, 185, 135], dtype=np.float64)
        for y in range(horizon):
            a = y / max(horizon, 1)
            img[y, :] = (sky_top * (1 - a) + sky_bot * a).astype(np.uint8)
        ground_top = np.array([95, 110, 85], dtype=np.float64)
        ground_bot = np.array([45, 55, 42], dtype=np.float64)
        for y in range(horizon, self.h):
            a = (y - horizon) / max(self.h - horizon, 1)
            img[y, :] = (ground_top * (1 - a) + ground_bot * a).astype(np.uint8)
        # texture noise
        noise = self._rng.normal(0, 3, img.shape).astype(np.int16)
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

    def _make_map_texture(self, tw: int, th: int) -> np.ndarray:
        """Текстура карты: спокойные зелёно-серые тона, без белого/ярких цветов."""
        tex = np.zeros((th, tw, 3), dtype=np.uint8)
        tex[:] = (78, 92, 82)
        for _ in range(60):
            rw = int(self._rng.uniform(40, 170))
            rh = int(self._rng.uniform(30, 130))
            x = int(self._rng.uniform(0, tw - rw))
            y = int(self._rng.uniform(0, th - rh))
            shade = int(self._rng.integers(-14, 16))
            color = (78 + shade, 96 + shade, 88 + shade)
            cv2.ellipse(tex, (x + rw // 2, y + rh // 2), (rw // 2, rh // 2), 0, 0, 360, color, -1)
        noise = self._rng.normal(0, 3, tex.shape).astype(np.int16)
        return np.clip(tex.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _draw_bar(self, img, x1, y1, x2, y2, frac, fill_bgr, bg_bgr=(42, 42, 48)):
        """Горизонтальная полоска со светлой рамкой (как в игровом HUD)."""
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (175, 175, 178), -1)  # светлая рамка
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_bgr, -1)
        fx = int(x1 + max(0.0, min(1.0, frac)) * (x2 - x1))
        if fx > x1:
            cv2.rectangle(img, (x1, y1), (fx, y2), fill_bgr, -1)

    def heading_deg(self) -> float:
        return float((10.0 * self._t) % 360.0)

    # ------------------------------------------------------------ minimap ---
    def _draw_minimap(self, img: np.ndarray) -> dict[str, Any]:
        r = self._mm_r
        th, tw = self._map_tex.shape[:2]
        px, py = self.player_pos % [tw, th]
        x1, y1 = int(px) - r, int(py) - r
        ys = np.arange(y1, y1 + 2 * r) % th
        xs = np.arange(x1, x1 + 2 * r) % tw
        crop = self._map_tex[np.ix_(ys, xs)].copy()

        # иконки в координатах кадра (сдвигаются вместе с картой)
        icons_truth = []
        icon_colors = {
            "teleport": hsv_color(110, 200, 235),
            "statue": hsv_color(110, 80, 235),
            "quest": hsv_color(27, 210, 240),
        }
        icon_r = {"teleport": 7, "statue": 8, "quest": 6}
        for wx, wy, typ in self._mm_icons_world:
            ix, iy = wx - px + r, wy - py + r
            dist = ((ix - r) ** 2 + (iy - r) ** 2) ** 0.5
            if dist > r * 0.88:
                continue
            rad = icon_r[typ]
            cv2.circle(crop, (int(ix), int(iy)), rad, icon_colors[typ], -1)
            cv2.circle(crop, (int(ix), int(iy)), rad, (30, 30, 34), 1)
            icons_truth.append({
                "type": typ,
                "x": round((ix - r) / r, 3),
                "y": round((iy - r) / r, 3),
            })

        # стрелка игрока в центре (белая, остриё — направление курса)
        heading = self.heading_deg()
        ang = np.radians(heading)
        dirv = np.array([np.sin(ang), -np.cos(ang)])
        perp = np.array([np.cos(ang), np.sin(ang)])
        c = np.array([r, r], dtype=np.float64)
        tip = c + 0.28 * r * dirv
        back = c - 0.10 * r * dirv
        bl = back + 0.13 * r * perp
        br = back - 0.13 * r * perp
        pts = np.array([tip, bl, br], dtype=np.int32)
        cv2.fillConvexPoly(crop, pts, (245, 245, 245))

        # вклеиваем круглую миникарту в кадр (с обрезкой по краям кадра)
        mask = np.zeros((2 * r, 2 * r), dtype=np.uint8)
        cv2.circle(mask, (r, r), r, 255, -1)
        y0, x0 = self._mm_cy - r, self._mm_cx - r
        cy0, cx0 = max(0, y0), max(0, x0)
        cy1, cx1 = min(self.h, y0 + 2 * r), min(self.w, x0 + 2 * r)
        if cy1 > cy0 and cx1 > cx0:
            sub = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
            roi = img[cy0:cy1, cx0:cx1]
            inv = cv2.bitwise_not(mask[sub])
            bg = cv2.bitwise_and(roi, roi, mask=inv)
            fg = cv2.bitwise_and(crop[sub], crop[sub], mask=mask[sub])
            img[cy0:cy1, cx0:cx1] = cv2.add(bg, fg)
        cv2.circle(img, (self._mm_cx, self._mm_cy), r, (225, 225, 225), 2)

        return {
            "heading_deg": heading,
            "icons": icons_truth,
            "motion_px_s": [float(self.player_vel[0]), float(self.player_vel[1])],
        }

    # ------------------------------------------------------------- objects --
    def _draw_objects(self, img: np.ndarray) -> list[dict[str, Any]]:
        s = self._s
        truth = []
        for obj in self.objects:
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            if obj["class"] == "chest":
                gold = hsv_color(24, 210, 230)
                gold_dark = hsv_color(24, 210, 190)
                cv2.rectangle(img, (int(x), int(y + 0.35 * h)), (int(x + w), int(y + h)), gold_dark, -1)
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + 0.4 * h)), gold, -1)
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (40, 40, 45), 2)
                cv2.rectangle(img, (int(x + 0.42 * w), int(y + 0.3 * h)),
                              (int(x + 0.58 * w), int(y + 0.55 * h)), (50, 50, 55), -1)
            elif obj["class"] == "teleport_waypoint":
                cy_p = hsv_color(110, 200, 255)
                cy_dim = hsv_color(110, 170, 235)
                cv2.ellipse(img, (int(x + w / 2), int(y + h)), (int(w * 1.1), int(0.32 * w)),
                            0, 0, 360, cy_dim, -1)                      # основание-кольцо
                cv2.rectangle(img, (int(x), int(y + 0.12 * h)), (int(x + w), int(y + h)), cy_p, -1)
                cv2.circle(img, (int(x + w / 2), int(y + 0.1 * h)), int(w * 0.32), cy_p, -1)
            elif obj["class"] == "oculus":
                col = {"anemo": hsv_color(55, 175, 235), "electro": hsv_color(150, 175, 230)}
                core = col.get(obj["element"], hsv_color(55, 175, 235))
                glow = {"anemo": hsv_color(55, 110, 255), "electro": hsv_color(150, 110, 255)}
                outer = glow.get(obj["element"], hsv_color(55, 110, 255))
                cx, cy = int(x + w / 2), int(y + h / 2)
                bob = int(4 * s * np.sin(self._t * 2.0 + x))
                cv2.circle(img, (cx, cy + bob), int(w / 2), outer, -1)      # свечение
                cv2.circle(img, (cx, cy + bob), int(w / 3.2), core, -1)     # ядро
            elif obj["class"] == "puzzle":
                stone = (120, 125, 135)
                cv2.rectangle(img, (int(x + 0.15 * w), int(y + 0.3 * h)),
                              (int(x + 0.85 * w), int(y + h)), stone, -1)
                cv2.rectangle(img, (int(x + 0.15 * w), int(y + 0.3 * h)),
                              (int(x + 0.85 * w), int(y + h)), (60, 62, 66), 2)
                sym = hsv_color(11, 210, 245)
                cv2.circle(img, (int(x + w / 2), int(y + 0.22 * h)), int(w * 0.42), sym, -1)
                cv2.circle(img, (int(x + w / 2), int(y + 0.22 * h)), int(w * 0.2),
                           hsv_color(11, 210, 185), -1)
            truth.append({
                "class": obj["class"],
                "element": obj["element"],
                "bbox": [x, y, x + w, y + h],
            })
        return truth

    # ------------------------------------------------------------------ step
    def read(self) -> Optional[tuple[float, np.ndarray]]:
        if self.max_frames and self._frame_idx >= self.max_frames:
            return None
        dt = 1.0 / self.fps
        self._t += dt
        self._frame_idx += 1
        img = self._bg.copy()
        s = self._s

        # --- гуманоиды (враги с полосками HP) ---------------------------
        truth_enemies = []
        for e in self.enemies:
            e["cx"] += e["vx"] * dt
            e["cy"] += e["vy"] * dt
            lo_x, hi_x = e["x_range"]
            m = e["body_w"] / 2 + 10 * s
            lo_y = max(self._zone[1] + self._bar_lift + 10, 250 * s)
            hi_y = self._zone[3] - 140 * s
            if e["cx"] < lo_x + m or e["cx"] > hi_x - m:
                e["vx"] *= -1
                e["cx"] = float(np.clip(e["cx"], lo_x + m, hi_x - m))
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
                    "enemy_type": "humanoid",
                    "element": None,
                }
            )

        # --- слизни (враги без полосок HP, рисуются поверх) ------------
        for sl in self.slimes:
            sl["cx"] += sl["vx"] * dt
            sl["cy"] += sl["vy"] * dt
            lo_x, hi_x = sl["x_range"]
            lo_y, hi_y = 528 * s, 548 * s  # нижняя полоса: не перекрывает гуманоидов
            if sl["cx"] < lo_x + sl["r"] or sl["cx"] > hi_x - sl["r"]:
                sl["vx"] *= -1
                sl["cx"] = float(np.clip(sl["cx"], lo_x + sl["r"], hi_x - sl["r"]))
            if sl["cy"] < lo_y or sl["cy"] > hi_y:
                sl["vy"] *= -1
                sl["cy"] = float(np.clip(sl["cy"], lo_y, hi_y))
            cx, cy, r = int(sl["cx"]), int(sl["cy"]), int(sl["r"])
            cv2.circle(img, (cx, cy), r + 2, (25, 25, 30), -1)          # обводка
            cv2.circle(img, (cx, cy), r, sl["color"], -1)              # тело
            cv2.ellipse(img, (cx - int(0.3 * r), cy - int(0.35 * r)),
                        (int(0.45 * r), int(0.3 * r)), 0, 0, 360, sl["hi"], -1)  # блик
            truth_enemies.append({
                "id": sl["id"],
                "bbox": [cx - r, cy - r, cx + r, cy + r],
                "bar_bbox": None,
                "velocity": [sl["vx"], sl["vy"]],
                "hp_frac": None,
                "enemy_type": "slime",
                "element": sl["element"],
            })

        # --- объекты мира ---------------------------------------------
        truth_objects = self._draw_objects(img)

        # --- босс ------------------------------------------------------
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

        # --- игрок едет по карте --------------------------------------
        self.player_pos = self.player_pos + self.player_vel * dt

        # --- HUD игрока ------------------------------------------------
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

        # --- миникарта -------------------------------------------------
        minimap_truth = self._draw_minimap(img)

        self.last_truth = {
            "player_hp_frac": self.player_hp_frac,
            "player_hp_cur": cur,
            "enemies": truth_enemies,
            "objects": truth_objects,
            "boss": boss_truth,
            "minimap": minimap_truth,
        }

        if self.realtime:
            time.sleep(1.0 / self.fps)
        return self._t, img

    def close(self) -> None:  # source protocol
        pass
