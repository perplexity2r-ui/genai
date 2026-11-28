"""Core data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Bounding box in absolute pixel coordinates: (x1, y1, x2, y2).
BBox = tuple[float, float, float, float]


@dataclass
class Detection:
    """Single-frame detection produced by a detector backend."""

    bbox: BBox
    score: float
    cls_id: int
    cls_name: str
    # Extra payload, e.g. {"hp_frac": 0.42} attached to a bar or entity.
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def wh(self) -> tuple[float, float]:
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass
class HudReading:
    """State read from the calibrated HUD regions."""

    player_hp_frac: Optional[float] = None
    player_hp_cur: Optional[int] = None
    player_hp_max: Optional[int] = None
    stamina_frac: Optional[float] = None
    boss_hp_frac: Optional[float] = None
    party_hp_fracs: list[Optional[float]] = field(default_factory=list)
    burst_ready: list[bool] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_hp_frac": self.player_hp_frac,
            "player_hp_cur": self.player_hp_cur,
            "player_hp_max": self.player_hp_max,
            "stamina_frac": self.stamina_frac,
            "boss_hp_frac": self.boss_hp_frac,
            "party_hp_fracs": self.party_hp_fracs,
            "burst_ready": self.burst_ready,
        }


@dataclass
class EnemyState:
    """A tracked enemy (or boss) with kinematics."""

    track_id: int
    cls_name: str
    bbox: BBox
    score: float
    hp_frac: Optional[float] = None
    velocity: tuple[float, float] = (0.0, 0.0)  # px/s in screen space
    predicted_bbox: Optional[BBox] = None  # extrapolated into the future
    age: float = 0.0  # seconds since first seen
    hits: int = 0
    distance_m: Optional[float] = None  # оценка расстояния до объекта, м
    element: Optional[str] = None       # стихия: anemo/geo/electro/hydro/cryo/pyro/dendro
    enemy_type: Optional[str] = None    # тонкий тип: slime/hilichurl/ruin_guard/...
    aggro: Optional[bool] = None        # атакует/заметил (видна полоска HP)
    threat: Optional[str] = None        # high / medium / low
    closing_px_s: Optional[float] = None  # скорость сближения с игроком (+ = идёт на вас)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.track_id,
            "class": self.cls_name,
            "bbox": [round(v, 1) for v in self.bbox],
            "score": round(self.score, 3),
            "hp_frac": None if self.hp_frac is None else round(self.hp_frac, 4),
            "velocity_px_s": [round(self.velocity[0], 1), round(self.velocity[1], 1)],
            "speed_px_s": round(float((self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5), 1),
            "predicted_bbox": (
                None if self.predicted_bbox is None else [round(v, 1) for v in self.predicted_bbox]
            ),
            "age_s": round(self.age, 2),
            "hits": self.hits,
        }
        if self.distance_m is not None:
            out["distance_m"] = round(self.distance_m, 1)
        if self.element is not None:
            out["element"] = self.element
        if self.enemy_type is not None:
            out["enemy_type"] = self.enemy_type
        if self.aggro is not None:
            out["aggro"] = self.aggro
        if self.threat is not None:
            out["threat"] = self.threat
        if self.closing_px_s is not None:
            out["closing_px_s"] = round(self.closing_px_s, 1)
        return out


@dataclass
class GameState:
    """Full per-frame world state exposed to consumers (overlay, bots, logs)."""

    timestamp: float
    frame_size: tuple[int, int]  # (w, h)
    fps: float = 0.0
    hud: HudReading = field(default_factory=HudReading)
    enemies: list[EnemyState] = field(default_factory=list)
    interactables: list[EnemyState] = field(default_factory=list)  # сундуки/телепорты/окулиусы/головоломки
    minimap: dict[str, Any] = field(default_factory=dict)  # курс, иконки, движение
    enemies_summary: dict[str, Any] = field(default_factory=dict)  # сводка по всем врагам
    in_combat: bool = False
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "frame_size": list(self.frame_size),
            "fps": round(self.fps, 1),
            "player": {
                "hp_frac": self.hud.player_hp_frac,
                "hp_cur": self.hud.player_hp_cur,
                "hp_max": self.hud.player_hp_max,
                "stamina_frac": self.hud.stamina_frac,
            },
            "enemies": [e.to_dict() for e in self.enemies],
            "enemies_summary": self.enemies_summary,
            "interactables": [i.to_dict() for i in self.interactables],
            "minimap": self.minimap,
            "ui": {
                "in_combat": self.in_combat,
                "boss_hp_frac": self.hud.boss_hp_frac,
                "party_hp_fracs": self.hud.party_hp_fracs,
                "burst_ready": self.hud.burst_ready,
            },
            "backend": self.backend,
        }
