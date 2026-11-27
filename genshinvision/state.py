"""Assemble per-frame :class:`GameState` from detections, tracks and HUD readings."""
from __future__ import annotations

from .track import TrackState
from .types import Detection, EnemyState, GameState, HudReading

_ENTITY_CLASSES = ("enemy", "elite", "boss")


class StateBuilder:
    """Merges all signals into one JSON-serialisable world state."""

    def __init__(self, backend_name: str = ""):
        self.backend_name = backend_name

    def build(
        self,
        timestamp: float,
        frame_size: tuple[int, int],
        hud: HudReading,
        tracks: list[TrackState],
        detections: list[Detection] | None = None,
        fps: float = 0.0,
    ) -> GameState:
        enemies: list[EnemyState] = []
        for tr in tracks:
            if tr.cls_name not in _ENTITY_CLASSES:
                continue
            hp_frac = tr.attrs.get("hp_frac")
            enemies.append(
                EnemyState(
                    track_id=tr.track_id,
                    cls_name=tr.cls_name,
                    bbox=tr.bbox,
                    score=tr.score,
                    hp_frac=hp_frac,
                    velocity=tr.velocity,
                    predicted_bbox=tr.predicted_bbox(0.5),  # где окажется через 0.5 c
                    age=tr.age,
                    hits=tr.hits,
                )
            )

        in_combat = bool(enemies) or (hud.boss_hp_frac is not None and hud.boss_hp_frac > 0.01)

        return GameState(
            timestamp=timestamp,
            frame_size=frame_size,
            fps=fps,
            hud=hud,
            enemies=enemies,
            in_combat=in_combat,
            backend=self.backend_name,
        )
