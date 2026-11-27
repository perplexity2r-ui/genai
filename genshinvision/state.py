"""Assemble per-frame :class:`GameState` from detections, tracks, HUD and minimap."""
from __future__ import annotations

from typing import Any, Optional

from .detect import INTERACTABLE_CLASSES
from .track import TrackState
from .types import Detection, EnemyState, GameState, HudReading

_ENTITY_CLASSES = ("enemy", "elite", "boss")


class StateBuilder:
    """Merges all signals into one JSON-serialisable world state."""

    def __init__(self, backend_name: str = "", calibration: Optional[dict[str, Any]] = None):
        self.backend_name = backend_name
        self.calib = calibration or {}
        self.focal_px = float(self.calib.get("focal_px", 900.0))
        self.size_refs = dict(self.calib.get("object_real_size_m") or {})

    def _distance_m(self, cls_name: str, bbox) -> Optional[float]:
        real = self.size_refs.get(cls_name)
        h_px = bbox[3] - bbox[1]
        if not real or h_px <= 1:
            return None
        return real * self.focal_px / h_px

    @staticmethod
    def _tracked_to_state(tr: TrackState, cls_name: str) -> EnemyState:
        hp_frac = tr.attrs.get("hp_frac")
        return EnemyState(
            track_id=tr.track_id,
            cls_name=cls_name,
            bbox=tr.bbox,
            score=tr.score,
            hp_frac=hp_frac,
            velocity=tr.velocity,
            predicted_bbox=tr.predicted_bbox(0.5),  # где окажется через 0.5 c
            age=tr.age,
            hits=tr.hits,
            element=tr.attrs.get("element"),
        )

    def build(
        self,
        timestamp: float,
        frame_size: tuple[int, int],
        hud: HudReading,
        tracks: list[TrackState],
        detections: list[Detection] | None = None,
        fps: float = 0.0,
        minimap: Optional[dict[str, Any]] = None,
    ) -> GameState:
        enemies: list[EnemyState] = []
        interactables: list[EnemyState] = []
        for tr in tracks:
            if tr.cls_name in _ENTITY_CLASSES:
                st = self._tracked_to_state(tr, tr.cls_name)
                st.distance_m = self._distance_m(tr.cls_name, tr.bbox)
                enemies.append(st)
            elif tr.cls_name in INTERACTABLE_CLASSES:
                if tr.hits < 2:  # однокадровые ложные срабатывания не показываем
                    continue
                st = self._tracked_to_state(tr, tr.cls_name)
                st.distance_m = self._distance_m(tr.cls_name, tr.bbox)
                interactables.append(st)

        # чем ближе — тем раньше в списке
        interactables.sort(key=lambda e: e.distance_m if e.distance_m is not None else 1e9)

        in_combat = bool(enemies) or (hud.boss_hp_frac is not None and hud.boss_hp_frac > 0.01)

        return GameState(
            timestamp=timestamp,
            frame_size=frame_size,
            fps=fps,
            hud=hud,
            enemies=enemies,
            interactables=interactables,
            minimap=dict(minimap or {}),
            in_combat=in_combat,
            backend=self.backend_name,
        )
