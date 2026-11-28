"""Assemble per-frame :class:`GameState` from detections, tracks, HUD and minimap."""
from __future__ import annotations

from typing import Any, Optional

from .detect import INTERACTABLE_CLASSES, tier_of
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
        anchor = self.calib.get("player_anchor") or {"x": 0.5, "y": 0.6}
        self.player_anchor = (float(anchor["x"]), float(anchor["y"]))

    def _distance_m(self, cls_name: str, bbox) -> Optional[float]:
        real = self.size_refs.get(cls_name)
        if real is None:
            # для тонких типов берём уровень (tier)
            real = self.size_refs.get(tier_of(cls_name))
        h_px = bbox[3] - bbox[1]
        if not real or h_px <= 1:
            return None
        if h_px < 25:
            # слишком мелкий бокс (прокси по полоске HP босса) — дистанцию
            # по размеру не оцениваем
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
            enemy_type=tr.attrs.get("enemy_type"),
        )

    def _threat(self, st: EnemyState, frame_size: tuple[int, int]) -> None:
        """Угроза: дистанция + сближение с игроком (якорь — центр экрана)."""
        w, h = frame_size
        ax, ay = self.player_anchor[0] * w, self.player_anchor[1] * h
        cx = (st.bbox[0] + st.bbox[2]) / 2.0
        cy = (st.bbox[1] + st.bbox[3]) / 2.0
        dx, dy = ax - cx, ay - cy
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        # скорость сближения: положительная — враг идёт на игрока
        st.closing_px_s = (st.velocity[0] * dx + st.velocity[1] * dy) / norm
        dist = st.distance_m
        closing = st.closing_px_s or 0.0
        if (dist is not None and dist < 9.0) or closing > 80.0:
            st.threat = "high"
        elif (dist is not None and dist < 17.0) or closing > 35.0:
            st.threat = "medium"
        else:
            st.threat = "low"

    @staticmethod
    def _summary(enemies: list[EnemyState]) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        by_element: dict[str, int] = {}
        by_tier: dict[str, int] = {"enemy": 0, "elite": 0, "boss": 0}
        aggro = 0
        nearest: Optional[float] = None
        threats = {"high": 0, "medium": 0, "low": 0}
        for e in enemies:
            by_tier[e.cls_name] = by_tier.get(e.cls_name, 0) + 1
            if e.enemy_type:
                by_type[e.enemy_type] = by_type.get(e.enemy_type, 0) + 1
            if e.element:
                by_element[e.element] = by_element.get(e.element, 0) + 1
            if e.aggro:
                aggro += 1
            if e.distance_m is not None:
                nearest = e.distance_m if nearest is None else min(nearest, e.distance_m)
            if e.threat:
                threats[e.threat] = threats.get(e.threat, 0) + 1
        level = "high" if threats["high"] else ("medium" if threats["medium"] else "low")
        return {
            "total": len(enemies),
            "by_tier": by_tier,
            "by_type": by_type,
            "by_element": by_element,
            "aggro_count": aggro,
            "nearest_m": None if nearest is None else round(nearest, 1),
            "threat_level": level if enemies else "none",
            "threat_counts": threats,
        }

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
                st.distance_m = self._distance_m(st.enemy_type or tr.cls_name, tr.bbox)
                # аггро: видна полоска HP (враг замечен/в бою)
                st.aggro = st.hp_frac is not None
                self._threat(st, frame_size)
                enemies.append(st)
            elif tr.cls_name in INTERACTABLE_CLASSES:
                if tr.hits < 2:  # однокадровые ложные срабатывания не показываем
                    continue
                st = self._tracked_to_state(tr, tr.cls_name)
                st.distance_m = self._distance_m(tr.cls_name, tr.bbox)
                interactables.append(st)

        # чем ближе — тем раньше в списке
        enemies.sort(key=lambda e: e.distance_m if e.distance_m is not None else 1e9)
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
            enemies_summary=self._summary(enemies),
            in_combat=in_combat,
            backend=self.backend_name,
        )
