"""SORT-style multi-object tracker with a constant-velocity Kalman filter.

Tracks are maintained in screen space. The filter state is

    x = [cx, cy, a, h, vcx, vcy, va, vh]

where ``a = w / h`` is the aspect ratio and velocities are expressed in
pixels (and units of ``a``) per *second*, so variable frame rates are
handled naturally via the ``dt`` passed to :meth:`SortTracker.update`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import BBox, Detection

__all__ = ["SortTracker", "TrackState", "iou_matrix"]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``[x1, y1, x2, y2]`` boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    a = boxes_a[:, None, :]  # (N,1,4)
    b = boxes_b[None, :, :]  # (1,M,4)
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _bbox_to_z(bbox: BBox) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array(
        [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0, w / max(h, 1e-6), h],
        dtype=np.float64,
    )


def _z_to_bbox(z: np.ndarray) -> BBox:
    cx, cy, a, h = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _KalmanBox:
    """Constant-velocity Kalman filter over the 8-dim box state."""

    def __init__(self, bbox: BBox):
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = _bbox_to_z(bbox)

        self.F = np.eye(8, dtype=np.float64)  # dt is written into F on predict
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        self.R = np.eye(4, dtype=np.float64)
        self.R[2, 2] = 5.0
        self.R[3, 3] = 5.0

        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # unobserved velocity starts uncertain

        self.Q = np.eye(8, dtype=np.float64)
        self.Q[4:, 4:] *= 0.25

    def predict(self, dt: float) -> None:
        self.F = np.eye(8, dtype=np.float64)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt
        q = self.Q * max(dt, 1e-3)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + q

    def update(self, bbox: BBox) -> None:
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def bbox(self) -> BBox:
        return _z_to_bbox(self.x)

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    def extrapolate(self, dt: float) -> BBox:
        z = self.x[:4].copy()
        z[0] += self.x[4] * dt
        z[1] += self.x[5] * dt
        z[2] += self.x[6] * dt
        z[3] += self.x[7] * dt
        return _z_to_bbox(z)


class TrackState:
    """Public snapshot of one live track."""

    def __init__(
        self,
        track_id: int,
        cls_name: str,
        bbox: BBox,
        velocity: tuple[float, float],
        score: float,
        age: float,
        hits: int,
        time_since_update: float,
        extrapolate,
        attrs: dict | None = None,
    ):
        self.track_id = track_id
        self.cls_name = cls_name
        self.bbox = bbox
        self.velocity = velocity
        self.score = score
        self.age = age
        self.hits = hits
        self.time_since_update = time_since_update
        self._extrapolate = extrapolate
        self.attrs = dict(attrs or {})

    def predicted_bbox(self, dt: float) -> BBox:
        return self._extrapolate(dt)


class _Track:
    _next_id = 1

    def __init__(self, det: Detection, timestamp: float):
        self.kf = _KalmanBox(det.bbox)
        self.cls_name = det.cls_name
        self.score = det.score
        self.attrs = dict(det.attrs)
        self.track_id = _Track._next_id
        _Track._next_id += 1
        self.time_since_update = 0.0
        self.hits = 1
        self.first_ts = timestamp
        self.last_ts = timestamp

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)

    def update(self, det: Detection, timestamp: float) -> None:
        self.kf.update(det.bbox)
        self.score = det.score
        self.cls_name = det.cls_name
        # hp_frac «липкий» (приходит с полосок), остальные атрибуты — свежие
        self.attrs = {**{k: v for k, v in self.attrs.items() if k == "hp_frac"}, **det.attrs}
        self.time_since_update = 0.0
        self.hits += 1
        self.last_ts = timestamp

    def snapshot(self, now: float) -> TrackState:
        return TrackState(
            track_id=self.track_id,
            cls_name=self.cls_name,
            bbox=self.kf.bbox,
            velocity=self.kf.velocity,
            score=self.score,
            age=now - self.first_ts,
            hits=self.hits,
            time_since_update=self.time_since_update,
            extrapolate=self.kf.extrapolate,
            attrs=self.attrs,
        )


class SortTracker:
    """Two-stage (ByteTrack-lite) assignment on top of SORT dynamics."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        low_iou_threshold: float = 0.15,
        max_age_s: float = 1.0,
        min_hits: int = 2,
        min_det_score: float = 0.2,
    ):
        self.iou_threshold = iou_threshold
        self.low_iou_threshold = low_iou_threshold
        self.max_age_s = max_age_s
        self.min_hits = min_hits
        self.min_det_score = min_det_score
        self._tracks: list[_Track] = []
        self._last_ts: float | None = None

    # ------------------------------------------------------------------ api
    def update(self, detections: list[Detection], timestamp: float) -> list[TrackState]:
        """Advance the tracker with this frame's detections.

        Returns confirmed track snapshots (predicted tracks are dropped after
        ``max_age_s`` seconds without a matching detection).
        """
        if self._last_ts is None:
            dt = 1.0 / 60.0
        else:
            dt = float(np.clip(timestamp - self._last_ts, 1.0 / 240.0, 0.5))
        self._last_ts = timestamp

        for t in self._tracks:
            t.predict(dt)
            t.time_since_update += dt

        high = [d for d in detections if d.score >= self.min_det_score]
        low = [d for d in detections if d.score < self.min_det_score]

        matched, unmatched_dets = self._assign(self._tracks, high, self.iou_threshold)
        for ti, di in matched:
            self._tracks[ti].update(high[di], timestamp)

        # Second chance for occluded / low-confidence detections.
        remaining = [self._tracks[i] for i in range(len(self._tracks)) if i not in {m[0] for m in matched}]
        matched_low, _ = self._assign(remaining, low, self.low_iou_threshold)
        for ti, di in matched_low:
            remaining[ti].update(low[di], timestamp)

        # Spawn tracks from unmatched strong detections.
        for di in unmatched_dets:
            self._tracks.append(_Track(high[di], timestamp))

        # Drop dead tracks.
        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age_s]

        now = timestamp
        out: list[TrackState] = []
        for t in self._tracks:
            confirmed = t.hits >= self.min_hits or t.time_since_update <= dt * 1.5
            if confirmed:
                out.append(t.snapshot(now))
        return out

    def reset(self) -> None:
        self._tracks.clear()
        self._last_ts = None

    # ------------------------------------------------------------- internals
    @staticmethod
    def _assign(tracks: list[_Track], dets: list[Detection], iou_thr: float):
        if not tracks or not dets:
            return [], list(range(len(dets)))
        det_boxes = np.array([d.bbox for d in dets], dtype=np.float64)
        trk_boxes = np.array([t.kf.bbox for t in tracks], dtype=np.float64)
        iou = iou_matrix(trk_boxes, det_boxes)
        cost = 1.0 - iou
        rows, cols = linear_sum_assignment(cost)
        matched, used_det = [], set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= iou_thr:
                matched.append((int(r), int(c)))
                used_det.add(int(c))
        unmatched = [i for i in range(len(dets)) if i not in used_det]
        return matched, unmatched
