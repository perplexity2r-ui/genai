"""Тесты трекера: стабильные ID, скорость, предсказание."""
import numpy as np

from genshinvision.track import SortTracker, iou_matrix
from genshinvision.types import Detection


def make_det(x, y, w, h, score=0.9, name="enemy"):
    return Detection(bbox=(x, y, x + w, y + h), score=score, cls_id=1, cls_name=name)


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1.0) < 1e-6
    assert abs(m[0, 1] - 25 / 175) < 1e-6


def test_two_objects_stable_ids_and_velocity():
    """Два объекта движутся равномерно: ID стабильны, скорость близка к истинной."""
    tracker = SortTracker(min_hits=1)
    fps = 30.0
    dt = 1.0 / fps
    v = [(120.0, 0.0), (-80.0, 60.0)]  # px/s
    pos = [(100.0, 100.0), (600.0, 300.0)]
    size = (60.0, 80.0)

    seen_ids = {0: set(), 1: set()}
    last_state = None
    for f in range(60):
        t = f * dt
        dets = []
        for i in range(2):
            cx = pos[i][0] + v[i][0] * t
            cy = pos[i][1] + v[i][1] * t
            dets.append(make_det(cx - size[0] / 2, cy - size[1] / 2, *size))
        states = tracker.update(dets, t)
        assert len(states) == 2, f"frame {f}: ожидалось 2 трека, найдено {len(states)}"
        for i, st in enumerate(states):
            cx = st.bbox[0] + (st.bbox[2] - st.bbox[0]) / 2
            best = min(range(2), key=lambda k: abs(cx - (pos[k][0] + v[k][0] * t)))
            seen_ids[best].add(st.track_id)
        if f == 59:
            last_state = states

    # у каждого объекта ровно один стабильный ID
    assert all(len(ids) == 1 for ids in seen_ids.values()), seen_ids
    assert len(set.union(*seen_ids.values())) == 2

    # скорость после схождения фильтра близка к истинной
    for st in last_state:
        cx = st.bbox[0] + (st.bbox[2] - st.bbox[0]) / 2
        k = 0 if abs(cx - (pos[0][0] + v[0][0] * 59 * dt)) < abs(cx - (pos[1][0] + v[1][0] * 59 * dt)) else 1
        assert abs(st.velocity[0] - v[k][0]) < 35, (st.velocity, v[k])
        assert abs(st.velocity[1] - v[k][1]) < 35, (st.velocity, v[k])

    # предсказание на 0.5 с вперёд
    st = last_state[0]
    pred = st.predicted_bbox(0.5)
    cx_pred = (pred[0] + pred[2]) / 2
    cx_now = (st.bbox[0] + st.bbox[2]) / 2
    assert abs((cx_pred - cx_now) - st.velocity[0] * 0.5) < 2.0


def test_track_survives_missing_frames():
    """Трек переживает пропуски детекций и возвращается к тому же ID."""
    tracker = SortTracker(min_hits=1, max_age_s=0.5)
    dt = 1 / 30.0
    ids = []
    for f in range(30):
        det = make_det(100 + f * 2, 100, 50, 60)
        dets = [] if 10 <= f < 16 else [det]
        states = tracker.update(dets, f * dt)
        if states and 5 <= f < 10:
            ids.append(states[0].track_id)
    assert len(set(ids)) == 1
