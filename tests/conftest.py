import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "da_vits14_int8.onnx")


@pytest.fixture(scope="session")
def has_model():
    return os.path.exists(MODEL)


@pytest.fixture(scope="session")
def estimator(has_model):
    if not has_model:
        pytest.skip("ONNX-модель не скачана: python scripts/fetch_model.py")
    from parkour.depth import DepthEstimator
    return DepthEstimator().load()


@pytest.fixture
def synthetic_wall_frame():
    """Простой синтетический кадр: тёмная стена справа-слева и светлый пол."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[220:, :, :] = 150
    img[:, :180, :] = 90
    img[:, 460:, :] = 90
    return img
