"""Genshin-Parkour: экранный агент локальной навигации по карте глубины."""
from .config import ParkourConfig, DepthConfig
from .depth import DepthEstimator
from .perception import Perception, SceneAnalysis
from .policy import ParkourBrain, ActionFrame, State

__all__ = [
    "ParkourConfig",
    "DepthConfig",
    "DepthEstimator",
    "Perception",
    "SceneAnalysis",
    "ParkourBrain",
    "ActionFrame",
    "State",
]
__version__ = "0.1.0"
