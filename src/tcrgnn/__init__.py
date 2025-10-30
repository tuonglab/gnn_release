from .training.config import TrainConfig, TrainPaths
from .training.loop import train

__all__ = ["GATv2", "TrainConfig", "TrainPaths", "train"]


def __getattr__(name: str):
    if name == "GATv2":
        from .models.gatv2 import GATv2

        return GATv2
    raise AttributeError(f"module {__name__} has no attribute {name}")
