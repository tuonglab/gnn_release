from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainPaths:
    model_dir: Path = Path("model_2025_boltz_111")
    best_name: str = "best_model.pt"

    @property
    def best_path(self) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir / self.best_name

@dataclass
class TrainConfig:
    epochs: int = 100
    patience: int = 15
    lr: float = 5e-4
    weight_decay: float = 0.25
    min_delta_loss: float = 0.01
    min_delta_acc: float = 0.01
    batch_size: int = 256
    seed: int = 111
