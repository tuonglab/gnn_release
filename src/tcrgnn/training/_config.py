from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainPaths:
    """
    Container for model output paths used during training.

    Attributes:
        model_dir: Directory where model checkpoints will be stored.
        best_name: Filename for the best performing checkpoint.

    Behavior:
        - model_dir is coerced to a Path during initialization.
        - Accessing best_path ensures the directory exists.
    """

    model_dir: Path | str
    best_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_dir, Path):
            self.model_dir = Path(self.model_dir)

    @property
    def best_path(self) -> Path:
        """
        Path to the best performing model checkpoint.

        Creates the model directory if necessary.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir / self.best_name


@dataclass
class TrainConfig:
    """
    Basic hyperparameters and training control settings.

    Attributes:
        epochs: Maximum number of training epochs.
        patience: Early stopping patience for validation metrics.
        lr: Learning rate.
        weight_decay: L2 regularization factor.
        min_delta_loss: Minimal improvement in loss to reset patience.
        min_delta_acc: Minimal improvement in accuracy to reset patience.
        batch_size: Mini batch size for training.
        seed: Random seed for reproducibility.
    """

    epochs: int = 100
    patience: int = 15
    lr: float = 5e-4
    weight_decay: float = 0.25
    min_delta_loss: float = 0.01
    min_delta_acc: float = 0.01
    batch_size: int = 256
    seed: int = 111
