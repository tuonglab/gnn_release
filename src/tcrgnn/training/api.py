from __future__ import annotations

from pathlib import Path

import torch

from tcrgnn.models.gatv2 import GATv2

from ..utils._data_loading import load_train_data
from ._config import TrainConfig, TrainPaths
from ._loop import train


def train_model(
    cancer_dirs: list[str | Path],
    control_dirs: list[str | Path],
    cfg: TrainConfig,
    save_path: TrainPaths,
) -> None:
    train_set = load_train_data(cancer_dirs, control_dirs)
    if not train_set:
        raise ValueError("Training set is empty")

    nfeat = train_set[0][0].num_node_features
    model = GATv2(nfeat=nfeat, nhid=375, nclass=2, dropout=0.17)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # match test’s expected train signature
    train(model, train_set, cfg, save_path, device)

    print(f"Training complete. Model saved to {save_path.model_dir}")
