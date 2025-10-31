from pathlib import Path

import torch

from tcrgnn.models.gatv2 import GATv2

from ..utils.data_loading import load_train_data
from .config import TrainConfig, TrainPaths
from .loop import train


def train_model(
    cancer_dirs: list[str | Path],
    control_dirs: list[str | Path],
    cfg: TrainConfig,
    save_path: TrainPaths,
) -> None:
    # Your training code here
    train_set = load_train_data(cancer_dirs, control_dirs)

    model = GATv2(
        nfeat=train_set[0][0].num_node_features, nhid=375, nclass=2, dropout=0.17
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model on the training samples
    train(model, train_set, num_epochs=500, cfg=cfg, device=device, save_path=save_path)
    print("Training complete. Model saved to", save_path.model_dir)
    return
