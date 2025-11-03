from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ._config import TrainConfig, TrainPaths


@dataclass
class TrainState:
    best_loss: float = float("inf")
    best_acc: float = 0.0
    patience_count: int = 0


def train(
    model: torch.nn.Module,
    samples: list[list[Data]],
    cfg: TrainConfig,
    save_path: TrainPaths,
    device: torch.device,
) -> None:
    """
    Train a GNN with early stopping based on loss and accuracy improvements.

    Assumptions
        - samples is a list of samples, where each sample is a list of PyG Data objects
        - the model forward signature is model(x, edge_index, batch) and returns logits of shape [B, 2]
        - data.y is a binary label tensor of shape [B]

    Args:
        model: Torch module producing two class logits per graph.
        samples: List of samples, where each sample is a list of PyG Data.
        cfg: Training configuration hyperparameters.
        save_path: Paths helper; best_path is used to save the best state dict.
        device: Compute device.

    Returns:
        None
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Keep your batching scheme: each batch item is a list[Data]
    pin = device.type == "cuda"
    loader = DataLoader(
        samples, batch_size=cfg.batch_size, shuffle=True, pin_memory=pin
    )

    state = TrainState()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sample in loader:
            # each sample is a list[Data]
            for data in sample:
                data = data.to(device, non_blocking=pin)

                optim.zero_grad()
                logits = model(data.x, data.edge_index, data.batch)[:, 1]  # shape [B]
                loss = criterion(logits, data.y.float())
                loss.backward()
                optim.step()

                with torch.no_grad():
                    bs = data.y.size(0)
                    total_loss += loss.item() * bs
                    pred = torch.round(torch.sigmoid(logits))
                    total_correct += (pred == data.y).sum().item()
                    total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f} acc={acc:.4f}")

        improved = False
        if avg_loss < state.best_loss - cfg.min_delta_loss:
            state.best_loss = avg_loss
            improved = True
        if acc > state.best_acc + cfg.min_delta_acc:
            state.best_acc = acc
            improved = True

        if improved:
            state.patience_count = 0
            torch.save(model.state_dict(), save_path.best_path)
        else:
            state.patience_count += 1
            if state.patience_count >= cfg.patience:
                print(f"Early stopping after no improvement for {cfg.patience} epochs")
                break
