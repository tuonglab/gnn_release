from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.loader import DataLoader

from .config import TrainConfig, TrainPaths


@dataclass
class TrainState:
    best_loss: float = float("inf")
    best_acc: float = 0.0
    patience_count: int = 0


def train(
    model: torch.nn.Module,
    samples: list[list[torch.Tensor]],
    cfg: TrainConfig,
    paths: TrainPaths,
    device: torch.device,
) -> None:
    criterion = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # samples is a list of lists of Data objects, keep your batching scheme
    loader = DataLoader(samples, batch_size=cfg.batch_size, shuffle=True)
    state = TrainState()

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for sample in loader:
            for data in sample:
                data = data.to(device)
                optim.zero_grad()
                out = model(data.x, data.edge_index, data.batch)[:, 1]
                loss = criterion(out, data.y.float())
                loss.backward()
                optim.step()

                bs = data.y.size(0)
                total_loss += loss.item() * bs
                pred = torch.round(torch.sigmoid(out.detach()))
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
            torch.save(model.state_dict(), paths.best_path)
        else:
            state.patience_count += 1
            if state.patience_count >= cfg.patience:
                print(f"Early stopping after no improvement for {cfg.patience} epochs")
                break
