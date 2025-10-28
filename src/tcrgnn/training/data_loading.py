from __future__ import annotations
from pathlib import Path
import os
import torch

# keep your existing loader signature
def load_graphs(file: str):
    # local import to avoid hard coupling if user moves modules
    from graph_generation.graph import load_graphs as _load
    return _load(file)

def load_train_data(cancer_dirs: list[str | os.PathLike], control_dirs: list[str | os.PathLike]) -> list[list[torch.Tensor]]:
    training_set: list[list[torch.Tensor]] = []

    for d in cancer_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                graphs = load_graphs(os.path.join(d, fname))
                training_set.append(graphs)

    for d in control_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                graphs = load_graphs(os.path.join(d, fname))
                training_set.append(graphs)

    return training_set
