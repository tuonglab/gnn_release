from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def load_graphs(file: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """
    Load graphs from a file.

    Args:
        file: Path to the file containing the graphs.
        map_location: Passed to torch.load. Default "cpu".

    Returns:
        The loaded dataset.
    """
    return torch.load(str(file), map_location=map_location)


def load_train_data(
    cancer_dirs: list[str | os.PathLike], control_dirs: list[str | os.PathLike]
) -> list[list[torch.Tensor]]:
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
