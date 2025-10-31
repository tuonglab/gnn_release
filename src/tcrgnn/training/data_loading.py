from __future__ import annotations

import os

import torch


def load_graphs(file: str):
    """
    Load graphs from a file.

    Args:
        file (str): The path to the file containing the graphs.

    Returns:
        dataset: The loaded dataset containing the graphs.
    """
    dataset = torch.load(file, map_location=torch.device)
    return dataset


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
