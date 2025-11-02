from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch
from torch_geometric.data import Data


def load_graphs(
    file: str | Path,
    map_location: str | torch.device = "cpu",
) -> list[Data] | Iterable[Data] | torch.Tensor:
    """
    Load a serialized graph bundle from disk.

    Args:
        file: Path to a serialized graph file, typically produced by PyTorch.
        map_location: Passed through to torch.load.

    Returns:
        Whatever object was stored in the file, commonly:
            - list[Data]
            - iterable of Data
            - PyTorch tensor
    """
    return torch.load(str(file), map_location=map_location)


def load_train_data(
    cancer_dirs: list[str | os.PathLike],
    control_dirs: list[str | os.PathLike],
) -> list[list[Data]]:
    """
    Load training samples from cancer and control directories.

    Each directory is expected to contain files where each file loads
    to a list of PyTorch Geometric Data objects. Each *file* becomes
    one sample in the training set.

    Args:
        cancer_dirs: Directories for positive samples.
        control_dirs: Directories for negative samples.

    Returns:
        A list of samples, where each sample is a list[Data].
    """
    training_set: list[list[Data]] = []

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


def load_test_file(file_path: str | Path) -> list[Data] | Iterable[Data]:
    """
    Load a test graph bundle from a file.

    Args:
        file_path: Path to a saved graph list.

    Returns:
        List (or iterable) of Data objects.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)

    if file_path.is_file():
        return load_graphs(file_path)

    raise FileNotFoundError(f"Test file not found: {file_path}")
