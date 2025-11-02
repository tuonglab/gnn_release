from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.loader import DataLoader

# A helpful alias for common PyG graph containers
GraphLike = Data | HeteroData | Batch


def get_device() -> torch.device:
    """
    Select an available compute device.

    Returns:
        torch.device: "cuda" if at least one CUDA device is available, else "cpu".
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(
    model: torch.nn.Module,
    model_file: str,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load a serialized state dict into a model and move it to a device.

    The function calls model.eval before returning.

    Args:
        model: Instantiated torch module matching the saved architecture.
        model_file: Path to a file saved via torch.save(model.state_dict(), path).
        device: Target device for all model parameters.

    Returns:
        torch.nn.Module: The model with loaded weights on the given device.

    Raises:
        FileNotFoundError: If model_file does not exist.
        RuntimeError: If state dict keys do not match the model architecture.
    """
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def move_graph_to_device(graph: GraphLike, device: torch.device) -> GraphLike:
    """
    Move a PyG graph structure to a target device.

    Works with Data, HeteroData, or Batch. Attributes such as x, y,
    edge_index, and batch tensors are copied accordingly.

    Args:
        graph: A PyG graph container.
        device: Target device where tensors should reside.

    Returns:
        GraphLike: The same graph object with tensors on device.
    """
    return graph.to(device)


def make_loader(
    test_data: Dataset | Iterable,
) -> DataLoader:
    """
    Create a PyG DataLoader for deterministic per sample iteration.

    Uses batch_size 1 and shuffle False. The dataset may yield single
    graphs or collections of graphs per item.

    Args:
        test_data: Iterable or Dataset producing items to evaluate.

    Returns:
        DataLoader: A configured loader suitable for inference.
    """
    return DataLoader(test_data, batch_size=1, shuffle=False)


def write_scores_to_txt(
    scores: Sequence[float],
    filename: str,
    sequences: Sequence[str] | None = None,
) -> None:
    """
    Write numeric scores to a text file, optionally pairing them with sequences.

    When sequences are provided, output lines are comma separated. If the
    lengths do not match, missing values are padded with "N/A" and a warning
    is printed.

    Args:
        scores: Numeric values to write.
        filename: Output path to write the results.
        sequences: Optional strings aligned to each score.

    Returns:
        None

    Notes:
        This function overwrites the file if it exists.
    """
    length_scores = len(scores)
    length_sequences = len(sequences) if sequences is not None else 0

    if sequences is not None and length_sequences != length_scores:
        print(
            "Warning: Length of sequences does not match length of scores. "
            "Missing entries will be marked as N/A."
        )

    with open(filename, "w") as f:
        if sequences is None:
            for s in scores:
                f.write(f"{s}\n")
        else:
            max_len = max(length_scores, length_sequences)
            for i in range(max_len):
                seq = sequences[i] if i < length_sequences else "N/A"
                sc = scores[i] if i < length_scores else "N/A"
                f.write(f"{seq},{sc}\n")
