from collections.abc import Iterable

import torch

from tcrgnn.evaluate._predict import predict_on_graph_list
from tcrgnn.evaluate._utils import (
    get_device,
    load_trained_model,
    make_loader,
)
from tcrgnn.models.gatv2 import GATv2


def evaluate_model(
    model_file: str,
    test_data,
    device: torch.device | None = None,
) -> list[float]:
    """
    Evaluate a single sample composed of a list of graphs.

    Assumptions:
        - The loader will yield exactly one item.
        - That item is a list or iterable of PyG graph objects.
        - Returns flat per graph scores and labels for that one sample.

    Args:
        model_file: Path to a state dict saved via torch.save(model.state_dict(), path).
        test_data: Dataset or indexable container where test_data[0][0] is a PyG Data.
        device: Optional device override. Uses CUDA if available, else CPU.

    Returns:
        list[float]: Per graph scores
    """
    # Infer feature dimension from the first graph of the first sample
    nfeat = test_data[0][0].num_node_features
    base_model = GATv2(nfeat=nfeat, nhid=375, nclass=2, dropout=0.17)

    device = device or get_device()
    model = load_trained_model(base_model, model_file, device)

    # Your make_loader already fixes batch_size=1 and shuffle=False
    loader = make_loader(test_data)

    # Optional safeguard if the DataLoader implements __len__
    if hasattr(loader, "__len__"):
        assert len(loader) == 1, "Loader must contain exactly one batch"

    try:
        batch = next(iter(loader))
    except StopIteration:
        return []

    # Unwrap if DataLoader returned a single element inside a list or tuple
    sample_graphs: Iterable = (
        batch[0] if isinstance(batch, (list, tuple)) and len(batch) == 1 else batch
    )

    scores = predict_on_graph_list(model, sample_graphs, device)
    return scores
