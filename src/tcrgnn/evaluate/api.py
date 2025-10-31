import torch

from tcrgnn.evaluate.predict import predict_on_graph_list
from tcrgnn.evaluate.utils import (
    get_device,
    load_trained_model,
    make_loader,
)
from tcrgnn.models.gatv2 import GATv2


def evaluate(
    model_file: str,
    test_data,
    device: torch.device | None = None,
) -> tuple[list[float], list[int]]:
    """
    Single-sample evaluation:
      - loader has exactly one item, which is a list of graphs
      - returns flat per-graph scores and labels for that single sample
    """
    base_model = GATv2(
        nfeat=test_data[0][0].num_node_features, nhid=375, nclass=2, dropout=0.17
    )
    device = device or get_device()
    model = load_trained_model(base_model, model_file, device)
    loader = make_loader(test_data, batch_size=1, shuffle=False)

    # Optional safeguard if your DataLoader is length-aware
    if hasattr(loader, "__len__"):
        assert len(loader) == 1, "Loader must contain exactly one batch"

    try:
        batch = next(iter(loader))
    except StopIteration:
        return [], []

    # Unwrap the single item if DataLoader wrapped it
    sample_graphs = (
        batch[0] if isinstance(batch, (list, tuple)) and len(batch) == 1 else batch
    )

    sample_scores = predict_on_graph_list(model, sample_graphs, device)
    return sample_scores
