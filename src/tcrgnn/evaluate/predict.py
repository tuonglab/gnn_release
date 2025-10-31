from collections.abc import Iterable

import torch

from tcrgnn.evaluate.utils import move_graph_to_device

# If you use PyG types, uncomment these lines for better type hints
# from torch_geometric.data import Data


@torch.inference_mode()
def predict_on_graph_list(
    model: torch.nn.Module,
    graphs: Iterable,  # Iterable of PyG Data objects
    device: torch.device,
) -> tuple[list[float], list[int]]:
    """
    Run inference on a list of graphs that belong to a single sample.
    Returns per-graph positive class scores and labels.
    """
    sample_scores: list[float] = []

    for g in graphs:
        g = move_graph_to_device(g, device)

        # Basic shape sanity check
        if not hasattr(g, "x"):
            raise ValueError("Graph is missing node features 'x'")
        if g.x.dim() != 2:
            raise ValueError(
                f"Input features must be 2-dimensional. Current shape: {g.x.shape}"
            )

        out = model(g.x, g.edge_index, getattr(g, "batch", None))
        # Use sigmoid for binary logits. If your model outputs probabilities already, skip this.
        probs = torch.sigmoid(out)

        if probs.ndim == 2 and probs.size(1) > 1:
            # Multi-logit output. Take positive class column 1.
            pos_scores = probs[:, 1]
        else:
            # Single logit or single probability. Convert to positive class prob.
            pos_scores = probs.squeeze()

        # Collect scores
        sample_scores.extend(pos_scores.tolist())
    return sample_scores
