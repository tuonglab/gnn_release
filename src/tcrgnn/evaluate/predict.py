from collections.abc import Iterable

import torch

from tcrgnn.evaluate.utils import move_graph_to_device


@torch.inference_mode()
def predict_on_graph_list(
    model: torch.nn.Module,
    graphs: Iterable,  # Iterable of PyG Data objects
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[list[float], list[int]]:
    """
    Run inference on a list of graphs that belong to a single sample.
    Returns per-graph positive class scores and predicted labels.
    """
    sample_scores: list[float] = []
    sample_labels: list[int] = []

    model.eval()

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
        # If your model already outputs probabilities, skip the sigmoid line
        probs = torch.sigmoid(out)

        # Get positive class probability per item
        if probs.ndim == 2 and probs.size(1) > 1:
            pos = probs[:, 1]
        else:
            # 0D or 1D tensor. Do not squeeze to avoid creating a Python float.
            pos = probs

        # Ensure a 1D tensor so .tolist() returns a list, never a float
        pos = pos.reshape(-1)

        scores = pos.detach().float().cpu().tolist()
        sample_scores.extend(scores)
        sample_labels.extend([int(s >= threshold) for s in scores])

    return sample_scores
