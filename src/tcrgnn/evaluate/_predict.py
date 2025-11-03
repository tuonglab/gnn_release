from collections.abc import Iterable

import torch

from tcrgnn.evaluate._utils import move_graph_to_device


@torch.inference_mode()
def predict_on_graph_list(
    model: torch.nn.Module,
    graphs: Iterable,  # Iterable of PyG Data objects
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[list[float], list[int]]:
    """
    Run inference on an iterable of graphs belonging to a single sample.

    Each graph is forwarded individually through the model.
    The function collects positive class probabilities and binary labels.

    Args:
        model: A graph neural network that accepts arguments (x, edge_index, batch).
        graphs: Iterable of PyG Data objects.
        device: Target device for inference.
        threshold: Decision cutoff for assigning a positive label.

    Returns:
        tuple[list[float], list[int]]:
            A pair containing:
            - Per-graph positive class probabilities
            - Per-graph predicted labels based on threshold
    """
    sample_scores: list[float] = []

    model.eval()

    for g in graphs:
        g = move_graph_to_device(g, device)

        if not hasattr(g, "x"):
            raise ValueError("Graph is missing node features 'x'")
        if g.x.dim() != 2:
            raise ValueError(
                f"Input features must be 2-dimensional. Current shape: {g.x.shape}"
            )

        out = model(g.x, g.edge_index, getattr(g, "batch", None))
        probs = torch.sigmoid(out)

        if probs.ndim == 2 and probs.size(1) > 1:
            pos = probs[:, 1]
        else:
            pos = probs

        pos = pos.reshape(-1)

        scores = pos.detach().float().cpu().tolist()
        sample_scores.extend(scores)

    return sample_scores
