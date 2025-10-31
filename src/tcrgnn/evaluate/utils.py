import torch
from torch_geometric.loader import DataLoader


def get_device() -> torch.device:
    """Pick CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(
    model: torch.nn.Module, model_file: str, device: torch.device
) -> torch.nn.Module:
    """Load state dict and move model to device."""
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def move_graph_to_device(graph, device: torch.device):
    """
    Move a single PyG graph object to device.
    Assumes standard fields x, edge_index, y, batch are present where applicable.
    """
    graph = graph.to(device)
    return graph


def make_loader(test_data) -> DataLoader:
    """
    Build a DataLoader for datasets that yield one item per sample,
    where each item is a list of graphs.
    """
    return DataLoader(test_data, batch_size=1, shuffle=False)


def write_scores_to_txt(
    scores: list[float], filename: str, sequences: list | None = None
) -> None:
    """Write scores to a text file, optionally paired with sequences."""
    length_scores = len(scores)
    length_sequences = len(sequences) if sequences is not None else 0

    # Warn if lengths do not match
    if sequences is not None and length_sequences != length_scores:
        print(
            "Warning: Length of sequences does not match length of scores. Missing entries will be marked as N/A."
        )

    with open(filename, "w") as f:
        if sequences is None:
            # No sequences provided
            for s in scores:
                f.write(f"{s}\n")
        else:
            max_len = max(length_scores, length_sequences)
            for i in range(max_len):
                seq = sequences[i] if i < length_sequences else "N/A"
                sc = scores[i] if i < length_scores else "N/A"
                f.write(f"{seq},{sc}\n")
