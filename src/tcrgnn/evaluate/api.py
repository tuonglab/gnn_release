from torch_geometric.data import Data

from tcrgnn.evaluate._predict import predict_on_graph_list
from tcrgnn.evaluate._utils import (
    get_device,
    load_trained_model,
)
from tcrgnn.models.gatv2 import GATv2


def evaluate_model(model_file: str, graphs: list[Data], device=None):
    """
    Evaluate a trained model on a provided list of PyG Data graphs.

    Args:
        model_file: Path to the saved model checkpoint.
        graphs: A list of PyG Data objects to evaluate.
        device: Optional torch device ('cpu' or 'cuda').

    Returns:
        Model prediction scores or evaluation metrics.
    """
    # Input validation
    if not isinstance(graphs, list):
        raise TypeError(f"Expected a list of Data objects, got {type(graphs).__name__}")
    if len(graphs) == 0:
        raise ValueError("The input list of graphs is empty.")
    if not all(isinstance(g, Data) for g in graphs):
        raise TypeError("All elements in the input list must be PyG Data objects.")

    nfeat = graphs[0].num_node_features
    base_model = GATv2(nfeat=nfeat, nhid=375, nclass=2, dropout=0.17)

    device = device or get_device()
    model = load_trained_model(base_model, model_file, device)

    # Load model and move to device
    model.eval()

    # Perform prediction
    scores = predict_on_graph_list(model, graphs, device)

    return scores
