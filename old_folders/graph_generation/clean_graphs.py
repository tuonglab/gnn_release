import torch
from graph import load_graphs

# Path to the .pt file
file_path = "/scratch/project/tcr_ml/gnn_release/test_data_v2/20241106_WGS_20241106_sc_PICA0033-PICA0069_Pool_3/processed/20241106_WGS_20241106_sc_PICA0033-PICA0069_Pool_3_0_cdr3_results_edges.tar.pt"

# Load graphs
graphs = load_graphs(file_path)

# Filter out invalid graphs
filtered_graphs = []
for i, graph in enumerate(graphs):
    if not hasattr(graph, 'x'):
        print(f"Graph {i} has no 'x' attribute.")
        continue
    if graph.x is None:
        print(f"Graph {i} has 'x' = None.")
        continue
    if graph.x.numel() == 0:
        print(f"Graph {i} has empty 'x' tensor with shape: {graph.x.shape}")
        print(f"Label: {graph.y}, Characters: {getattr(graph, 'original_characters', 'N/A')}")
        continue
    if graph.x.ndim != 2:
        print(f"Graph {i} has 'x' with incorrect dimension: {graph.x.shape}")
        continue
    filtered_graphs.append(graph)

print(f"Filtered {len(graphs) - len(filtered_graphs)} graphs. Saving {len(filtered_graphs)} graphs back.")

# Overwrite the original file
torch.save(filtered_graphs, file_path)
print("Done saving filtered graphs.")
