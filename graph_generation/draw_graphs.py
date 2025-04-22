from graph import load_graphs, draw_graph

# Path to the .pt file
file_path = "/scratch/project/tcr_ml/gnn_release/test_data_v2/seekgene/processed/S1_cdr3_results_edges.tar.pt"

# file_path = "/scratch/project/tcr_ml/gnn_release/test_data_v2/seekgene_boltz/processed/boltz_results_S1_edges.tar.pt"
# Target sequences you're interested in
target_sequences = {
    "CASSSLLPQGWGLDGYTF"
}

# Load graphs once
graphs = load_graphs(file_path)

# Use list comprehension to filter matching graphs
matching_graphs = [(i, g) for i, g in enumerate(graphs) if getattr(g, "original_characters", None) in target_sequences]

# Draw the matching graphs
for i, graph in matching_graphs:
    seq = graph.original_characters
    print(f"Drawing graph {i} with sequence {seq}")
    draw_graph(graph, sequence=seq, filename=f"graph_{i}_{seq}.pdf")
