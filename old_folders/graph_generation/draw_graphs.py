from graph import load_graphs

# Path to the .pt file
file_path = "/scratch/project/tcr_ml/gnn_release/test_data_v2/20241106_WGS_20241106_sc_PICA0033-PICA0069_Pool_6/processed/20241106_WGS_20241106_sc_PICA0033-PICA0069_Pool_6_2_cdr3_results_edges.tar.pt"

# file_path = "/scratch/project/tcr_ml/gnn_release/test_data_v2/seekgene_boltz/processed/boltz_results_S1_edges.tar.pt"
# Target sequences you're interested in
target_sequences = {"CASSQEDRRVDEQYF"}

# Load graphs once
graphs = load_graphs(file_path)
for i, graph in enumerate(graphs):
    if not hasattr(graph, "x"):
        print(f"Graph {i} has no 'x' attribute.")
    elif graph.x is None:
        print(f"Graph {i} has 'x' = None.")
    elif graph.x.numel() == 0:
        print(f"Graph {i} has empty 'x' tensor with shape: {graph.x.shape}")
        print(graph.y)
        print(graph.original_characters)
    elif graph.x.ndim != 2:
        print(f"Graph {i} has 'x' with incorrect dimension: {graph.x.shape}")
print(i)

# # Use list comprehension to filter matching graphs
# matching_graphs = [(i, g) for i, g in enumerate(graphs) if getattr(g, "original_characters", None) in target_sequences]

# # Draw the matching graphs
# for i, graph in matching_graphs:
#     seq = graph.original_characters
#     print(f"Drawing graph {i} with sequence {seq}")
#     draw_graph(graph, sequence=seq, filename=f"graph_{i}_{seq}.pdf")
