from graph_generation.graph import load_graphs,draw_graph
# Specify the path to your .pt file
file_path = "/scratch/project/tcr_ml/GNN/test_data_v2/cancer/processed/Sample_18885402-2850-4f6d-9b3b-acccc06a6ca0_cdr3_results_edges.tar.pt"  # replace with your file path

# Load the graphs
graphs = load_graphs(file_path)
for i in range(4):
    sequence = graphs[i].original_characters
    print(sequence)
    print(len(graphs[i].original_characters))
    draw_graph(graphs[i],sequence=sequence,filename='graph'+str(i)+'.pdf')
