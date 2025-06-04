import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.decomposition import PCA
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from graph_generation.graph import load_graphs

# === Set Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(46)
np.random.seed(46)

# === GATv2 Model Definition ===
class GATv2(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, temperature=1.0):
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(nfeat, nhid, heads=16, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(nhid, nhid, heads=16, dropout=dropout, concat=False)
        self.classifier = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.temperature = temperature

    def forward(self, x, edge_index, batch):
        x = torch.nn.functional.leaky_relu(self.conv1(x, edge_index))
        x = torch.nn.functional.leaky_relu(self.conv2(x, edge_index))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x /= self.temperature
        return x

    def extract_embedding(self, x, edge_index, batch):
        x = torch.nn.functional.leaky_relu(self.conv1(x, edge_index))
        x = torch.nn.functional.leaky_relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

# === Load Graphs from Directory with Limit per Dataset ===
def load_graphs_with_sources(dataset_dirs: dict, max_graphs_per_source: int = None):
    graphs = []
    sources = []

    for source_name, paths in dataset_dirs.items():
        count = 0
        for path in paths:
            if not os.path.isdir(path):
                continue
            for filename in os.listdir(path):
                if max_graphs_per_source is not None and count >= max_graphs_per_source:
                    break
                file_path = os.path.join(path, filename)
                graph_list = load_graphs(file_path)
                for graph in graph_list:
                    if max_graphs_per_source is not None and count >= max_graphs_per_source:
                        break
                    graphs.append(graph)
                    sources.append(source_name)
                    count += 1
            if max_graphs_per_source is not None and count >= max_graphs_per_source:
                break
    return graphs, sources

# === Main Function ===
def main():
    MODEL_PATH = "model_2025_ccdi_only"
    MODEL_NAME = "best_model.pt"
    MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
    MAX_GRAPHS_PER_SOURCE = 5000
    MAX_POINTS = 20000  # Max to plot for UMAP

    dataset_dirs = {
        "blood":   ["/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed"],
        "ccdi":    ["/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed"],
        "PICA":    ["/scratch/project/tcr_ml/gnn_release/test_data_v2/20240918_WGS_20240924_sc_PICA0008-PICA0032_Pool_1/processed"],
        "scTRB":   ["/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed"],
        "tumor":   ["/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed"],
        "control": ["/scratch/project/tcr_ml/gnn_release/dataset_v2/control/processed"]
    }

    # Biological label mapping
    source_to_biolabel = {
        "blood": "cancer",
        "ccdi": "cancer",
        "PICA": "control",
        "scTRB": "cancer",
        "tumor": "cancer",
        "control": "control",
    }
    biolabel_to_int = {"control": 0, "cancer": 1}

    print("Loading graphs (max 5000 per source)...")
    start = time.time()
    graph_list, source_list = load_graphs_with_sources(dataset_dirs, max_graphs_per_source=MAX_GRAPHS_PER_SOURCE)
    print(f"Loaded {len(graph_list)} graphs in {time.time() - start:.2f} seconds.")

    source_names = sorted(set(source_list))
    source_to_int = {name: idx for idx, name in enumerate(source_names)}
    int_labels = [source_to_int[src] for src in source_list]
    biolabels = [biolabel_to_int[source_to_biolabel[src]] for src in source_list]

    sample_graph = graph_list[0]
    model = GATv2(
        nfeat=sample_graph.num_node_features,
        nhid=375,
        nclass=2,
        dropout=0.17,
    )
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()

    loader = DataLoader(graph_list, batch_size=256, shuffle=False, num_workers=4)

    print("Extracting embeddings...")
    all_embeddings = []
    all_source_labels = []
    all_biolabels = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            emb = model.extract_embedding(batch.x, batch.edge_index, batch.batch)
            all_embeddings.append(emb.cpu())
            all_source_labels.extend(int_labels[i * batch.num_graphs:(i + 1) * batch.num_graphs])
            all_biolabels.extend(biolabels[i * batch.num_graphs:(i + 1) * batch.num_graphs])

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    source_labels = np.array(all_source_labels)
    biolabels = np.array(all_biolabels)
    print(f"Extracted embeddings for {embeddings.shape[0]} graphs.")

    # PCA
    print("Running PCA...")
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)

    # Subsample if necessary
    if embeddings_pca.shape[0] > MAX_POINTS:
        print(f"Subsampling to {MAX_POINTS} points for UMAP...")
        idx = np.random.choice(len(embeddings_pca), size=MAX_POINTS, replace=False)
        embeddings_pca = embeddings_pca[idx]
        source_labels = source_labels[idx]
        biolabels = biolabels[idx]

    # UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings_pca)

    # Plot 1: Dataset Source
    print("Plotting UMAP by dataset source...")
    cmap = plt.cm.get_cmap("tab10", len(source_to_int))
    colors = [cmap(i) for i in range(len(source_to_int))]

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=source_labels, cmap="tab10", s=6
    )
    sorted_sources = sorted(source_to_int.items(), key=lambda x: x[1])
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=name,
                   markerfacecolor=colors[idx], markersize=8)
        for name, idx in sorted_sources
    ]
    plt.legend(handles=handles, title="Dataset Source")
    plt.title("UMAP of Graph Embeddings Colored by Dataset Source")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig("umap_by_source.png", dpi=300)
    plt.show()

    # Plot 2: Biological label
    print("Plotting UMAP by biological label...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0], embedding_2d[:, 1],
        c=biolabels, cmap="coolwarm", s=6
    )
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=label,markerfacecolor=plt.cm.coolwarm(idx / (len(biolabel_to_int) - 1))
)
        for label, idx in biolabel_to_int.items()
    ]
    plt.legend(handles=handles, title="Biological Label")
    plt.title("UMAP of Graph Embeddings Colored by Biological Label")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig("umap_by_biolabel.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
