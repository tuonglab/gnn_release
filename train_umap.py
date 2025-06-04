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
        x = x / self.temperature
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
    MAX_GRAPHS_PER_SOURCE = 400000
    MAX_POINTS = 100000         # Max to plot for UMAP
    MAX_EPOCHS = 400           # Train up to 400 epochs if no early stop
    PATIENCE = 15              # Early stopping patience on training loss
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4

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

    # Map source names to integer indices (for coloring by source)
    source_names = sorted(set(source_list))
    source_to_int = {name: idx for idx, name in enumerate(source_names)}
    source_labels = np.array([source_to_int[src] for src in source_list])

    # Compute biological labels (0 or 1) and assign to each graph as graph.y
    biolabels = np.array([biolabel_to_int[source_to_biolabel[src]] for src in source_list])
    for graph, lbl in zip(graph_list, biolabels):
        graph.y = torch.tensor([lbl], dtype=torch.long)

    # Create DataLoader for training (shuffle) and for embedding extraction (no shuffle)
    train_loader = DataLoader(
        graph_list,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    embed_loader = DataLoader(
        graph_list,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Sample a graph to get feature dimension
    sample_graph = graph_list[0]
    nfeat = sample_graph.num_node_features

    # Initialize model from scratch
    model = GATv2(nfeat=nfeat, nhid=375, nclass=2, dropout=0.17).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    def extract_and_visualize(epoch, embeddings, source_labels, biolabels):
        # PCA
        print(f"[Epoch {epoch}] Running PCA...")
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)

        # Subsample if necessary
        if embeddings_pca.shape[0] > MAX_POINTS:
            print(f"[Epoch {epoch}] Subsampling to {MAX_POINTS} points for UMAP...")
            idx = np.random.choice(len(embeddings_pca), size=MAX_POINTS, replace=False)
            embeddings_pca = embeddings_pca[idx]
            src_lbls = source_labels[idx]
            bio_lbls = biolabels[idx]
        else:
            src_lbls = source_labels
            bio_lbls = biolabels

        # UMAP
        print(f"[Epoch {epoch}] Running UMAP...")
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1)
        embedding_2d = reducer.fit_transform(embeddings_pca)

        # Plot by dataset source
        print(f"[Epoch {epoch}] Plotting UMAP by dataset source...")
        cmap = plt.cm.get_cmap("tab10", len(source_to_int))
        plt.figure(figsize=(10, 8))
        plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=src_lbls, cmap="tab10", s=6
        )
        sorted_sources = sorted(source_to_int.items(), key=lambda x: x[1])
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=name,
                       markerfacecolor=cmap(idx), markersize=8)
            for name, idx in sorted_sources
        ]
        plt.legend(handles=handles, title="Dataset Source")
        plt.title(f"UMAP of Graph Embeddings by Source (Epoch {epoch})")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(f"umap_by_source_epoch_{epoch}.png", dpi=300)
        plt.close()

        # Plot by biological label
        print(f"[Epoch {epoch}] Plotting UMAP by biological label...")
        plt.figure(figsize=(10, 8))
        plt.scatter(
            embedding_2d[:, 0], embedding_2d[:, 1],
            c=bio_lbls, cmap="coolwarm", s=6
        )
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=label,
                       markerfacecolor=plt.cm.coolwarm(idx / (len(biolabel_to_int) - 1)),
                       markersize=8)
            for label, idx in biolabel_to_int.items()
        ]
        plt.legend(handles=handles, title="Biological Label")
        plt.title(f"UMAP of Graph Embeddings by Biological Label (Epoch {epoch})")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(f"umap_by_biolabel_epoch_{epoch}.png", dpi=300)
        plt.close()

    best_loss = float("inf")
    epochs_no_improve = 0

    # === Training Loop with Early Stopping ===
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.num_graphs

        avg_loss = epoch_loss / len(graph_list)
        print(f"Epoch {epoch}/{MAX_EPOCHS} - Training Loss: {avg_loss:.4f}")

        # Check for improvement
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Extract & visualize after first epoch
        if epoch == 1:
            model.eval()
            all_embeddings = []
            with torch.no_grad():
                for batch in embed_loader:
                    batch = batch.to(device)
                    emb = model.extract_embedding(batch.x, batch.edge_index, batch.batch)
                    all_embeddings.append(emb.cpu())
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            extract_and_visualize(epoch, embeddings, source_labels, biolabels)

        # Early stopping check
        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs. Stopping early at epoch {epoch}.")
            # Extract & visualize at stopping epoch
            model.eval()
            all_embeddings = []
            with torch.no_grad():
                for batch in embed_loader:
                    batch = batch.to(device)
                    emb = model.extract_embedding(batch.x, batch.edge_index, batch.batch)
                    all_embeddings.append(emb.cpu())
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            extract_and_visualize(epoch, embeddings, source_labels, biolabels)
            break

        # If reached max epoch, extract & visualize
        if epoch == MAX_EPOCHS:
            model.eval()
            all_embeddings = []
            with torch.no_grad():
                for batch in embed_loader:
                    batch = batch.to(device)
                    emb = model.extract_embedding(batch.x, batch.edge_index, batch.batch)
                    all_embeddings.append(emb.cpu())
            embeddings = torch.cat(all_embeddings, dim=0).numpy()
            extract_and_visualize(epoch, embeddings, source_labels, biolabels)

    # Optionally, save the final model
    os.makedirs("trained_models", exist_ok=True)
    torch.save(model.state_dict(), "trained_models/gatv2_from_scratch.pt")
    print("Training complete. Model saved to 'trained_models/gatv2_from_scratch.pt'.")

if __name__ == "__main__":
    main()
