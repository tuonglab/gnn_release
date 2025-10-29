import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph_generation.graph import load_graphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(46)
np.random.seed(46)

MODEL_PATH = "model_2025_uncertainty_curated"
MODEL_NAME = "best_model.pt"
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)


class GATv2Heteroscedastic(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.17, temperature=1.0):
        super().__init__()
        self.conv1 = GATv2Conv(nfeat, nhid, heads=16, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(nhid, nhid, heads=16, dropout=dropout, concat=False)
        self.classifier = torch.nn.Linear(nhid, nclass)
        self.log_var_layer = torch.nn.Linear(nhid, 1)
        self.dropout = dropout
        self.temperature = temperature

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=True)  # Enable dropout for MC Dropout
        x = global_mean_pool(x, batch)
        logits = self.classifier(x) / self.temperature
        log_var = self.log_var_layer(x)
        return logits, log_var


def heteroscedastic_bce_loss(logits, log_var, labels):
    out_1 = logits[:, 1]
    bce = F.binary_cross_entropy_with_logits(out_1, labels.float(), reduction="none")
    # log_var = torch.clamp(log_var, min=-10.0, max=10.0)
    precision = torch.exp(-log_var.squeeze(-1))
    loss = 0.5 * precision * bce + 0.5 * log_var.squeeze(-1)
    return loss.mean()


def train_hetero(model, loader, num_epochs=100, patience=15):
    os.makedirs(MODEL_PATH, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.25)
    best_loss = float("inf")
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for sample in loader:
            for data in sample:
                data = data.to(device)
                optimizer.zero_grad()
                logits, log_var = model(data.x, data.edge_index, data.batch)
                loss = heteroscedastic_bce_loss(logits, log_var, data.y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_size = data.y.size(0)
                total_loss += loss.item() * batch_size
                predicted = torch.round(torch.sigmoid(logits[:, 1]))
                total_correct += (predicted == data.y).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.5f}, Acc: {accuracy:.5f}")

        min_delta_loss = 0.02
        min_delta_acc = 0.01

        improved = False

        if avg_loss < best_loss - min_delta_loss:
            best_loss = avg_loss
            improved = True

        if accuracy > best_accuracy + min_delta_acc:
            best_accuracy = accuracy
            improved = True

        if improved:
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_FILE)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping due to no improvement in loss or accuracy after {patience} epochs"
            )
            break


def load_train_data(cancer_paths, control_paths):
    training_set = []
    for path_list in [cancer_paths, control_paths]:
        for path in path_list:
            if os.path.isdir(path):
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    graphs = load_graphs(file_path)
                    training_set.append(graphs)
    return training_set


def main():
    train_cancer_directories = [
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed",
    ]
    train_control_directories = [
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/control+pica/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/control/processed",
        # "/scratch/project/tcr_ml/gnn_release/test_data_v2/pica_complete/processed"
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/curated/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/single_cell_control/processed"
    ]

    train_set = load_train_data(train_cancer_directories, train_control_directories)
    model = GATv2Heteroscedastic(
        nfeat=train_set[0][0].num_node_features,
        nhid=375,
        nclass=2,
        dropout=0.17,
        temperature=1.0,
    ).to(device)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    train_hetero(model, train_loader, num_epochs=100, patience=15)


if __name__ == "__main__":
    main()
