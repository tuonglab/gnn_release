import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from graph_generation.graph import load_graphs

# Set the random seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(46)
np.random.seed(46)

# Load model path to save here
MODEL_PATH = "model_2025_sc_all_cancer"
MODEL_NAME = "best_model.pt"
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)


class GATv2(torch.nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        temperature: float = 1,
    ) -> None:
        super(GATv2, self).__init__()
        self.conv1 = GATv2Conv(nfeat, nhid, heads=16, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(nhid, nhid, heads=16, dropout=dropout, concat=False)
        self.classifier = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.temperature = temperature

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x /= self.temperature
        return x


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    num_epochs: int = 100,
    patience: int = 15,
) -> None:
    os.makedirs(MODEL_PATH, exist_ok=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.25)

    best_loss = float("inf")
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for sample in loader:
            for data in sample:
                data = data.to(device)  # Move data to same device as model
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                out = out[:, 1]

                loss = criterion(out, data.y.float())
                batch_size = data.y.size(0)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size

                predicted = torch.round(torch.sigmoid(out.data))
                total_correct += (predicted == data.y).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"Epoch: {epoch+1}, Loss: {avg_loss}, Accuracy: {accuracy}")

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



def load_train_data(cancer_paths: list, control_paths: list):
    training_set = []

    for cancer_path in cancer_paths:
        if os.path.isdir(cancer_path):
            for filename in os.listdir(cancer_path):
                file_path = os.path.join(cancer_path, filename)
                graphs = load_graphs(file_path)
                training_set.append(graphs)

    for control_path in control_paths:
        if os.path.isdir(control_path):
            for filename in os.listdir(control_path):
                file_path = os.path.join(control_path, filename)
                graphs = load_graphs(file_path)
                training_set.append(graphs)

    return training_set


def main() -> None:
    train_cancer_directories = [
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed",
    ]
    train_control_directories = [
        "/scratch/project/tcr_ml/gnn_release/dataset_v2/single_cell_control/processed"
    ]

    train_set = load_train_data(train_cancer_directories, train_control_directories)

    model = GATv2(
        nfeat=train_set[0][0].num_node_features, nhid=375, nclass=2, dropout=0.17
    )
    model.to(device)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)

    train(model, train_loader, num_epochs=500)


if __name__ == "__main__":
    main()
