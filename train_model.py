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
MODEL_PATH = "model_2025_prem_boltz"
MODEL_NAME = "best_model.pt"
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)


class GATv2(torch.nn.Module):
    """
    Graph Attention Network version 2 (GATv2) module.

    Args:
        nfeat (int): Number of input features.
        nhid (int): Number of hidden units.
        nclass (int): Number of output classes.
        dropout (float): Dropout rate.
        temperature (float, optional): Temperature parameter to scale the confidence of output probabilities. Defaults to 1.1.

    Attributes:
        conv1 (GATv2Conv): First GATv2 convolutional layer.
        conv2 (GATv2Conv): Second GATv2 convolutional layer.
        classifier (torch.nn.Linear): Linear layer for classification.
        dropout (float): Dropout rate.
        temperature (float): Temperature parameter.

    Methods:
        forward(x, edge_index, batch): Forward pass of the GATv2 module.


        Forward pass of the GATv2 module.

        Args:
            x (torch.Tensor): Input features.
            edge_index (torch.Tensor): Graph edge indices.
            batch (torch.Tensor): Batch indices.

        Returns:
            torch.Tensor: Output tensor after the forward pass.

    """

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
        self.temperature = temperature  # scale confidence of output prob

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout layer
        x = global_mean_pool(x, batch)  # Global pooling
        x = self.classifier(x)
        x /= self.temperature
        return x


def train(
    model: torch.nn.Module,
    loader: DataLoader,
    num_epochs: int = 100,
    patience: int = 15,
) -> None:
    """
    Trains the given model using the provided data loader for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader): The data loader containing the training data.
        num_epochs (int, optional): The number of epochs to train the model (default: 100).
        patience (int, optional): The number of epochs to wait for improvement in loss or accuracy before early stopping (default: 15).

    Returns:
        None
    """
    os.makedirs(MODEL_PATH, exist_ok=True)
    # pos_weight = torch.tensor([28])  # Convert the float to a tensor
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([25]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.25)

    best_loss = float("inf")  # Initialize the best loss
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = (
            0  # Initialize a counter for the total number of correct predictions
        )
        total_samples = 0  # Initialize a counter for the total number of samples
        for sample in loader:
            for data in sample:
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
        avg_loss = total_loss / total_samples  # Calculate average loss
        accuracy = total_correct / total_samples  # Calculate accuracy
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


import os

def load_train_data(cancer_paths: list, control_paths: list):
    """
    Load training data from the specified paths.

    Args:
        cancer_paths (list): A list of directories containing cancer data files.
        control_paths (list): A list of directories containing control data files.

    Returns:
        list: A list of training samples.
    """
    training_set = []

    # Process cancer graphs
    for cancer_path in cancer_paths:
        if os.path.isdir(cancer_path):
            for filename in os.listdir(cancer_path):
                file_path = os.path.join(cancer_path, filename)
                graphs = load_graphs(file_path)
                training_set.append(graphs)

    # Process control graphs
    for control_path in control_paths:
        if os.path.isdir(control_path):
            for filename in os.listdir(control_path):
                file_path = os.path.join(control_path, filename)
                graphs = load_graphs(file_path)
                training_set.append(graphs)

    return training_set



def main() -> None:

    # Specify the directory you want to traverse
    # replace with your directory path
    train_cancer_directories = [
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/blood_tissue/processed",
        # '/scratch/project/tcr_ml/gnn_release/dataset_v2/ccdi/processed',
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/d360/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/scTRB/processed",
        # "/scratch/project/tcr_ml/gnn_release/test_data_v2/seekgene/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/tumor_tissue/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/ccdi_boltz/processed"
    ]
    train_control_directories = [
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/curated/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/control/processed"
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/single_cell_control/processed",
        # "/scratch/project/tcr_ml/gnn_release/dataset_v2/control+pica/processed",
        "/scratch/project/tcr_ml/gnn_release/dataset_boltz/control_training/processed"
    ]

    train_set = load_train_data(train_cancer_directories, train_control_directories)


    model = GATv2(
        nfeat=train_set[0][0].num_node_features, nhid=375, nclass=2, dropout=0.17
    )
    model.to(device)

    train_loader = DataLoader(
        train_set,
        batch_size=256,
        shuffle=True,
    )

    # Train the model on the training samples
    train(model, train_loader, num_epochs=500)


if __name__ == "__main__":
    main()
