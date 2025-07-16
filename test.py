import heapq
import os
import shutil
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from graph_generation.graph import load_graphs

from train_model import GATv2

# Set the random seed for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(46)
np.random.seed(46)

# Load model path to save here
MODEL_NAME = "best_model.pt"

from typing import List, Tuple, Any

def load_test_data(data_path: str) -> Tuple[List[str], List[Any]]:
    test_set = []
    file_set = []

    if os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            if os.path.isfile(file_path):
                graphs = load_graphs(file_path)
                prefix = filename.split("_cdr3")[0]
                new_filename = prefix + "_cdr3.csv"
                test_set.append(graphs)
                file_set.append(new_filename)

    return file_set, test_set

def test(
    model: torch.nn.Module,
    model_file: str,
    loader: DataLoader,
    filenames: List[str],
    scores_dir: str,
) -> float:
    def create_filename(label: int, original_filename: str, iteration: int, suffix: str) -> str:
        label_map = {0: "control", 1: "cancer"}
        label_str = label_map[label]
        prefix = original_filename.split("_cdr3")[0]
        new_filename = f"{prefix}_{iteration}_{label_str}_cdr3_{suffix}"
        return new_filename

    def save_scores_to_file(scores: list, original_filename: str, filename: str, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, filename)
        base_dir = "/scratch/project/tcr_ml/colabfold"

        sequences = None
        found_file = None

        for root, _, files in os.walk(base_dir):
            csv_files = [f for f in files if f.endswith('.csv')]
            if not csv_files:
                continue

            if original_filename in csv_files:
                try:
                    df = pd.read_csv(os.path.join(root, original_filename))
                    sequences = df["sequence"].tolist()
                    found_file = os.path.join(root, original_filename)
                    break
                except Exception as e:
                    print(f"Error reading CSV file {os.path.join(root, original_filename)}: {str(e)}")
                    continue

        if sequences is None:
            raise FileNotFoundError(
                f"Error: The CSV file {original_filename} was not found in {base_dir} or any of its subdirectories."
            )

        print(f"Found and read CSV file from: {found_file}")
        max_length = max(len(sequences), len(scores))

        with open(file_path, "w") as file:
            for i in range(max_length):
                sequence = sequences[i] if i < len(sequences) else "N/A"
                score = scores[i] if i < len(scores) else "N/A"
                file.write(f"{sequence},{score}\n")

    # Load model state and move to device
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    sample_reprensatative_pred = []
    sample_reprensatative_label = []

    for i, sample in enumerate(loader):
        original_filename = filenames[i]
        prefix = original_filename.split("_cdr3")[0]


        # Check if any file in scores_dir matches: prefix_..._cdr3_scores.txt
        existing_scores = [
            f for f in os.listdir(scores_dir)
            if f.startswith(f"{prefix}_") and f.endswith("_cdr3_scores.txt")
        ] if os.path.exists(scores_dir) else []

        if existing_scores:
            print(f"Skipping already processed file: {original_filename} (found: {existing_scores[0]})")
            continue
        sample_scores = []
        sample_labels = []
        try:
            for j, data in enumerate(sample):
                data = data.to(device)
                if data.x.dim() != 2:
                    raise ValueError(f"Input features must be 2-dimensional. Current shape: {data.x.shape}")
                out = model(data.x, data.edge_index, data.batch)
                probabilities = torch.sigmoid(out)
                sample_scores.extend(probabilities[:, 1].tolist())
                sample_labels.extend(data.y.tolist())
        except Exception as e:
            print(f"\nError processing file: {filenames[i]}")
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            continue

        mean_sample_scores = np.mean(sample_scores)
        mean_sample_labels = np.mean(sample_labels)

        scores_filename = create_filename(int(mean_sample_labels), filenames[i], i, suffix="scores.txt")
        save_scores_to_file(sample_scores, filenames[i], scores_filename, scores_dir)

        sample_reprensatative_pred.append(float(mean_sample_scores))
        sample_reprensatative_label.append(float(mean_sample_labels))

    return "Done"

def test_trained_model(dataset_path, model_path, dataset_name) -> None:
    filenames, test_set = load_test_data(dataset_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model_file = os.path.join(model_path, MODEL_NAME)
    scores_dir = os.path.join(model_path, f"{dataset_name}_scores")

    model = GATv2(nfeat=test_set[0][0].num_node_features, nhid=375, nclass=2, dropout=0.17)
    model.to(device)

    test(model, model_file, test_loader, filenames, scores_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model with datasets.")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the saved model file.")
    parser.add_argument('--dataset-name', type=str, required=True, help="Name of the dataset (used for naming output folders).")

    args = parser.parse_args()

    test_trained_model(args.dataset_path, args.model_path, args.dataset_name)

    # Optionally remove train.out if it exists
    if os.path.exists(os.path.join(args.model_path, "train.out")):
        os.remove(os.path.join(args.model_path, "train.out"))
