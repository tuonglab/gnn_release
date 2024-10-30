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

import os
from typing import List, Tuple, Any

def load_test_data(data_path: str) -> Tuple[List[str], List[Any]]:
    """
    Load test data from the specified path.

    Args:
        data_path (str): The path to the directory containing both cancer and control data files.

    Returns:
        Tuple[List[str], List[Any]]: A tuple containing two lists. The first list contains the modified filenames,
        and the second list contains the loaded test data graphs.
    """
    test_set = []
    file_set = []
    
    if os.path.isdir(data_path):
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            
            # Check if it's a file and skip directories
            if os.path.isfile(file_path):
                graphs = load_graphs(file_path)
                prefix = filename.split("_cdr3")[0]
                new_filename = prefix + "_cdr3.csv"
                test_set.append(graphs)
                file_set.append(new_filename)

    return file_set, test_set


def repr_plot(
    sample_representative_pred: list, sample_representative_label: list, output_dir: str
) -> None:
    """
    Generate a boxplot comparing the predictions of control and cancer samples.

    Args:
        sample_representative_pred (list): List of predicted values for each sample.
        sample_representative_label (list): List of labels for each sample (0 for control, 1 for cancer).
        output_dir (str): Directory where the boxplot image will be saved.

    Returns:
        None
    """
    control_preds = [
        pred
        for label, pred in zip(sample_representative_label, sample_representative_pred)
        if label == 0
    ]
    cancer_preds = [
        pred
        for label, pred in zip(sample_representative_label, sample_representative_pred)
        if label == 1
    ]

    print(control_preds, cancer_preds)

    # Create boxplots
    plt.figure(figsize=(12, 6))
    boxplot_elements = plt.boxplot([control_preds, cancer_preds], widths=0.4)

    # Add individual data points on top of the boxplot
    for i, element in enumerate(boxplot_elements["medians"]):
        xdata = element.get_xdata()
        ydata = element.get_ydata()
        ymean = ydata.mean()
        xmean = xdata.mean()

        # plot points
        if i == 0:
            plt.plot([xmean] * len(control_preds), control_preds, "r.", alpha=0.2)
        elif i == 1:
            plt.plot([xmean] * len(cancer_preds), cancer_preds, "r.", alpha=0.2)

    # Set y-axis limits
    plt.ylim(0.3, 1)
    plt.xticks([1, 2], ["Control", "Cancer"])
    plt.title("Control vs Cancer")
    plt.show()
    plt.savefig(os.path.join(output_dir, "boxplot.png"))
    plt.close()

def test(
    model: torch.nn.Module, model_file: str, loader: DataLoader, filenames: list[str]
) -> float:
    """
    Test the model on the given samples and calculate the AUC.

    Args:
        model (torch.nn.Module): The trained model.
        model_path (str): The path to the saved model.
        loader (torch.utils.data.DataLoader): The data loader for loading the test data.
        filenames (list): The list of filenames corresponding to the test data.

    Returns:
        float: The sample AUC (Area Under the Curve) score.
    """

    def create_filename(label: int, original_filename: str, iteration: int, suffix: str) -> str:
        """
        Creates a new filename based on the given parameters.

        Parameters:
            label (int): The numeric label.
            original_filename (str): The original filename.
            iteration (int): The iteration number.
            suffix (str): The suffix to be added to the new filename.

        Returns:
            str: The new filename.
        """
        # Map the numeric label to a string
        label_map = {0: "control", 1: "cancer"}
        label_str = label_map[label]

        # Strip the suffix until "cdr3" and add the specified suffix
        prefix = original_filename.split("_cdr3")[0]
        new_filename = f"{prefix}_{iteration}_{label_str}_cdr3_{suffix}"

        return new_filename

    def save_scores_to_file(
        scores: list, original_filename: str, filename: str, directory: str
    ) -> None:
        """
        Save scores to a file along with corresponding sequences.

        Args:
            scores (list): List of scores.
            original_filename (str): Name of the original file.
            filename (str): Name of the file to be saved.
            directory (str): Directory where the file will be saved.

        Returns:
            None

        Raises:
            FileNotFoundError: If the original file is not found in any of the directories.

        """
        os.makedirs(directory, exist_ok=True)
        # Create the full path for the file
        file_path = os.path.join(directory, filename)

        # Define the directories where the original files might be
        directories = [
            "/scratch/project/tcr_ml/colabfold/cancer",
            "/scratch/project/tcr_ml/colabfold/control",
            "/scratch/project/tcr_ml/colabfold/phs002517",
        ]

        # Initialize sequences as an empty list
        sequences = []

        # Try to read the sequences from the original file in each directory
        for dir in directories:
            try:
                df = pd.read_csv(os.path.join(dir, original_filename))
                sequences = df["sequence"].tolist()  # Assuming 'sequence' is the column name
                break  # If the file is successfully read, break the loop
            except FileNotFoundError:
                continue  # If the file is not found in the current directory, continue to the next directory

        # If the sequences list is still empty after trying all directories, print an error message
        if not sequences:
            print(f"Error: The file {original_filename} was not found in any of the directories.")
            return

        # Open the file and write the sequences and scores
        with open(file_path, "w") as file:
            for sequence, score in zip(sequences, scores):
                file.write(f"{sequence},{score}\n")

    # Load the model from the file
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.eval()  # put in eval mode

    sample_reprensatative_pred = []
    sample_reprensatative_label = []

    for i, sample in enumerate(loader):  # iterate over samples
        sample_scores = []
        sample_labels = []
        for j, data in enumerate(sample):  # iterate over TCR sequences in each sample
            out = model(data.x, data.edge_index, data.batch)
            probabilities = torch.sigmoid(out)
            sample_scores.extend(probabilities[:, 1].tolist())
            sample_labels.extend(data.y.tolist())

        # calculate the mean feature importance score for the sample
        mean_sample_scores = np.mean(sample_scores)
        mean_sample_labels = np.mean(sample_labels)

        scores_filename = create_filename(
            int(mean_sample_labels), filenames[i], i, suffix="scores.txt"
        )
        # Save the scores to a file in the model path directory
        save_scores_to_file(sample_scores, filenames[i], scores_filename, MODEL_PATH + "/scores")

        sample_reprensatative_pred.append(float(mean_sample_scores))
        sample_reprensatative_label.append(float(mean_sample_labels))

    # sample_auc = roc_auc_score(sample_reprensatative_label, sample_reprensatative_pred)
    # repr_plot(sample_reprensatative_pred, sample_reprensatative_label, MODEL_PATH)
    return "Done"

def test_trained_model(dataset_path, model_path) -> None:
    """
    Function to test a trained model using test data from cancer and control paths

    Args:
        cancer_path (str): Path to the cancer test dataset.
        control_path (str): Path to the control test dataset.
        model_path (str): Path where the model is saved.
    """
    filenames, test_set = load_test_data(dataset_path)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    model_file = os.path.join(model_path, MODEL_NAME)

    model = GATv2(nfeat=test_set[0][0].num_node_features, nhid=256, nclass=2, dropout=0.17)
    model.to(device)

    test(model, model_file, test_loader, filenames)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test a trained model with datasets.")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the test dataset.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the saved model file.")

    args = parser.parse_args()
    MODEL_PATH = args.model_path

    # Call the test function with the arguments
    test_trained_model(args.dataset_path,args.model_path)

    # After the function is done
    if os.path.exists(args.model_path + "/train.out"):
        os.remove(args.model_path + "/train.out")
    # shutil.move("train.out", args.model_path)