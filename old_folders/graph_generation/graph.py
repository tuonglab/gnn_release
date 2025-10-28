import os
import tarfile

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Mapping from three-letter code to one-letter code for amino acids
AMINO_ACID_MAPPING = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}
# Read the PCA encoding matrix
PCA_ENCODING = pd.read_csv("/scratch/project/tcr_ml/gnn_release/graph_generation/AAidx_PCA_2024.txt", sep="\t", index_col=0)

CANCEROUS = 1
CONTROL = 0


class MultiGraphDataset(Dataset):
    """
    A dataset class for multi-graph data.

    Args:
        root (str): The root directory of the dataset.
        samples (list): A list of sample file names.
        cancer (bool, optional): Whether the data is cancerous or not. Defaults to False.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
        pre_transform (callable, optional): A function/transform that takes in a sample and returns a preprocessed version. Defaults to None.

        Initializes a MultiGraphDataset object.

        Args:
            root (str): The root directory of the dataset.
            samples (list): A list of sample file names.
            cancer (bool, optional): Whether the data is cancerous or not. Defaults to False.
            transform (callable, optional): A function/transform that takes in a sample and returns a transformed version. Defaults to None.
            pre_transform (callable, optional): A function/transform that takes in a sample and returns a preprocessed version. Defaults to None.

    """

    def __init__(self, root, samples, cancer=False, transform=None, pre_transform=None):
        self.samples = samples
        self.root = root # root directory of the dataset
        self.cancer = cancer
        super(MultiGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.samples

    @property
    def processed_file_names(self):
        return ["processed_data.pt"]

    def process(self):
        processed_dir = os.path.join(self.root, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # Process each file in a loop
        for file in self.raw_paths:
            self.process_file(file)

    def process_file(self, raw_path):
        tmp_dir = os.getenv("TMPDIR")
        processed_dir = os.path.join(self.root, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # Check if the file is a tar.gz file
        if raw_path.endswith(".tar.gz"):
            print(f"Processing raw file: {raw_path}")  # Debug print
            # Extract the tar file
            with tarfile.open(raw_path, "r:gz") as tar:
                tar.extractall(path=tmp_dir)

            # Get the list of extracted files
            extracted_files = [f for f in os.listdir(tmp_dir) if f.endswith(".txt")]

            # Initialize a list to hold all data objects for this sample
            data_objects = []

            # Process each extracted file
            for file in extracted_files:
                print(f"Processing extracted file: {file}")  # Debug print
                file_path = os.path.join(tmp_dir, file)
                data = self.create_data_object(
                    file_path, PCA_ENCODING, AMINO_ACID_MAPPING, label=self.cancer
                )

                # Add the data object to the list
                data_objects.append(data)

                # Remove the file as soon as it's processed
                os.remove(file_path)

            # Get the base name of the raw file (without directories)
            base_name = os.path.basename(raw_path)
            # Remove the .tar.gz extension
            name_without_extension = os.path.splitext(base_name)[0]
            # Save all data objects for this sample to a .pt file
            torch.save(
                data_objects,
                os.path.join(processed_dir, f"{name_without_extension}.pt"),
            )

            print(
                f"Saving processed data to: {os.path.join(processed_dir, f'{os.path.splitext(raw_path)[0]}.pt')}"
            )

    def len(self):
        processed_dir = os.path.join(self.root, "processed")
        return len(os.listdir(processed_dir))

    def get(self, idx):
        processed_dir = os.path.join(self.root, "processed")
        file_name = os.listdir(processed_dir)[idx]
        data = torch.load(os.path.join(processed_dir, file_name))
        return data

    def download(self):
        pass  # download raw data here (could be implemented later)

    def create_data_object(self, edge_file, pca_encoding, amino_acid_mapping, label=CANCEROUS):
        if not self.cancer:
            label = CONTROL

        with open(edge_file, "r") as f:
            edgelist = [line.strip().split() for line in f]

        # Initialize nodes, node_mapping, and original_characters as empty lists and dictionaries
        nodes = []
        node_mapping = {}
        node_position_dict = {}  # To store nodes and their positions
        position_counter = 0  # Position counter to keep track of sequence position

        for edge in edgelist:
            for node in [(edge[0], edge[1]), (edge[2], edge[3])]:
                # Only add the node if it's not already in the mapping
                if node not in node_mapping:
                    node_mapping[node] = len(nodes)
                    nodes.append(node)
                    # Store the node with its sequence position
                    node_position_dict[int(node[1])] = amino_acid_mapping[node[0]]
                    position_counter += 1

        # Reconstruct the sequence based on positions
        sorted_positions = sorted(node_position_dict.keys())
        reconstructed_sequence = ''.join([node_position_dict[pos] for pos in sorted_positions])

        # Create the source and target node lists
        source_nodes = [node_mapping[(edge[0], edge[1])] for edge in edgelist]
        target_nodes = [node_mapping[(edge[2], edge[3])] for edge in edgelist]

        # Add reverse edges for undirected graph
        source_nodes += [node_mapping[(edge[2], edge[3])] for edge in edgelist]
        target_nodes += [node_mapping[(edge[0], edge[1])] for edge in edgelist]

        # Convert to a tensor
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # Create the node feature matrix
        x = torch.tensor(
            np.array([pca_encoding.loc[amino_acid_mapping[node[0]]].values for node in nodes]),
            dtype=torch.float,
        )

        y = torch.tensor([label], dtype=torch.long)

        # Create the Data object with an additional 'original_characters' attribute
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            original_characters=reconstructed_sequence,
        )

        return data


def draw_graph(data, sequence=None, filename="graph.pdf"):  # Default filename with .pdf extension
    import matplotlib.pyplot as plt
    import networkx as nx

    # Convert to NetworkX graph
    G = nx.from_edgelist(data.edge_index.numpy().T)

    # If sequence string is provided, map nodes to characters
    if sequence:
        # Create a dictionary mapping node numbers (0, 1, 2, ...) to characters
        labels = {i: char for i, char in enumerate(sequence)}
    else:
        # If no sequence provided, just use the node numbers as labels
        labels = {i: i for i in range(len(G.nodes))}

    # Use spring_layout for better spacing, adjust the k parameter to space out nodes
    pos = nx.spring_layout(G, k=2.5, iterations=150)  # Adjust k for more/less spacing
    
    # Draw the graph with node labels and better spacing
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=10, edge_color='gray')

    # If sequence is provided, add it as a legend
    if sequence:
        legend_label = f"Sequence: {sequence}"
        plt.legend([legend_label], loc='upper right', fontsize=12, frameon=False)

    # Save the plot to a file in PDF format
    plt.savefig(filename, format='pdf')  # Save as PDF
    plt.clf()  # Clear the figure to avoid overlapping plots

def load_graphs(file:str):
    """
    Load graphs from a file.

    Args:
        file (str): The path to the file containing the graphs.

    Returns:
        dataset: The loaded dataset containing the graphs.
    """
    dataset = torch.load(file,map_location=device)
    return dataset
