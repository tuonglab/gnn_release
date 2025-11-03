import logging
from pathlib import Path

import pandas as pd
from torch_geometric.data import Data

from ._build_graph import build_graph_from_edgelist
from ._io import list_edge_txts, parse_edges, save_graphs_to_disk

LOG = logging.getLogger(__name__)


def generate_graphs_from_edge_dir(
    edge_dir: Path,
    pca_encoding: pd.DataFrame,
    aa_map: dict[str, str],
    label: int,
    save_to_disk: bool = False,
    filename: str = "generated_graphs.pt",
) -> list[Data]:
    """
    Build graphs from all edge list text files in a directory.

    Parameters
    ----------
    edge_dir : pathlib.Path
        Directory containing edge list .txt files.
    pca_encoding : pandas.DataFrame
        PCA encoding table indexed by single letter amino acids.
    aa_map : dict[str, str]
        Mapping from three letter to single letter amino acids.
    label : int
        Integer class label attached to each graph.
    save_to_disk : bool, optional
        If True, the generated graphs are written to disk.
    filename : str, optional
        Output file name used when saving graphs to disk.

    Returns
    -------
    list[torch_geometric.data.Data]
        One graph per edge text file.
    """
    # Collect files and keep order stable
    edge_txt_files = sorted(list_edge_txts(edge_dir))
    graphs = [
        build_graph_from_edgelist(
            parse_edges(p),  # convert file to edgelist
            pca_encoding=pca_encoding,
            aa_map=aa_map,
            label=label,
        )
        for p in edge_txt_files
    ]

    if save_to_disk:
        save_graphs_to_disk(graphs, filename)
    return graphs


def generate_graph_from_edge_file(
    edge_file: Path, pca_encoding: pd.DataFrame, aa_map: dict[str, str], label: int
) -> Data:
    """
    Generate a single PyTorch Geometric graph from an edge list text file.

    This function reads a text file containing whitespace-separated edge
    specifications, parses them into an edgelist, and constructs a graph
    object suitable for downstream graph neural network processing.

    Parameters
    ----------
    edge_file : pathlib.Path
        Path to a text file where each non-empty line contains four
        whitespace-separated tokens representing an edge
        (amino acid code, position, amino acid code, position).

    Returns
    -------
    torch_geometric.data.Data
        A graph object constructed from all edges described in the file.
    """
    edge_lst = parse_edges(edge_file)
    graph = build_graph_from_edgelist(
        edge_lst, pca_encoding=pca_encoding, aa_map=aa_map, label=label
    )
    return graph
