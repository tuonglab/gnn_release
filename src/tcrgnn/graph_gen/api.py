import logging
from pathlib import Path

import pandas as pd
from torch_geometric.data import Data

from ._io import list_edge_txts, parse_edges
from .build_graph import build_graph_from_edgelist

LOG = logging.getLogger(__name__)


def generate_graphs_from_edge_dir(
    edge_dir: Path,
    pca_encoding: pd.DataFrame,
    aa_map: dict[str, str],
    label: int,
) -> list[Data]:
    """
    Build graphs from all edge text files in a directory.

    Parameters
    ----------
    edge_dir : pathlib.Path
        Directory containing edge list .txt files.
    pca_encoding : pandas.DataFrame
        PCA encoding table indexed by single letter amino acids.
    aa_map : dict[str, str]
        Mapping from three letter to single letter amino acids.
    label : int
        Integer class label to attach to each graph.

    Returns
    -------
    list[torch_geometric.data.Data]
        One graph per edge text file.
    """
    # Collect files and keep order stable
    edge_txt_files = sorted(list_edge_txts(edge_dir))
    return [
        build_graph_from_edgelist(
            parse_edges(p),  # convert file to edgelist
            pca_encoding=pca_encoding,
            aa_map=aa_map,
            label=label,
        )
        for p in edge_txt_files
    ]


def generate_graph_from_edge_file(edge_file: Path):
    """Placeholder function to build a graph from a single edge text file."""

    edge_lst = parse_edges(edge_file)
    graph = build_graph_from_edgelist(edge_lst)
    return graph
