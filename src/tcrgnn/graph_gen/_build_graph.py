from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _index_nodes_and_edges(
    edgelist: Iterable[tuple[str, str, str, str]],
    aa_map: dict[str, str],
) -> tuple[
    list[tuple[str, str]],
    dict[tuple[str, str], int],
    dict[int, str],
    list[tuple[int, int]],
]:
    """
    Convert an edgelist of (a3, i_str, b3, j_str) into indexed graph parts.

    Each tuple represents a directed edge from residue (a3, i_str) to (b3, j_str).
    Residue codes use three letter amino acid abbreviations mapped to single letter
    via aa_map. Positions are provided as strings and cast to int.

    Parameters
    ----------
    edgelist
        Iterable of 4-tuples (a3, i_str, b3, j_str).
    aa_map
        Mapping from three letter amino acid codes to single letter codes.

    Returns
    -------
    nodes
        Stable ordered list of nodes as (a3, i_str).
    node_map
        Mapping from node tuple to integer index.
    pos_to_char
        Mapping from integer position to single letter amino acid.
    edge_pairs
        Directed edges as index pairs (src_idx, dst_idx).
    """
    nodes: list[tuple[str, str]] = []
    node_map: dict[tuple[str, str], int] = {}
    pos_to_char: dict[int, str] = {}
    edge_pairs: list[tuple[int, int]] = []

    for a3, i_str, b3, j_str in edgelist:
        # register nodes
        for node in ((a3, i_str), (b3, j_str)):
            if node not in node_map:
                node_map[node] = len(nodes)
                nodes.append(node)
                pos_to_char[int(node[1])] = aa_map[node[0]]
        # record directed edge
        edge_pairs.append((node_map[(a3, i_str)], node_map[(b3, j_str)]))

    return nodes, node_map, pos_to_char, edge_pairs


def _assemble_graph(
    nodes: list[tuple[str, str]],
    edge_pairs: list[tuple[int, int]],
    pca_encoding: pd.DataFrame,
    aa_map: dict[str, str],
    pos_to_char: dict[int, str],
    label: int,
    *,
    undirected: bool = True,
) -> Data:
    """
    Assemble a torch_geometric.data.Data graph from node and edge components.

    Parameters
    ----------
    nodes
        Ordered nodes as (a3, i_str).
    edge_pairs
        Directed edges as index pairs (src_idx, dst_idx).
    pca_encoding
        DataFrame indexed by single letter amino acids with PCA feature columns.
    aa_map
        Mapping from three letter codes to single letter amino acids.
    pos_to_char
        Mapping from position to single letter amino acid for sequence reconstruction.
    label
        Integer class label for the whole graph.
    undirected
        If True, add reverse edges to symmetrize the graph.

    Returns
    -------
    Data
        PyG Data object with fields x, edge_index, y, and original_characters.
    """
    if len(nodes) == 0:
        raise ValueError("Cannot assemble a graph with no nodes")

    # Build edge_index
    if undirected:
        src = [s for s, _ in edge_pairs] + [t for _, t in edge_pairs]
        dst = [t for _, t in edge_pairs] + [s for s, _ in edge_pairs]
    else:
        src = [s for s, _ in edge_pairs]
        dst = [t for _, t in edge_pairs]

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node features using PCA encoding indexed by single letter AA
    try:
        x_mat = np.array([pca_encoding.loc[aa_map[a3]].values for a3, _ in nodes])
    except KeyError as e:
        raise KeyError(
            f"Amino acid not found in pca_encoding index or aa_map is missing a key: {e}"
        ) from e

    x = torch.tensor(x_mat, dtype=torch.float)

    # Graph label
    y = torch.tensor([label], dtype=torch.long)

    # Reconstruct sequence in positional order
    seq = "".join(pos_to_char[i] for i in sorted(pos_to_char))

    return Data(x=x, edge_index=edge_index, y=y, original_characters=seq)


def build_graph_from_edgelist(
    edgelist: list[tuple[str, str, str, str]],
    pca_encoding: pd.DataFrame,
    aa_map: dict[str, str],
    label: int,
) -> Data:
    """
    Build a PyTorch Geometric graph from a pre-parsed edgelist.

    Parameters
    ----------
    edgelist
        Edges as (a3, i_str, b3, j_str).
    pca_encoding
        DataFrame indexed by single letter amino acids with PCA feature columns.
    aa_map
        Mapping from three letter codes to single letter amino acids.
    label
        Integer class label.

    Returns
    -------
    Data
        A graph object with node features from pca_encoding and reconstructed sequence.
    """
    nodes, _node_map, pos_to_char, edge_pairs = _index_nodes_and_edges(edgelist, aa_map)
    return _assemble_graph(nodes, edge_pairs, pca_encoding, aa_map, pos_to_char, label)
