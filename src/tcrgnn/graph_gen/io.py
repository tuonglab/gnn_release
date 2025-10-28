from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch_geometric.data import Data
import pandas as pd

CANCEROUS = 1
CONTROL = 0

def parse_edges(edge_file: Path) -> list[list[str]]:
    with edge_file.open() as f:
        return [line.strip().split() for line in f if line.strip()]

def build_graph_from_edge_txt(
    edge_file: Path,
    pca_encoding: pd.DataFrame,
    aa_map: dict[str, str],
    label: int = CANCEROUS,
) -> Data:
    edgelist = parse_edges(edge_file)

    nodes: list[tuple[str, str]] = []
    node_map: dict[tuple[str, str], int] = {}
    pos_to_char: dict[int, str] = {}

    for a3, i_str, b3, j_str in edgelist:
        for node in [(a3, i_str), (b3, j_str)]:
            if node not in node_map:
                node_map[node] = len(nodes)
                nodes.append(node)
                pos_to_char[int(node[1])] = aa_map[node[0]]

    seq = "".join(pos_to_char[i] for i in sorted(pos_to_char))

    src = [node_map[(a3, i)] for a3, i, _, _ in edgelist]
    dst = [node_map[(b3, j)] for _, _, b3, j in edgelist]
    # undirected
    src += dst
    dst += src[:len(dst)]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(
        np.array([pca_encoding.loc[aa_map[a3]].values for a3, _ in nodes]),
        dtype=torch.float,
    )
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, original_characters=seq)
    return data
