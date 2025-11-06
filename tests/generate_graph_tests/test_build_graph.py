import numpy as np
import pandas as pd
import torch

from tcrgnn.graph_gen._build_graph import (
    _assemble_graph,
    _index_nodes_and_edges,
    build_graph_from_edgelist,
)


def make_pca_and_map():
    # Minimal PCA-like encoding for four amino acids
    pca = pd.DataFrame(
        {
            "pc1": [0.1, 0.2, 0.3, 0.4],
            "pc2": [1.1, 1.2, 1.3, 1.4],
        },
        index=["A", "C", "G", "T"],
    )
    aa_map = {"ALA": "A", "CYS": "C", "GLY": "G", "THR": "T"}
    return pca, aa_map


def test_index_nodes_and_edges_basic_and_stability():
    pca, aa_map = make_pca_and_map()
    # Includes a duplicate edge and repeated nodes to test stability and de-duplication
    edgelist = [
        ("ALA", "0", "CYS", "2"),
        ("CYS", "2", "GLY", "1"),
        ("ALA", "0", "CYS", "2"),  # duplicate
        (
            "GLY",
            "1",
            "THR",
            "03",
        ),  # position string with leading zero to test int() parsing
    ]

    nodes, node_map, pos_to_char, edge_pairs = _index_nodes_and_edges(edgelist)

    # Node order must follow first appearance
    assert nodes == [("ALA", "0"), ("CYS", "2"), ("GLY", "1"), ("THR", "03")]
    assert node_map == {
        ("ALA", "0"): 0,
        ("CYS", "2"): 1,
        ("GLY", "1"): 2,
        ("THR", "03"): 3,
    }

    # Positions are coerced to int keys and map to single letter characters
    # positions: 0 -> A, 2 -> C, 1 -> G, 3 -> T
    assert pos_to_char == {0: "A", 2: "C", 1: "G", 3: "T"}

    # Directed edge indices reference node_map and include the duplicate edge
    # Expected pairs:
    #   (ALA0 -> CYS2) -> (0 -> 1)
    #   (CYS2 -> GLY1) -> (1 -> 2)
    #   duplicate       -> (0 -> 1)
    #   (GLY1 -> THR03) -> (2 -> 3)
    assert edge_pairs == [(0, 1), (1, 2), (0, 1), (2, 3)]


def test_assemble_graph_undirected_true():
    pca, aa_map = make_pca_and_map()

    nodes = [("ALA", "0"), ("CYS", "2"), ("GLY", "1")]
    pos_to_char = {0: "A", 2: "C", 1: "G"}
    edge_pairs = [(0, 1), (1, 2)]  # ALA0->CYS2, CYS2->GLY1

    data = _assemble_graph(
        nodes=nodes,
        edge_pairs=edge_pairs,
        pca_encoding=pca,
        pos_to_char=pos_to_char,
        label=7,
        undirected=True,
    )

    # Undirected duplicates edges in reverse direction
    # Forward: 0->1, 1->2
    # Reverse: 1->0, 2->1
    expected_src = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    expected_dst = torch.tensor([1, 2, 0, 1], dtype=torch.long)
    assert torch.equal(
        data.edge_index, torch.stack([expected_src, expected_dst], dim=0)
    )

    # Node features pulled in node order using single letter index
    expected_x = np.array(
        [
            pca.loc["A"].values,  # ALA
            pca.loc["C"].values,  # CYS
            pca.loc["G"].values,  # GLY
        ],
        dtype=float,
    )
    assert np.allclose(data.x.numpy(), expected_x)

    # Label is a 1D tensor with single class id
    assert torch.equal(data.y, torch.tensor([7], dtype=torch.long))

    # Sequence reconstruction uses sorted positional ints: positions 0,1,2 -> A, G, C -> "AGC"
    assert data.original_characters == "AGC"


def test_assemble_graph_directed_false():
    pca, aa_map = make_pca_and_map()

    nodes = [("ALA", "0"), ("CYS", "2"), ("GLY", "1")]
    pos_to_char = {0: "A", 2: "C", 1: "G"}
    edge_pairs = [(0, 1), (1, 2)]

    data = _assemble_graph(
        nodes=nodes,
        edge_pairs=edge_pairs,
        pca_encoding=pca,
        pos_to_char=pos_to_char,
        label=3,
        undirected=False,  # exercise the else branch
    )

    expected_src = torch.tensor([0, 1], dtype=torch.long)
    expected_dst = torch.tensor([1, 2], dtype=torch.long)
    assert torch.equal(
        data.edge_index, torch.stack([expected_src, expected_dst], dim=0)
    )
    assert torch.equal(data.y, torch.tensor([3], dtype=torch.long))
    assert data.original_characters == "AGC"


def test_build_graph_from_edgelist_integration_matches_manual_undirected():
    pca, aa_map = make_pca_and_map()
    edgelist = [
        ("ALA", "0", "CYS", "2"),
        ("CYS", "2", "GLY", "1"),
    ]

    # Run the public builder
    built = build_graph_from_edgelist(
        edgelist=edgelist,
        pca_encoding=pca,
        label=99,
    )

    # Recreate expected intermediate results to compare
    nodes, _, pos_to_char, edge_pairs = _index_nodes_and_edges(edgelist)
    manual = _assemble_graph(
        nodes=nodes,
        edge_pairs=edge_pairs,
        pca_encoding=pca,
        pos_to_char=pos_to_char,
        label=99,
        undirected=True,  # default path exercised by build_graph_from_edgelist
    )

    # Compare tensors and fields
    assert torch.equal(built.edge_index, manual.edge_index)
    assert np.allclose(built.x.numpy(), manual.x.numpy())
    assert torch.equal(built.y, manual.y)
    assert built.original_characters == manual.original_characters
