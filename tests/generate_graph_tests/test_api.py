# test_edge_dir_generators.py
from pathlib import Path

import pandas as pd
import pytest
from torch_geometric.data import Data

from tcrgnn.graph_gen.api import (
    generate_graph_from_edge_file,
    generate_graphs_from_edge_dir,
)


@pytest.fixture
def dummy_args():
    # Minimal placeholders for pass through testing
    pca = pd.DataFrame({"pc1": [0.1], "pc2": [0.2]}, index=["A"])
    aa_map = {"ALA": "A"}
    label = 42
    return pca, aa_map, label


def test_generate_graphs_from_edge_dir_calls_dependencies_in_sorted_order(
    monkeypatch, tmp_path, dummy_args
):
    pca, aa_map, label = dummy_args

    # Create some fake files in an unsorted order
    files = [tmp_path / "b.txt", tmp_path / "a.txt", tmp_path / "c.txt"]
    for fp in files:
        fp.write_text("dummy")

    # Stub list_edge_txts to return the unsorted list
    call_order_parse = []

    def fake_list_edge_txts(edge_dir: Path):
        assert edge_dir == tmp_path
        return [files[0], files[1], files[2]]  # b, a, c

    def fake_parse_edges(path: Path):
        # Record call order to verify sorting happened before parsing
        call_order_parse.append(path.name)
        # Return a unique edgelist per file so we can trace it
        return [(path.stem.upper(), "0", path.stem.upper(), "1")]

    build_calls = []

    def fake_build_graph_from_edgelist(edgelist, *, pca_encoding, aa_map, label):
        # Record that args were passed through correctly
        build_calls.append(
            {
                "edgelist": tuple(edgelist),
                "pca_encoding_id": id(pca_encoding),
                "aa_map_id": id(aa_map),
                "label": label,
            }
        )
        # Return a distinct Data object for each call
        return Data(name="graph_" + edgelist[0][0])

    # Patch the internal imports used by the module under test
    import tcrgnn.graph_gen._io  # noqa: F401  - update if needed

    # Patch where the functions are looked up by the module under test
    monkeypatch.setattr("tcrgnn.graph_gen.api.list_edge_txts", fake_list_edge_txts)
    monkeypatch.setattr("tcrgnn.graph_gen.api.parse_edges", fake_parse_edges)
    monkeypatch.setattr(
        "tcrgnn.graph_gen.api.build_graph_from_edgelist", fake_build_graph_from_edgelist
    )

    # Run
    out = generate_graphs_from_edge_dir(tmp_path, pca, aa_map, label)

    # Verify we got one Data per file
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(g, Data) for g in out)

    # Sorting should make parse_edges run on a.txt, b.txt, c.txt in that order
    assert call_order_parse == ["a.txt", "b.txt", "c.txt"]

    # Check pass through of args and association with each file
    # Because we made edgelist depend on filename, we can match them
    expected_names = ["A", "B", "C"]  # stems uppercased
    assert [g.name for g in out] == [f"graph_{n}" for n in expected_names]

    # Every build call should have received the exact same pca, aa_map by identity
    assert all(call["pca_encoding_id"] == id(pca) for call in build_calls)
    assert all(call["aa_map_id"] == id(aa_map) for call in build_calls)
    assert all(call["label"] == label for call in build_calls)


def test_generate_graph_from_edge_file_calls_parse_then_builder(monkeypatch, tmp_path):
    # Create a single fake edge file
    edge_file = tmp_path / "edges.txt"
    edge_file.write_text("ALA 0 ALA 1\n")

    # Stubs that record how they were called
    calls = {"parse": None, "build": None}

    def fake_parse_edges(path: Path):
        calls["parse"] = path
        # Return a sentinel edgelist
        return [("ALA", "0", "ALA", "1")]

    def fake_build_graph_from_edgelist(*args, **kwargs):
        # Record exactly what was forwarded
        calls["build"] = {"args": args, "kwargs": kwargs}
        # Return a simple Data object so we can assert on the return value
        return Data(tag="sentinel")

    # IMPORTANT: patch where the functions are *looked up* (the api module),
    # not where they are defined elsewhere.
    monkeypatch.setattr("tcrgnn.graph_gen.api.parse_edges", fake_parse_edges)
    monkeypatch.setattr(
        "tcrgnn.graph_gen.api.build_graph_from_edgelist",
        fake_build_graph_from_edgelist,
    )

    # Inputs that should be forwarded to the builder
    pca_df = pd.DataFrame()
    aa_map = {}
    label = 7

    # Run
    out = generate_graph_from_edge_file(
        edge_file, pca_encoding=pca_df, aa_map=aa_map, label=label
    )

    # It should return whatever the builder returned
    assert isinstance(out, Data)
    assert out.tag == "sentinel"

    # parse_edges must be called with the file path
    assert calls["parse"] == edge_file

    # The builder should get the edgelist as the ONLY positional arg
    assert calls["build"]["args"] == ([("ALA", "0", "ALA", "1")],)

    # And the extra params should be forwarded as kwargs, unchanged
    kwargs = calls["build"]["kwargs"]
    assert kwargs["pca_encoding"] is pca_df
    assert kwargs["aa_map"] is aa_map
    assert kwargs["label"] == label
