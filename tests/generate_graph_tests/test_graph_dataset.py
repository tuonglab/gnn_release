# tests/test_multigraph_dataset.py
import io
import os
import tarfile
from pathlib import Path

import pytest
import torch

from tcrgnn.graph_gen.graph_dataset import CANCEROUS, CONTROL, MultiGraphDataset


@pytest.fixture
def aa_map():
    # minimal believable 3-letter to 1-letter mapping
    return {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F"}


@pytest.fixture
def pca_path():
    # Point to your real PCA encoding file
    # Example for running locally:
    #   export TCRGNN_PCA_PATH=/path/to/pca_encoding.pt
    p = os.environ.get("TCRGNN_PCA_PATH")
    assert p, "Set TCRGNN_PCA_PATH to the real PCA encoding file"
    path = Path(p)
    assert path.exists(), f"PCA file not found at {path}"
    return path


def make_edge_txt(dir_path: Path, name: str, content: str = "u v\nv w\n") -> Path:
    f = dir_path / name
    f.write_text(content)
    return f


def make_tar_gz_with_txts(dir_path: Path, name: str, inner_files: list[str]) -> Path:
    tar_gz = dir_path / name
    memfile = io.BytesIO()
    with tarfile.open(fileobj=memfile, mode="w:gz") as tar:
        for fname in inner_files:
            data = f"u v\n{fname} x\n"
            info = tarfile.TarInfo(name=fname)
            encoded = data.encode("utf-8")
            info.size = len(encoded)
            tar.addfile(info, io.BytesIO(encoded))
    tar_gz.write_bytes(memfile.getvalue())
    return tar_gz


def test_raw_and_processed_file_names_normalization(
    tmp_path: Path, aa_map, pca_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    txt_a = make_edge_txt(raw_dir, "edges_a.txt")
    txt_b = make_edge_txt(raw_dir, "edges_b.txt")
    tar = make_tar_gz_with_txts(raw_dir, "batch.tar.gz", ["g1.txt", "g2.txt"])

    # put files under root/raw because PyG Dataset expects that
    root = tmp_path
    for p in [txt_a, txt_b, tar]:
        target = root / "raw" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    # stub non essential heavy parts of process
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.build_graph_from_edgelist",
        lambda *a, **k: torch.tensor([1]),
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.parse_edges",
        lambda p: [("u", "v"), ("v", "w")],
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.list_edge_txts",
        lambda extracted: [extracted / "g1.txt", extracted / "g2.txt"],
    )

    def _extract(_tar_path, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "g1.txt").write_text("a b\n")
        (outdir / "g2.txt").write_text("b c\n")
        return outdir

    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.safe_extract_tar_gz", _extract)
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.tmp_root", lambda: tmp_path / "tmp"
    )
    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.cleanup", lambda _: None)

    ds = MultiGraphDataset(
        root=str(root),
        samples=[txt_a, txt_b, tar],
        pca_path=str(pca_path),
        aa_map=aa_map,
        cancer=False,
    )

    # raw names are basenames only
    assert ds.raw_file_names == ["edges_a.txt", "edges_b.txt", "batch.tar.gz"]

    # processed names strip .tar.gz correctly
    assert ds.processed_file_names == ["edges_a.pt", "edges_b.pt", "batch.pt"]


def test_process_single_txt_and_len_get(tmp_path: Path, aa_map, pca_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    txt = make_edge_txt(raw_dir, "one.txt", "u v\nv w\n")

    # relocate into root/raw
    root = tmp_path
    (root / "raw" / txt.name).parent.mkdir(parents=True, exist_ok=True)
    (root / "raw" / txt.name).write_text(txt.read_text())

    calls = {"labels": [], "paths": []}

    def fake_build_graph(edges, pca, aa_map_local, label):
        calls["labels"].append(label)
        calls["paths"].append(len(edges))
        return {"edges": edges, "label": label}

    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.build_graph_from_edgelist", fake_build_graph
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.parse_edges", lambda p: [("u", "v"), ("v", "w")]
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.tmp_root", lambda: tmp_path / "tmp"
    )
    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.cleanup", lambda _: None)

    ds = MultiGraphDataset(
        root=str(root),
        samples=[txt],
        pca_path=str(pca_path),
        aa_map=aa_map,
        cancer=True,
    )

    assert ds.len() == 1
    objs = ds.get(0)
    assert isinstance(objs, list)
    assert objs and objs[0]["label"] == CANCEROUS
    assert calls["labels"] == [CANCEROUS]
    assert calls["paths"] == [2]


def test_process_tar_gz_multiple_graphs(tmp_path: Path, aa_map, pca_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    tar = make_tar_gz_with_txts(raw_dir, "batch.tar.gz", ["g1.txt", "g2.txt", "g3.txt"])

    # move into root/raw for PyG Dataset
    root = tmp_path
    (root / "raw" / tar.name).parent.mkdir(parents=True, exist_ok=True)
    (root / "raw" / tar.name).write_bytes(tar.read_bytes())

    def _extract(_tar, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        for name in ["g1.txt", "g2.txt", "g3.txt"]:
            (outdir / name).write_text("a b\n")
        return outdir

    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.safe_extract_tar_gz", _extract)
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.list_edge_txts",
        lambda extracted: sorted(extracted.glob("*.txt")),
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.parse_edges", lambda p: [("a", "b")]
    )
    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.tmp_root", lambda: tmp_path / "tmp"
    )
    monkeypatch.setattr("tcrgnn.graph_gen.graph_dataset.cleanup", lambda _: None)

    built = []

    def fake_build_graph(edges, pca, aa_map_local, label):
        built.append({"n_edges": len(edges), "label": label})
        return {"edges": edges, "label": label}

    monkeypatch.setattr(
        "tcrgnn.graph_gen.graph_dataset.build_graph_from_edgelist", fake_build_graph
    )

    ds = MultiGraphDataset(
        root=str(root),
        samples=[tar],
        pca_path=str(pca_path),
        aa_map=aa_map,
        cancer=False,
    )

    assert ds.len() == 1
    objs = ds.get(0)
    assert isinstance(objs, list) and len(objs) == 3
    assert all(obj["label"] == CONTROL for obj in objs)
    assert [b["n_edges"] for b in built] == [1, 1, 1]
