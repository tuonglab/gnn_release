# test_data_loading.py
import os
from pathlib import Path

import pytest
import torch

from tcrgnn.training.data_loading import load_train_data

try:
    from torch_geometric.data import Data

    HAS_PYG = True
except Exception:
    HAS_PYG = False

from tcrgnn.training.data_loading import load_graphs


@pytest.mark.parametrize("use_path_obj", [False, True])
def test_load_graphs_roundtrip(tmp_path, use_path_obj):
    if HAS_PYG:
        ds = [
            Data(
                x=torch.randn(4, 3),
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                y=torch.tensor([1]),
            ),
            Data(
                x=torch.randn(2, 3),
                edge_index=torch.tensor([[0, 0], [1, 1]]),
                y=torch.tensor([0]),
            ),
        ]
    else:
        ds = {"a": torch.randn(3), "b": torch.tensor([1, 2, 3])}

    path = tmp_path / "graphs.pt"
    torch.save(ds, path)

    arg = path if use_path_obj else str(path)
    out = load_graphs(arg)  # default map_location is "cpu"

    assert type(out) is type(ds)
    if HAS_PYG and isinstance(ds, list):
        assert len(out) == len(ds)
        for go, gr in zip(out, ds, strict=False):
            assert torch.allclose(go.x, gr.x)
            assert torch.equal(go.edge_index, gr.edge_index)
            assert torch.equal(go.y, gr.y)
    else:
        assert out.keys() == ds.keys()
        for k in ds:
            assert torch.equal(out[k], ds[k])


def test_load_graphs_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        load_graphs(str(missing))


@pytest.mark.parametrize("ml", ["cpu", torch.device("cpu")])
def test_load_graphs_passes_map_location(monkeypatch, tmp_path, ml):
    obj = {"v": torch.tensor([1])}
    path = tmp_path / "graphs.pt"
    torch.save(obj, path)

    called = {}

    real_load = torch.load

    def spy_load(file, map_location=None):
        called["file"] = file
        called["map_location"] = map_location
        return real_load(file, map_location=map_location)

    monkeypatch.setattr(torch, "load", spy_load)

    out = load_graphs(path, map_location=ml)
    assert out["v"].equal(obj["v"])
    assert Path(called["file"]) == path
    # normalize to device string for assertion
    ml_str = ml if isinstance(ml, str) else ml.type
    called_str = (
        called["map_location"]
        if isinstance(called["map_location"], str)
        else getattr(called["map_location"], "type", None)
    )
    assert called_str == ml_str


def test_load_train_data_traverses_dirs_and_appends_graph_lists(monkeypatch, tmp_path):
    cancer_dir = tmp_path / "cancer"
    control_dir = tmp_path / "control"
    cancer_dir.mkdir()
    control_dir.mkdir()

    (cancer_dir / "a.pt").write_text("x")
    (cancer_dir / "b.pt").write_text("x")
    (control_dir / "c.pt").write_text("x")

    # Save original listdir before patch
    orig_listdir = os.listdir

    def fake_listdir(path):
        # deterministically sort names from the real filesystem
        return sorted(orig_listdir(path))

    def fake_load_graphs(file):
        name = Path(file).name
        if name == "a.pt":
            return [torch.tensor([1]), torch.tensor([2])]
        if name == "b.pt":
            return [torch.tensor([3])]
        if name == "c.pt":
            return [torch.tensor([4]), torch.tensor([5]), torch.tensor([6])]
        raise AssertionError(f"Unexpected file: {name}")

    monkeypatch.setattr(os, "listdir", fake_listdir)
    monkeypatch.setattr(os.path, "isdir", lambda d: Path(d).is_dir())
    monkeypatch.setattr("tcrgnn.training.data_loading.load_graphs", fake_load_graphs)

    out = load_train_data([str(cancer_dir)], [str(control_dir)])

    assert len(out) == 3
    assert [len(lst) for lst in out] == [2, 1, 3]
    assert torch.equal(out[0][0], torch.tensor([1]))
    assert torch.equal(out[1][0], torch.tensor([3]))
    assert torch.equal(out[2][-1], torch.tensor([6]))


def test_load_train_data_skips_non_dirs(monkeypatch, tmp_path):
    # One real dir and one non-dir
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "file.pt").write_text("x")

    not_a_dir = tmp_path / "not_a_dir.txt"
    not_a_dir.write_text("hello")

    calls = []

    def fake_load_graphs(file):
        calls.append(Path(file).name)
        return [torch.tensor([0])]

    # Real is dir, not_a_dir is not
    monkeypatch.setattr(os.path, "isdir", lambda d: Path(d).is_dir())
    monkeypatch.setattr("tcrgnn.training.data_loading.load_graphs", fake_load_graphs)

    out = load_train_data([str(real_dir), str(not_a_dir)], [str(not_a_dir)])

    # Only file in real_dir should have been processed once
    assert calls == ["file.pt"]
    assert len(out) == 1
    assert len(out[0]) == 1
