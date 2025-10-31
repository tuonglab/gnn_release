# test_data_loading.py
import os
from pathlib import Path

import torch

from tcrgnn.training.data_loading import load_train_data


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
