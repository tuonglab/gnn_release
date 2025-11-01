from unittest import mock

import torch
from torch_geometric.data import Data

from tcrgnn.evaluate import utils


def test_get_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = utils.get_device()
    assert device.type == "cuda"


def test_get_device_returns_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = utils.get_device()
    assert device.type == "cpu"


def test_load_trained_model_loads_state_and_sets_eval(monkeypatch):
    captured = {}

    def fake_load(path, map_location=None):
        captured["path"] = path
        captured["map_location"] = map_location
        return {"weights": torch.tensor([1.0])}

    monkeypatch.setattr(torch, "load", fake_load)
    device = torch.device("cpu")

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.tensor([0.0]))

        def forward(self, x):
            return x

    model = DummyModel()
    model.load_state_dict = mock.Mock()
    model.to = mock.Mock(return_value=model)
    model.eval = mock.Mock(return_value=model)

    result = utils.load_trained_model(model, "dummy.pt", device)

    assert result is model
    assert captured == {"path": "dummy.pt", "map_location": device}
    model.load_state_dict.assert_called_once_with({"weights": torch.tensor([1.0])})
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once()


def test_move_graph_to_device_uses_graph_to_method():
    class DummyGraph:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    graph = DummyGraph()
    device = torch.device("cpu")
    moved = utils.move_graph_to_device(graph, device)
    assert moved is graph
    assert graph.moved_to == device


def test_make_loader_produces_ordered_batches():
    data_items = [
        Data(x=torch.tensor([[0.0]])),
        Data(x=torch.tensor([[1.0]])),
    ]
    loader = utils.make_loader(data_items)
    batches = list(loader)
    assert loader.batch_size == 1
    assert len(batches) == 2
    assert batches[0].x.flatten().tolist() == [0.0]
    assert batches[1].x.flatten().tolist() == [1.0]


def test_write_scores_to_txt_without_sequences(tmp_path):
    path = tmp_path / "scores.txt"
    utils.write_scores_to_txt([0.1, 0.2], str(path))
    assert path.read_text().splitlines() == ["0.1", "0.2"]


def test_write_scores_to_txt_with_sequence_mismatch(tmp_path, capsys):
    path = tmp_path / "scores_with_seq.txt"
    utils.write_scores_to_txt([0.5], str(path), ["AAA", "BBB"])
    captured = capsys.readouterr().out.strip()
    assert "Warning: Length of sequences does not match length of scores." in captured
    assert path.read_text().splitlines() == ["AAA,0.5", "BBB,N/A"]
