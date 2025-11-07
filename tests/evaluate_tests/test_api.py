from unittest import mock

import pytest
import torch
from torch_geometric.data import Data

from tcrgnn.evaluate import api


@pytest.fixture
def sample_graphs():
    return [Data(x=torch.randn(3, 5)), Data(x=torch.randn(2, 5))]


def test_evaluate_model_calls_dependencies(monkeypatch, sample_graphs):
    records = {}

    class DummyGAT:
        def __init__(self, nfeat, nhid, nclass, dropout):
            records["base_model_args"] = {
                "nfeat": nfeat,
                "nhid": nhid,
                "nclass": nclass,
                "dropout": dropout,
            }

    fake_model = mock.Mock()

    def fake_get_device():
        records["get_device_calls"] = records.get("get_device_calls", 0) + 1
        return "cuda:0"

    def fake_load_trained_model(base_model, model_file, device):
        records["load_args"] = (base_model, model_file, device)
        return fake_model

    def fake_predict_on_graph_list(model, graphs, device):
        records["predict_args"] = (model, graphs, device)
        return ["score-1", "score-2"]

    monkeypatch.setattr(api, "GATv2", DummyGAT)
    monkeypatch.setattr(api, "get_device", fake_get_device)
    monkeypatch.setattr(api, "load_trained_model", fake_load_trained_model)
    monkeypatch.setattr(api, "predict_on_graph_list", fake_predict_on_graph_list)

    result = api.evaluate_model("checkpoint.pt", sample_graphs)

    assert result == ["score-1", "score-2"]
    assert records["base_model_args"] == {
        "nfeat": 5,
        "nhid": 375,
        "nclass": 2,
        "dropout": 0.17,
    }
    assert records["get_device_calls"] == 1
    assert records["load_args"][1:] == ("checkpoint.pt", "cuda:0")
    assert records["predict_args"][0] is fake_model
    assert records["predict_args"][1] == sample_graphs
    assert records["predict_args"][2] == "cuda:0"
    fake_model.eval.assert_called_once()


def test_evaluate_model_respects_explicit_device(monkeypatch, sample_graphs):
    monkeypatch.setattr(api, "GATv2", lambda **_: mock.Mock())
    monkeypatch.setattr(
        api, "load_trained_model", lambda base_model, model_file, device: mock.Mock()
    )
    monkeypatch.setattr(
        api, "predict_on_graph_list", lambda model, graphs, device: ["ok"]
    )
    monkeypatch.setattr(
        api, "get_device", mock.Mock(side_effect=AssertionError("should not call"))
    )

    result = api.evaluate_model("checkpoint.pt", sample_graphs, device="cpu")
    assert result == ["ok"]


def test_evaluate_model_requires_list(sample_graphs):
    with pytest.raises(TypeError, match="Expected a list of Data objects"):
        api.evaluate_model("checkpoint.pt", sample_graphs[0])


def test_evaluate_model_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        api.evaluate_model("checkpoint.pt", [])


def test_evaluate_model_requires_data_instances(sample_graphs):
    with pytest.raises(TypeError, match="PyG Data objects"):
        api.evaluate_model("checkpoint.pt", [sample_graphs[0], object()])
