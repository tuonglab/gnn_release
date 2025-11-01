from unittest.mock import Mock

import pytest
import torch

import tcrgnn.evaluate.predict as predict_module


class DummyGraph:
    def __init__(self, x=None, edge_index=None, batch=None):
        self.x = x
        self.edge_index = edge_index
        if batch is not None:
            self.batch = batch


def test_predict_on_graph_list_single_output(monkeypatch):
    calls = []

    def fake_move(graph, dev):
        calls.append((graph, dev))
        return graph

    monkeypatch.setattr(predict_module, "move_graph_to_device", fake_move)

    graph1 = DummyGraph(torch.zeros((2, 3)))
    graph2 = DummyGraph(torch.ones((1, 3)))
    model = Mock(
        side_effect=[
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0]),
        ]
    )
    device = torch.device("cpu")

    result = predict_module.predict_on_graph_list(model, [graph1, graph2], device)

    assert len(calls) == 2
    assert [g for g, _ in calls] == [graph1, graph2]
    assert all(dev is device for _, dev in calls)
    assert result == pytest.approx(
        [0.5, torch.sigmoid(torch.tensor(1.0)).item(), 0.5], rel=1e-6
    )
    first_args = model.call_args_list[0].args
    second_args = model.call_args_list[1].args
    assert first_args == (graph1.x, graph1.edge_index, getattr(graph1, "batch", None))
    assert second_args == (graph2.x, graph2.edge_index, getattr(graph2, "batch", None))


def test_predict_on_graph_list_selects_positive_class(monkeypatch):
    monkeypatch.setattr(predict_module, "move_graph_to_device", lambda graph, _: graph)

    graph = DummyGraph(torch.zeros((2, 4)))
    model = Mock(return_value=torch.tensor([[0.0, 1.0], [1.0, -1.0]]))
    device = torch.device("cpu")

    result = predict_module.predict_on_graph_list(model, [graph], device)

    expected = [
        torch.sigmoid(torch.tensor(1.0)).item(),
        torch.sigmoid(torch.tensor(-1.0)).item(),
    ]
    assert result == pytest.approx(expected, rel=1e-6)


def test_predict_on_graph_list_missing_features_raises(monkeypatch):
    monkeypatch.setattr(predict_module, "move_graph_to_device", lambda graph, _: graph)

    class NoFeatureGraph:
        def __init__(self):
            self.edge_index = None

    model = Mock()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="missing node features"):
        predict_module.predict_on_graph_list(model, [NoFeatureGraph()], device)


def test_predict_on_graph_list_invalid_feature_dim_raises(monkeypatch):
    monkeypatch.setattr(predict_module, "move_graph_to_device", lambda graph, _: graph)

    graph = DummyGraph(torch.zeros(3))
    model = Mock()
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Input features must be 2-dimensional"):
        predict_module.predict_on_graph_list(model, [graph], device)
