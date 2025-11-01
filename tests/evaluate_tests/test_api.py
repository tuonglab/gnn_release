from types import SimpleNamespace
from unittest.mock import MagicMock

from tcrgnn.evaluate.api import evaluate


class DummyLoaderWithLen:
    def __init__(self, batch):
        self._batch = batch

    def __iter__(self):
        return iter([self._batch])

    def __len__(self):
        return 1


class EmptyLoader:
    def __iter__(self):
        return iter([])


def test_evaluate_predicts_on_unwrapped_batch(monkeypatch):
    graph = SimpleNamespace(num_node_features=42)
    test_data = [[graph]]
    sample_graphs = ["graph_a", "graph_b"]
    sample_scores = [0.7, 0.3]

    mock_gat = MagicMock(return_value="base_model")
    mock_device = object()
    mock_loader = DummyLoaderWithLen((sample_graphs,))
    mock_trained_model = "trained_model"

    mock_get_device = MagicMock(return_value=mock_device)
    mock_load_trained = MagicMock(return_value=mock_trained_model)
    mock_make_loader = MagicMock(return_value=mock_loader)
    mock_predict = MagicMock(return_value=sample_scores)

    monkeypatch.setattr("tcrgnn.evaluate.api.GATv2", mock_gat)
    monkeypatch.setattr("tcrgnn.evaluate.api.get_device", mock_get_device)
    monkeypatch.setattr("tcrgnn.evaluate.api.load_trained_model", mock_load_trained)
    monkeypatch.setattr("tcrgnn.evaluate.api.make_loader", mock_make_loader)
    monkeypatch.setattr("tcrgnn.evaluate.api.predict_on_graph_list", mock_predict)

    result = evaluate("model.pt", test_data)

    assert result == sample_scores
    mock_gat.assert_called_once_with(nfeat=42, nhid=375, nclass=2, dropout=0.17)
    mock_load_trained.assert_called_once_with("base_model", "model.pt", mock_device)
    mock_make_loader.assert_called_once_with(test_data, batch_size=1, shuffle=False)
    mock_predict.assert_called_once_with(mock_trained_model, sample_graphs, mock_device)


def test_evaluate_returns_empty_results_when_no_batches(monkeypatch):
    graph = SimpleNamespace(num_node_features=10)
    test_data = [[graph]]

    monkeypatch.setattr(
        "tcrgnn.evaluate.api.GATv2", MagicMock(return_value="base_model")
    )
    monkeypatch.setattr(
        "tcrgnn.evaluate.api.get_device", MagicMock(return_value="device")
    )
    monkeypatch.setattr(
        "tcrgnn.evaluate.api.load_trained_model",
        MagicMock(return_value="trained_model"),
    )
    monkeypatch.setattr(
        "tcrgnn.evaluate.api.make_loader", MagicMock(return_value=EmptyLoader())
    )
    predict_mock = MagicMock()
    monkeypatch.setattr("tcrgnn.evaluate.api.predict_on_graph_list", predict_mock)

    result = evaluate("model.pt", test_data)

    assert result == ([], [])
    predict_mock.assert_not_called()
