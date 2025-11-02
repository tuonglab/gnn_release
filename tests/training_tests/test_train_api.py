# test_train_model.py
import torch
from torch_geometric.data import Data

from tcrgnn.training.config import TrainConfig, TrainPaths
from tcrgnn.training.train_api import train_model


def test_train_model_calls_pipeline_with_expected_args(monkeypatch, tmp_path, capsys):
    # Create a tiny fake dataset: two graphs in two lists
    g1 = Data(x=torch.randn(3, 7))  # num_node_features == 7
    g2 = Data(x=torch.randn(4, 7))
    fake_train_set = [[g1], [g2]]

    # Stub load_train_data -> returns our fake set
    def fake_load_train_data(cancer_dirs, control_dirs):
        # Verify dirs are passed in as given
        assert isinstance(cancer_dirs, list)
        assert isinstance(control_dirs, list)
        return fake_train_set

    # Capture GATv2 init args and the instance
    ctor_calls = {}

    class FakeGATv2:
        def __init__(self, nfeat, nhid, nclass, dropout):
            ctor_calls["nfeat"] = nfeat
            ctor_calls["nhid"] = nhid
            ctor_calls["nclass"] = nclass
            ctor_calls["dropout"] = dropout
            self._to_device = None

        def to(self, device):
            self._to_device = device
            return self

    # Stub train to record its arguments
    train_calls = {}

    def fake_train(model, train_set, num_epochs, cfg, device, save_path):
        train_calls["model"] = model
        train_calls["train_set"] = train_set
        train_calls["num_epochs"] = num_epochs
        train_calls["cfg"] = cfg
        train_calls["device"] = device
        train_calls["save_path"] = save_path

    # Force CPU path
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Patch symbols where they are USED
    monkeypatch.setattr(
        "tcrgnn.training.train_api.load_train_data", fake_load_train_data
    )
    monkeypatch.setattr("tcrgnn.training.train_api.GATv2", FakeGATv2)
    monkeypatch.setattr("tcrgnn.training.train_api.train", fake_train)

    # Prepare config and save path
    cfg = TrainConfig(epochs=3, patience=1)  # values do not affect test except identity
    save_path = TrainPaths(model_dir=tmp_path / "models", best_name="best.pt")

    # Call
    ret = train_model(
        cancer_dirs=[str(tmp_path / "cancerA")],
        control_dirs=[str(tmp_path / "controlB")],
        cfg=cfg,
        save_path=save_path,
    )

    # Return is None
    assert ret is None

    # GATv2 should have used num_node_features from first graph in first list
    assert ctor_calls == {"nfeat": 7, "nhid": 375, "nclass": 2, "dropout": 0.17}

    # Device should be CPU based on our patch
    assert isinstance(train_calls["device"], torch.device)
    assert str(train_calls["device"]) == "cpu"

    # Model.to should have been called with the same device
    assert isinstance(train_calls["model"], FakeGATv2)
    assert str(train_calls["model"].__dict__["_to_device"]) == "cpu"

    # Train should receive our exact train set object and expected args
    assert train_calls["train_set"] is fake_train_set
    assert train_calls["num_epochs"] == cfg.epochs
    assert train_calls["cfg"] is cfg
    assert train_calls["save_path"] is save_path

    # It should print a completion message mentioning the model_dir
    out = capsys.readouterr().out
    assert "Training complete. Model saved to" in out
    assert str(save_path.model_dir) in out
