import os

import torch
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from tcrgnn.training._config import TrainConfig, TrainPaths
from tcrgnn.training._loop import train


# Graph-level classifier: pool by batch to get one embedding per graph
class DummyModel(torch.nn.Module):
    def __init__(self, in_dim=8, hid=16, out_dim=2):
        super().__init__()
        self.enc = torch.nn.Linear(in_dim, hid)
        self.cls = torch.nn.Linear(hid, out_dim)

    def forward(self, x, edge_index, batch):
        h = torch.relu(self.enc(x))
        g = global_mean_pool(h, batch)  # [num_graphs, hid]
        return self.cls(g)  # [num_graphs, 2]


def make_sample_graph(num_nodes=10, in_dim=8):
    x = torch.randn(num_nodes, in_dim)
    # simple 2-edge graph is fine for a smoke test
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # one label per graph
    y = torch.randint(0, 2, (1,), dtype=torch.long)
    # batch will be constructed by DataLoader when graphs are collated
    return Data(x=x, edge_index=edge_index, y=y)


def make_dataset(num_graphs=8):
    # your loop expects list[list[Data]]
    # keep inner lists of size 1 to match your current batching scheme
    return [[make_sample_graph()] for _ in range(num_graphs)]


def test_train_runs(tmp_path):
    model = DummyModel()
    samples = make_dataset(8)
    device = torch.device("cpu")

    cfg = TrainConfig(
        lr=1e-3,
        weight_decay=0.0,
        batch_size=2,
        epochs=3,
        patience=5,
        min_delta_loss=0.0,
        min_delta_acc=0.0,
    )

    save_path = TrainPaths(
        model_dir=str(tmp_path),
        best_name="best.pt",
    )

    train(model, samples, cfg, save_path, device)

    best_path = os.path.join(save_path.model_dir, save_path.best_name)
    assert os.path.exists(best_path)


def test_early_stopping(tmp_path):
    model = DummyModel()
    samples = make_dataset(4)
    device = torch.device("cpu")

    cfg = TrainConfig(
        lr=1e-3,
        weight_decay=0.0,
        batch_size=2,
        epochs=20,
        patience=1,  # trigger early stop quickly
        min_delta_loss=0.0,
        min_delta_acc=0.0,
    )

    save_path = TrainPaths(
        model_dir=str(tmp_path),
        best_name="best.pt",
    )

    train(model, samples, cfg, save_path, device)

    best_path = os.path.join(save_path.model_dir, save_path.best_name)
    assert os.path.exists(best_path)


def test_model_weights_change(tmp_path):
    model = DummyModel()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    samples = make_dataset(8)
    device = torch.device("cpu")

    cfg = TrainConfig(
        lr=1e-2,
        weight_decay=0.0,
        batch_size=2,
        epochs=2,
        patience=5,
        min_delta_loss=0.0,
        min_delta_acc=0.0,
    )

    save_path = TrainPaths(
        model_dir=str(tmp_path),
        best_name="best.pt",
    )

    train(model, samples, cfg, save_path, device)

    changed = any(not torch.equal(initial[k], model.state_dict()[k]) for k in initial)
    assert changed


class ConstantModel(torch.nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        # tiny param so the optimizer is valid
        self.bias = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x, edge_index, batch):
        g = global_mean_pool(x, batch)
        return self.bias.expand(g.size(0), -1)


def make_graph(n=4, d=8):
    x = torch.randn(n, d)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.randint(0, 2, (1,), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def test_early_stopping_branch(tmp_path, capsys, monkeypatch):
    model = ConstantModel()
    samples = make_dataset(4)
    device = torch.device("cpu")

    cfg = TrainConfig(
        lr=0.0,  # freeze updates so metrics never improve after epoch 1
        weight_decay=0.0,
        batch_size=2,
        epochs=50,  # large upper bound
        patience=1,  # stop after first non improvement epoch
        min_delta_loss=0.0,
        min_delta_acc=0.0,
    )

    save_path = TrainPaths(model_dir=tmp_path, best_name="best.pt")

    # capture checkpoint saves
    saves = []
    real_save = torch.save

    def fake_save(state, path):
        saves.append(path)
        real_save({}, path)

    monkeypatch.setattr(torch, "save", fake_save)

    train(model, samples, cfg, save_path, device)

    out = capsys.readouterr().out

    # early stop message was printed
    assert f"Early stopping after no improvement for {cfg.patience} epochs" in out

    # we should only see epoch 1 and epoch 2 logs, then stop
    assert "epoch=1 " in out
    assert "epoch=2 " in out
    assert "epoch=3 " not in out

    # only the first epoch should have saved a best checkpoint
    assert len(saves) == 1
    assert os.path.exists(save_path.best_path)
