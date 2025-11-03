# test_train_config.py
import pytest

from tcrgnn.training._config import TrainConfig, TrainPaths  # update import path


def test_trainpaths_best_path_creates_dir_and_returns_path(tmp_path):
    custom_dir = tmp_path / "models_subdir"
    tp = TrainPaths(model_dir=custom_dir, best_name="custom_best.pt")

    best = tp.best_path
    assert best == custom_dir / "custom_best.pt"
    assert best.parent.exists() and best.parent.is_dir()
    # Directory should be exactly what we set
    assert best.parent == custom_dir


def test_trainpaths_best_path_idempotent(tmp_path):
    tp = TrainPaths(model_dir=tmp_path / "model_here", best_name="best.pt")
    first = tp.best_path
    second = tp.best_path
    assert first == second
    assert first.parent.exists()


def test_trainconfig_defaults():
    cfg = TrainConfig()
    assert cfg.epochs == 100
    assert cfg.patience == 15
    assert cfg.lr == 5e-4
    assert cfg.weight_decay == 0.25
    assert cfg.min_delta_loss == 0.01
    assert cfg.min_delta_acc == 0.01
    assert cfg.batch_size == 256
    assert cfg.seed == 111


def test_trainconfig_custom_values():
    cfg = TrainConfig(
        epochs=5,
        patience=2,
        lr=1e-3,
        weight_decay=0.0,
        min_delta_loss=0.001,
        min_delta_acc=0.002,
        batch_size=32,
        seed=7,
    )
    assert (cfg.epochs, cfg.patience, cfg.batch_size, cfg.seed) == (5, 2, 32, 7)
    assert pytest.approx(cfg.lr) == 1e-3
    assert pytest.approx(cfg.weight_decay) == 0.0
    assert pytest.approx(cfg.min_delta_loss) == 0.001
    assert pytest.approx(cfg.min_delta_acc) == 0.002
