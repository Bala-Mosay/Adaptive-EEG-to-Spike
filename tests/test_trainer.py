import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.snn.lif_net import BaselineLIFNet
from src.snn.trainer import BaselineTrainer, TrainingConfig


def test_loss_decreases_on_toy_data():
    """Loss must decrease over a few epochs on separable toy data."""
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # Two easily separable classes
    X = np.concatenate([
        rng.standard_normal((50, 16)).astype(np.float32) + 2.0,
        rng.standard_normal((50, 16)).astype(np.float32) - 2.0,
    ])
    y = np.concatenate([np.zeros(50, dtype=np.int64), np.ones(50, dtype=np.int64)])

    model = BaselineLIFNet(n_features=16, n_hidden=32, n_classes=2, T=4)
    cfg = TrainingConfig(n_epochs=30, batch_size=25, learning_rate=0.01, T=4, random_state=42, device="cpu")
    trainer = BaselineTrainer(model, cfg)
    history = trainer.fit(X, y, verbose=False)

    first_loss = history[0]["ce_loss"]
    last_loss = history[-1]["ce_loss"]
    assert last_loss < first_loss, f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"


def test_evaluate_returns_predictions():
    """evaluate must return predictions array and spike_count."""
    model = BaselineLIFNet(n_features=8, n_hidden=16, n_classes=2, T=4)
    cfg = TrainingConfig(T=4, device="cpu", random_state=0)
    trainer = BaselineTrainer(model, cfg)
    X = np.random.randn(10, 8).astype(np.float32)
    y = np.zeros(10, dtype=np.int64)
    out = trainer.evaluate(X, y)
    assert "predictions" in out
    assert "total_spike_count" in out
    assert len(out["predictions"]) == 10
    assert out["total_spike_count"] >= 0.0


def test_trainer_deterministic():
    """Two identical runs with same seed must produce same loss history."""
    X = np.random.randn(40, 8).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    cfg = TrainingConfig(n_epochs=10, batch_size=20, T=4, random_state=99, device="cpu")
    torch.manual_seed(0)
    model1 = BaselineLIFNet(n_features=8, n_hidden=16, n_classes=2, T=4)
    h1 = BaselineTrainer(model1, cfg).fit(X, y, verbose=False)

    torch.manual_seed(0)
    model2 = BaselineLIFNet(n_features=8, n_hidden=16, n_classes=2, T=4)
    h2 = BaselineTrainer(model2, cfg).fit(X, y, verbose=False)

    losses1 = [r["ce_loss"] for r in h1]
    losses2 = [r["ce_loss"] for r in h2]
    np.testing.assert_array_almost_equal(losses1, losses2, decimal=4)
