import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.snn.lif_net import BaselineLIFNet
from src.snn.synops import count_synops


def test_synops_returns_dict():
    """count_synops must return a dict with the three required keys."""
    model = BaselineLIFNet(n_features=32, n_hidden=16, n_classes=4, T=4)
    X = np.random.randn(8, 32).astype(np.float32)
    result = count_synops(model, X, T=4)
    for key in ["total_synops", "synops_per_trial", "spike_density"]:
        assert key in result, f"Missing key: {key}"


def test_synops_nonnegative():
    """SynOps values must be non-negative."""
    model = BaselineLIFNet(n_features=32, n_hidden=16, n_classes=4, T=4)
    X = np.random.randn(8, 32).astype(np.float32)
    result = count_synops(model, X, T=4)
    assert result["total_synops"] >= 0
    assert result["spike_density"] >= 0
    assert result["synops_per_trial"] >= 0


def test_synops_per_trial_consistent():
    """synops_per_trial must equal total_synops / n_trials."""
    model = BaselineLIFNet(n_features=32, n_hidden=16, n_classes=4, T=4)
    X = np.random.randn(8, 32).astype(np.float32)
    result = count_synops(model, X, T=4, device="cpu")
    expected = result["total_synops"] / 8
    assert abs(result["synops_per_trial"] - expected) < 1e-3
