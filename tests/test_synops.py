import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from spikingjelly.activation_based import layer

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


def test_synops_analytic_all_spiking():
    """Hand-computed SynOps must include every neuron and every timestep.

    With all-positive weights, a zero threshold, and unit input, every LIF
    neuron spikes at every timestep.  For n_hidden=4, n_classes=2, T=4, and
    3 trials:

    - LIF 1: 4 neurons x 4 timesteps x 3 trials = 48 spikes, fan-out 2
    - LIF 2: 2 neurons x 4 timesteps x 3 trials = 24 spikes, fan-out 2
    - total = 2*48 + 2*24 = 144, per trial = 48
    """
    T = 4
    n_trials = 3
    model = BaselineLIFNet(n_features=2, n_hidden=4, n_classes=2, T=T)
    for module in model.net:
        if isinstance(module, layer.Linear):
            torch.nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    X = np.ones((n_trials, 2), dtype=np.float32)
    result = count_synops(model, X, T=T, device="cpu", threshold=0.0)

    assert result["total_synops"] == 144.0
    assert result["synops_per_trial"] == 48.0
    assert result["spike_density"] == 1.0
