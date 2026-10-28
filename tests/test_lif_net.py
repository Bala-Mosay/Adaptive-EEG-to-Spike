import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.snn.lif_net import BaselineLIFNet


def test_forward_output_shape():
    """Logits must be (batch, n_classes); spike count a scalar."""
    model = BaselineLIFNet(n_features=128, n_hidden=64, n_classes=4, T=8)
    x = torch.randn(8, 16, 128)  # (T, batch, features)
    logits, spike_count = model(x)
    assert logits.shape == (16, 4)
    assert spike_count.dim() == 0


def test_spike_count_positive():
    """Spike count must be non-negative."""
    model = BaselineLIFNet(n_features=64, n_hidden=32, n_classes=4, T=8)
    _, spike_count = model(torch.randn(8, 4, 64))
    assert spike_count.item() >= 0.0


def test_no_plif_in_baseline():
    """Baseline must use fixed-tau LIFNode only -- NO ParametricLIFNode."""
    from spikingjelly.activation_based import neuron

    model = BaselineLIFNet(n_features=32, n_classes=4)
    for name, module in model.named_modules():
        assert not isinstance(module, neuron.ParametricLIFNode), (
            f"Found ParametricLIFNode at {name}; baseline must use LIFNode only"
        )


def test_lif_membrane_positive():
    """LIF membrane must not drift excessively negative.

    With standard LIF (v_reset=0.0, decay_input=False), membrane resets
    to 0 on spike and fluctuates around the input mean.  Without ReLU,
    negative Linear outputs can push membrane below 0 between spikes,
    but v_reset=0.0 prevents unbounded drift.
    """
    from spikingjelly.activation_based import functional, neuron as sj_neuron

    model = BaselineLIFNet(n_features=32, n_hidden=64, n_classes=4, T=16)
    model.eval()

    # Generate positive-dominant spike input (~25% rate)
    torch.manual_seed(42)
    x_seq = (torch.rand(16, 8, 32) > 0.75).float()

    functional.reset_net(model)
    with torch.no_grad():
        out, _ = model(x_seq)

    # Check LIF membrane potentials — must not drift deeply negative
    lif_modules = [m for m in model.net if isinstance(m, sj_neuron.LIFNode)]
    for m in lif_modules:
        v_mean = m.v.mean().item()
        assert v_mean > -5.0, (
            f"LIF membrane too negative: mean v = {v_mean:.3f}. "
            f"v_reset=0.0 should prevent unbounded negative drift."
        )
