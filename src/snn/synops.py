"""Synaptic Operations (SynOps) counter — standardised energy proxy.

SynOps is the mathematically validated standard for SNN energy-proxy
estimation (Eq. 2 of the research paper; Merolla et al. 2014 [24] and
Rueckauer et al. 2017 [25])::

    SynOps = sum_l ( fanout_l * total_spikes_l )

where ``fanout_l`` is the number of output connections of the spiking layer
l (i.e. the neuron count of the NEXT linear layer) and ``total_spikes_l`` is
the number of spikes emitted by layer l over all neurons, trials, and
timesteps.

The earlier implementation averaged over neurons *and* trials and then
divided by the trial count again, which undercounted SynOps by roughly
``n_neurons * n_trials``.  This version sums spikes directly and divides
only once.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron


def _lif_fanouts(model: nn.Module) -> Dict[nn.Module, int]:
    """Map each spiking neuron layer to its fan-out.

    For a sequential architecture ``Linear -> LIF -> ... -> Linear``, the
    fan-out of a spiking layer is the output dimension of the following
    Linear layer.  A spiking layer with no following Linear is assigned a
    fan-out of 1 (its spikes have no downstream synaptic cost).
    """
    modules = list(model.net) if hasattr(model, "net") else list(model)
    fanouts: Dict[nn.Module, int] = {}
    for i, module in enumerate(modules):
        if isinstance(module, neuron.BaseNode):
            fanout = None
            for later in modules[i + 1:]:
                if isinstance(later, layer.Linear):
                    fanout = later.out_features
                    break
            if fanout is None:
                fanout = 1
            fanouts[module] = int(fanout)
    return fanouts


def count_synops(
    model: nn.Module,
    X_sample: np.ndarray,
    T: int = 16,
    device: str = "cpu",
    threshold: float = 0.0,
) -> Dict[str, float]:
    """Count Synaptic Operations (SynOps) for a batch of samples.

    Parameters
    ----------
    model : nn.Module
        Trained (or untrained) SNN with a ``.net`` Sequential containing
        ``spikingjelly`` spiking-neuron layers and exposing
        ``forward(x_seq)``.
    X_sample : np.ndarray
        Sample input batch, shape ``(n_trials, n_features)``.
    T : int
        Number of timesteps.
    device : str
        Compute device.
    threshold : float
        Fixed encoding threshold used to build the input spike train.

    Returns
    -------
    dict with keys:
        ``total_synops`` : float — total SynOps across all spiking layers
        for the whole batch (Eq. 2).
        ``synops_per_trial`` : float — ``total_synops / n_trials``.
        ``spike_density`` : float — average fraction of neurons firing per
        timestep, averaged over spiking layers.
    """
    if T < 1:
        raise ValueError(f"T must be positive, got {T}")

    model.eval()
    model.to(device)
    fanouts = _lif_fanouts(model)

    spike_sums: list = []
    densities: list = []

    def make_hook(fanout: int):
        def hook(module, inputs, output):
            spikes = output.detach()
            # Multi-step output is (T, batch, neurons); single-step is
            # (batch, neurons).  Normalise to the former for counting.
            if spikes.dim() == 2:
                spikes = spikes.unsqueeze(0)
            total_spikes = spikes.sum().item()
            spike_sums.append(fanout * total_spikes)
            densities.append((spikes > 0).float().mean().item())
        return hook

    hooks = []
    for module, fanout in fanouts.items():
        hooks.append(module.register_forward_hook(make_hook(fanout)))

    x = torch.as_tensor(X_sample, dtype=torch.float32, device=device)
    x = (x.abs() > threshold).float()
    x_seq = x.unsqueeze(0).repeat(T, 1, 1)
    try:
        functional.reset_net(model)
        with torch.no_grad():
            model(x_seq)
    finally:
        for h in hooks:
            h.remove()

    n_trials = len(X_sample)
    total_synops = float(sum(spike_sums))
    return {
        "total_synops": total_synops,
        "synops_per_trial": total_synops / max(n_trials, 1),
        "spike_density": float(np.mean(densities)) if densities else 0.0,
    }
