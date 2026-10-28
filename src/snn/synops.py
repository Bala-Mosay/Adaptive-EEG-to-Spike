"""Synaptic Operations (SynOps) counter — standardised energy proxy.

SynOps is the mathematically validated standard for SNN energy-proxy
estimation (Eq. 2 of the research paper; Merolla et al. 2014 [24] and
Rueckauer et al. 2017 [25])::

    SynOps = sum_l ( fanout_l * mean_spikes_l )

where ``fanout_l`` is the number of output connections of the spiking layer
l (i.e. the neuron count of the NEXT linear layer) and ``mean_spikes_l`` is
the average number of spikes emitted per neuron in layer l.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron


def _lif_fanouts(model: nn.Module) -> Dict[nn.Module, int]:
    """Map each LIFNode to its fan-out (neurons in the next Linear layer).

    For the architecture Linear -> LIF -> ... -> Linear, the fan-out of a
    spiking layer is the output dimension of the following Linear layer.
    """
    modules = list(model.net) if hasattr(model, "net") else list(model)
    fanouts: Dict[nn.Module, int] = {}
    for i, module in enumerate(modules):
        if isinstance(module, neuron.LIFNode):
            fanout = None
            for later in modules[i + 1:]:
                if isinstance(later, layer.Linear):
                    fanout = later.out_features
                    break
            if fanout is None:
                fanout = 1  # spiking layer feeds nothing further (readout done)
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
        ``spikingjelly`` LIFNode layers, exposing ``forward(x_seq)``.
    X_sample : np.ndarray
        Sample input batch, shape ``(n_trials, n_features)``.
    T : int
        Number of timesteps.
    device : str
        Compute device.

    Returns
    -------
    dict with keys:
        ``total_synops`` : float — total SynOps across all spiking layers
        for the whole batch (Eq. 2).
        ``synops_per_trial`` : float — ``total_synops / n_trials``.
        ``spike_density`` : float — average fraction of neurons firing per
        timestep, averaged over spiking layers.
    """
    model.eval()
    model.to(device)
    fanouts = _lif_fanouts(model)

    spike_sums: list = []
    densities: list = []

    def make_hook(fanout: int):
        def hook(module, inputs, output):
            # output: (T, batch, n_neurons)
            spikes = output.detach()
            mean_spikes_per_neuron = spikes.mean().item()  # avg over T, batch, neurons
            spike_sums.append(fanout * mean_spikes_per_neuron * T)
            densities.append((spikes > 0).float().mean().item())
        return hook

    hooks = []
    for module, fanout in fanouts.items():
        hooks.append(module.register_forward_hook(make_hook(fanout)))

    x = torch.as_tensor(X_sample, dtype=torch.float32, device=device)
    x = (x.abs() > threshold).float()  # binary spike encoding (|x| > threshold)
    x_seq = x.unsqueeze(0).repeat(T, 1, 1)
    with torch.no_grad():
        model(x_seq)

    for h in hooks:
        h.remove()

    n_trials = len(X_sample)
    total_synops = float(sum(spike_sums))
    return {
        "total_synops": total_synops,
        "synops_per_trial": total_synops / max(n_trials, 1),
        "spike_density": float(np.mean(densities)) if densities else 0.0,
    }
