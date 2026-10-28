"""Baseline single-timescale LIF SNN classifier (Stage 5, simplified version).

The research paper (Gap 4, Section 4.4) notes that ALL existing SNN-BCIs use
a single membrane time constant — this baseline reproduces that standard
architecture. Phase 3 Contribution 3 replaces it with the dual-branch
fast/slow Parametric LIF network with attention fusion.
"""
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron, surrogate


class BaselineLIFNet(nn.Module):
    """Baseline SNN using standard single-tau LIF neurons (SpikingJelly).

    Architecture (multi-step, spike-vote readout)::

        Linear -> LIF -> Dropout -> Linear -> LIF -> Linear

    Standard LIF architecture: Linear feeds directly into LIF neurons.
    v_reset=0.0 with decay_input=False prevents negative membrane drift.

    All LIF neurons share ONE fixed membrane time constant ``tau``. Surrogate
    gradients use the arctangent approximation (Eq. 3 of the paper,
    SpikingJelly default). Classification is by summing output logits over
    the T timesteps (spike voting over T=16, Section 5.6).

    Parameters
    ----------
    n_features : int
        Input feature dimension (n_bands * n_classes * n_components).
    n_hidden : int
        First hidden layer size. Default 64.
    n_classes : int
        Number of output classes. Default 4.
    tau : float
        LIF membrane time constant (SpikingJelly parametrisation, tau > 1).
        Single fixed value for ALL neurons — the baseline limitation.
    T : int
        Number of timesteps. Default 16.
    dropout : float
        Dropout probability after the first hidden block. Default 0.5.

    Note
    ----
    ``ParametricLIFNode`` is deliberately NOT used anywhere in this module
    (guarded by ``tests/test_lif_net.py``) — learnable tau belongs to Phase 3.
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int = 64,
        n_classes: int = 4,
        tau: float = 20.0,
        T: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.T = int(T)
        self.n_classes = int(n_classes)
        self.tau = float(tau)

        surrogate_fn = surrogate.ATan()
        lif_kwargs = dict(
            tau=self.tau,
            surrogate_function=surrogate_fn,
            detach_reset=True,
            decay_input=False,  # standard LIF: v = v*(1-1/tau) + x
            v_reset=0.0,  # hard reset: v -> 0 on spike, prevents negative drift
        )

        # spikingjelly layer.* wrappers support step_mode='m' so plain
        # (T, N, C) tensors flow through without manual reshaping.
        # Standard LIF baseline: Linear -> LIF -> Dropout -> Linear -> LIF -> Linear.
        # v_reset=0.0 with decay_input=False prevents negative membrane drift.
        self.net = nn.Sequential(
            layer.Linear(n_features, n_hidden, step_mode="m"),
            neuron.LIFNode(**lif_kwargs, step_mode="m"),
            layer.Dropout(p=dropout, step_mode="m"),
            layer.Linear(n_hidden, n_hidden // 2, step_mode="m"),
            neuron.LIFNode(**lif_kwargs, step_mode="m"),
            layer.Linear(n_hidden // 2, n_classes, step_mode="m"),
        )

        functional.set_step_mode(self, step_mode="m")

        # Hook-based spike counting across ALL LIF layers. In multi-step
        # mode a LIFNode's internal `.spike` only reflects the last
        # timestep, so we accumulate per-forward sums via hooks instead.
        self._lif_modules = [m for m in self.net if isinstance(m, neuron.LIFNode)]
        self._spike_sums: list = []
        self._hooks = []
        for m in self._lif_modules:
            self._hooks.append(m.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook(module, inputs, output):
            # output: (T, batch, n_neurons) in multi-step mode
            self._spike_sums.append(output.detach().abs().sum())
        return hook

    def forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over T timesteps.

        Parameters
        ----------
        x_seq : torch.Tensor
            Spike input of shape ``(T, batch, n_features)``.

        Returns
        -------
        logits : torch.Tensor
            Output summed over timesteps (spike vote), shape ``(batch, n_classes)``.
        spike_count : torch.Tensor
            Scalar tensor: total spikes across ALL LIF layers and timesteps
            for this batch (energy-proxy diagnostic; see also
            :mod:`src.snn.synops` for the SynOps measure).
        """
        functional.reset_net(self)
        self._spike_sums = []
        out_seq = self.net(x_seq)  # (T, batch, n_classes)
        logits = out_seq.sum(dim=0)

        total_spikes = torch.zeros((), dtype=x_seq.dtype, device=x_seq.device)
        for s in self._spike_sums:
            total_spikes = total_spikes + s.to(x_seq.device)
        return logits, total_spikes

    def get_layer_spike_rates(self, x_seq: torch.Tensor) -> Dict[str, float]:
        """Diagnostic: get average spike rate per LIF layer.

        Parameters
        ----------
        x_seq : torch.Tensor
            Spike input, shape ``(T, batch, n_features)``.

        Returns
        -------
        dict with ``lif1_rate`` and ``lif2_rate`` (0.0–1.0).
        """
        self.eval()
        lif_modules = [m for m in self.net if isinstance(m, neuron.LIFNode)]
        rates: Dict[str, float] = {}
        hooks: list = []

        def make_h(name):
            def h(mod, inp, out):
                rates[name] = (out > 0).float().mean().item()
            return h

        for i, m in enumerate(lif_modules):
            hooks.append(m.register_forward_hook(make_h(f"lif{i}")))

        functional.reset_net(self)
        with torch.no_grad():
            self.net(x_seq)

        for h in hooks:
            h.remove()
        return rates
