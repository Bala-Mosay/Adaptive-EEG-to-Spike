"""Fixed-threshold spike encoder (Stage 4, baseline/fixed version).

This is the deliberate BASELINE encoder described in the research paper
(Gap 3, Section 4.3): the encoding threshold is set to the 75th percentile
of absolute training-feature amplitude, giving ~25% spike density.
Phase 3 Contribution 2 replaces this with a learnable threshold optimised
jointly with a spike-count sparsity penalty.
"""
from typing import Union

import numpy as np
import torch
import torch.nn as nn


class FixedThresholdEncoder(nn.Module):
    """Convert continuous features to spike trains using a FIXED threshold.

    The threshold is set ONCE from the 75th percentile of absolute
    training-fold feature values (via :meth:`set_threshold_from_data`) and
    never updated. It is intentionally NOT an ``nn.Parameter`` — the baseline
    threshold must not be learnable (enforced by ``tests/test_encoder.py``).

    Parameters
    ----------
    threshold : float
        Fixed encoding threshold.
    T : int
        Number of timesteps. Default 16 (250 ms windows over a 4 s epoch).
    """

    def __init__(self, threshold: float = 0.0, T: int = 16):
        super().__init__()
        self.threshold = float(threshold)
        self.T = int(T)

    def encode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a feature tensor to binary spikes (no T expansion).

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Binary spike tensor of same shape ``(batch, n_features)``.
        """
        return (torch.abs(x) > self.threshold).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features into a constant spike train across T timesteps.

        Eq. 9 of the paper: S(t) = H(x - theta), with x held fixed across
        timesteps (baseline; features are per-epoch scalars, not time series).

        Parameters
        ----------
        x : torch.Tensor
            Feature tensor of shape ``(batch, n_features)``.

        Returns
        -------
        torch.Tensor
            Binary spike tensor of shape ``(T, batch, n_features)``.
        """
        x_expanded = x.unsqueeze(0).repeat(self.T, 1, 1)
        return (torch.abs(x_expanded) > self.threshold).float()

    def set_threshold_from_data(self, X_train: Union[np.ndarray, torch.Tensor]) -> None:
        """Set the threshold to the 75th percentile of absolute feature values.

        Paper Section 5.4: "The threshold θ is initialised at the 75th
        percentile of the absolute signal amplitude on the training set,
        providing approximately 25% spike density at initialization."

        Must be called ONCE per fold, on training data only (never on
        val/test) to avoid leakage.

        Parameters
        ----------
        X_train : np.ndarray or torch.Tensor
            Training feature matrix, shape ``(n_trials, n_features)``.
        """
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        self.threshold = float(np.percentile(np.abs(X_train), 75))

    def extra_repr(self) -> str:
        return f"threshold={self.threshold:.6f}, T={self.T}"
