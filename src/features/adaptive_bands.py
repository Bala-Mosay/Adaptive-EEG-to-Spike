"""Adaptive-band CSP features (Phase 3, Contribution 1).

This module keeps the Phase 2 FBCSP backbone unchanged: one regularized CSP
is fitted per filter-bank band on the training fold, and log-variance
features are computed exactly as in :mod:`src.features.bandpass`.  The only
new step is that each subject keeps the ``top_k`` bands selected by
:class:`src.bands.selector.AdaptiveBandSelector`.

Output layout
-------------
For a trial from subject ``s``, the feature vector is the concatenation of
the CSP log-variance blocks for ``s``'s selected bands, in filter-bank order.
The output dimension is ``top_k * n_classes * n_components`` for every
subject, so a single classifier can be trained across subjects.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.bands.bank import SEVEN_BANDS, normalize_bands
from src.bands.selector import AdaptiveBandSelector
from src.features.bandpass import FixedBandCSPFeatures


class AdaptiveBandCSPFeatures(BaseEstimator, TransformerMixin):
    """Per-band CSP features with per-subject adaptive band selection.

    Parameters
    ----------
    bands : list of (name, low_hz, high_hz), optional
        Filter bank.  Defaults to :data:`src.bands.bank.SEVEN_BANDS`.
    n_components : int
        CSP components per class (OvR), per band.
    top_k : int
        Number of bands kept per subject.
    sfreq : float
        Sampling frequency in Hz.
    n_neighbors : int
        KNN parameter for the MI estimator.
    min_trials : int
        Minimum training trials for a subject-specific ranking.
    random_state : int, optional
        Passed to CSP and the MI estimator for determinism.

    Attributes
    ----------
    base_ : FixedBandCSPFeatures
        Fitted FBCSP extractor over the full filter bank.
    selector_ : AdaptiveBandSelector
        Fitted per-subject band selector.
    band_slices_ : dict
        ``{band_name: slice}`` into the full-bank feature matrix.
    n_features_out_ : int
        Output feature dimension.
    """

    def __init__(
        self,
        bands: Optional[List[Tuple[str, float, float]]] = None,
        n_components: int = 4,
        top_k: int = 3,
        sfreq: float = 160.0,
        n_neighbors: int = 5,
        min_trials: int = 20,
        random_state: Optional[int] = 42,
    ):
        self.bands = bands
        self.n_components = n_components
        self.top_k = top_k
        self.sfreq = sfreq
        self.n_neighbors = n_neighbors
        self.min_trials = min_trials
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: Sequence[str],
    ) -> "AdaptiveBandCSPFeatures":
        """Fit the full-bank CSP and per-subject band selector.

        Parameters
        ----------
        X : np.ndarray
            Training epochs, shape ``(n_trials, n_channels, n_timepoints)``.
        y : np.ndarray
            Integer labels, shape ``(n_trials,)``.
        subjects : sequence of str
            Subject ID per training trial.

        Returns
        -------
        AdaptiveBandCSPFeatures
            self
        """
        y = np.asarray(y)
        subjects = np.asarray(subjects)
        if X.ndim != 3:
            raise ValueError(
                f"X must have shape (n_trials, n_channels, n_timepoints), got {X.shape}"
            )
        if len(X) != len(y) or len(X) != len(subjects):
            raise ValueError("X, y, and subjects must have the same length")

        bands = SEVEN_BANDS if self.bands is None else self.bands
        self.bands_ = normalize_bands(bands)
        if self.top_k < 1 or self.top_k > len(self.bands_):
            raise ValueError(
                f"top_k must be in [1, {len(self.bands_)}], got {self.top_k}"
            )

        self.base_ = FixedBandCSPFeatures(
            bands=self.bands_,
            n_components=self.n_components,
            sfreq=self.sfreq,
            random_state=self.random_state,
        )
        self.base_.fit(X, y)

        F_all = self.base_.transform(X)
        classes = np.unique(y)
        block = len(classes) * self.n_components
        expected = len(self.bands_) * block
        if F_all.shape[1] != expected:
            raise RuntimeError(
                f"Unexpected full-bank feature dimension: {F_all.shape[1]} != {expected}"
            )

        self.band_slices_ = {
            name: slice(i * block, (i + 1) * block)
            for i, (name, _, _) in enumerate(self.bands_)
        }
        band_features = {
            name: F_all[:, self.band_slices_[name]] for name, _, _ in self.bands_
        }

        self.selector_ = AdaptiveBandSelector(
            top_k=self.top_k,
            n_neighbors=self.n_neighbors,
            min_trials=self.min_trials,
            random_state=self.random_state or 0,
        )
        self.selector_.fit(band_features, y, subjects)

        self.n_features_out_ = self.top_k * block
        return self

    def transform(self, X: np.ndarray, subjects: Sequence[str]) -> np.ndarray:
        """Select each trial's subject-specific band blocks.

        Parameters
        ----------
        X : np.ndarray
            Epochs, shape ``(n_trials, n_channels, n_timepoints)``.
        subjects : sequence of str
            Subject ID per trial.

        Returns
        -------
        np.ndarray
            Features of shape ``(n_trials, top_k * n_classes * n_components)``.
        """
        if not hasattr(self, "selector_"):
            raise RuntimeError("AdaptiveBandCSPFeatures must be fit before transform")
        subjects = np.asarray(subjects)
        if X.ndim != 3:
            raise ValueError(
                f"X must have shape (n_trials, n_channels, n_timepoints), got {X.shape}"
            )
        if len(X) != len(subjects):
            raise ValueError("X and subjects must have the same length")

        F_all = self.base_.transform(X)
        out = np.empty((len(X), self.n_features_out_), dtype=F_all.dtype)
        for subject in np.unique(subjects):
            mask = subjects == subject
            bands = self.selector_.get_bands(str(subject))
            cols = np.concatenate(
                [np.arange(F_all.shape[1])[self.band_slices_[name]] for name in bands]
            )
            out[mask] = F_all[mask][:, cols]
        return out

    def get_selected_bands(self) -> Dict[str, List[str]]:
        """Return ``{subject: selected_bands}`` from the fitted selector."""
        if not hasattr(self, "selector_"):
            raise RuntimeError("AdaptiveBandCSPFeatures must be fit before use")
        return {
            subject: list(bands)
            for subject, bands in self.selector_.selected_bands_.items()
        }
