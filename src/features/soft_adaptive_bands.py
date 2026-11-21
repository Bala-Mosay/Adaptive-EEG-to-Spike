"""Soft adaptive-band CSP features (Phase 3, revised Contribution 1).

Same FBCSP backbone as :mod:`src.features.adaptive_bands`, but instead of
deleting all but ``top_k`` band blocks, every band is kept and scaled by a
per-subject weight from :class:`src.bands.soft_weights.SoftBandWeighter`.

With ``lambda_=0`` and ``shrink_to="uniform"`` the output is bitwise identical
to the unweighted full-bank features (``all7``), so the control condition is a
strict special case of this model.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.bands.bank import SEVEN_BANDS, normalize_bands
from src.bands.soft_weights import SoftBandWeighter
from src.features.bandpass import FixedBandCSPFeatures


class SoftAdaptiveBandCSPFeatures(BaseEstimator, TransformerMixin):
    """Per-band CSP features with soft, shrunk per-subject band weights.

    Parameters
    ----------
    bands : list of (name, low_hz, high_hz), optional
        Filter bank.  Defaults to :data:`src.bands.bank.SEVEN_BANDS`.
    n_components : int
        CSP components per class (OvR), per band.
    sfreq : float
        Sampling frequency in Hz.
    lambda_ : float
        Trust in the per-subject KNN-MI ranking (0 = no adaptation).
    temperature : float
        Softmax temperature for the weighting.
    shrink_to : {"uniform", "global"}
        Prior the weights are shrunk toward.
    n_neighbors, min_trials : int
        KNN-MI parameters and the small-sample fallback threshold.
    random_state : int, optional
        Seed for CSP and the MI estimator.

    Attributes
    ----------
    base_ : FixedBandCSPFeatures
        Fitted FBCSP extractor over the full bank.
    weighter_ : SoftBandWeighter
        Fitted per-subject weight lookup.
    band_slices_ : dict
        ``{band_name: slice}`` into the full-bank feature matrix.
    n_features_out_ : int
        Output dimension (``n_bands * n_classes * n_components``).
    """

    def __init__(
        self,
        bands: Optional[List[Tuple[str, float, float]]] = None,
        n_components: int = 4,
        sfreq: float = 160.0,
        lambda_: float = 0.3,
        temperature: float = 1.0,
        shrink_to: str = "uniform",
        n_neighbors: int = 5,
        min_trials: int = 20,
        random_state: Optional[int] = 42,
    ):
        self.bands = bands
        self.n_components = n_components
        self.sfreq = sfreq
        self.lambda_ = lambda_
        self.temperature = temperature
        self.shrink_to = shrink_to
        self.n_neighbors = n_neighbors
        self.min_trials = min_trials
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: Sequence[str],
    ) -> "SoftAdaptiveBandCSPFeatures":
        """Fit full-bank CSP and the per-subject band weights on train data."""
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

        self.base_ = FixedBandCSPFeatures(
            bands=self.bands_,
            n_components=self.n_components,
            sfreq=self.sfreq,
            random_state=self.random_state,
        )
        self.base_.fit(X, y)

        F_all = self.base_.transform(X)
        block = len(np.unique(y)) * self.n_components
        self.band_slices_ = {
            name: slice(index * block, (index + 1) * block)
            for index, (name, _, _) in enumerate(self.bands_)
        }
        expected = len(self.bands_) * block
        if F_all.shape[1] != expected:
            raise RuntimeError(
                f"Unexpected full-bank feature dimension: {F_all.shape[1]} != {expected}"
            )
        band_features = {
            name: F_all[:, self.band_slices_[name]] for name, _, _ in self.bands_
        }

        self.weighter_ = SoftBandWeighter(
            lambda_=self.lambda_,
            temperature=self.temperature,
            shrink_to=self.shrink_to,
            n_neighbors=self.n_neighbors,
            min_trials=self.min_trials,
            random_state=self.random_state or 0,
        ).fit(band_features, y, subjects)

        self.n_features_out_ = len(self.bands_) * block
        return self

    def transform(self, X: np.ndarray, subjects: Sequence[str]) -> np.ndarray:
        """Return full-bank features with each band block scaled per subject."""
        if not hasattr(self, "weighter_"):
            raise RuntimeError(
                "SoftAdaptiveBandCSPFeatures must be fit before transform"
            )
        subjects = np.asarray(subjects)
        if X.ndim != 3:
            raise ValueError(
                f"X must have shape (n_trials, n_channels, n_timepoints), got {X.shape}"
            )
        if len(X) != len(subjects):
            raise ValueError("X and subjects must have the same length")

        F_all = self.base_.transform(X)
        out = F_all.copy()
        for subject in np.unique(subjects):
            mask = subjects == subject
            weights = self.weighter_.get_weights(str(subject))
            for index, (name, _, _) in enumerate(self.bands_):
                out[mask, self.band_slices_[name]] *= weights[index]
        return out

    def get_weights(self) -> Dict[str, List[float]]:
        """Return ``{subject: [weight per band]}`` from the fitted weighter."""
        if not hasattr(self, "weighter_"):
            raise RuntimeError(
                "SoftAdaptiveBandCSPFeatures must be fit before use"
            )
        return {
            subject: [float(value) for value in weights]
            for subject, weights in self.weighter_.weights_.items()
        }
