"""Fixed-band feature extraction (Stage 3, baseline/fixed version).

The research paper (Section 5.3) describes Stage 3 as: decompose the EEG into
frequency sub-bands and compute log-variance features per sub-band, with CSP
spatial filtering within each band (the FBCSP pattern). In Phase 2 the
sub-bands are FIXED (mu 8-12 Hz, beta 13-30 Hz) and identical for every
subject — this is the deliberate baseline limitation that Phase 3's adaptive
per-subject KNN-MI band selection (Contribution 1) will replace.

All fitting (per-band CSP filters, feature scaling) happens on the training
fold only, inside the cross-validation loop.
"""
from typing import List, Optional, Tuple

import numpy as np
from mne.filter import filter_data
from sklearn.base import BaseEstimator, TransformerMixin

from src.spatial.rcsp import RegularizedCSP

#: Fixed frequency bands — same for ALL subjects (baseline).
#: This is the "problem" that Phase 3 Contribution 1 (adaptive bands) fixes.
FIXED_BANDS: List[Tuple[str, float, float]] = [
    ("mu", 8.0, 12.0),
    ("beta", 13.0, 30.0),
]


def extract_band_features(
    X: np.ndarray,
    sfreq: float = 160.0,
    bands: Optional[List[Tuple[str, float, float]]] = None,
) -> np.ndarray:
    """Compute log-variance features per band per channel (no spatial filtering).

    Parameters
    ----------
    X : np.ndarray
        Epochs of shape ``(n_trials, n_channels, n_timepoints)``.
    sfreq : float
        Sampling frequency in Hz.
    bands : list of (name, low_hz, high_hz), optional
        Bands to extract. Defaults to :data:`FIXED_BANDS`.

    Returns
    -------
    np.ndarray
        Features of shape ``(n_trials, n_channels * n_bands)`` where feature
        blocks are concatenated in band order. Each feature is
        ``log(var(bandpass_filtered))`` per trial per channel.
    """
    if bands is None:
        bands = FIXED_BANDS
    n_trials, n_ch, _ = X.shape
    features = []
    for _, low, high in bands:
        filtered = filter_data(
            X.reshape(n_trials * n_ch, -1).astype(np.float64),
            sfreq=sfreq,
            l_freq=low,
            h_freq=high,
            method="fir",
            verbose=False,
        ).reshape(n_trials, n_ch, -1)
        # Eq. 7 of the paper: f_b = log(Var(x_b))
        log_var = np.log(np.var(filtered, axis=-1) + 1e-8)
        features.append(log_var)
    return np.concatenate(features, axis=-1)


class FixedBandCSPFeatures(BaseEstimator, TransformerMixin):
    """Per-band CSP feature extractor with FIXED bands (baseline Stage 3).

    For each band: bandpass-filter the epochs, fit a fresh
    :class:`~src.spatial.rcsp.RegularizedCSP` on the TRAINING fold only
    (One-vs-Rest over all classes), and emit log-variance features of the
    CSP-projected signals. Feature blocks from all bands are concatenated.

    Output feature dimension: ``n_bands * n_classes * n_components``.

    This follows the paper's FBCSP pattern with fixed bands; Phase 3 will
    replace the fixed band set with per-subject KNN-MI selected bands.

    Parameters
    ----------
    bands : list of (name, low_hz, high_hz), optional
        Fixed bands. Defaults to :data:`FIXED_BANDS`.
    n_components : int
        CSP components per class (OvR), per band.
    sfreq : float
        Sampling frequency in Hz.
    random_state : int, optional
        Passed to each per-band CSP estimator for determinism.

    Attributes
    ----------
    csp_per_band_ : dict
        Fitted ``RegularizedCSP`` per band name.
    """

    def __init__(
        self,
        bands: Optional[List[Tuple[str, float, float]]] = None,
        n_components: int = 4,
        sfreq: float = 160.0,
        random_state: Optional[int] = None,
    ):
        self.bands = bands
        self.n_components = n_components
        self.sfreq = sfreq
        self.random_state = random_state

    def _band_filtered(self, X: np.ndarray, band: Tuple[str, float, float]) -> np.ndarray:
        _, low, high = band
        n_trials, n_ch, _ = X.shape
        return filter_data(
            X.reshape(n_trials * n_ch, -1).astype(np.float64),
            sfreq=self.sfreq,
            l_freq=low,
            h_freq=high,
            method="fir",
            verbose=False,
        ).reshape(n_trials, n_ch, -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FixedBandCSPFeatures":
        """Fit one RegularizedCSP per band on the training data.

        Parameters
        ----------
        X : np.ndarray
            Training epochs, shape ``(n_trials, n_channels, n_timepoints)``.
        y : np.ndarray
            Integer class labels (0-indexed), shape ``(n_trials,)``.

        Returns
        -------
        FixedBandCSPFeatures
            self
        """
        if self.bands is None:
            bands = FIXED_BANDS
        else:
            bands = self.bands
        self.bands_ = list(bands)
        self.csp_per_band_ = {}
        for name, low, high in self.bands_:
            X_band = self._band_filtered(X, (name, low, high))
            csp = RegularizedCSP(n_components=self.n_components, random_state=self.random_state)
            csp.fit(X_band, y)
            self.csp_per_band_[name] = csp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute concatenated per-band CSP log-variance features.

        Parameters
        ----------
        X : np.ndarray
            Epochs, shape ``(n_trials, n_channels, n_timepoints)``.

        Returns
        -------
        np.ndarray
            Features, shape ``(n_trials, n_bands * n_classes * n_components)``.
        """
        feats = []
        for name, low, high in self.bands_:
            X_band = self._band_filtered(X, (name, low, high))
            feats.append(self.csp_per_band_[name].transform(X_band))
        return np.concatenate(feats, axis=1)
