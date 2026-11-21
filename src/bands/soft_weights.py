"""Noise-robust soft per-subject band weighting (Phase 3, revised C1).

Hard top-k selection deletes whole bands from a noisy ranking; with ~70
training trials per subject the KNN-MI estimate is unstable, so deletion is
often wrong.  This module keeps every band and instead assigns each subject a
*weight* per band.

Weights are a convex combination of two mean-one vectors::

    w = (1 - lambda) * prior + lambda * softmax(z / temperature)

where ``z`` is the per-subject KNN-MI score vector standardized over bands.
The mean-one scaling gives two useful limits:

* ``lambda = 0`` with ``shrink_to="uniform"`` -> every weight is exactly 1,
  i.e. the unweighted full-bank features (the ``all7`` control);
* ``lambda = 1`` with a small temperature -> the ranking collapses onto the
  best band, the same failure mode as hard selection.

So the method *contains* both the control and the hard-selection extreme, and
``lambda`` / ``temperature`` interpolate between them.  This is why soft
weighting cannot discard a band by accident.

Leakage contract
----------------
``fit`` receives only training-fold trials, labels, and subject IDs.  The
weights are frozen at fit time; ``get_weights`` is a pure lookup.
"""
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator

from .knn_mi import band_mi_scores


class SoftBandWeighter(BaseEstimator):
    """Fit per-subject band weights shrunk toward a stable prior.

    Parameters
    ----------
    lambda_ : float
        Trust in the per-subject ranking, in ``[0, 1]``.  ``0`` ignores the
        ranking (weights = prior), ``1`` uses it fully.
    temperature : float
        Softmax temperature applied to standardized scores.  Smaller values
        make the weighting peakier; must be positive.
    shrink_to : {"uniform", "global"}
        Prior to shrink toward.  ``"uniform"`` gives every band equal weight;
        ``"global"`` uses the ranking computed over all training subjects.
    n_neighbors, min_trials, random_state
        Passed to the KNN-MI estimator / used for the small-sample fallback.

    Attributes
    ----------
    band_names_ : list of str
        Band names in filter-bank order.
    global_scores_ : dict
        MI scores over all training trials.
    scores_ : dict
        ``{subject: {band: score}}`` for subjects with enough trials.
    weights_ : dict
        ``{subject: np.ndarray}`` with one mean-one weight per band, in
        ``band_names_`` order.
    """

    def __init__(
        self,
        lambda_: float = 0.3,
        temperature: float = 1.0,
        shrink_to: str = "uniform",
        n_neighbors: int = 5,
        min_trials: int = 20,
        random_state: int = 42,
    ):
        self.lambda_ = lambda_
        self.temperature = temperature
        self.shrink_to = shrink_to
        self.n_neighbors = n_neighbors
        self.min_trials = min_trials
        self.random_state = random_state

    def fit(
        self,
        band_features: Mapping[str, np.ndarray],
        y: np.ndarray,
        subjects: Sequence[str],
    ) -> "SoftBandWeighter":
        """Fit the global prior and every training subject's weights."""
        if not band_features:
            raise ValueError("band_features must contain at least one band")
        if not 0.0 <= float(self.lambda_) <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {self.lambda_}")
        if float(self.temperature) <= 0.0:
            raise ValueError(
                f"temperature must be positive, got {self.temperature}"
            )
        if self.shrink_to not in {"uniform", "global"}:
            raise ValueError(
                f"shrink_to must be 'uniform' or 'global', got {self.shrink_to!r}"
            )

        self.band_names_ = list(band_features.keys())
        y = np.asarray(y)
        subjects = np.asarray(subjects)

        self.global_scores_ = band_mi_scores(
            band_features,
            y,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        self.prior_ = self._prior()
        self.global_weights_ = self._combine(self.global_scores_)

        self.weights_ = {}
        self.scores_ = {}
        for subject in np.unique(subjects):
            mask = subjects == subject
            if mask.sum() < self.min_trials or len(np.unique(y[mask])) < 2:
                # Too few trials for a subject-specific estimate: fall back to
                # the prior, which is exactly what "no adaptation" means.
                self.weights_[str(subject)] = np.array(self.prior_, dtype=float)
                continue
            scores = band_mi_scores(
                {name: values[mask] for name, values in band_features.items()},
                y[mask],
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
            self.scores_[str(subject)] = scores
            self.weights_[str(subject)] = self._combine(scores)

        return self

    def get_weights(self, subject: str) -> np.ndarray:
        """Return mean-one band weights for ``subject`` in filter-bank order.

        Unknown subjects (e.g. a held-out person) receive the global weights.
        """
        if not hasattr(self, "weights_"):
            raise RuntimeError("SoftBandWeighter must be fit before use")
        return np.asarray(
            self.weights_.get(str(subject), self.global_weights_), dtype=float
        )

    # -- internals ----------------------------------------------------------
    def _prior(self) -> np.ndarray:
        """Mean-one prior vector for the shrinkage target."""
        n_bands = len(self.band_names_)
        if self.shrink_to == "uniform":
            return np.ones(n_bands, dtype=float)
        return self._soft_weights(self.global_scores_)

    def _soft_weights(self, scores: Mapping[str, float]) -> np.ndarray:
        """Softmax over standardized scores, rescaled to mean one."""
        values = np.array([scores[name] for name in self.band_names_], dtype=float)
        spread = float(values.std())
        if spread <= 0.0:
            return np.ones(len(values), dtype=float)
        z = (values - values.mean()) / spread
        # Subtract the max for numerical stability; softmax is shift-invariant.
        exp = np.exp((z - z.max()) / float(self.temperature))
        probabilities = exp / exp.sum()
        return probabilities * len(values)

    def _combine(self, scores: Mapping[str, float]) -> np.ndarray:
        """Shrink this subject's soft weights toward the prior."""
        lam = float(self.lambda_)
        return (1.0 - lam) * self.prior_ + lam * self._soft_weights(scores)
