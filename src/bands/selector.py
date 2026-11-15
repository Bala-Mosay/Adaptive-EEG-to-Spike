"""Per-subject adaptive band selection (Phase 3, Contribution 1).

The selector is fitted on **training-fold trials only**.  For each subject it
ranks the filter-bank bands by :func:`src.bands.knn_mi.band_mi_scores` and
keeps the top ``top_k`` bands.  Subjects with too few training trials, or
with fewer than two classes in the training split, fall back to the global
ranking computed over all training subjects.

Leakage contract
----------------
``fit`` receives only training trials, labels, and subject IDs.  ``get_bands``
is a pure lookup; it never sees test labels.  The returned band list is in
filter-bank order (not MI-rank order), so feature-block positions have the
same meaning for every subject.
"""
from typing import Dict, List, Sequence

import numpy as np
from sklearn.base import BaseEstimator

from .knn_mi import band_mi_scores


class AdaptiveBandSelector(BaseEstimator):
    """Rank and select frequency bands separately for each subject.

    Parameters
    ----------
    top_k : int
        Number of bands to keep per subject.
    n_neighbors : int
        KNN parameter for the MI estimator.
    min_trials : int
        Minimum number of training trials required to fit a subject-specific
        ranking.  Below this threshold the global ranking is used.
    random_state : int
        Seed for the MI estimator.

    Attributes
    ----------
    band_names_ : list of str
        Band names in filter-bank order.
    global_scores_ : dict
        MI scores computed on all training trials.
    global_bands_ : list of str
        Global top-``top_k`` bands, in filter-bank order.
    selected_bands_ : dict
        ``{subject_id: [band names]}`` for every training subject.
    scores_ : dict
        ``{subject_id: {band: score}}`` for subjects with a fitted ranking.
    """

    def __init__(
        self,
        top_k: int = 3,
        n_neighbors: int = 5,
        min_trials: int = 20,
        random_state: int = 42,
    ):
        self.top_k = top_k
        self.n_neighbors = n_neighbors
        self.min_trials = min_trials
        self.random_state = random_state

    def fit(
        self,
        band_features: Dict[str, np.ndarray],
        y: np.ndarray,
        subjects: Sequence[str],
    ) -> "AdaptiveBandSelector":
        """Fit per-subject band rankings on training data.

        Parameters
        ----------
        band_features : dict
            ``{band_name: X_band}`` with ``X_band`` shape
            ``(n_trials, n_features_in_band)``.
        y : np.ndarray
            Integer labels, shape ``(n_trials,)``.
        subjects : sequence of str
            Subject ID per trial, length ``n_trials``.

        Returns
        -------
        AdaptiveBandSelector
            self
        """
        if not band_features:
            raise ValueError("band_features must contain at least one band")
        self.band_names_ = list(band_features.keys())
        if self.top_k < 1 or self.top_k > len(self.band_names_):
            raise ValueError(
                f"top_k must be in [1, {len(self.band_names_)}], got {self.top_k}"
            )

        y = np.asarray(y)
        subjects = np.asarray(subjects)
        if len(subjects) != len(y):
            raise ValueError("subjects and y must have the same length")
        for name, Xb in band_features.items():
            if len(Xb) != len(y):
                raise ValueError(
                    f"Band {name!r} has {len(Xb)} rows but y has {len(y)}"
                )

        self.global_scores_ = band_mi_scores(
            band_features,
            y,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        self.global_bands_ = self._select(self.global_scores_)

        self.selected_bands_ = {}
        self.scores_ = {}
        for subject in np.unique(subjects):
            mask = subjects == subject
            y_sub = y[mask]
            if mask.sum() < self.min_trials or len(np.unique(y_sub)) < 2:
                self.selected_bands_[str(subject)] = list(self.global_bands_)
                continue

            sub_features = {
                name: Xb[mask] for name, Xb in band_features.items()
            }
            scores = band_mi_scores(
                sub_features,
                y_sub,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
            self.scores_[str(subject)] = scores
            self.selected_bands_[str(subject)] = self._select(scores)

        return self

    def get_bands(self, subject: str) -> List[str]:
        """Return the selected bands for ``subject``.

        Unknown subjects receive the global top-``top_k`` bands.  The list is
        always in filter-bank order.
        """
        if not hasattr(self, "selected_bands_"):
            raise RuntimeError("AdaptiveBandSelector must be fit before use")
        return list(self.selected_bands_.get(str(subject), self.global_bands_))

    def _select(self, scores: Dict[str, float]) -> List[str]:
        """Pick top-k by score, then return them in filter-bank order."""
        ranked = sorted(
            scores,
            key=lambda name: (-scores[name], self.band_names_.index(name)),
        )
        chosen = set(ranked[: self.top_k])
        return [name for name in self.band_names_ if name in chosen]
