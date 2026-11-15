"""KNN mutual-information scoring for frequency bands (Phase 3, C1).

The paper's Eq. 8 specifies a Kraskov KNN-MI estimator for ranking bands.
This module uses scikit-learn's :func:`~sklearn.feature_selection.mutual_info_classif`,
which is itself a k-nearest-neighbour MI estimator for continuous features
and discrete labels.

For a band with several CSP components, the band score is the **mean of the
per-component KNN-MI values**.  This is a deliberate numerical choice: a
joint 16-dimensional KNN-MI estimate is unstable with the ~60 training
trials available per subject, while the mean of per-component estimates is
stable and still ranks bands by their label information.  The score is used
only for *ranking* bands; it is not reported as an absolute MI value.
"""
from typing import Dict, Mapping

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def band_mi_scores(
    band_features: Mapping[str, np.ndarray],
    y: np.ndarray,
    *,
    n_neighbors: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute a KNN-MI score for each band's feature block.

    Parameters
    ----------
    band_features : mapping
        ``{band_name: X_band}`` where ``X_band`` has shape
        ``(n_trials, n_features_in_band)``.
    y : np.ndarray
        Integer class labels, shape ``(n_trials,)``.
    n_neighbors : int
        Number of neighbours for the KNN-MI estimator.
    random_state : int
        Seed for the small amount of jitter added by scikit-learn's
        estimator (keeps rankings deterministic).

    Returns
    -------
    dict
        ``{band_name: mean_mi}``.  Bands with constant features receive
        ``0.0``.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
    if len(np.unique(y)) < 2:
        raise ValueError("KNN-MI requires at least two classes")

    scores: Dict[str, float] = {}
    for name, Xb in band_features.items():
        Xb = np.asarray(Xb, dtype=np.float64)
        if Xb.ndim == 1:
            Xb = Xb[:, None]
        if Xb.shape[0] != len(y):
            raise ValueError(
                f"Band {name!r} has {Xb.shape[0]} rows but y has {len(y)}"
            )

        # scikit-learn jitters continuous features internally to break ties.
        # A constant component would therefore receive a spurious non-zero
        # KNN-MI estimate, so exclude it explicitly (a constant feature
        # carries no label information by construction).
        active = np.flatnonzero(np.var(Xb, axis=0) > 0.0)
        if len(active) == 0:
            scores[name] = 0.0
            continue

        mi = mutual_info_classif(
            Xb[:, active],
            y,
            discrete_features=False,
            n_neighbors=int(n_neighbors),
            random_state=random_state,
        )
        scores[name] = float(np.mean(mi)) if len(mi) else 0.0
    return scores
