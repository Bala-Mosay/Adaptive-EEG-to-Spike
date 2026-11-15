"""Leakage guards for Phase 3 C1 band selection.

The selector must be fit on training trials only.  These tests make the
leakage failure mode observable: changing the labels used for selection
changes the selected bands.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.bands.bank import SEVEN_BANDS
from src.bands.selector import AdaptiveBandSelector


def _band_features(y: np.ndarray, informative: str, seed: int = 0):
    """Seven band blocks; only ``informative`` depends on the labels."""
    n = len(y)
    rng = np.random.default_rng(seed)
    bands = {}
    for i, (name, _, _) in enumerate(SEVEN_BANDS):
        X = rng.standard_normal((n, 4))
        if name == informative:
            X[np.arange(n), y] += 6.0
        bands[name] = X
    return bands


def test_selection_depends_on_training_labels():
    y_a = np.tile(np.arange(4), 20)
    # Random relabelling so gamma is informative for y_b but not for y_a.
    y_b = np.random.default_rng(8).permutation(y_a)
    bands = _band_features(y_a, informative="theta", seed=1)
    # Rebuild gamma so it is informative for the second label vector.
    rng = np.random.default_rng(2)
    bands["gamma"] = rng.standard_normal((len(y_a), 4))
    bands["gamma"][np.arange(len(y_b)), y_b] += 6.0
    subjects = np.array(["S1"] * len(y_a))

    sel_a = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(bands, y_a, subjects)
    sel_b = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(bands, y_b, subjects)

    assert "theta" in sel_a.get_bands("S1")
    assert "gamma" in sel_b.get_bands("S1")
    assert sel_a.get_bands("S1") != sel_b.get_bands("S1")


def test_train_only_selection_differs_from_full_data_selection():
    """Fitting on train+test labels changes the selection for at least one subject."""
    y = np.tile(np.arange(4), 30)
    bands = _band_features(y, informative="mu", seed=3)
    subjects = np.array(["S1"] * 80 + ["S2"] * 40)

    train_mask = np.r_[0:60, 80:110]
    sel_train = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(
        {name: X[train_mask] for name, X in bands.items()},
        y[train_mask],
        subjects[train_mask],
    )
    sel_full = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(bands, y, subjects)

    changed = any(
        sel_train.get_bands(subj) != sel_full.get_bands(subj)
        for subj in np.unique(subjects)
    )
    assert changed, "Selection was identical on train and full data; guard is not informative"


def test_get_bands_is_pure_lookup():
    y = np.tile(np.arange(4), 20)
    bands = _band_features(y, informative="beta", seed=4)
    subjects = np.array(["S1"] * len(y))
    selector = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(bands, y, subjects)

    first = selector.get_bands("S1")
    _ = selector.get_bands("S1")
    assert first == selector.get_bands("S1")
