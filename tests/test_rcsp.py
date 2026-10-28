import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from src.spatial.rcsp import RegularizedCSP


def test_rcsp_fit_2class_and_4class():
    """Test RegularizedCSP.fit() on synthetic 2-class and 4-class datasets."""
    rng = np.random.RandomState(42)

    # 2-class dataset
    X_2class = rng.randn(12, 4, 100)
    y_2class = np.array([0, 1] * 6)
    rcsp_2 = RegularizedCSP(n_components=2)
    rcsp_2.fit(X_2class, y_2class)
    assert set(rcsp_2.filters_.keys()) == {"0", "1"}

    # 4-class dataset
    X_4class = rng.randn(20, 6, 100)
    y_4class = np.array([0, 1, 2, 3] * 5)
    rcsp_4 = RegularizedCSP(n_components=4)
    rcsp_4.fit(X_4class, y_4class)
    assert set(rcsp_4.filters_.keys()) == {"0", "1", "2", "3"}


def test_rcsp_transform_output_shape():
    """Test RegularizedCSP.transform() output shape matches (n_trials, n_classes * n_components)."""
    rng = np.random.RandomState(42)

    # 4-class, 4 n_components -> 4 classes * 4 components = 16 features
    X = rng.randn(16, 8, 200)
    y = np.array([0, 1, 2, 3] * 4)

    rcsp = RegularizedCSP(n_components=4)
    rcsp.fit(X, y)
    feats = rcsp.transform(X)
    assert feats.shape == (16, 16)

    # 2-class, 2 n_components -> 2 classes * 2 components = 4 features
    rcsp_2comp = RegularizedCSP(n_components=2)
    rcsp_2comp.fit(X, y)
    feats_2comp = rcsp_2comp.transform(X)
    assert feats_2comp.shape == (16, 8)  # 4 classes * 2 components = 8 features total across OvR


def test_rcsp_leakage_guard_filters_differ_across_folds():
    """Test that RegularizedCSP filters fit on fold 1 differ from fold 2."""
    rng = np.random.RandomState(10)
    X = rng.randn(20, 4, 100)
    y = np.array([0, 1, 2, 3] * 5)

    idx1 = np.arange(0, 10)
    idx2 = np.arange(10, 20)

    rcsp1 = RegularizedCSP(n_components=2).fit(X[idx1], y[idx1])
    rcsp2 = RegularizedCSP(n_components=2).fit(X[idx2], y[idx2])

    f1 = rcsp1.filters_.get("0")
    f2 = rcsp2.filters_.get("0")

    assert f1 is not None and f2 is not None
    assert not np.allclose(f1, f2), "RCSP filters should differ across training folds"


def test_rcsp_eigenvalue_spread():
    """CSP eigenvalues must have spread > 1 (class-discriminative filters).

    With the ledoit_wolf(cov) bug, shrinkage was always 1.0, collapsing
    covariance to scaled identity and producing eigenvalue ratio = 1.0
    (no class separation). After the fix (ledoit_wolf(X.T)), the spread
    must be significantly > 1.
    """
    rng = np.random.RandomState(42)
    n_ch, n_time = 8, 200

    # Create 2 classes with DIFFERENT spatial covariance structures
    # Class 0: channels 0-3 active
    A0 = np.zeros((n_ch, n_ch))
    A0[:4, :4] = np.diag([3.0, 2.5, 2.0, 1.5])
    # Class 1: channels 4-7 active
    A1 = np.zeros((n_ch, n_ch))
    A1[4:, 4:] = np.diag([3.0, 2.5, 2.0, 1.5])

    X0 = rng.randn(20, n_ch) @ A0 + rng.randn(20, n_ch) * 0.1  # (20, 8)
    X1 = rng.randn(20, n_ch) @ A1 + rng.randn(20, n_ch) * 0.1  # (20, 8)
    X = np.concatenate([X0, X1], axis=0)  # (40, 8)
    y = np.array([0]*20 + [1]*20)

    # Reshape to (n_trials, n_channels, n_time) by repeating
    X_trial = np.stack([X]*n_time, axis=-1)  # (40, 8, 200)

    rcsp = RegularizedCSP(n_components=4)
    rcsp.fit(X_trial, y)

    # Check that the two class-conditional covariance matrices have
    # different eigenvalue spreads (i.e., CSP found discriminative filters)
    from scipy.linalg import eigh
    for cls in rcsp.classes_:
        W = rcsp.filters_[str(cls)]
        # Project: (40, 8, 200) x (8, 4) -> check variance structure
        proj = np.tensordot(X_trial, W, axes=([1], [0]))  # (40, 200, 4)
        var = proj.var(axis=2)  # (40, 4)
        # Class 0 and class 1 should have DIFFERENT variance patterns
        var0 = var[:20].mean(axis=0)
        var1 = var[20:].mean(axis=0)
        ratio = var0 / (var1 + 1e-10)
        # At least one component should have >2x difference between classes
        assert ratio.max() > 2.0 or ratio.min() < 0.5, (
            f"CSP filters not discriminative for class {cls}: "
            f"var0={var0.round(3)}, var1={var1.round(3)}"
        )
