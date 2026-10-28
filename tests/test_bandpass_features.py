import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.features.bandpass import FixedBandCSPFeatures, extract_band_features, FIXED_BANDS
from src.labels import LABEL_ORDER


def _make_epochs(n_per_class=10, n_ch=64, n_times=640, seed=0):
    """Synthetic epochs (n_trials, n_ch, n_times) + 0-indexed labels."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls_idx, _ in enumerate(LABEL_ORDER):
        X.append(rng.standard_normal((n_per_class, n_ch, n_times)).astype(np.float32))
        y.append(np.full(n_per_class, cls_idx, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y)


def test_extract_band_features_output_shape():
    """Band features must be (n_trials, n_bands * n_channels)."""
    X, _ = _make_epochs(n_per_class=5, n_ch=8, n_times=100)
    feats = extract_band_features(X, bands=FIXED_BANDS)
    assert feats.shape == (5 * 4, 8 * len(FIXED_BANDS))


def test_fixed_band_csp_features_output_shape():
    """CSP features must be (n_trials, n_bands * n_classes * n_components)."""
    X, y = _make_epochs(n_per_class=10, n_ch=8, n_times=100)
    extractor = FixedBandCSPFeatures(bands=FIXED_BANDS, n_components=2, sfreq=160.0)
    extractor.fit(X, y)
    feats = extractor.transform(X)
    n_classes = len(LABEL_ORDER)
    expected = len(FIXED_BANDS) * n_classes * 2
    assert feats.shape == (40, expected)


def test_leakage_guard_fit_train_vs_full():
    """CSP fitted on train subset must differ from full-data fit."""
    X, y = _make_epochs(n_per_class=12, n_ch=8, n_times=100)
    # Split: first 9 per-class = 36 train, last 3 per-class = 12 test
    # (ensure all 4 classes present in train)
    train_mask = np.array([True] * 9 + [False] * 3) * 4
    train_mask = np.tile([True] * 9 + [False] * 3, 4).astype(bool)
    X_train, y_train = X[train_mask], y[train_mask]
    X_full, y_full = X, y

    ext_train = FixedBandCSPFeatures(bands=FIXED_BANDS[:1], n_components=2, sfreq=160.0)
    ext_train.fit(X_train, y_train)

    ext_full = FixedBandCSPFeatures(bands=FIXED_BANDS[:1], n_components=2, sfreq=160.0)
    ext_full.fit(X_full, y_full)

    # Transform a shared test sample with both extractors
    X_test = X[30:]
    feats_train = ext_train.transform(X_test)
    feats_full = ext_full.transform(X_test)

    assert not np.allclose(feats_train, feats_full), (
        "CSP trained on train-only must differ from full-data CSP — possible leakage"
    )
