"""
Leakage guard for Phase 2 pipeline: band features and z-score statistics
computed on a train fold must differ from those that would be computed on
the full dataset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.normalization import FeatureScaler
from src.features.bandpass import extract_band_features, FIXED_BANDS


def test_feature_scaler_fit_on_train_vs_full():
    """FeatureScaler fitted on train must differ from full-data fit."""
    rng = np.random.default_rng(42)
    X_full = rng.standard_normal((100, 64)).astype(np.float32)
    X_train = X_full[:80]
    X_test = X_full[80:]

    scaler_train = FeatureScaler().fit(X_train)
    scaler_full = FeatureScaler().fit(X_full)

    feats_train = scaler_train.transform(X_test)
    feats_full = scaler_full.transform(X_test)

    assert not np.allclose(feats_train, feats_full), (
        "FeatureScaler fitted on train == full: possible leakage"
    )
    assert not np.allclose(scaler_train.mean_, scaler_full.mean_, atol=1e-6)


def test_band_features_fold_independence():
    """Band features from two different train folds must differ."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((20, 8, 100)).astype(np.float32)
    y = np.array([0, 1, 2, 3] * 5, dtype=np.int64)

    # Fold 1: first 10, Fold 2: last 10
    train1 = np.arange(0, 10)
    train2 = np.arange(10, 20)

    feats1 = extract_band_features(X[train1], bands=FIXED_BANDS[:1])
    feats2 = extract_band_features(X[train2], bands=FIXED_BANDS[:1])

    mean1 = feats1.mean(axis=0)
    mean2 = feats2.mean(axis=0)

    assert not np.allclose(mean1, mean2, atol=1e-6), (
        "Band features identical across folds: unexpected"
    )
