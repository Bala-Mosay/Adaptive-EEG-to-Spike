import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.bands.bank import SEVEN_BANDS
from src.features.adaptive_bands import AdaptiveBandCSPFeatures


def _make_epochs(n_per_subject=40, n_ch=8, n_times=320, seed=0):
    """Synthetic epochs for two subjects, balanced across four classes."""
    rng = np.random.default_rng(seed)
    n = n_per_subject * 2
    X = rng.standard_normal((n, n_ch, n_times)).astype(np.float32)
    y = np.tile(np.arange(4), n // 4).astype(np.int64)
    subjects = np.array(["A"] * n_per_subject + ["B"] * n_per_subject)
    return X, y, subjects


def _split(X, y, subjects):
    train = np.r_[0:30, 40:70]
    test = np.r_[30:40, 70:80]
    return X[train], y[train], subjects[train], X[test], y[test], subjects[test]


def test_adaptive_features_output_shape_and_finiteness():
    X, y, subjects = _make_epochs()
    X_tr, y_tr, s_tr, X_te, _, s_te = _split(X, y, subjects)

    extractor = AdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=2,
        top_k=3,
        sfreq=160.0,
        n_neighbors=5,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)
    F_te = extractor.transform(X_te, s_te)

    assert F_te.shape == (20, 3 * 4 * 2)
    assert np.isfinite(F_te).all()
    assert extractor.n_features_out_ == 24


def test_adaptive_features_match_selected_blocks():
    """Each row must equal the full-bank row restricted to selected bands."""
    X, y, subjects = _make_epochs(seed=1)
    X_tr, y_tr, s_tr, X_te, _, s_te = _split(X, y, subjects)

    extractor = AdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=2,
        top_k=3,
        sfreq=160.0,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)

    F_all = extractor.base_.transform(X_te)
    F_te = extractor.transform(X_te, s_te)

    for subject in np.unique(s_te):
        mask = s_te == subject
        bands = extractor.selector_.get_bands(str(subject))
        cols = np.concatenate(
            [np.arange(F_all.shape[1])[extractor.band_slices_[b]] for b in bands]
        )
        assert np.allclose(F_te[mask], F_all[mask][:, cols])


def test_adaptive_features_selected_bands_per_subject():
    X, y, subjects = _make_epochs(seed=2)
    X_tr, y_tr, s_tr, _, _, _ = _split(X, y, subjects)

    extractor = AdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=2,
        top_k=3,
        sfreq=160.0,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)

    selected = extractor.get_selected_bands()
    assert set(selected) == {"A", "B"}
    for bands in selected.values():
        assert len(bands) == 3
        assert len(set(bands)) == 3
        assert set(bands).issubset({b[0] for b in SEVEN_BANDS})


def test_adaptive_features_unseen_subject_uses_global_bands():
    X, y, subjects = _make_epochs(seed=3)
    X_tr, y_tr, s_tr, X_te, _, _ = _split(X, y, subjects)

    extractor = AdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=2,
        top_k=3,
        sfreq=160.0,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)

    F_new = extractor.transform(X_te[:5], np.array(["UNSEEN"] * 5))
    assert F_new.shape == (5, 24)
    assert extractor.selector_.get_bands("UNSEEN") == extractor.selector_.global_bands_
