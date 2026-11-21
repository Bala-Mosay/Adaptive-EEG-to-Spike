"""Guards for the band-condition screening harness.

The screen is only useful if it computes exactly what the production pipeline
computes.  These tests fail if the single-pass extractor, the per-subject
column selection, or the guarded selection ever diverge from
``FixedBandCSPFeatures`` / ``AdaptiveBandCSPFeatures``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_band_conditions import (  # noqa: E402
    CACHE_KEYS,
    PairBandCSPExtractor,
    band_slices,
    columns_for,
    guarded_selection,
    load_fold_cache,
    per_subject_band_center,
    per_subject_band_zscore,
    per_subject_matrix,
    permuted_weights,
    save_fold_cache,
)
from src.bands.bank import SEVEN_BANDS  # noqa: E402
from src.features.adaptive_bands import AdaptiveBandCSPFeatures  # noqa: E402
from src.features.bandpass import FixedBandCSPFeatures  # noqa: E402

N_COMPONENTS = 2
BLOCK = 4 * N_COMPONENTS
BAND_NAMES = [name for name, _, _ in SEVEN_BANDS]


def _synthetic():
    rng = np.random.default_rng(0)
    n_per_subject = 40
    n = n_per_subject * 2
    X = rng.standard_normal((n, 8, 320)).astype(np.float32)
    y = np.tile(np.arange(4), n // 4).astype(np.int64)
    subjects = np.array(["A"] * n_per_subject + ["B"] * n_per_subject)
    train = np.r_[0:30, 40:70]
    test = np.r_[30:40, 70:80]
    return (
        X[train], y[train], subjects[train], X[test], subjects[test]
    )


def test_pair_extractor_matches_fit_then_transform():
    """Single-pass features must equal the production fit/transform path."""
    X_tr, y_tr, _, X_te, _ = _synthetic()

    reference = FixedBandCSPFeatures(
        bands=SEVEN_BANDS, n_components=N_COMPONENTS, sfreq=160.0, random_state=0
    ).fit(X_tr, y_tr)
    F_tr_ref = reference.transform(X_tr)
    F_te_ref = reference.transform(X_te)

    pair = PairBandCSPExtractor(SEVEN_BANDS, N_COMPONENTS, 160.0, 0)
    F_tr, F_te = pair.fit_transform_pair(X_tr, y_tr, X_te)

    assert F_tr.shape == F_tr_ref.shape
    assert F_te.shape == F_te_ref.shape
    assert np.array_equal(F_tr, F_tr_ref)
    assert np.array_equal(F_te, F_te_ref)


def test_screen_selection_matches_adaptive_features():
    """Screen column picking must reproduce AdaptiveBandCSPFeatures exactly."""
    X_tr, y_tr, s_tr, X_te, s_te = _synthetic()

    adaptive = AdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=N_COMPONENTS,
        top_k=3,
        sfreq=160.0,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)
    F_ref = adaptive.transform(X_te, s_te)

    base = FixedBandCSPFeatures(
        bands=SEVEN_BANDS, n_components=N_COMPONENTS, sfreq=160.0, random_state=0
    ).fit(X_tr, y_tr)
    F_all = base.transform(X_te)
    slices = band_slices([name for name, _, _ in base.bands_], BLOCK)
    selection = {
        str(subject): adaptive.selector_.get_bands(str(subject))
        for subject in np.unique(s_te)
    }
    F_screen = per_subject_matrix(F_all, s_te, selection, slices)

    assert np.array_equal(F_ref, F_screen)


def test_guarded_selection_pins_sensorimotor_bands():
    """mu and beta must survive regardless of the MI ranking."""
    X_tr, y_tr, s_tr, _, _ = _synthetic()

    adaptive = AdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=N_COMPONENTS,
        top_k=3,
        sfreq=160.0,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)

    guarded = guarded_selection(adaptive.selector_, ["A", "B"], k=3)
    bank_order = [name for name, _, _ in SEVEN_BANDS]
    for subject, bands in guarded.items():
        assert len(bands) == 3, f"{subject} kept {len(bands)} bands"
        assert len(set(bands)) == 3
        assert "mu" in bands and "beta" in bands
        # Output must stay in filter-bank order so block positions match.
        assert bands == [name for name in bank_order if name in bands]


def test_columns_for_matches_slice_bounds():
    """Fixed band subsets must map to contiguous filter-bank blocks."""
    slices = band_slices([name for name, _, _ in SEVEN_BANDS], BLOCK)
    cols = columns_for(["mu", "beta"], slices)
    expected = np.r_[slices["mu"], slices["beta"]]
    assert np.array_equal(cols, expected)


# --------------------------------------------------------------------------
# Fold feature cache
# --------------------------------------------------------------------------
def _cache_arrays() -> dict:
    rng = np.random.default_rng(3)
    n_train, n_test = 30, 12
    return {
        "F_train_all": rng.standard_normal((n_train, 7 * BLOCK)).astype(np.float32),
        "F_test_all": rng.standard_normal((n_test, 7 * BLOCK)).astype(np.float32),
        "F_broad_train": rng.standard_normal((n_train, 2 * BLOCK)),
        "F_broad_test": rng.standard_normal((n_test, 2 * BLOCK)),
        "y_train": np.tile(np.arange(4), n_train // 4).astype(np.int64),
        "y_test": np.tile(np.arange(4), n_test // 4).astype(np.int64),
        "subj_train": np.array(["A"] * 15 + ["B"] * 15),
        "subj_test": np.array(["A"] * 6 + ["B"] * 6),
    }


def test_cache_round_trip_is_bitwise_identical(tmp_path):
    """Cached features must be reused exactly, or the screen is not the screen."""
    arrays = _cache_arrays()
    path = tmp_path / "fold0.npz"

    save_fold_cache(path, arrays)
    loaded = load_fold_cache(path)

    assert set(loaded) == set(CACHE_KEYS)
    for key in CACHE_KEYS:
        assert loaded[key].dtype == arrays[key].dtype, key
        assert loaded[key].shape == arrays[key].shape, key
        assert np.array_equal(loaded[key], arrays[key]), key


def test_cache_round_trip_preserves_float32(tmp_path):
    """A dtype upcast on save would silently change the screened numbers."""
    arrays = _cache_arrays()
    path = tmp_path / "fold0_dtype.npz"

    save_fold_cache(path, arrays)
    loaded = load_fold_cache(path)
    assert loaded["F_train_all"].dtype == np.float32


# --------------------------------------------------------------------------
# Ablations: per-subject band calibration with no mutual information
# --------------------------------------------------------------------------
def _subject_matrices():
    rng = np.random.default_rng(4)
    n_cols = 7 * BLOCK
    subj_train = np.array(["A"] * 20 + ["B"] * 20)
    subj_test = np.array(["A"] * 10 + ["B"] * 10)
    return (
        rng.standard_normal((40, n_cols)),
        rng.standard_normal((20, n_cols)),
        subj_train,
        subj_test,
    )


def test_subject_band_center_zeroes_train_block_means():
    F_train, F_test, subj_train, subj_test = _subject_matrices()
    slices = band_slices(BAND_NAMES, BLOCK)

    out_train, out_test = per_subject_band_center(
        F_train, F_test, subj_train, subj_test, slices, BAND_NAMES
    )

    for subject in ("A", "B"):
        train_mask = subj_train == subject
        test_mask = subj_test == subject
        for name in BAND_NAMES:
            block = slices[name]
            assert np.isclose(out_train[train_mask, block].mean(), 0.0, atol=1e-12)
            # Test trials are shifted by the *training* mean of that subject.
            expected = F_test[test_mask, block] - F_train[train_mask, block].mean()
            assert np.allclose(out_test[test_mask, block], expected)


def test_subject_band_center_ignores_test_trials():
    """The offsets must be a function of training trials only."""
    F_train, F_test, subj_train, subj_test = _subject_matrices()
    slices = band_slices(BAND_NAMES, BLOCK)

    _, out_test = per_subject_band_center(
        F_train, F_test, subj_train, subj_test, slices, BAND_NAMES
    )
    _, out_test_modified = per_subject_band_center(
        F_train, F_test + 100.0, subj_train, subj_test, slices, BAND_NAMES
    )

    # Shifting the test set must shift the output by exactly the same amount.
    assert np.allclose(out_test_modified - out_test, 100.0)


def test_subject_band_zscore_standardizes_train_trials():
    F_train, F_test, subj_train, subj_test = _subject_matrices()
    slices = band_slices(BAND_NAMES, BLOCK)

    out_train, out_test = per_subject_band_zscore(
        F_train, F_test, subj_train, subj_test, slices, BAND_NAMES
    )

    for subject in ("A", "B"):
        train_mask = subj_train == subject
        test_mask = subj_test == subject
        for name in BAND_NAMES:
            block = slices[name]
            standardized = out_train[train_mask, block]
            assert np.allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
            assert np.allclose(standardized.std(axis=0), 1.0, atol=1e-9)
            mean = F_train[train_mask, block].mean(axis=0)
            std = F_train[train_mask, block].std(axis=0)
            expected = (F_test[test_mask, block] - mean) / std
            assert np.allclose(out_test[test_mask, block], expected)


# --------------------------------------------------------------------------
# Shuffled-ranking control
# --------------------------------------------------------------------------
def test_permuted_weights_preserve_the_weight_distribution():
    rng = np.random.default_rng(5)
    weights = {
        "A": rng.standard_normal(7),
        "B": rng.standard_normal(7),
    }

    shuffled = permuted_weights(weights, ["A", "B"], seed=0)

    assert set(shuffled) == {"A", "B"}
    for subject in ("A", "B"):
        assert np.allclose(np.sort(shuffled[subject]), np.sort(weights[subject]))
    # With distinct weights the permutation must actually reorder them.
    assert not np.allclose(shuffled["A"], weights["A"])
    assert not np.allclose(shuffled["B"], weights["B"])


def test_permuted_weights_are_deterministic():
    rng = np.random.default_rng(6)
    weights = {"A": rng.standard_normal(7), "B": rng.standard_normal(7)}

    first = permuted_weights(weights, ["A", "B"], seed=42)
    second = permuted_weights(weights, ["B", "A"], seed=42)

    # Order of the subject list must not change the result.
    for subject in ("A", "B"):
        assert np.array_equal(first[subject], second[subject])
