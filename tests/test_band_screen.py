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
    PairBandCSPExtractor,
    band_slices,
    columns_for,
    guarded_selection,
    per_subject_matrix,
)
from src.bands.bank import SEVEN_BANDS  # noqa: E402
from src.features.adaptive_bands import AdaptiveBandCSPFeatures  # noqa: E402
from src.features.bandpass import FixedBandCSPFeatures  # noqa: E402

N_COMPONENTS = 2
BLOCK = 4 * N_COMPONENTS


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
