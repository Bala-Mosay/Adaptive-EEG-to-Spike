"""Guards for the soft adaptive band-weighting modules (revised C1).

Hard top-k selection deletes bands, so it can only lose information.  Soft
weighting keeps every band and scales its blocks instead, which makes the
unweighted full bank (``all7``) an exact special case.  These tests fail if
that nesting breaks, if the weights stop being mean-one/positive, if the
ranking stops driving the weighting, or if anything is fit on held-out data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from screen_band_conditions import apply_band_weights, band_slices  # noqa: E402
from src.bands.bank import SEVEN_BANDS  # noqa: E402
from src.bands.soft_weights import SoftBandWeighter  # noqa: E402
from src.features.bandpass import FixedBandCSPFeatures  # noqa: E402
from src.features.soft_adaptive_bands import (  # noqa: E402
    SoftAdaptiveBandCSPFeatures,
)

N_COMPONENTS = 2
BLOCK = 4 * N_COMPONENTS
BAND_NAMES = [name for name, _, _ in SEVEN_BANDS]


def _band_features(
    n_per_subject: int = 300, n_components: int = 4, informative: str = "mu"
) -> tuple:
    """Synthetic per-band blocks with a class signal planted in one band."""
    rng = np.random.default_rng(0)
    y = np.tile(np.arange(4), n_per_subject // 4)
    features = {}
    for name in BAND_NAMES:
        noise = rng.standard_normal((len(y), n_components))
        if name == informative:
            # Class-dependent offset: high mutual information with y.
            offset = (y[:, None] * 1.5).astype(float)
            features[name] = noise + offset
        else:
            features[name] = noise
    return features, y


def _synthetic_eeg() -> tuple:
    """Small EEG-like epochs with two subjects, split into train/test."""
    rng = np.random.default_rng(1)
    n_per_subject = 40
    n = n_per_subject * 2
    X = rng.standard_normal((n, 8, 320)).astype(np.float32)
    y = np.tile(np.arange(4), n // 4).astype(np.int64)
    subjects = np.array(["A"] * n_per_subject + ["B"] * n_per_subject)
    train = np.r_[0:30, 40:70]
    test = np.r_[30:40, 70:80]
    return X[train], y[train], subjects[train], X[test], subjects[test]


# --------------------------------------------------------------------------
# Nesting: lambda=0 + uniform prior must be the `all7` control, bitwise.
# --------------------------------------------------------------------------
def test_lambda_zero_uniform_reproduces_all7_bitwise():
    X_tr, y_tr, s_tr, X_te, s_te = _synthetic_eeg()

    reference = FixedBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=N_COMPONENTS,
        sfreq=160.0,
        random_state=0,
    ).fit(X_tr, y_tr)
    F_ref = reference.transform(X_te)

    soft = SoftAdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=N_COMPONENTS,
        sfreq=160.0,
        lambda_=0.0,
        shrink_to="uniform",
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)
    F_soft = soft.transform(X_te, s_te)

    assert F_soft.shape == F_ref.shape
    # Not approximately: the control condition must be a strict special case.
    assert np.array_equal(F_soft, F_ref)


def test_apply_band_weights_with_unit_weights_is_identity():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((50, len(BAND_NAMES) * BLOCK))
    subjects = np.tile(["A", "B"], 25)
    slices = band_slices(BAND_NAMES, BLOCK)
    weights = {subject: np.ones(len(BAND_NAMES)) for subject in ("A", "B")}

    out = apply_band_weights(F, subjects, weights, slices, BAND_NAMES)
    assert np.array_equal(out, F)


# --------------------------------------------------------------------------
# Weight shape: positive, mean-one, and a convex combination of prior/softmax.
# --------------------------------------------------------------------------
def test_weights_are_positive_and_mean_one():
    features, y = _band_features()
    subjects = np.array(["A"] * len(y))

    weighter = SoftBandWeighter(lambda_=0.5, random_state=0).fit(
        features, y, subjects
    )
    weights = weighter.get_weights("A")

    assert weights.shape == (len(BAND_NAMES),)
    assert np.all(weights > 0.0)
    assert np.isclose(weights.mean(), 1.0, atol=1e-9)
    # Mean-one weights must not scale the feature matrix up or down globally.
    assert np.isclose(weights.sum(), len(BAND_NAMES), atol=1e-9)


def test_weights_lie_between_prior_and_softmax():
    features, y = _band_features()
    subjects = np.array(["A"] * len(y))
    lam = 0.4

    weighter = SoftBandWeighter(
        lambda_=lam, shrink_to="uniform", random_state=0
    ).fit(features, y, subjects)
    combined = weighter.get_weights("A")
    soft = weighter._soft_weights(weighter.scores_["A"])
    prior = np.ones_like(soft)

    expected = (1.0 - lam) * prior + lam * soft
    assert np.allclose(combined, expected)
    # A convex combination can never leave the interval spanned by its inputs.
    assert np.all(combined >= np.minimum(prior, soft) - 1e-12)
    assert np.all(combined <= np.maximum(prior, soft) + 1e-12)


def test_lambda_one_puts_most_weight_on_the_informative_band():
    features, y = _band_features(informative="mu")
    subjects = np.array(["A"] * len(y))

    weighter = SoftBandWeighter(
        lambda_=1.0, shrink_to="uniform", temperature=1.0, random_state=0
    ).fit(features, y, subjects)
    weights = weighter.get_weights("A")

    assert BAND_NAMES[int(np.argmax(weights))] == "mu"
    assert weights.max() > weights.min()


def test_lambda_zero_ignores_the_ranking():
    features, y = _band_features(informative="mu")
    subjects = np.array(["A"] * len(y))

    weighter = SoftBandWeighter(lambda_=0.0, shrink_to="uniform", random_state=0)
    weighter.fit(features, y, subjects)

    assert np.allclose(weighter.get_weights("A"), 1.0)


# --------------------------------------------------------------------------
# Shrinkage targets.
# --------------------------------------------------------------------------
def test_global_prior_is_shared_across_subjects_at_lambda_zero():
    features, y = _band_features(n_per_subject=160)
    subjects = np.array(["A", "B"] * (len(y) // 2))

    weighter = SoftBandWeighter(lambda_=0.0, shrink_to="global", random_state=0)
    weighter.fit(features, y, subjects)

    weights_a = weighter.get_weights("A")
    weights_b = weighter.get_weights("B")
    assert np.allclose(weights_a, weights_b)
    assert np.allclose(weights_a, weighter.global_weights_)
    # The global prior is peaked, not uniform: it is derived from the scores.
    assert not np.allclose(weights_a, 1.0)
    assert np.isclose(weights_a.mean(), 1.0, atol=1e-9)


def test_small_sample_subject_falls_back_to_the_prior():
    features, y = _band_features(n_per_subject=200)
    # Subject B has only 8 trials, below ``min_trials``.
    subjects = np.array(["A"] * 192 + ["B"] * 8)

    weighter = SoftBandWeighter(
        lambda_=0.7, shrink_to="uniform", min_trials=20, random_state=0
    ).fit(features, y, subjects)

    assert np.allclose(weighter.get_weights("B"), weighter.prior_)
    assert not np.allclose(weighter.get_weights("A"), weighter.prior_)


def test_unseen_subject_gets_the_global_weights():
    features, y = _band_features(n_per_subject=160)
    subjects = np.array(["A"] * 80 + ["B"] * 80)

    weighter = SoftBandWeighter(lambda_=0.5, shrink_to="global", random_state=0)
    weighter.fit(features, y, subjects)

    assert np.allclose(
        weighter.get_weights("never-seen"), weighter.global_weights_
    )


# --------------------------------------------------------------------------
# Leakage, determinism, and validation.
# --------------------------------------------------------------------------
def test_fit_is_deterministic():
    features, y = _band_features(n_per_subject=160)
    subjects = np.array(["A", "B"] * 80)

    first = SoftBandWeighter(lambda_=0.5, random_state=7).fit(
        features, y, subjects
    )
    second = SoftBandWeighter(lambda_=0.5, random_state=7).fit(
        features, y, subjects
    )

    assert np.array_equal(first.get_weights("A"), second.get_weights("A"))
    assert np.array_equal(first.get_weights("B"), second.get_weights("B"))


def test_weights_depend_on_the_fitted_data_only():
    """Different training folds must produce different fitted weights.

    This is the inverse of the usual leakage test: if a parameter were
    estimated once outside cross-validation, the weights would be identical
    across folds.  Here we require them to differ, proving the weighter is
    actually re-fitted per fold.
    """
    features, y = _band_features(n_per_subject=200)
    subjects = np.array(["A"] * 200)

    full = SoftBandWeighter(lambda_=0.8, random_state=0).fit(
        features, y, subjects
    )
    half = SoftBandWeighter(lambda_=0.8, random_state=0).fit(
        {name: values[:100] for name, values in features.items()},
        y[:100],
        subjects[:100],
    )

    assert not np.allclose(full.get_weights("A"), half.get_weights("A"))


def test_feature_transform_uses_frozen_weights():
    """Test trials must not influence the weights used to transform them."""
    X_tr, y_tr, s_tr, X_te, s_te = _synthetic_eeg()

    soft = SoftAdaptiveBandCSPFeatures(
        bands=SEVEN_BANDS,
        n_components=N_COMPONENTS,
        sfreq=160.0,
        lambda_=0.6,
        min_trials=10,
        random_state=0,
    ).fit(X_tr, y_tr, s_tr)
    before = {key: weights.copy() for key, weights in soft.get_weights().items()}

    soft.transform(X_te, s_te)
    after = soft.get_weights()

    assert set(before) == set(after)
    for subject in before:
        assert np.array_equal(before[subject], np.asarray(after[subject]))


def test_invalid_hyperparameters_raise():
    features, y = _band_features(n_per_subject=40)
    subjects = np.array(["A"] * 40)

    with pytest.raises(ValueError):
        SoftBandWeighter(lambda_=1.5).fit(features, y, subjects)
    with pytest.raises(ValueError):
        SoftBandWeighter(temperature=0.0).fit(features, y, subjects)
    with pytest.raises(ValueError):
        SoftBandWeighter(shrink_to="median").fit(features, y, subjects)
    with pytest.raises(RuntimeError):
        SoftBandWeighter(lambda_=0.3).get_weights("A")
