import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.bands.bank import SEVEN_BANDS
from src.bands.knn_mi import band_mi_scores
from src.bands.selector import AdaptiveBandSelector


def _class_dependent_block(n: int, y: np.ndarray, strength: float, seed: int):
    """Noise block with a class-dependent shift in one column."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    X[np.arange(n), y] += strength
    return X


def _make_subject_bands(
    n_per_class: int = 15,
    informative_band: str = "theta",
    strength: float = 6.0,
    seed: int = 0,
):
    """Build 7-band features for one synthetic subject."""
    n = n_per_class * 4
    y = np.tile(np.arange(4), n_per_class)
    bands = {}
    for i, (name, _, _) in enumerate(SEVEN_BANDS):
        if name == informative_band:
            bands[name] = _class_dependent_block(n, y, strength, seed + i)
        else:
            rng = np.random.default_rng(seed + 100 + i)
            bands[name] = rng.standard_normal((n, 4))
    return bands, y


def test_band_mi_ranks_informative_band_above_noise():
    """A class-dependent band must score above a pure-noise band."""
    n = 160
    y = np.tile(np.arange(4), n // 4)
    informative = _class_dependent_block(n, y, strength=6.0, seed=1)
    rng = np.random.default_rng(2)
    noise = rng.standard_normal((n, 4))

    scores = band_mi_scores(
        {"informative": informative, "noise": noise},
        y,
        n_neighbors=5,
        random_state=0,
    )

    assert scores["informative"] > scores["noise"]
    assert scores["informative"] > 0.0


def test_band_mi_constant_band_scores_zero():
    """A constant band carries no label information."""
    n = 40
    y = np.tile(np.arange(4), n // 4)
    scores = band_mi_scores(
        {"constant": np.ones((n, 3)), "noise": np.random.default_rng(0).standard_normal((n, 3))},
        y,
        n_neighbors=5,
        random_state=0,
    )
    assert scores["constant"] == 0.0


def test_band_mi_is_deterministic():
    """Repeated scoring with the same seed must be identical."""
    n = 80
    y = np.tile(np.arange(4), n // 4)
    X = _class_dependent_block(n, y, strength=4.0, seed=3)
    first = band_mi_scores({"b": X}, y, n_neighbors=5, random_state=7)
    second = band_mi_scores({"b": X}, y, n_neighbors=5, random_state=7)
    assert first == second


def test_band_mi_requires_two_classes():
    with pytest.raises(ValueError, match="at least two classes"):
        band_mi_scores({"b": np.zeros((10, 2))}, np.zeros(10, dtype=int))


def test_selector_picks_subject_specific_bands():
    """Subject A should keep theta; subject B should keep gamma."""
    bands_a, y_a = _make_subject_bands(
        informative_band="theta", strength=6.0, seed=10
    )
    bands_b, y_b = _make_subject_bands(
        informative_band="gamma", strength=6.0, seed=20
    )

    band_features = {
        name: np.concatenate([bands_a[name], bands_b[name]], axis=0)
        for name, _, _ in SEVEN_BANDS
    }
    y = np.concatenate([y_a, y_b])
    subjects = np.array(["A"] * len(y_a) + ["B"] * len(y_b))

    selector = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(band_features, y, subjects)

    bands_for_a = selector.get_bands("A")
    bands_for_b = selector.get_bands("B")

    assert "theta" in bands_for_a
    assert "gamma" in bands_for_b
    assert bands_for_a != bands_for_b
    assert len(bands_for_a) == 3
    assert len(bands_for_b) == 3


def test_selector_falls_back_for_unseen_subject():
    bands, y = _make_subject_bands(informative_band="mu", seed=30)
    subjects = np.array(["S1"] * len(y))
    selector = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=20, random_state=0
    ).fit(bands, y, subjects)

    assert selector.get_bands("S999") == selector.global_bands_


def test_selector_falls_back_when_trials_too_few():
    bands, y = _make_subject_bands(informative_band="mu", seed=40)
    subjects = np.array(["S1"] * len(y))
    selector = AdaptiveBandSelector(
        top_k=3, n_neighbors=5, min_trials=1000, random_state=0
    ).fit(bands, y, subjects)

    assert selector.selected_bands_["S1"] == selector.global_bands_


def test_selector_rejects_bad_top_k():
    bands, y = _make_subject_bands(seed=50)
    subjects = np.array(["S1"] * len(y))
    with pytest.raises(ValueError, match="top_k"):
        AdaptiveBandSelector(top_k=99).fit(bands, y, subjects)
