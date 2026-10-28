import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data.balancer import balance_classes
from src.labels import LABEL_ORDER


def _make_trials(counts: dict, n_ch: int = 4, n_times: int = 100, seed: int = 0):
    """Create synthetic trials (n_trials, n_ch, n_times) + 0-indexed labels."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, n in counts.items():
        idx = LABEL_ORDER.index(label)
        X.append(rng.standard_normal((n, n_ch, n_times)).astype(np.float32))
        y.append(np.full(n, idx, dtype=np.int64))
    return np.concatenate(X), np.concatenate(y)


def test_rest_is_capped():
    """Rest class must be capped to the smallest MI class count."""
    X, y = _make_trials({"left_hand": 8, "right_hand": 7, "feet": 15, "rest": 42})
    _, yb, _ = balance_classes(X, y, random_state=42)
    counts = {LABEL_ORDER[c]: int(np.sum(yb == c)) for c in np.unique(yb)}
    assert counts["rest"] <= min(counts["left_hand"], counts["right_hand"], counts["feet"])


def test_mi_classes_not_inflated():
    """MI classes must never be inflated beyond their original size."""
    X, y = _make_trials({"left_hand": 8, "right_hand": 7, "feet": 15, "rest": 42})
    _, yb, _ = balance_classes(X, y)
    assert int(np.sum(yb == LABEL_ORDER.index("right_hand"))) <= 7
    assert int(np.sum(yb == LABEL_ORDER.index("left_hand"))) <= 8


def test_balanced_total_and_determinism():
    """Total trials = 4 * min_MI_count; repeated calls give identical subsets."""
    X, y = _make_trials({"left_hand": 8, "right_hand": 7, "feet": 15, "rest": 42})
    Xb1, yb1, idx1 = balance_classes(X, y, random_state=42)
    Xb2, yb2, idx2 = balance_classes(X, y, random_state=42)
    assert len(yb1) == 4 * 7
    assert np.array_equal(idx1, idx2)
    assert np.array_equal(Xb1, Xb2)
    assert set(np.unique(yb1)) == {0, 1, 2, 3}


def test_unknown_strategy_raises():
    X, y = _make_trials({"left_hand": 2, "right_hand": 2, "feet": 2, "rest": 2})
    with pytest.raises(ValueError):
        balance_classes(X, y, strategy="oversample")
