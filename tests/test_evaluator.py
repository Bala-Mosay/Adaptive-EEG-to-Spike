import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.snn.evaluator import compute_metrics
from src.labels import LABEL_ORDER, N_CLASSES


def test_accuracy_perfect():
    """Perfect predictions must yield accuracy=1, kappa=1, f1=1."""
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    result = compute_metrics(y, y, fold=0)
    assert abs(result.accuracy - 1.0) < 1e-6
    assert abs(result.kappa - 1.0) < 1e-6
    assert abs(result.f1_macro - 1.0) < 1e-6
    assert len(result.f1_per_class) == N_CLASSES
    for f1 in result.f1_per_class:
        assert abs(f1 - 1.0) < 1e-6


def test_accuracy_worst():
    """Worst-case predictions must yield accuracy < 0.5 and kappa < 0."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    y_pred = np.array([3, 3, 0, 0, 1, 1, 2, 2], dtype=np.int64)
    result = compute_metrics(y_true, y_pred, fold=0)
    assert result.accuracy < 0.5
    assert result.kappa < 0


def test_to_dict_has_correct_f1_keys():
    """to_dict must have f1_left_hand, f1_right_hand, f1_feet, f1_rest."""
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    result = compute_metrics(y, y, fold=0)
    d = result.to_dict()
    for label in LABEL_ORDER:
        assert f"f1_{label}" in d, f"Missing f1_{label} in to_dict"
