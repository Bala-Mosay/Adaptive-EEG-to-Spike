"""Evaluation metrics for the baseline SNN (accuracy, kappa, F1, SynOps).

Metrics follow the research paper's evaluation protocol (Section 6.3):
classification accuracy, Cohen's Kappa, F1 per class, and the SynOps energy
proxy. Per-class F1 columns are always ordered per
:data:`src.labels.LABEL_ORDER` (left_hand, right_hand, feet, rest).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from src.labels import LABEL_ORDER, N_CLASSES


@dataclass
class EvalResult:
    """All evaluation metrics for one cross-validation fold."""

    fold: int
    accuracy: float
    kappa: float
    f1_per_class: List[float]
    f1_macro: float
    confusion_matrix: np.ndarray
    synops_per_trial: float
    spike_density: float
    n_trials_test: int

    def to_dict(self) -> Dict[str, object]:
        """Flatten to a JSON/CSV-friendly dict (confusion matrix excluded)."""
        d: Dict[str, object] = {
            "fold": self.fold,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "f1_macro": self.f1_macro,
            "synops_per_trial": self.synops_per_trial,
            "spike_density": self.spike_density,
            "n_trials_test": self.n_trials_test,
        }
        for i, label in enumerate(LABEL_ORDER):
            d[f"f1_{label}"] = self.f1_per_class[i]
        return d


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold: int,
    synops_per_trial: float = 0.0,
    spike_density: float = 0.0,
) -> EvalResult:
    """Compute all evaluation metrics for one fold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0-indexed per :data:`src.labels.LABEL_ORDER`).
    y_pred : np.ndarray
        Predicted labels (same indexing).
    fold : int
        CV fold index (for logging).
    synops_per_trial : float
        SynOps count from :mod:`src.snn.synops`.
    spike_density : float
        Average spike density from :mod:`src.snn.synops`.

    Returns
    -------
    EvalResult
    """
    labels = list(range(N_CLASSES))
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1_per = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvalResult(
        fold=fold,
        accuracy=float(acc),
        kappa=float(kappa),
        f1_per_class=f1_per,
        f1_macro=float(f1_mac),
        confusion_matrix=cm,
        synops_per_trial=float(synops_per_trial),
        spike_density=float(spike_density),
        n_trials_test=len(y_true),
    )
