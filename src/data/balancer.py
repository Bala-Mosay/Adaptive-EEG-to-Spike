"""Class balancing for the imbalanced 4-class EEGMMIDB scheme.

Phase 1 revealed a strong class imbalance per subject (e.g. S001:
left_hand=8, right_hand=7, feet=15, rest=42). Without balancing, any
classifier is biased toward the majority "rest" class. This module caps the
rest class down to the smallest motor-imagery (MI) class count per subject.

Note on leakage: balancing uses ONLY class counts (not feature statistics)
and is applied per subject before cross-validation splitting, so it does not
leak label-predictive information across folds.
"""
from typing import List, Optional, Tuple

import numpy as np

from src.labels import LABEL_ORDER


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "cap_to_min",
    random_state: int = 42,
    mi_classes: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance class trial counts by downsampling over-represented classes.

    Parameters
    ----------
    X : np.ndarray
        Trial data of shape ``(n_trials, ...)``. Only the first axis is
        indexed; remaining axes are passed through unchanged.
    y : np.ndarray
        Integer class label per trial, 0-indexed per
        :data:`src.labels.LABEL_ORDER` (0=left_hand, 1=right_hand, 2=feet,
        3=rest).
    strategy : str
        ``"cap_to_min"`` — cap EVERY class (including rest) to the size of
        the smallest MI class (left/right/feet). MI classes are never
        inflated.
    random_state : int
        Seed for reproducible random subsampling.
    mi_classes : tuple of int, optional
        Indices of the MI classes used to determine the cap. Defaults to
        ``(0, 1, 2)`` = (left_hand, right_hand, feet).

    Returns
    -------
    X_balanced : np.ndarray
        Balanced trials (sorted by original index for determinism).
    y_balanced : np.ndarray
        Balanced integer labels.
    keep_idx : np.ndarray
        Original indices of the kept trials.

    Raises
    ------
    ValueError
        If ``strategy`` is unknown or any MI class has zero trials.
    """
    if strategy != "cap_to_min":
        raise ValueError(f"Unknown balancing strategy: {strategy!r}")
    if mi_classes is None:
        mi_classes = (LABEL_ORDER.index("left_hand"), LABEL_ORDER.index("right_hand"), LABEL_ORDER.index("feet"))

    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    mi_counts = [int(np.sum(y == c)) for c in mi_classes]
    if min(mi_counts) == 0:
        raise ValueError(f"An MI class has zero trials; MI counts: {mi_counts}")
    cap = min(mi_counts)

    keep: List[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)

    keep_idx = np.sort(np.concatenate(keep))
    return X[keep_idx], y[keep_idx], keep_idx
