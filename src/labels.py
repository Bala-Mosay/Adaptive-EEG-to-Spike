"""Canonical class-label constants shared across all pipeline stages.

The research paper (Survey_Paper.pdf, Section 6.1) defines a 4-class motor
imagery scheme: left_hand, right_hand, feet, rest. All Phase 2+ components
(balancer, evaluator, runner) index class labels 0-3 in the order given by
:data:`LABEL_ORDER`. This module is the single source of truth for that
ordering, so the F1-per-class column names and confusion-matrix axes are
always consistent.

These 0-indexed labels are INDEPENDENT of the MNE event codes used in Phase 1
(:data:`src.data.events.CLASS_CODES`, where rest=1, left_hand=2, right_hand=3,
feet=4). Use :func:`labels_to_indices` to convert Phase 1 string labels to
0-indexed integers.
"""
from typing import Dict, List

#: Canonical class order for all 0-indexed integer labels (Phase 2+).
LABEL_ORDER: List[str] = ["left_hand", "right_hand", "feet", "rest"]

#: 0-indexed integer code for each class label.
LABEL_INDICES: Dict[str, int] = {label: i for i, label in enumerate(LABEL_ORDER)}

#: Number of classes.
N_CLASSES: int = len(LABEL_ORDER)


def labels_to_indices(labels: List[str]) -> "List[int]":
    """Convert Phase 1 string labels to canonical 0-indexed integers.

    Parameters
    ----------
    labels : List[str]
        Class label per trial (e.g. from ``Preprocessor.process_subject``).

    Returns
    -------
    List[int]
        0-indexed integer label per trial, ordered per :data:`LABEL_ORDER`.

    Raises
    ------
    ValueError
        If a label is not one of the four canonical classes.
    """
    out = []
    for label in labels:
        if label not in LABEL_INDICES:
            raise ValueError(f"Unknown class label: {label!r} (expected one of {LABEL_ORDER})")
        out.append(LABEL_INDICES[label])
    return out
