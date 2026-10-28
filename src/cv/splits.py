from typing import Iterator, Tuple
import numpy as np
from sklearn.model_selection import KFold


def subject_mixed_kfold(n_splits: int = 5, random_state: int = 42) -> KFold:
    """Return a KFold object configured for subject-mixed splitting.

    Consumer should call `split(X)` on the returned KFold object.
    """
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def loso_split(subjects: list) -> Iterator[Tuple[list, list]]:
    """Yield train/test subject lists for Leave-One-Subject-Out.

    This is a stub implementing the generator signature; actual usage will
    map subjects back to trial indices externally.
    """
    for i, subj in enumerate(subjects):
        train = [s for s in subjects if s != subj]
        test = [subj]
        yield train, test
