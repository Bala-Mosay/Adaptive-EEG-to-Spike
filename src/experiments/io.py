from typing import List, Tuple
import numpy as np
import os


def load_outputs_for_subject(subject_dir: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load `epochs.npz` and `metadata.csv` from a subject output directory.

    Returns (X, labels, subjects) where:
      - X: (n_trials, n_channels, n_time)
      - labels: list of string labels length n_trials
      - subjects: list of subject ids length n_trials
    """
    npz_path = os.path.join(subject_dir, 'epochs.npz')
    meta_path = os.path.join(subject_dir, 'metadata.csv')
    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing outputs in {subject_dir}")
    data = np.load(npz_path)
    X = data['X']
    labels = []
    subjects = []
    with open(meta_path, 'r', encoding='utf8') as fh:
        next(fh)
        for line in fh:
            subj, lab, idx = line.strip().split(',')
            labels.append(lab)
            subjects.append(subj)
    return X, labels, subjects


def assemble_dataset(subject_dirs: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Combine outputs from multiple subjects into X, y, subjects lists.

    Labels are mapped to integer classes by order of appearance.
    """
    Xs = []
    labs = []
    subs = []
    for sd in subject_dirs:
        X, labels, subjects = load_outputs_for_subject(sd)
        Xs.append(X)
        labs.extend(labels)
        subs.extend(subjects)
    if not Xs:
        raise ValueError("No subject outputs provided")
    X_all = np.concatenate(Xs, axis=0)
    # map labels to ints
    unique = sorted(list(dict.fromkeys(labs)))
    mapping = {lab: i for i, lab in enumerate(unique)}
    y = np.array([mapping[l] for l in labs])
    return X_all, y, subs
