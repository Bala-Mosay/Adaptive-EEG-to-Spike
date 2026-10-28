from typing import List, Optional
import numpy as np


class PerSubjectZScore:
    """Fit/transform per-subject z-score normalization.

    Usage:
    - Call `fit(X, subjects)` with training trials only.
    - Call `transform(X, subjects)` to normalize new trials.

    Notes: X is expected shape (n_trials, n_channels, n_timepoints).
    """

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, X: np.ndarray, subjects: List[str]) -> "PerSubjectZScore":
        subjects = list(subjects)
        for subj in set(subjects):
            idx = [i for i, s in enumerate(subjects) if s == subj]
            data = X[idx]
            self.means[subj] = data.mean(axis=(0, 2))
            self.stds[subj] = data.std(axis=(0, 2), ddof=0)
        return self

    def transform(self, X: np.ndarray, subjects: List[str]) -> np.ndarray:
        out = np.empty_like(X)
        for i, subj in enumerate(subjects):
            mean = self.means.get(subj)
            std = self.stds.get(subj)
            if mean is None or std is None:
                raise ValueError(f"Subject {subj} missing in fitted normalizer")
            # shape alignment: mean/std are per-channel
            out[i] = (X[i] - mean[None, :, None]) / (std[None, :, None] + 1e-12)
        return out

    def fit_transform(self, X: np.ndarray, subjects: List[str]) -> np.ndarray:
        self.fit(X, subjects)
        return self.transform(X, subjects)


class FeatureScaler:
    """Fit/transform flat z-score normalization over feature vectors.

    Used in Phase 2+ to standardize CSP/band log-variance feature matrices
    (shape ``(n_trials, n_features)``) using statistics computed EXCLUSIVELY
    on the training fold.

    Usage:
    - Call `fit(X_train)` with training trials only.
    - Call `transform(X)` on train/val/test feature matrices.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / (self.std_ + 1e-12)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
