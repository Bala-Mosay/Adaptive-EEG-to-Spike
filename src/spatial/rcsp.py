from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf
from scipy.linalg import eigh


class RegularizedCSP(BaseEstimator, TransformerMixin):
    """Regularized CSP with Ledoit-Wolf shrinkage, extended to OvR for multi-class.

    Produces log-variance features per class (OvR) using the top `n_components` filters.

    # TODO(phase 2): Currently RegularizedCSP applies to the broadband signal (Stage 2).
    # In Stage 3 (Filter Bank CSP), RegularizedCSP will be instantiated fresh per sub-band
    # across a 7-band filter bank (4-32 Hz). The current class interface supports being called
    # independently for each sub-band without structural modifications.
    """

    def __init__(self, n_components: int = 4, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state
        self.filters_ = None
        self.classes_ = None

    def _cov(self, X: np.ndarray) -> np.ndarray:
        # X: (n_channels, n_time) -> X.T: (n_time, n_channels)
        # ledoit_wolf expects (n_samples, n_features) data matrix, NOT a
        # pre-computed covariance matrix.  Passing a (64,64) cov matrix
        # would set shrinkage=1.0 always, destroying all class structure.
        cov_shrunk, _ = ledoit_wolf(X.T)
        return cov_shrunk

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedCSP":
        # X: (n_trials, n_channels, n_time)
        self.classes_ = np.unique(y)
        self.filters_ = {}
        for cls in self.classes_:
            cls_idx = y == cls
            X1 = X[cls_idx]
            X2 = X[~cls_idx]
            S1 = np.mean([self._cov(tr) for tr in X1], axis=0)
            S2 = np.mean([self._cov(tr) for tr in X2], axis=0)
            S1 = (S1 + S1.T) / 2
            S2 = (S2 + S2.T) / 2
            eigvals, eigvecs = eigh(S1, S2)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            nc = self.n_components
            if nc % 2 == 0:
                picks = np.r_[0 : nc // 2, - (nc // 2) : 0]
            else:
                picks = np.r_[0 : (nc // 2) + 1, - (nc // 2) : 0]
            W = eigvecs[:, picks]
            self.filters_[str(cls)] = W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_trials = X.shape[0]
        feats = []
        for cls in self.classes_:
            W = self.filters_[str(cls)]
            # project each trial with W: use tensordot for clarity
            # X: (n_trials, n_channels, n_time), W: (n_channels, n_components)
            proj = np.tensordot(X, W, axes=([1], [0]))  # -> (n_trials, n_time, n_components)
            proj = np.transpose(proj, (0, 2, 1))  # -> (n_trials, n_components, n_time)
            var = proj.var(axis=2)
            logvar = np.log(var + 1e-10)
            feats.append(logvar)
        feats = np.concatenate(feats, axis=1)
        return feats
