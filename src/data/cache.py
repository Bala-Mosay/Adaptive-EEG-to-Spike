"""Validated per-subject preprocessing cache.

Phase 1 writes one compressed ``.npz`` per subject.  A cache can silently go
bad (all-zero arrays, NaNs, a truncated file).  This module validates every
cache before use, re-processes a subject once if the cache is invalid, and
writes replacements atomically.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.data.loader import load_subject_raw_with_events
from src.data.preprocessing import Preprocessor
from src.labels import LABEL_ORDER, labels_to_indices

log = logging.getLogger(__name__)


class InvalidCacheError(ValueError):
    """Raised when a preprocessed subject cache fails validation."""


def validate_preprocessed(
    X: np.ndarray,
    y: np.ndarray,
    subject_id: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate one subject's preprocessed trials.

    Parameters
    ----------
    X : np.ndarray
        Trials, shape ``(n_trials, n_channels, n_timepoints)``.
    y : np.ndarray
        Integer labels, shape ``(n_trials,)``.
    subject_id : str
        Used in error messages.

    Returns
    -------
    X : np.ndarray
        Float32 trials.
    y : np.ndarray
        Int64 labels.

    Raises
    ------
    InvalidCacheError
        If the arrays have the wrong shape, non-finite values, zero/constant
        variance, mismatched lengths, or out-of-range labels.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 3:
        raise InvalidCacheError(
            f"{subject_id}: X must be 3-D (trials, channels, time), got {X.shape}"
        )
    if y.ndim != 1:
        raise InvalidCacheError(f"{subject_id}: y must be 1-D, got {y.shape}")
    if len(X) != len(y):
        raise InvalidCacheError(
            f"{subject_id}: X has {len(X)} trials but y has {len(y)} labels"
        )
    if len(X) == 0:
        raise InvalidCacheError(f"{subject_id}: cache contains zero trials")
    if X.shape[1] == 0 or X.shape[2] == 0:
        raise InvalidCacheError(f"{subject_id}: empty channel/time axis {X.shape}")

    X = X.astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise InvalidCacheError(f"{subject_id}: X contains NaN or Inf")
    if not np.issubdtype(y.dtype, np.integer):
        raise InvalidCacheError(f"{subject_id}: y must be integer, got {y.dtype}")
    y = y.astype(np.int64, copy=False)

    # All-zero or constant data carries no class information and usually
    # signals a failed preprocessing run (S091 is the known historical case).
    finite_max = float(np.max(np.abs(X)))
    if finite_max == 0.0 or float(np.std(X)) == 0.0:
        raise InvalidCacheError(
            f"{subject_id}: X is all-zero or constant (max |x|={finite_max:g})"
        )

    if y.min() < 0 or y.max() >= len(LABEL_ORDER):
        raise InvalidCacheError(
            f"{subject_id}: labels outside [0, {len(LABEL_ORDER) - 1}]: "
            f"min={y.min()}, max={y.max()}"
        )
    if len(np.unique(y)) < 2:
        raise InvalidCacheError(f"{subject_id}: fewer than two classes present")

    return X, y


def load_or_process_subject(
    subject_id: str,
    preprocess_cfg: Dict,
    cache_dir: Path,
    *,
    force_reprocess: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a validated cache entry, or run Stage 1 and cache the result.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. ``"S001"``.
    preprocess_cfg : dict
        Phase 1 preprocessing config.
    cache_dir : Path
        Directory containing ``{subject_id}.npz``.
    force_reprocess : bool
        Ignore an existing cache and re-run Stage 1.

    Returns
    -------
    X : np.ndarray
        Float32 trials, shape ``(n_trials, n_channels, n_timepoints)``.
    y : np.ndarray
        Int64 labels, shape ``(n_trials,)``.
    """
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{subject_id}.npz"

    if cache_file.exists() and not force_reprocess:
        try:
            with np.load(cache_file) as blob:
                X = blob["X"]
                y = blob["y"]
            return validate_preprocessed(X, y, subject_id)
        except (InvalidCacheError, KeyError, OSError, ValueError) as exc:
            log.warning("Invalid cache for %s (%s); re-processing once", subject_id, exc)

    # Run Stage 1.  Copy the config so setting subject_id cannot mutate the
    # caller's dictionary.
    cfg = dict(preprocess_cfg)
    cfg["subject_id"] = subject_id
    raw, run_files, events, event_id = load_subject_raw_with_events(subject_id)
    pre = Preprocessor(cfg)
    X_raw, labels, _ = pre.process_subject(
        raw,
        run_files=run_files,
        events=events,
        event_id=event_id,
    )
    y_raw = np.asarray(labels_to_indices(labels), dtype=np.int64)
    X, y = validate_preprocessed(X_raw, y_raw, subject_id)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_name(f"{cache_file.stem}.tmp.npz")
    np.savez_compressed(tmp_file, X=X, y=y)
    tmp_file.replace(cache_file)
    log.info("Wrote validated cache %s (%d trials)", cache_file, len(y))
    return X, y
