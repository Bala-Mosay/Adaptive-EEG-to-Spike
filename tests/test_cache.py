import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.data.cache import InvalidCacheError, validate_preprocessed


def _valid():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 4, 20)).astype(np.float32)
    y = np.tile(np.arange(4), 3).astype(np.int64)
    return X, y


def test_valid_cache_accepted():
    X, y = _valid()
    X_out, y_out = validate_preprocessed(X, y, "S001")
    assert X_out.dtype == np.float32
    assert y_out.dtype == np.int64
    assert X_out.shape == X.shape


def test_all_zero_cache_rejected():
    X, y = _valid()
    with pytest.raises(InvalidCacheError, match="all-zero or constant"):
        validate_preprocessed(np.zeros_like(X), y, "S091")


def test_constant_cache_rejected():
    X, y = _valid()
    with pytest.raises(InvalidCacheError, match="all-zero or constant"):
        validate_preprocessed(np.ones_like(X), y, "S001")


def test_nan_cache_rejected():
    X, y = _valid()
    X[0, 0, 0] = np.nan
    with pytest.raises(InvalidCacheError, match="NaN or Inf"):
        validate_preprocessed(X, y, "S001")


def test_wrong_ndim_rejected():
    X, y = _valid()
    with pytest.raises(InvalidCacheError, match="3-D"):
        validate_preprocessed(X[..., 0], y, "S001")


def test_length_mismatch_rejected():
    X, y = _valid()
    with pytest.raises(InvalidCacheError, match="labels"):
        validate_preprocessed(X, y[:-1], "S001")


def test_out_of_range_labels_rejected():
    X, y = _valid()
    y = y.copy()
    y[0] = 99
    with pytest.raises(InvalidCacheError, match="outside"):
        validate_preprocessed(X, y, "S001")


def test_single_class_rejected():
    X, y = _valid()
    y = np.zeros_like(y)
    with pytest.raises(InvalidCacheError, match="fewer than two classes"):
        validate_preprocessed(X, y, "S001")
