import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.snn.encoder import FixedThresholdEncoder


def test_threshold_set_from_data():
    """Threshold must equal the 75th percentile of |training data|."""
    enc = FixedThresholdEncoder(T=16)
    X = np.random.randn(100, 64).astype(np.float32)
    enc.set_threshold_from_data(X)
    expected = float(np.percentile(np.abs(X), 75))
    assert abs(enc.threshold - expected) < 1e-6


def test_output_shape():
    """Output spike tensor shape must be (T, batch, n_features)."""
    enc = FixedThresholdEncoder(threshold=0.0, T=16)
    X = torch.randn(32, 64)
    spikes = enc(X)
    assert spikes.shape == (16, 32, 64)


def test_binary_output():
    """All values in the spike tensor must be 0 or 1."""
    enc = FixedThresholdEncoder(threshold=0.0, T=8)
    spikes = enc(torch.randn(10, 32))
    assert set(torch.unique(spikes).tolist()).issubset({0.0, 1.0})


def test_not_learnable():
    """Baseline threshold must NOT be an nn.Parameter (Phase 3 makes it one)."""
    enc = FixedThresholdEncoder()
    param_names = [n for n, _ in enc.named_parameters()]
    assert "threshold" not in param_names
    assert len(param_names) == 0


def test_absolute_value_encoding():
    """Negative features MUST produce spikes if |feature| > threshold."""
    enc = FixedThresholdEncoder(threshold=0.5, T=4)
    X = torch.tensor([[-2.0, 0.1, -1.5, 0.3]])
    spikes = enc(X)
    # Feature 0 (-2.0): |-2.0| = 2.0 > 0.5 -> must fire
    assert spikes[:, 0, 0].all(), "Feature -2.0: |-2.0|=2.0 > 0.5, must spike"
    # Feature 1 (0.1): |0.1| = 0.1 < 0.5 -> must not fire
    assert not spikes[:, 0, 1].any(), "Feature 0.1: |0.1|=0.1 < 0.5, must not spike"
    # Feature 2 (-1.5): |-1.5| = 1.5 > 0.5 -> must fire
    assert spikes[:, 0, 2].all(), "Feature -1.5: |-1.5|=1.5 > 0.5, must spike"
    # Feature 3 (0.3): |0.3| = 0.3 < 0.5 -> must not fire
    assert not spikes[:, 0, 3].any(), "Feature 0.3: |0.3|=0.3 < 0.5, must not spike"


def test_spike_density_near_25_percent():
    """After P75 threshold from |X_train|, spike density should be ~25%."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((500, 128)).astype(np.float32)
    enc = FixedThresholdEncoder(T=16)
    enc.set_threshold_from_data(X_train)
    # Compute spike density on the training data itself
    spikes = enc(torch.as_tensor(X_train, dtype=torch.float32))
    density = spikes.mean().item()
    assert 0.15 < density < 0.35, (
        f"Expected ~25% spike density, got {density:.1%}"
    )
