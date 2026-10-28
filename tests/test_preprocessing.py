import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mne
import pytest
from src.data.preprocessing import Preprocessor


def test_bandpass_filter():
    """Test that bandpass filtering attenuates frequencies outside 0.5-40 Hz."""
    sfreq = 160.0
    t = np.arange(0, 10, 1 / sfreq)
    # Signal with 10 Hz (in-band) + 60 Hz (out-of-band)
    signal = np.sin(2 * np.pi * 10 * t) + 1.0 * np.sin(2 * np.pi * 60 * t)
    info = mne.create_info(ch_names=["CZ"], sfreq=sfreq, ch_types="eeg")

    # Compute original FFT before filtering (and copy array to prevent in-place mutation of signal)
    fft_orig = np.abs(np.fft.rfft(signal))
    raw = mne.io.RawArray(signal[np.newaxis, :].copy(), info, verbose=False)

    cfg = {
        "bandpass": {
            "l_freq": 0.5,
            "h_freq": 40.0,
            "method": "iir",
            "iir_params": {"order": 4, "ftype": "butter"},
        }
    }
    prep = Preprocessor(cfg)
    filtered_raw = prep.bandpass(raw)
    data = filtered_raw.get_data()[0]

    # Power spectrum comparison via FFT
    fft_filt = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(len(t), 1 / sfreq)

    idx_10hz = np.argmin(np.abs(freqs - 10.0))
    idx_60hz = np.argmin(np.abs(freqs - 60.0))

    ratio_10hz = float(fft_filt[idx_10hz] / fft_orig[idx_10hz])
    ratio_60hz = float(fft_filt[idx_60hz] / fft_orig[idx_60hz])

    # 10 Hz power should be preserved (>= 80%)
    assert ratio_10hz > 0.8, f"Expected 10 Hz ratio > 0.8, got {ratio_10hz}"
    # 60 Hz power should be strongly attenuated (<= 10%)
    assert ratio_60hz < 0.1, f"Expected 60 Hz ratio < 0.1, got {ratio_60hz}"


def test_epoching_trial_count():
    """Test that epoching produces expected number of trials from mock raw."""
    sfreq = 160.0
    n_samples = int(20 * sfreq)
    data = np.random.randn(2, n_samples)
    info = mne.create_info(ch_names=["C3", "C4"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    # 5 event markers at sample positions
    events = np.array([
        [int(2 * sfreq), 0, 1],
        [int(5 * sfreq), 0, 1],
        [int(8 * sfreq), 0, 2],
        [int(11 * sfreq), 0, 2],
        [int(14 * sfreq), 0, 1],
    ])
    event_id = {"left_hand": 1, "right_hand": 2}

    cfg = {"epoch": {"tmin": 0.0, "tmax": 2.0, "baseline": None}}
    prep = Preprocessor(cfg)
    epochs = prep.epoch(raw, events, event_id)

    assert len(epochs) == 5
    assert len(epochs["left_hand"]) == 3
    assert len(epochs["right_hand"]) == 2


def test_autoreject_reject_log_labels():
    """Test that autoreject's reject_log distinguishes clean, interpolated, and dropped epochs via reject_log.labels."""
    from autoreject import AutoReject

    sfreq = 160.0
    ch_names = ["Fz", "Cz", "Pz", "C3", "C4", "F3", "F4", "P3"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage)

    n_epochs = 10
    n_channels = len(ch_names)
    n_times = int(sfreq)
    rng = np.random.RandomState(42)

    data = rng.randn(n_epochs, n_channels, n_times) * 1e-5

    # Epoch 1: Channel 0 has moderate artifact -> candidate for interpolation
    data[1, 0, :] += 5e-4
    # Epoch 2: All channels have extreme artifact -> candidate for drop
    data[2, :, :] += 5e-3

    events = np.column_stack([
        np.arange(10, 10 + n_epochs * 20, 20),
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    raw = mne.io.RawArray(data.reshape(n_channels, -1), info, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"task": 1},
        tmin=0.0,
        tmax=0.99,
        baseline=None,
        preload=True,
        verbose=False,
    )

    ar = AutoReject(n_jobs=1, verbose=False)
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    # reject_log.labels shape: (n_epochs, n_channels)
    # Values: 0 = clean, 1 = interpolated, 2 = dropped
    assert reject_log.labels.shape == (n_epochs, n_channels)
    unique_vals = set(np.unique(reject_log.labels))
    assert unique_vals.issubset({0, 1, 2}), f"Unexpected labels in reject_log: {unique_vals}"
    assert isinstance(reject_log.bad_epochs, np.ndarray)
    assert reject_log.bad_epochs.shape == (n_epochs,)
