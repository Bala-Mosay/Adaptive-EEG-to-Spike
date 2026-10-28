import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable when running tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import mne
import pytest

from src.data.events import (
    CLASS_CODES,
    annotation_map_for_run,
    build_run_aware_events,
    run_number_from_filename,
)


def _make_raw_with_annotations(descriptions: list, sfreq: float = 160.0, duration_s: float = 12.0) -> mne.io.RawArray:
    """Create a small raw with one EEG channel and the given annotation descriptions."""
    n_samples = int(duration_s * sfreq)
    data = np.random.randn(1, n_samples) * 1e-5
    info = mne.create_info(ch_names=["Cz"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    onsets = np.arange(len(descriptions), dtype=float) * 2.0 + 1.0
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[1.0] * len(descriptions), description=descriptions))
    return raw


def test_run_number_from_filename():
    assert run_number_from_filename("S001R04.edf") == 4
    assert run_number_from_filename("S042R12.edf") == 12


def test_annotation_map_for_run_semantics():
    """Task-1 runs map T1/T2 to left/right hand; task-2 runs map T2 to feet."""
    # Task 1: left/right hand imagery (R04, R08, R12)
    for run in [4, 8, 12]:
        assert annotation_map_for_run(run) == {"T0": "rest", "T1": "left_hand", "T2": "right_hand"}, \
            f"R{run:02d} should be Task 1 (left/right)"
    # Task 2: both fists / feet imagery (R06, R10, R14)
    for run in [6, 10, 14]:
        assert annotation_map_for_run(run) == {"T0": "rest", "T1": "both_fists", "T2": "feet"}, \
            f"R{run:02d} should be Task 2 (both fists/feet)"
    # Baseline / execution runs should raise
    with pytest.raises(ValueError):
        annotation_map_for_run(2)  # baseline run, not an imagery run


def test_build_run_aware_events_excludes_both_fists_and_offsets():
    """Run-aware events must (a) exclude task-2 'both fists' trials, (b) map
    task-2 T2 to 'feet', and (c) offset task-2 events into the concatenated
    timeline. This is the regression test for the Phase 1 label bug."""
    # Run 4: 2x T1 (left), 2x T2 (right), 1x T0 (rest)
    raw_r04 = _make_raw_with_annotations(["T1", "T2", "T0", "T1", "T2"])
    # Run 6 (task 2): 2x T1 (both fists -> EXCLUDED), 2x T2 (feet)
    raw_r06 = _make_raw_with_annotations(["T1", "T2", "T1", "T2"])

    events, event_id, labels = build_run_aware_events([(raw_r04, 4), (raw_r06, 6)])

    # both_fists excluded, feet present, canonical codes used
    assert "both_fists" not in event_id
    assert set(labels) == {"rest", "left_hand", "right_hand", "feet"}
    assert event_id == {k: CLASS_CODES[k] for k in event_id}
    assert labels.count("left_hand") == 2
    assert labels.count("right_hand") == 2
    assert labels.count("feet") == 2
    assert labels.count("rest") == 1
    assert len(events) == len(labels)

    # task-2 events must be offset by the duration of run 4 (samples)
    offset = len(raw_r04.times)
    run2_samples = events[events[:, 2] == CLASS_CODES["feet"]][:, 0]
    assert np.all(run2_samples >= offset), "feet events must be shifted into run-8 timeline"

    # events sorted chronologically on the concatenated timeline
    assert np.all(np.diff(events[:, 0]) >= 0)
