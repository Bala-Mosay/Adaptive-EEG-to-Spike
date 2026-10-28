from typing import List, Tuple
import mne
import numpy as np
from mne.datasets import eegbci

from .events import build_run_aware_events, run_number_from_filename


def list_subjects() -> List[str]:
    return [f"S{idx:03d}" for idx in range(1, 110)]


def load_subject_raw(subject_id: str, runs: List[int] = None, preload: bool = True) -> Tuple[mne.io.BaseRaw, List[str]]:
    """Load and concatenate the imagery runs for a subject.

    Note: this returns only the concatenated raw. For run-aware event
    extraction (required for correct 4-class labels), use
    :func:`load_subject_raw_with_events` instead.
    """
    raw, file_paths, _, _ = load_subject_raw_with_events(subject_id, runs=runs, preload=preload)
    return raw, file_paths


def load_subject_raw_with_events(
    subject_id: str, runs: List[int] = None, preload: bool = True
) -> Tuple[mne.io.BaseRaw, List[str], np.ndarray, dict]:
    """Load a subject's imagery runs and build run-aware events.

    Events are extracted from each run's annotations BEFORE concatenation,
    because EEGMMIDB annotation descriptions (T0/T1/T2) have run-dependent
    meanings (see ``src/data/events.py``). Extracting events after
    concatenation silently mislabels task-2 trials — this is the documented
    fix for the Phase 1 label-mapping bug.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. ``"S001"``.
    runs : List[int], optional
        EEGMMIDB run numbers. Defaults to ``[4, 6, 8, 10, 12, 14]`` (the six
        imagery runs: 3 left/right hand + 3 both fists/feet).
    preload : bool
        Whether to preload raw data.

    Returns
    -------
    raw : mne.io.BaseRaw
        Concatenated raw recording (standardised channel names + montage).
    file_paths : List[str]
        Paths of the loaded EDF files, in concatenation order.
    events : np.ndarray, shape (n_events, 3)
        Run-aware events on the concatenated timeline; third column holds
        the canonical class codes (rest=1, left_hand=2, right_hand=3, feet=4).
        "Both fists" trials (task-2 T1) are excluded — not a class in the
        paper's 4-class scheme.
    event_id : Dict[str, int]
        Mapping from class label to integer code (only classes present).
    """
    sub_idx = int(subject_id[1:])
    if runs is None:
        runs = [4, 6, 8, 10, 12, 14]
    file_paths = mne.datasets.eegbci.load_data(sub_idx, runs)
    run_raws = []
    for fp, run_number in zip(file_paths, runs):
        raw_i = mne.io.read_raw_edf(fp, preload=preload, verbose=False)
        run_raws.append((raw_i, run_number))
    events, event_id, _ = build_run_aware_events(run_raws)
    # concatenate (order matches run_raws, as assumed by build_run_aware_events)
    raw = mne.concatenate_raws([r for r, _ in run_raws])
    # Standardize eegbci channel names and set a dataset-appropriate montage
    try:
        eegbci.standardize(raw)
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage)
    except Exception:
        # If standardization/montage fails, proceed without raising — caller may handle
        pass
    return raw, file_paths, events, event_id
