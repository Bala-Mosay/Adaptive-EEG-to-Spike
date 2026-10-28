from typing import Dict, Tuple, Optional, List
import mne
import numpy as np

# ---------------------------------------------------------------------------
# Run-aware event mapping for the PhysioNet EEG Motor Movement/Imagery dataset
# (EEGMMIDB). Annotation semantics are RUN-DEPENDENT (see the EEGMMIDB
# documentation and Shuqfa et al. 2024 [31] of the survey paper):
#
#   Imagery task-1 runs (R04, R08, R12):  T1 = imagine LEFT fist,   T2 = imagine RIGHT fist
#   Imagery task-2 runs (R06, R10, R14):  T1 = imagine BOTH fists,  T2 = imagine BOTH feet
#
# The research paper (Survey_Paper.pdf, Section 6.1) defines a 4-class scheme:
# {left_hand, right_hand, feet, rest}. "Both fists" is NOT one of the paper's
# classes, so those trials are EXCLUDED from the 4-class scheme.
#
# Motor execution runs (R03, R05, R07, R09, R11, R13) are NOT loaded.
# Baseline runs (R01, R02) are NOT loaded.
# ---------------------------------------------------------------------------

#: EEGMMIDB imagery run numbers for task 1 (left/right fist imagery).
#: R04, R08, R12 all share the same annotation semantics: T1=left, T2=right.
IMAGERY_TASK1_RUNS = frozenset({4, 8, 12})
#: EEGMMIDB imagery run numbers for task 2 (both-fists / both-feet imagery).
#: R06, R10, R14 all share the same annotation semantics: T1=both_fists, T2=feet.
IMAGERY_TASK2_RUNS = frozenset({6, 10, 14})

#: Canonical integer codes for the paper's 4-class scheme (Section 6.1).
CLASS_CODES: Dict[str, int] = {
    "rest": 1,
    "left_hand": 2,
    "right_hand": 3,
    "feet": 4,
}


def run_number_from_filename(path) -> int:
    """Extract the EEGMMIDB run number from an EDF file path.

    Example: ``S001R04.edf`` -> ``4``.
    """
    from pathlib import Path

    stem = Path(path).stem  # e.g. "S001R04"
    return int(stem[5:])  # strip "S###R" prefix


def annotation_map_for_run(run_number: int) -> Dict[str, str]:
    """Return the ``{annotation_description: class_label}`` mapping for a run.

    Parameters
    ----------
    run_number : int
        EEGMMIDB run number (e.g. 4, 8, 12 for imagery runs).

    Returns
    -------
    Dict[str, str]
        Mapping from annotation description (``T0``/``T1``/``T2``) to one of
        the paper's 4 class labels, with task-2 ``T1`` ("both fists") mapped
        to ``both_fists`` (callers should exclude those trials).
    """
    if run_number in IMAGERY_TASK1_RUNS:
        return {"T0": "rest", "T1": "left_hand", "T2": "right_hand"}
    if run_number in IMAGERY_TASK2_RUNS:
        return {"T0": "rest", "T1": "both_fists", "T2": "feet"}
    raise ValueError(
        f"Run number {run_number} is not a recognised imagery run "
        f"(expected one of {sorted(IMAGERY_TASK1_RUNS | IMAGERY_TASK2_RUNS)})"
    )


def build_run_aware_events(
    run_raws: List[Tuple[mne.io.BaseRaw, int]],
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Build a single event array with run-aware class codes.

    Each run's events are extracted from its own annotations (BEFORE raw
    concatenation destroys the run context), mapped via
    :func:`annotation_map_for_run`, re-coded with the canonical
    :data:`CLASS_CODES`, and offset into the concatenated timeline.

    Parameters
    ----------
    run_raws : List[Tuple[BaseRaw, int]]
        List of ``(raw, run_number)`` pairs, in the same order the raws will
        be (or already were) concatenated by ``mne.concatenate_raws``.

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
        MNE-style events array on the concatenated timeline with canonical
        class codes in the third column.
    event_id : Dict[str, int]
        Only the class labels actually present, mapped to their codes.
    labels : List[str]
        Class label per event row (same order as ``events``).
    """
    all_events: List[np.ndarray] = []
    all_labels: List[str] = []
    sample_offset = 0
    for raw, run_number in run_raws:
        ann_map = annotation_map_for_run(run_number)
        ev, ann_desc = mne.events_from_annotations(raw, verbose=False)
        # ann_desc maps description -> code used in `ev`; invert it
        for desc, code in ann_desc.items():
            label = ann_map[desc]
            if label == "both_fists":
                continue  # not a class in the paper's 4-class scheme
            rows = ev[ev[:, 2] == code].copy()
            rows[:, 0] += sample_offset
            rows[:, 2] = CLASS_CODES[label]
            all_events.append(rows)
            all_labels.extend([label] * len(rows))
        sample_offset += len(raw.times)
    if not all_events:
        raise RuntimeError("No usable events found in any run")
    events = np.concatenate(all_events, axis=0)
    # restore chronological order (events were grouped per class per run)
    order = np.argsort(events[:, 0], kind="stable")
    events = events[order]
    labels = [all_labels[i] for i in order]
    present = sorted(set(labels))
    event_id = {label: CLASS_CODES[label] for label in present}
    return events, event_id, labels


def events_from_raw(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract events and event_id mapping from an MNE Raw object.

    Tries `mne.events_from_annotations` first (recommended), falling back
    to `mne.find_events` for older files. Returns (events, event_id).
    """
    try:
        events, event_id = mne.events_from_annotations(raw)
        return events, event_id
    except Exception:
        # fallback to peak-based event finding; this requires stimulus channel
        events = mne.find_events(raw, verbose=False)
        # Build event_id dict from unique event codes
        unique = np.unique(events[:, 2])
        event_id = {str(int(code)): int(code) for code in unique}
        return events, event_id


def map_event_id_to_labels(event_id: Dict[str, int], mapping: Optional[Dict[str, str]] = None) -> Dict[int, str]:
    """Map event_id keys to human-readable labels.

    .. deprecated::
        This global mapping is INCORRECT for the PhysioNet EEGMMIDB dataset
        because annotation semantics are run-dependent (see
        :func:`build_run_aware_events`). It is retained only for legacy
        callers and synthetic tests.

    - `event_id` typically maps annotation string -> int (e.g. 'T1'->2)
    - `mapping` is an optional user-provided dict mapping annotation strings
      or numeric strings to desired labels like 'left_hand'.

    If `mapping` is None, a conservative default mapping for common
    EEGBCI motor-imagery codes is used when possible. Any unmapped codes
    are returned as their original string names.
    """
    # Default mapping for common eegbci annotations (may vary by release)
    default_map = {
        '2': 'left_hand',
        '3': 'right_hand',
        '4': 'feet',
        '1': 'rest',
        'T1': 'left_hand',
        'T2': 'right_hand',
        'T3': 'feet',
        'T0': 'rest',
    }
    out: Dict[int, str] = {}
    for key, code in event_id.items():
        label = None
        if mapping and key in mapping:
            label = mapping[key]
        elif mapping and str(code) in mapping:
            label = mapping[str(code)]
        elif key in default_map:
            label = default_map[key]
        elif str(code) in default_map:
            label = default_map[str(code)]
        else:
            # fallback: use the annotation key as label
            label = key
        out[int(code)] = label
    return out
