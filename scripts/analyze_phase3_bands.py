"""Diagnostics for the Phase 3 C1 (adaptive KNN-MI band selection) result.

The C1 run underperformed the fixed-band baseline (29.28% vs 31.85%).  This
script answers the first question an examiner will ask: *what did the band
selector actually choose, and was it stable?*

It reads ``results/phase3_c1/selected_bands.json`` (written by
``experiments/run_phase3.py``) and reports, across all subject-fold pairs:

* how often each band in the 7-band bank is selected;
* how often the classic sensorimotor ``mu`` band is discarded;
* how many distinct band sets a single subject gets across the 5 folds (a
  proxy for KNN-MI ranking noise at ~70 trials/subject).

Usage::

    python scripts/analyze_phase3_bands.py
    python scripts/analyze_phase3_bands.py --json results/phase3_c1/band_diagnostics.json
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

#: Bands every MI-literature reader expects to dominate (paper Section 5.3).
SENSORIMOTOR_BANDS = ("mu", "low_beta", "beta")


def load_selected_bands(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Return ``{fold_name: {subject: [band, ...]}}``."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["per_fold"]


def analyze(per_fold: Dict[str, Dict[str, List[str]]]) -> dict:
    """Compute selection frequency and cross-fold stability statistics."""
    folds = sorted(per_fold)
    subjects = sorted({subject for fold in folds for subject in per_fold[fold]})

    band_counts: collections.Counter = collections.Counter()
    position_counts: collections.Counter = collections.Counter()
    subject_fold_sets: List[tuple] = []
    per_subject_sets: Dict[str, List[tuple]] = collections.defaultdict(list)

    for fold in folds:
        for subject, bands in per_fold[fold].items():
            if not bands:
                continue
            band_counts.update(bands)
            for position, band in enumerate(bands, start=1):
                position_counts[position] += 1
            chosen = tuple(bands)
            subject_fold_sets.append(chosen)
            per_subject_sets[subject].append(chosen)

    n_pairs = len(subject_fold_sets)
    missing_mu = sum(1 for bands in subject_fold_sets if "mu" not in bands)
    missing_sensorimotor = sum(
        1 for bands in subject_fold_sets if not set(bands) & set(SENSORIMOTOR_BANDS)
    )

    distinct_per_subject = [len(set(sets)) for sets in per_subject_sets.values() if sets]

    return {
        "n_folds": len(folds),
        "n_subjects": len(subjects),
        "n_subject_fold_pairs": n_pairs,
        "band_selection_counts": dict(band_counts.most_common()),
        "band_selection_fraction": {
            band: count / n_pairs for band, count in band_counts.most_common()
        },
        "position_counts": {str(k): v for k, v in sorted(position_counts.items())},
        "subject_folds_missing_mu": missing_mu,
        "subject_folds_missing_mu_fraction": missing_mu / n_pairs if n_pairs else 0.0,
        "subject_folds_missing_all_sensorimotor": missing_sensorimotor,
        "distinct_band_sets_per_subject": {
            "mean": statistics.mean(distinct_per_subject) if distinct_per_subject else 0.0,
            "median": statistics.median(distinct_per_subject) if distinct_per_subject else 0.0,
            "max": max(distinct_per_subject) if distinct_per_subject else 0,
            "n_subjects_with_all_five_identical": sum(
                1 for value in distinct_per_subject if value == 1
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bands",
        default=str(ROOT / "results" / "phase3_c1" / "selected_bands.json"),
        help="Path to selected_bands.json",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path to write the diagnostics as JSON",
    )
    args = parser.parse_args()

    per_fold = load_selected_bands(Path(args.bands))
    report = analyze(per_fold)

    print("=== Phase 3 C1 band-selection diagnostics ===")
    print(f"folds    : {report['n_folds']}")
    print(f"subjects : {report['n_subjects']}")
    print(f"pairs    : {report['n_subject_fold_pairs']} subject-fold selections")
    print("")
    print("Band selection frequency (each subject-fold picks 3 of 7):")
    for band, count in report["band_selection_counts"].items():
        frac = report["band_selection_fraction"][band]
        print(f"  {band:<10} {count:>4}  ({100.0 * frac:5.1f}% of subject-folds)")
    print("")
    print(
        "Subject-folds where the classic mu band was discarded: "
        f"{report['subject_folds_missing_mu']} "
        f"({100.0 * report['subject_folds_missing_mu_fraction']:.1f}%)"
    )
    print(
        "Subject-folds with no sensorimotor band at all "
        f"(mu/low_beta/beta): {report['subject_folds_missing_all_sensorimotor']}"
    )
    print("")
    stability = report["distinct_band_sets_per_subject"]
    print("Cross-fold stability of a subject's selected set:")
    print(f"  distinct sets per subject: mean={stability['mean']:.2f} "
          f"median={stability['median']:.0f} max={stability['max']}")
    print(f"  subjects with the same set in all 5 folds: "
          f"{stability['n_subjects_with_all_five_identical']}/{report['n_subjects']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
