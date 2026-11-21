"""Screen band-selection variants with a fast linear classifier.

Phase 3 C1 (per-subject KNN-MI top-3 of 7 bands) scored 29.28% against the
fixed-band baseline's 31.85%.  Before spending ~1 hour of SNN training per
fold on a candidate fix, this script measures the candidate feature sets with
logistic regression on the *same* folds.  Phase 2 showed the SNN tracks the
linear classifier closely on identical features (31.8% vs 34.6%), so this is a
cheap screen, not a replacement for the SNN run.

The expensive step (7-band CSP filtering) is done once per fold; every
condition is then a column subset of the same full-bank feature matrix.

Conditions
----------
======================  ====================================================
``broad_mu_beta``       Phase 2 features: mu 8-12, beta 13-30 (32 dims)
``all7``               All seven narrow bands, no selection (112 dims)
``adaptive3``          C1 as run: per-subject KNN-MI top-3 (48 dims)
``adaptive5``          Per-subject KNN-MI top-5 (80 dims)
``sensorimotor3``      Fixed mu + low_beta + beta (48 dims)
``guarded_adaptive``   Always mu + beta, plus the best remaining band (48 dims)
======================  ====================================================

Usage::

    python scripts/screen_band_conditions.py --folds 0
    python scripts/screen_band_conditions.py --folds 0,1 --json results/band_screen.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bands.bank import SEVEN_BANDS  # noqa: E402
from src.bands.selector import AdaptiveBandSelector  # noqa: E402
from src.cv.splits import subject_mixed_kfold  # noqa: E402
from src.data.balancer import balance_classes  # noqa: E402
from src.data.cache import load_or_process_subject  # noqa: E402
from src.data.normalization import FeatureScaler, PerSubjectZScore  # noqa: E402
from src.features.bandpass import FIXED_BANDS, FixedBandCSPFeatures  # noqa: E402

#: Bands that motor-imagery literature (and the Phase 2 baseline) rely on.
SENSORIMOTOR = ("mu", "low_beta", "beta")


def load_dataset(
    config: dict, preprocess_cfg: dict, cache_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, validate, and balanced-pool every configured subject."""
    excluded = set(config["data"]["excluded_subjects"])
    subjects = [
        f"S{i:03d}" for i in range(1, 110) if f"S{i:03d}" not in excluded
    ]
    bal = config["balancing"]

    X_parts, y_parts, subj_parts = [], [], []
    for index, subject in enumerate(subjects, start=1):
        try:
            X, y = load_or_process_subject(subject, preprocess_cfg, cache_dir)
        except Exception as exc:  # noqa: BLE001 - mirrors the runner behaviour
            print(f"  skipping {subject}: {exc}")
            continue
        X, y, _ = balance_classes(
            X, y, strategy=bal["strategy"], random_state=bal["random_state"]
        )
        X_parts.append(X)
        y_parts.append(y)
        subj_parts.append(np.full(len(y), subject))
        if index % 20 == 0 or index == len(subjects):
            print(f"  loaded {index}/{len(subjects)} subjects")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    subj_all = np.concatenate(subj_parts)
    del X_parts, y_parts, subj_parts
    return X_all, y_all, subj_all


def band_slices(band_names: Sequence[str], block: int) -> Dict[str, slice]:
    """Map each band to its column block in the concatenated feature matrix."""
    return {
        name: slice(index * block, (index + 1) * block)
        for index, name in enumerate(band_names)
    }


def columns_for(bands: Sequence[str], slices: Dict[str, slice]) -> np.ndarray:
    """Column indices for a fixed band set (same for every trial)."""
    return np.concatenate(
        [np.arange(slices[name].start, slices[name].stop) for name in bands]
    )


def per_subject_matrix(
    F_all: np.ndarray,
    subjects: np.ndarray,
    selected: Dict[str, List[str]],
    slices: Dict[str, slice],
) -> np.ndarray:
    """Restrict each trial to the bands selected for its subject."""
    subject_array = np.asarray(subjects)
    width = sum(slices[name].stop - slices[name].start for name in selected[str(subject_array[0])])
    out = np.empty((len(F_all), width), dtype=F_all.dtype)
    for subject in np.unique(subject_array):
        mask = subject_array == subject
        cols = columns_for(selected[str(subject)], slices)
        out[mask] = F_all[mask][:, cols]
    return out


def guarded_selection(
    selector: AdaptiveBandSelector,
    subjects: Sequence[str],
    k: int = 3,
    required: Sequence[str] = ("mu", "beta"),
) -> Dict[str, List[str]]:
    """Adaptive ranking with the sensorimotor bands pinned in.

    Keeps ``required`` unconditionally, then fills up to ``k`` with the
    highest-ranked remaining bands for that subject.  Output is always in
    filter-bank order so block positions stay consistent.
    """
    bank_order = list(selector.band_names_)
    pinned = [name for name in required if name in bank_order]
    selection: Dict[str, List[str]] = {}
    for subject in subjects:
        scores = selector.scores_.get(str(subject), selector.global_scores_)
        ranked = sorted(scores, key=lambda name: (-scores[name], bank_order.index(name)))
        chosen = list(pinned)
        for name in ranked:
            if len(chosen) >= k:
                break
            if name not in chosen:
                chosen.append(name)
        selection[str(subject)] = [name for name in bank_order if name in chosen]
    return selection


def evaluate(
    F_train: np.ndarray, y_train: np.ndarray, F_test: np.ndarray, y_test: np.ndarray
) -> dict:
    """Standardize on the training fold, fit logistic regression, score test."""
    scaler = FeatureScaler().fit(F_train)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(scaler.transform(F_train), y_train)
    pred = model.predict(scaler.transform(F_test))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "kappa": float(cohen_kappa_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "n_features": int(F_train.shape[1]),
    }


class PairBandCSPExtractor:
    """One filtering pass per band, producing train and test features together.

    Numerically identical to ``FixedBandCSPFeatures.fit`` followed by
    ``transform`` on train and test, but filters the training fold **once**
    instead of twice.  Filtering dominates screening runtime, so this halves
    the cost while holding only one filtered band array in memory at a time.
    """

    def __init__(
        self,
        bands: Sequence[tuple],
        n_components: int,
        sfreq: float,
        random_state: int,
    ):
        self.extractor = FixedBandCSPFeatures(
            bands=list(bands),
            n_components=n_components,
            sfreq=sfreq,
            random_state=random_state,
        )

    def fit_transform_pair(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit per-band CSP on train and return (train, test) feature blocks."""
        from src.spatial.rcsp import RegularizedCSP

        ext = self.extractor
        ext.bands_ = list(ext.bands if ext.bands is not None else FIXED_BANDS)
        ext.csp_per_band_ = {}
        train_blocks: List[np.ndarray] = []
        test_blocks: List[np.ndarray] = []

        for band in ext.bands_:
            X_band = ext._band_filtered(X_train, band)
            csp = RegularizedCSP(
                n_components=ext.n_components, random_state=ext.random_state
            )
            csp.fit(X_band, y_train)
            ext.csp_per_band_[band[0]] = csp
            train_blocks.append(csp.transform(X_band))
            del X_band

            X_band_test = ext._band_filtered(X_test, band)
            test_blocks.append(csp.transform(X_band_test))
            del X_band_test

        return (
            np.concatenate(train_blocks, axis=1),
            np.concatenate(test_blocks, axis=1),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "phase3_config.yaml"))
    parser.add_argument(
        "--preprocess-config", default=str(ROOT / "config" / "phase1_config.yaml")
    )
    parser.add_argument("--folds", default="0", help="Comma-separated fold indices")
    parser.add_argument("--json", default=None, help="Optional path for JSON output")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    with open(args.preprocess_config, encoding="utf-8") as fh:
        preprocess_cfg = yaml.safe_load(fh)

    cache_dir = ROOT / config["data"]["cache_dir"]
    feat = config["features"]
    sel_cfg = feat["band_selection"]
    n_components = int(feat["n_components"])
    sfreq = float(config["data"]["sfreq"])
    seed = int(config["experiment"]["seed"])
    n_classes = 4

    fold_indices = [int(value) for value in args.folds.split(",")]

    print("=== Loading subjects ===")
    started = time.time()
    X_all, y_all, subj_all = load_dataset(config, preprocess_cfg, cache_dir)
    print(f"loaded {len(y_all)} trials in {time.time() - started:.0f}s")

    splits = subject_mixed_kfold(
        n_splits=int(config["evaluation"]["cv_folds"]), random_state=seed
    ).split(X_all, y_all, subj_all)

    results: Dict[str, Dict[int, dict]] = {}

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        if fold_index not in fold_indices:
            continue
        fold_started = time.time()
        print(f"\n=== Fold {fold_index} (train {len(train_idx)} | test {len(test_idx)}) ===")

        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        subj_train, subj_test = subj_all[train_idx], subj_all[test_idx]

        zscore = PerSubjectZScore().fit(X_train, list(subj_train))
        X_train = zscore.transform(X_train, list(subj_train))
        X_test = zscore.transform(X_test, list(subj_test))

        # --- one expensive full-bank pass, reused by every narrow-band condition
        print("  extracting 7-band CSP features (single pass) ...")
        pair = PairBandCSPExtractor(SEVEN_BANDS, n_components, sfreq, seed)
        F_train_all, F_test_all = pair.fit_transform_pair(X_train, y_train, X_test)
        base = pair.extractor
        band_names = [name for name, _, _ in base.bands_]
        slices = band_slices(band_names, n_classes * n_components)
        print(f"  full-bank features: train {F_train_all.shape}, test {F_test_all.shape}")

        band_features_train = {
            name: F_train_all[:, slices[name]] for name in band_names
        }
        # Subject-mixed CV keeps every test subject in the training fold, but
        # resolve both sides explicitly so an unseen subject cannot KeyError.
        seen_subjects = np.unique(np.concatenate([subj_train, subj_test]))

        def selector_for(k: int) -> AdaptiveBandSelector:
            return AdaptiveBandSelector(
                top_k=k,
                n_neighbors=int(sel_cfg.get("n_neighbors", 5)),
                min_trials=int(sel_cfg.get("min_trials_per_subject", 20)),
                random_state=seed,
            ).fit(band_features_train, y_train, subj_train)

        print("  ranking bands (top-3 / top-5) ...")
        selector3 = selector_for(3)
        selector5 = selector_for(5)

        conditions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Phase 2 reference features (different band definitions).
        print("  extracting broad mu/beta (Phase 2) features ...")
        broad_pair = PairBandCSPExtractor(FIXED_BANDS, n_components, sfreq, seed)
        F_broad_train, F_broad_test = broad_pair.fit_transform_pair(
            X_train, y_train, X_test
        )
        conditions["broad_mu_beta"] = (F_broad_train, F_broad_test)

        # All seven bands, no selection.
        conditions["all7"] = (F_train_all, F_test_all)

        # Adaptive C1 and a wider top-5 variant.
        for label, selector in (("adaptive3", selector3), ("adaptive5", selector5)):
            selection = {
                str(subject): selector.get_bands(str(subject))
                for subject in seen_subjects
            }
            conditions[label] = (
                per_subject_matrix(F_train_all, subj_train, selection, slices),
                per_subject_matrix(F_test_all, subj_test, selection, slices),
            )

        # Fixed sensorimotor triple.
        fixed_cols = columns_for(list(SENSORIMOTOR), slices)
        conditions["sensorimotor3"] = (
            F_train_all[:, fixed_cols], F_test_all[:, fixed_cols]
        )

        # Adaptive with mu+beta pinned.
        guarded = guarded_selection(selector3, seen_subjects, k=3)
        conditions["guarded_adaptive"] = (
            per_subject_matrix(F_train_all, subj_train, guarded, slices),
            per_subject_matrix(F_test_all, subj_test, guarded, slices),
        )

        print(f"  scoring {len(conditions)} conditions ...")
        for label, (F_tr, F_te) in conditions.items():
            metrics = evaluate(F_tr, y_train, F_te, y_test)
            results.setdefault(label, {})[fold_index] = metrics
            print(
                f"    {label:<18} acc={metrics['accuracy']:.4f} "
                f"kappa={metrics['kappa']:.4f} f1={metrics['f1_macro']:.4f} "
                f"dims={metrics['n_features']}"
            )

        del X_train, X_test, F_train_all, F_test_all, base, pair, broad_pair
        print(f"  fold {fold_index} finished in {time.time() - fold_started:.0f}s")

    # --- summary -----------------------------------------------------------
    print("\n=== Summary (mean over evaluated folds) ===")
    print(f"{'condition':<20}{'accuracy':>10}{'kappa':>9}{'f1_macro':>10}{'dims':>6}")
    for label, per_fold in sorted(
        results.items(), key=lambda item: -np.mean([m["accuracy"] for m in item[1].values()])
    ):
        accs = [m["accuracy"] for m in per_fold.values()]
        kappas = [m["kappa"] for m in per_fold.values()]
        f1s = [m["f1_macro"] for m in per_fold.values()]
        dims = int(next(iter(per_fold.values()))["n_features"])
        print(
            f"{label:<20}{np.mean(accs):>10.4f}{np.mean(kappas):>9.4f}"
            f"{np.mean(f1s):>10.4f}{dims:>6}"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                {"folds": fold_indices, "conditions": results}, fh, indent=2
            )
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
