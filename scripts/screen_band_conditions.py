"""Screen band-selection variants with a fast linear classifier.

Phase 3 C1 (per-subject KNN-MI top-3 of 7 bands) scored 29.28% against the
fixed-band baseline's 31.85%.  Before spending ~1 hour of SNN training per
fold on a candidate fix, this script measures the candidate feature sets with
logistic regression on the *same* folds.  Phase 2 showed the SNN tracks the
linear classifier closely on identical features (31.8% vs 34.6%), so this is a
cheap screen, not a replacement for the SNN run.

The expensive step (7-band CSP filtering) is done once per fold and **cached**
in ``--features-dir``; every condition is then a column subset, a per-subject
scaling, or an offset of the same full-bank matrix.  Re-scoring a cached fold
takes seconds instead of ~22 minutes.

Conditions
----------
======================  ====================================================
``broad_mu_beta``       Phase 2 features: mu 8-12, beta 13-30 (32 dims)
``all7``               All seven narrow bands, no selection (112 dims)
``adaptive3``          C1 as run: per-subject KNN-MI top-3 (48 dims)
``adaptive5``          Per-subject KNN-MI top-5 (80 dims)
``sensorimotor3``      Fixed mu + low_beta + beta (48 dims)
``guarded_adaptive``   Always mu + beta, plus the best remaining band (48 dims)
``soft_u_l###``        Soft weights, shrunk to uniform, lambda = 0.15..0.60
``soft_g_l###``        Soft weights, shrunk to global, lambda = 0.15..0.60
``soft_shuf_u_l030``   Same weights, band correspondence permuted (control)
``subject_band_center``  Per-subject per-band mean offset, no MI (ablation)
``subject_band_zscore``  Per-subject per-band per-column z-score (ablation)
======================  ====================================================

The soft conditions keep all seven bands and scale each subject's band blocks
by mean-one weights, so they cannot delete a band the way ``adaptive3``/
``adaptive5`` do.  ``lambda=0`` with ``shrink_to="uniform"`` reproduces
``all7`` bitwise; the script asserts that every fold before scoring.

Why the ablations matter
------------------------
CSP features are log-variance, so scaling a band by ``w`` is *exactly* adding
``2*log(w)`` to every component in that block.  Soft weighting is therefore a
per-subject, per-band additive offset.  ``subject_band_center`` expresses the
same family with **no MI ranking at all**, and ``soft_shuf_u_l030`` keeps the
weight distribution but destroys which band got which weight.  If either
matches the soft conditions, the gain comes from per-subject band calibration
rather than from adaptive band ranking.

Usage::

    python scripts/screen_band_conditions.py --folds 0,1,2
    python scripts/screen_band_conditions.py --folds 0,1,2 --json results/band_screen.json
    python scripts/screen_band_conditions.py --folds 0 --recompute
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
from src.bands.soft_weights import SoftBandWeighter  # noqa: E402
from src.cv.splits import subject_mixed_kfold  # noqa: E402
from src.data.balancer import balance_classes  # noqa: E402
from src.data.cache import load_or_process_subject  # noqa: E402
from src.data.normalization import FeatureScaler, PerSubjectZScore  # noqa: E402
from src.features.bandpass import FIXED_BANDS, FixedBandCSPFeatures  # noqa: E402

#: Bands that motor-imagery literature (and the Phase 2 baseline) rely on.
SENSORIMOTOR = ("mu", "low_beta", "beta")

#: Softmax temperature for the soft band-weighting conditions.  Kept fixed
#: across the screen so that only lambda and the shrinkage target vary.
SOFT_TEMPERATURE = 1.0

#: (lambda, shrink_to) grid for the soft conditions.  ``lambda=0`` is skipped
#: because it is already scored as ``all7`` and is asserted to be identical.
SOFT_GRID: List[Tuple[float, str]] = [
    (0.15, "uniform"),
    (0.30, "uniform"),
    (0.45, "uniform"),
    (0.60, "uniform"),
    (0.15, "global"),
    (0.30, "global"),
    (0.45, "global"),
    (0.60, "global"),
]

#: Bump when the cached array layout changes, so stale caches are ignored.
CACHE_VERSION = 1

#: Arrays stored per fold; everything else is derived from them.
CACHE_KEYS = (
    "F_train_all",
    "F_test_all",
    "F_broad_train",
    "F_broad_test",
    "y_train",
    "y_test",
    "subj_train",
    "subj_test",
)


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


def apply_band_weights(
    F_all: np.ndarray,
    subjects: np.ndarray,
    weights: Dict[str, np.ndarray],
    slices: Dict[str, slice],
    band_names: Sequence[str],
) -> np.ndarray:
    """Scale each subject's band blocks by its mean-one weight vector.

    With all weights equal to 1 this returns the unweighted matrix, which is
    exactly the ``all7`` control.
    """
    subject_array = np.asarray(subjects)
    out = F_all.copy()
    for subject in np.unique(subject_array):
        weight_vector = weights.get(str(subject))
        if weight_vector is None:
            continue
        mask = subject_array == subject
        for index, name in enumerate(band_names):
            out[mask, slices[name]] *= weight_vector[index]
    return out


def per_subject_band_center(
    F_train: np.ndarray,
    F_test: np.ndarray,
    subj_train: np.ndarray,
    subj_test: np.ndarray,
    slices: Dict[str, slice],
    band_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtract one scalar per (subject, band), estimated on training trials.

    This is the additive family that soft weighting can express, computed with
    **no mutual information and no labels**.  It is the ablation that decides
    whether the soft gain comes from the ranking or from per-subject band
    calibration.

    The offsets are estimated from that subject's *training* trials only, which
    keeps the transform leakage-free even though it is per subject.
    """
    subject_train = np.asarray(subj_train)
    subject_test = np.asarray(subj_test)
    out_train = F_train.copy()
    out_test = F_test.copy()

    for subject in np.unique(subject_train):
        train_mask = subject_train == subject
        test_mask = subject_test == subject
        for name in band_names:
            block = slices[name]
            offset = float(F_train[train_mask, block].mean())
            out_train[train_mask, block] -= offset
            if test_mask.any():
                out_test[test_mask, block] -= offset
    return out_train, out_test


def per_subject_band_zscore(
    F_train: np.ndarray,
    F_test: np.ndarray,
    subj_train: np.ndarray,
    subj_test: np.ndarray,
    slices: Dict[str, slice],
    band_names: Sequence[str],
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize each (subject, band, component) column using train trials.

    A stronger, label-free recalibration than :func:`per_subject_band_center`:
    it also removes per-subject differences in the *scale* of each component.
    Statistics are estimated on training trials only.
    """
    subject_train = np.asarray(subj_train)
    subject_test = np.asarray(subj_test)
    out_train = F_train.copy()
    out_test = F_test.copy()

    for subject in np.unique(subject_train):
        train_mask = subject_train == subject
        test_mask = subject_test == subject
        for name in band_names:
            block = slices[name]
            values = F_train[train_mask, block]
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std = np.where(std < eps, 1.0, std)
            out_train[train_mask, block] = (values - mean) / std
            if test_mask.any():
                out_test[test_mask, block] = (
                    F_test[test_mask, block] - mean
                ) / std
    return out_train, out_test


def permuted_weights(
    weights: Dict[str, np.ndarray],
    subjects: Sequence[str],
    seed: int,
) -> Dict[str, np.ndarray]:
    """Return the same weight vectors with the band correspondence permuted.

    This is a control with the identical *distribution* of weights: it keeps
    how much each subject's bands are scaled, but randomly re-assigns which
    band receives which weight.  If this scores like the real ranking, the
    ranking is not what produces the gain.  The permutation depends only on
    (seed, subject order), so it is reproducible.
    """
    out: Dict[str, np.ndarray] = {}
    for index, subject in enumerate(sorted(str(name) for name in subjects)):
        rng = np.random.default_rng([int(seed), index])
        vector = np.asarray(weights[subject], dtype=float)
        out[subject] = vector[rng.permutation(len(vector))]
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


# ---------------------------------------------------------------------------
# Fold feature cache
# ---------------------------------------------------------------------------
def cache_path(features_dir: Path, fold_index: int) -> Path:
    """Location of the cached feature arrays for one fold."""
    return features_dir / f"fold{fold_index}_v{CACHE_VERSION}.npz"


def save_fold_cache(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Write one fold's feature arrays, preserving dtype exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [key for key in CACHE_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Missing cache arrays: {missing}")
    np.savez(path, **{key: arrays[key] for key in CACHE_KEYS})


def load_fold_cache(path: Path) -> Dict[str, np.ndarray]:
    """Load one fold's feature arrays exactly as they were written."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in CACHE_KEYS}


def build_fold_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subj_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    subj_test: np.ndarray,
    *,
    n_components: int,
    sfreq: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Compute the cached arrays for one fold (the expensive step)."""
    zscore = PerSubjectZScore().fit(X_train, list(subj_train))
    X_train = zscore.transform(X_train, list(subj_train))
    X_test = zscore.transform(X_test, list(subj_test))

    print("  extracting 7-band CSP features (single pass) ...")
    pair = PairBandCSPExtractor(SEVEN_BANDS, n_components, sfreq, seed)
    F_train_all, F_test_all = pair.fit_transform_pair(X_train, y_train, X_test)
    print(f"  full-bank features: train {F_train_all.shape}, test {F_test_all.shape}")

    print("  extracting broad mu/beta (Phase 2) features ...")
    broad_pair = PairBandCSPExtractor(FIXED_BANDS, n_components, sfreq, seed)
    F_broad_train, F_broad_test = broad_pair.fit_transform_pair(
        X_train, y_train, X_test
    )

    return {
        "F_train_all": F_train_all,
        "F_test_all": F_test_all,
        "F_broad_train": F_broad_train,
        "F_broad_test": F_broad_test,
        "y_train": y_train,
        "y_test": y_test,
        "subj_train": subj_train,
        "subj_test": subj_test,
    }


def score_conditions(
    arrays: Dict[str, np.ndarray],
    *,
    n_components: int,
    seed: int,
    sel_cfg: dict,
    n_classes: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build every condition's (train, test) matrices from cached features."""
    F_train_all = arrays["F_train_all"]
    F_test_all = arrays["F_test_all"]
    y_train = arrays["y_train"]
    subj_train = np.asarray(arrays["subj_train"])
    subj_test = np.asarray(arrays["subj_test"])

    band_names = [name for name, _, _ in SEVEN_BANDS]
    slices = band_slices(band_names, n_classes * n_components)

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

    conditions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Phase 2 reference features (different band definitions).
    conditions["broad_mu_beta"] = (
        arrays["F_broad_train"],
        arrays["F_broad_test"],
    )

    # All seven bands, no selection.
    conditions["all7"] = (F_train_all, F_test_all)

    # Adaptive C1 and a wider top-5 variant.
    print("  ranking bands (top-3 / top-5) ...")
    for label, k in (("adaptive3", 3), ("adaptive5", 5)):
        selector = selector_for(k)
        selection = {
            str(subject): selector.get_bands(str(subject))
            for subject in seen_subjects
        }
        conditions[label] = (
            per_subject_matrix(F_train_all, subj_train, selection, slices),
            per_subject_matrix(F_test_all, np.asarray(arrays["subj_test"]), selection, slices),
        )

    # Fixed sensorimotor triple.
    fixed_cols = columns_for(list(SENSORIMOTOR), slices)
    conditions["sensorimotor3"] = (
        F_train_all[:, fixed_cols], F_test_all[:, fixed_cols]
    )

    # Adaptive with mu+beta pinned.
    guarded = guarded_selection(selector_for(3), seen_subjects, k=3)
    conditions["guarded_adaptive"] = (
        per_subject_matrix(F_train_all, subj_train, guarded, slices),
        per_subject_matrix(F_test_all, np.asarray(arrays["subj_test"]), guarded, slices),
    )

    # --- soft adaptive weighting ------------------------------------------
    # Every band is kept; each subject's blocks are scaled by mean-one
    # weights shrunk toward a stable prior.  lambda=0 must reproduce the
    # unweighted full bank exactly, which is asserted below.
    print("  fitting soft band weights ...")
    soft_weights: Dict[str, Dict[str, np.ndarray]] = {}
    for lam, shrink in SOFT_GRID:
        label = f"soft_{shrink[0]}_l{int(round(lam * 100)):03d}"
        weighter = SoftBandWeighter(
            lambda_=lam,
            temperature=SOFT_TEMPERATURE,
            shrink_to=shrink,
            n_neighbors=int(sel_cfg.get("n_neighbors", 5)),
            min_trials=int(sel_cfg.get("min_trials_per_subject", 20)),
            random_state=seed,
        ).fit(band_features_train, y_train, subj_train)
        weights = {
            str(subject): weighter.get_weights(str(subject))
            for subject in seen_subjects
        }
        soft_weights[label] = weights
        conditions[label] = (
            apply_band_weights(F_train_all, subj_train, weights, slices, band_names),
            apply_band_weights(
                F_test_all, np.asarray(arrays["subj_test"]), weights, slices, band_names
            ),
        )

    # Control: identical weight distribution, permuted band correspondence.
    shuffled = permuted_weights(soft_weights["soft_u_l030"], seen_subjects, seed)
    conditions["soft_shuf_u_l030"] = (
        apply_band_weights(F_train_all, subj_train, shuffled, slices, band_names),
        apply_band_weights(
            F_test_all, np.asarray(arrays["subj_test"]), shuffled, slices, band_names
        ),
    )

    # Ablations: per-subject band calibration with no MI at all.
    conditions["subject_band_center"] = per_subject_band_center(
        F_train_all,
        F_test_all,
        subj_train,
        np.asarray(arrays["subj_test"]),
        slices,
        band_names,
    )
    conditions["subject_band_zscore"] = per_subject_band_zscore(
        F_train_all,
        F_test_all,
        subj_train,
        np.asarray(arrays["subj_test"]),
        slices,
        band_names,
    )

    # Nesting guard: lambda=0 with a uniform prior is exactly `all7`.
    nest = SoftBandWeighter(
        lambda_=0.0,
        temperature=SOFT_TEMPERATURE,
        shrink_to="uniform",
        n_neighbors=int(sel_cfg.get("n_neighbors", 5)),
        min_trials=int(sel_cfg.get("min_trials_per_subject", 20)),
        random_state=seed,
    ).fit(band_features_train, y_train, subj_train)
    nest_weights = {
        str(subject): nest.get_weights(str(subject)) for subject in seen_subjects
    }
    nest_train = apply_band_weights(
        F_train_all, subj_train, nest_weights, slices, band_names
    )
    nest_test = apply_band_weights(
        F_test_all, np.asarray(arrays["subj_test"]), nest_weights, slices, band_names
    )
    if not (
        np.array_equal(nest_train, F_train_all)
        and np.array_equal(nest_test, F_test_all)
    ):
        raise RuntimeError(
            "Nesting check failed: soft weights with lambda=0 must "
            "reproduce the unweighted all7 features exactly"
        )
    print("  nesting check ok: lambda=0 == all7 (bitwise)")

    return conditions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config" / "phase3_config.yaml"))
    parser.add_argument(
        "--preprocess-config", default=str(ROOT / "config" / "phase1_config.yaml")
    )
    parser.add_argument("--folds", default="0", help="Comma-separated fold indices")
    parser.add_argument("--json", default=None, help="Optional path for JSON output")
    parser.add_argument(
        "--features-dir",
        default=str(ROOT / "results" / "band_screen_cache"),
        help="Where per-fold feature arrays are cached (empty string disables)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore existing caches and re-extract features",
    )
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
    features_dir = Path(args.features_dir) if args.features_dir else None
    if features_dir is not None and not features_dir.is_absolute():
        features_dir = ROOT / features_dir

    # Only touch the EEG data when at least one fold is not cached.
    cached = {}
    if features_dir is not None and not args.recompute:
        for fold_index in fold_indices:
            path = cache_path(features_dir, fold_index)
            if path.exists():
                cached[fold_index] = load_fold_cache(path)
                print(f"using cached features: {path.name}")
    needs_extraction = len(cached) < len(fold_indices)

    splits = None
    if needs_extraction:
        print("=== Loading subjects ===")
        started = time.time()
        X_all, y_all, subj_all = load_dataset(config, preprocess_cfg, cache_dir)
        print(f"loaded {len(y_all)} trials in {time.time() - started:.0f}s")
        splits = list(
            subject_mixed_kfold(
                n_splits=int(config["evaluation"]["cv_folds"]), random_state=seed
            ).split(X_all, y_all, subj_all)
        )

    results: Dict[str, Dict[int, dict]] = {}

    for fold_index in fold_indices:
        fold_started = time.time()
        print(f"\n=== Fold {fold_index} ===")

        if fold_index in cached:
            arrays = cached[fold_index]
            print(
                f"  cached: train {arrays['F_train_all'].shape} "
                f"test {arrays['F_test_all'].shape}"
            )
        else:
            train_idx, test_idx = splits[fold_index]
            print(f"  train {len(train_idx)} | test {len(test_idx)}")
            arrays = build_fold_features(
                X_all[train_idx],
                y_all[train_idx],
                subj_all[train_idx],
                X_all[test_idx],
                y_all[test_idx],
                subj_all[test_idx],
                n_components=n_components,
                sfreq=sfreq,
                seed=seed,
            )
            if features_dir is not None:
                path = cache_path(features_dir, fold_index)
                save_fold_cache(path, arrays)
                print(f"  cached features written: {path.name}")

        conditions = score_conditions(
            arrays,
            n_components=n_components,
            seed=seed,
            sel_cfg=sel_cfg,
            n_classes=n_classes,
        )

        y_train = arrays["y_train"]
        y_test = arrays["y_test"]
        print(f"  scoring {len(conditions)} conditions ...")
        for label, (F_tr, F_te) in conditions.items():
            metrics = evaluate(F_tr, y_train, F_te, y_test)
            results.setdefault(label, {})[fold_index] = metrics
            print(
                f"    {label:<22} acc={metrics['accuracy']:.4f} "
                f"kappa={metrics['kappa']:.4f} f1={metrics['f1_macro']:.4f} "
                f"dims={metrics['n_features']}"
            )
        del conditions
        print(f"  fold {fold_index} finished in {time.time() - fold_started:.0f}s")

    # --- summary -----------------------------------------------------------
    def mean_metric(label: str, key: str) -> float:
        return float(np.mean([m[key] for m in results[label].values()]))

    print("\n=== Summary (mean over evaluated folds) ===")
    print(f"{'condition':<22}{'accuracy':>10}{'kappa':>9}{'f1_macro':>10}{'dims':>6}")
    for label in sorted(results, key=lambda name: -mean_metric(name, "accuracy")):
        dims = int(next(iter(results[label].values()))["n_features"])
        print(
            f"{label:<22}{mean_metric(label, 'accuracy'):>10.4f}"
            f"{mean_metric(label, 'kappa'):>9.4f}"
            f"{mean_metric(label, 'f1_macro'):>10.4f}{dims:>6}"
        )

    # The family average is the honest summary: picking the best of eight
    # lambdas after the fact would inflate the reported gain.
    soft_labels = [
        label for label in results if label.startswith("soft_") and "shuf" not in label
    ]
    if soft_labels:
        family = float(
            np.mean(
                [
                    np.mean([m["accuracy"] for m in results[label].values()])
                    for label in soft_labels
                ]
            )
        )
        print(
            f"\n{'soft family mean':<22}{family:>10.4f}"
            f"  ({len(soft_labels)} lambda/prior settings, not post-hoc selected)"
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
