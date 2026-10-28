#!/usr/bin/env python
"""Diagnostic: Logistic Regression on same features as baseline SNN.

Determines whether the bottleneck is the features (CSP) or the model (SNN).
Uses identical pipeline per fold: PerSubjectZScore -> CSP -> FeatureScaler
then fits LogisticRegression instead of the SNN.

Usage:
    python experiments/run_logreg_diagnostic.py

Outputs:
    results/logreg_diagnostic/summary.json
    results/logreg_diagnostic/fold_results.csv
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv.splits import subject_mixed_kfold
from src.data.balancer import balance_classes
from src.data.loader import load_subject_raw_with_events
from src.data.normalization import FeatureScaler, PerSubjectZScore
from src.features.bandpass import FixedBandCSPFeatures
from src.labels import LABEL_ORDER, labels_to_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXCLUDED = {"S038", "S088", "S089", "S092", "S100", "S104"}
SEED = 42


def load_or_process(subject_id: str, cache_dir: Path) -> tuple:
    """Load cached preprocessed trials for a subject."""
    cache_file = cache_dir / f"{subject_id}.npz"
    if cache_file.exists():
        blob = np.load(cache_file)
        return blob["X"], blob["y"]

    from src.data.preprocessing import Preprocessor

    config_path = Path(__file__).resolve().parents[1] / "config" / "phase1_config.yaml"
    preprocess_cfg = yaml.safe_load(open(config_path))

    raw, run_files, events, event_id = load_subject_raw_with_events(subject_id)
    pre = Preprocessor(preprocess_cfg)
    preprocess_cfg["subject_id"] = subject_id
    X, labels, _ = pre.process_subject(
        raw, run_files=run_files, events=events, event_id=event_id
    )
    y = np.asarray(labels_to_indices(labels), dtype=np.int64)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X.astype(np.float32), y=y)
    return X.astype(np.float32), y


def main():
    config_path = Path(__file__).resolve().parents[1] / "config" / "phase2_config.yaml"
    config = yaml.safe_load(open(config_path))

    cache_dir = Path(config["data"].get("cache_dir", "outputs/preprocessed"))
    output_dir = Path("results/logreg_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)

    # -- Load all subjects -----------------------------------------------
    all_subjects = [f"S{i:03d}" for i in range(1, 110) if f"S{i:03d}" not in EXCLUDED]
    log.info(f"Loading {len(all_subjects)} subjects...")

    X_parts, y_parts, subj_parts = [], [], []
    n_ok = 0
    for i, subj in enumerate(all_subjects, 1):
        try:
            X, y = load_or_process(subj, cache_dir)
            X, y, _ = balance_classes(X, y, strategy="cap_to_min", random_state=SEED)
            X_parts.append(X)
            y_parts.append(y)
            subj_parts.append([subj] * len(y))
            n_ok += 1
            if i % 20 == 0:
                log.info(f"  [{i}/{len(all_subjects)}] loaded {n_ok} subjects so far")
        except Exception as e:
            log.warning(f"Skipping {subj}: {e}")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    subjects_all = np.concatenate(subj_parts)
    log.info(
        f"Loaded {n_ok} subjects | {len(y_all)} trials | "
        f"class counts: {dict(zip(*np.unique(y_all, return_counts=True)))}"
    )

    # -- 5-fold CV -------------------------------------------------------
    bands = [(b[0], float(b[1]), float(b[2])) for b in config["features"]["bands"]]
    n_comp = config["features"]["n_components"]

    kfold = subject_mixed_kfold(n_splits=config["evaluation"]["cv_folds"], random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all)):
        log.info(f"--- Fold {fold_idx + 1}/5 (train {len(train_idx)} | test {len(test_idx)}) ---")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        subj_train, subj_test = subjects_all[train_idx], subjects_all[test_idx]

        # Per-subject z-score (fit on train only)
        zscore = PerSubjectZScore().fit(X_train, list(subj_train))
        X_train = zscore.transform(X_train, list(subj_train))
        X_test = zscore.transform(X_test, list(subj_test))

        # CSP features (fit on train only)
        extractor = FixedBandCSPFeatures(
            bands=bands, n_components=n_comp, random_state=SEED
        )
        extractor.fit(X_train, y_train)
        F_train = extractor.transform(X_train)
        F_test = extractor.transform(X_test)

        # Feature scaling (fit on train only)
        scaler = FeatureScaler().fit(F_train)
        F_train = scaler.transform(F_train)
        F_test = scaler.transform(F_test)

        # Logistic Regression (the only difference from baseline)
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial",
            random_state=SEED, n_jobs=-1,
        )
        clf.fit(F_train, y_train)
        y_pred = clf.predict(F_test)

        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Per-class F1
        f1_per_class = f1_score(y_test, y_pred, average=None)
        f1_dict = {LABEL_ORDER[i]: float(f1_per_class[i]) for i in range(len(LABEL_ORDER))}

        fold_results.append({
            "fold": fold_idx,
            "accuracy": float(acc),
            "kappa": float(kappa),
            "f1_macro": float(f1),
            **f1_dict,
        })
        log.info(f"Fold {fold_idx + 1}: acc={acc:.4f} | kappa={kappa:.4f} | f1_macro={f1:.4f}")

    # -- Save results ----------------------------------------------------
    df = pd.DataFrame(fold_results)
    df.to_csv(output_dir / "fold_results.csv", index=False)

    summary = {
        "experiment": "logreg_diagnostic",
        "classifier": "LogisticRegression(C=1.0, max_iter=1000, lbfgs, multinomial)",
        "pipeline": "PerSubjectZScore -> FixedBandCSP -> FeatureScaler -> LogReg",
        "n_folds": len(fold_results),
        "n_subjects": n_ok,
        "n_trials_total": int(len(y_all)),
        "cv_type": "subject_mixed",
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std()),
        "kappa_mean": float(df["kappa"].mean()),
        "kappa_std": float(df["kappa"].std()),
        "f1_macro_mean": float(df["f1_macro"].mean()),
        "f1_macro_std": float(df["f1_macro"].std()),
        "per_fold": fold_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=== LOGREG DIAGNOSTIC RESULTS ===")
    log.info(f"Accuracy : {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    log.info(f"Kappa    : {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")
    log.info(f"F1 Macro : {summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}")
    log.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
