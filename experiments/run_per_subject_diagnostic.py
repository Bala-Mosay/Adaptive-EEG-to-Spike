#!/usr/bin/env python
"""Diagnostic: Per-subject LogReg accuracy.

Tests whether individual subjects have discriminative CSP features
by training/testing within each subject using 5-fold CV.
Also tests cross-subject generalization.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.balancer import balance_classes
from src.data.normalization import FeatureScaler, PerSubjectZScore
from src.features.bandpass import FixedBandCSPFeatures
from src.labels import LABEL_ORDER, labels_to_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXCLUDED = {"S038", "S088", "S089", "S092", "S100", "S104"}
SEED = 42


def main():
    config_path = Path(__file__).resolve().parents[1] / "config" / "phase2_config.yaml"
    config = yaml.safe_load(open(config_path))
    cache_dir = Path(config["data"].get("cache_dir", "outputs/preprocessed"))

    bands = [(b[0], float(b[1]), float(b[2])) for b in config["features"]["bands"]]
    n_comp = config["features"]["n_components"]

    all_subjects = [f"S{i:03d}" for i in range(1, 110) if f"S{i:03d}" not in EXCLUDED]

    # -- Part 1: Per-subject within-subject accuracy ---------------------
    log.info("=== Part 1: Per-subject within-subject accuracy (5-fold CV) ===")
    per_subject_acc = []
    n_ok = 0
    for i, subj in enumerate(all_subjects, 1):
        cache_file = cache_dir / f"{subj}.npz"
        if not cache_file.exists():
            continue
        try:
            blob = np.load(cache_file)
            X, y = blob["X"], blob["y"]
            X, y, _ = balance_classes(X, y, strategy="cap_to_min", random_state=SEED)

            if len(np.unique(y)) < 2 or len(y) < 40:
                continue

            # 5-fold stratified CV within subject
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_accs = []
            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                # Z-score: fit on this subject's train fold
                zscore = PerSubjectZScore()
                zscore.fit(X_tr, [subj] * len(X_tr))
                X_tr = zscore.transform(X_tr, [subj] * len(X_tr))
                X_te = zscore.transform(X_te, [subj] * len(X_te))

                # CSP: fit on train
                extractor = FixedBandCSPFeatures(bands=bands, n_components=n_comp, random_state=SEED)
                extractor.fit(X_tr, y_tr)
                F_tr = extractor.transform(X_tr)
                F_te = extractor.transform(X_te)

                # Scale: fit on train
                scaler = FeatureScaler().fit(F_tr)
                F_tr = scaler.transform(F_tr)
                F_te = scaler.transform(F_te)

                clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                         random_state=SEED, n_jobs=-1)
                clf.fit(F_tr, y_tr)
                fold_accs.append(accuracy_score(y_te, clf.predict(F_te)))

            mean_acc = np.mean(fold_accs)
            per_subject_acc.append({"subject": subj, "n_trials": len(y), "accuracy": float(mean_acc)})
            n_ok += 1
            if i % 20 == 0:
                log.info(f"  [{i}/{len(all_subjects)}] done")
        except Exception as e:
            log.warning(f"Skipping {subj}: {e}")
            continue

    # Sort by accuracy
    per_subject_acc.sort(key=lambda x: x["accuracy"])

    accs = [d["accuracy"] for d in per_subject_acc]
    log.info(f"Per-subject within-subject accuracy ({n_ok} subjects):")
    log.info(f"  Mean:   {np.mean(accs):.4f}")
    log.info(f"  Median: {np.median(accs):.4f}")
    log.info(f"  Std:    {np.std(accs):.4f}")
    log.info(f"  Min:    {np.min(accs):.4f} ({per_subject_acc[0]['subject']})")
    log.info(f"  Max:    {np.max(accs):.4f} ({per_subject_acc[-1]['subject']})")
    log.info(f"  > 50%:  {sum(1 for a in accs if a > 0.50)}/{n_ok} subjects")
    log.info(f"  > 60%:  {sum(1 for a in accs if a > 0.60)}/{n_ok} subjects")
    log.info(f"  > 40%:  {sum(1 for a in accs if a > 0.40)}/{n_ok} subjects")
    log.info(f"  < 30%:  {sum(1 for a in accs if a < 0.30)}/{n_ok} subjects")

    # -- Part 2: Feature statistics --------------------------------------
    log.info("\n=== Part 2: Feature statistics ===")
    # Load a few subjects and check feature distributions
    sample_subjects = ["S001", "S004", "S010", "S050", "S100"]
    for subj in sample_subjects:
        if subj in EXCLUDED:
            continue
        cache_file = cache_dir / f"{subj}.npz"
        if not cache_file.exists():
            continue
        blob = np.load(cache_file)
        X, y = blob["X"], blob["y"]
        X, y, _ = balance_classes(X, y, strategy="cap_to_min", random_state=SEED)

        zscore = PerSubjectZScore().fit(X, [subj] * len(X))
        X_z = zscore.transform(X, [subj] * len(X))
        extractor = FixedBandCSPFeatures(bands=bands, n_components=n_comp, random_state=SEED)
        extractor.fit(X_z, y)
        F = extractor.transform(X_z)
        scaler = FeatureScaler().fit(F)
        F = scaler.transform(F)

        # Per-class feature means
        log.info(f"{subj}: features shape={F.shape}, "
                 f"mean={F.mean():.3f}, std={F.std():.3f}, "
                 f"min={F.min():.3f}, max={F.max():.3f}")
        for c in range(4):
            mask = y == c
            if mask.sum() > 0:
                log.info(f"  class {c} ({LABEL_ORDER[c]}): "
                         f"mean={F[mask].mean():.3f}, std={F[mask].std():.3f}")

    # Save results
    output_dir = Path("results/logreg_diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_subject_accuracy.json", "w") as f:
        json.dump(per_subject_acc, f, indent=2)
    log.info(f"\nPer-subject results saved to {output_dir / 'per_subject_accuracy.json'}")


if __name__ == "__main__":
    main()
