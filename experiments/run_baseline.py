#!/usr/bin/env python
"""Phase 2 baseline experiment: 5-fold subject-mixed CV for the baseline SNN.

Pipeline per the research paper's evaluation protocol (Section 6.2):
trials from all 103 subjects are POOLED and split into 5 folds by trial
(subject-mixed). Per-subject z-score, per-band CSP filters, feature scaling,
and the encoder threshold are fit EXCLUSIVELY on each fold's training trials.

Usage:
    python experiments/run_baseline.py --config config/phase2_config.yaml \
        [--subjects S001,S002,S003]   # optional subset for smoke runs

Outputs:
    results/baseline/fold_results.csv
    results/baseline/summary.json
    results/baseline/confusion_matrices.npy
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv.splits import subject_mixed_kfold
from src.data.balancer import balance_classes
from src.data.loader import load_subject_raw_with_events
from src.data.normalization import FeatureScaler, PerSubjectZScore
from src.features.bandpass import FixedBandCSPFeatures
from src.labels import LABEL_ORDER, labels_to_indices
from src.snn.encoder import FixedThresholdEncoder
from src.snn.evaluator import compute_metrics
from src.snn.lif_net import BaselineLIFNet
from src.snn.synops import count_synops
from src.snn.trainer import BaselineTrainer, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_or_process_subject(subject_id: str, preprocess_cfg: dict, cache_dir: Path) -> tuple:
    """Load cached preprocessed trials for a subject, or preprocess and cache.

    Caching is safe: Stage 1 preprocessing is deterministic (seeded ICA) and
    fits no label-dependent parameters, so it can run once per subject
    outside the CV loop (bandpass/ICA/epoching are leakage-safe per the
    paper's protocol; z-score/CSP/encoder-threshold are fit per fold).

    Returns
    -------
    (X, y) : (np.ndarray, np.ndarray)
        Trials ``(n_trials, n_channels, n_timepoints)`` float32 and
        0-indexed integer labels.
    """
    cache_file = cache_dir / f"{subject_id}.npz"
    if cache_file.exists():
        blob = np.load(cache_file)
        return blob["X"], blob["y"]

    from src.data.preprocessing import Preprocessor

    raw, run_files, events, event_id = load_subject_raw_with_events(subject_id)
    pre = Preprocessor(preprocess_cfg)
    preprocess_cfg["subject_id"] = subject_id
    X, labels, _ = pre.process_subject(raw, run_files=run_files, events=events, event_id=event_id)
    y = np.asarray(labels_to_indices(labels), dtype=np.int64)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X.astype(np.float32), y=y)
    return X.astype(np.float32), y


def run_baseline_cv(config: dict, preprocess_cfg: dict, subject_subset=None) -> None:
    """Full baseline pipeline: load -> balance -> 5-fold CV -> save results."""
    output_dir = Path(config["results"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["data"].get("cache_dir", "outputs/preprocessed"))
    seed = config["experiment"]["seed"]
    np.random.seed(seed)

    log.info("=== Phase 2: Baseline SNN 5-fold subject-mixed CV ===")

    # -- Step 1: load + balance all subjects ---------------------------
    excluded = set(config["data"]["excluded_subjects"])
    all_subjects = [f"S{i:03d}" for i in range(1, 110) if f"S{i:03d}" not in excluded]
    if subject_subset:
        all_subjects = [s for s in all_subjects if s in subject_subset]

    bal_cfg = config["balancing"]
    X_parts, y_parts, subj_parts = [], [], []
    n_ok = 0
    for i, subj in enumerate(all_subjects, 1):
        try:
            X, y = load_or_process_subject(subj, preprocess_cfg, cache_dir)
            X, y, _ = balance_classes(
                X, y,
                strategy=bal_cfg["strategy"],
                random_state=bal_cfg["random_state"],
            )
            X_parts.append(X)
            y_parts.append(y)
            subj_parts.append([subj] * len(y))
            n_ok += 1
            log.info(f"[{i}/{len(all_subjects)}] {subj}: {len(y)} balanced trials")
        except Exception as e:
            log.warning(f"Skipping {subj}: {e}")
    if n_ok == 0:
        raise RuntimeError("No subjects loaded successfully")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    subjects_all = np.concatenate(subj_parts)
    log.info(
        f"Loaded {n_ok} subjects | {len(y_all)} trials | "
        f"class counts: {dict(zip(*np.unique(y_all, return_counts=True)))} "
        f"(0={LABEL_ORDER[0]}, 1={LABEL_ORDER[1]}, 2={LABEL_ORDER[2]}, 3={LABEL_ORDER[3]})"
    )

    # -- Step 2: trial-pooled 5-fold CV (paper Section 6.2) ------------
    device = "cpu"
    if config["experiment"]["device"] == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    feat_cfg = config["features"]
    enc_cfg = config["encoder"]
    snn_cfg = config["snn"]
    train_cfg = config["training"]
    ev_cfg = config["evaluation"]

    fold_results = []
    confusion_matrices = []
    kfold = subject_mixed_kfold(n_splits=ev_cfg["cv_folds"], random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_all)):
        log.info(f"--- Fold {fold_idx + 1}/{ev_cfg['cv_folds']} "
                 f"(train {len(train_idx)} | test {len(test_idx)} trials) ---")
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        subj_train, subj_test = subjects_all[train_idx], subjects_all[test_idx]

        # Stage 1: per-subject z-score, fit on TRAIN fold only
        zscore = PerSubjectZScore().fit(X_train, list(subj_train))
        X_train = zscore.transform(X_train, list(subj_train))
        X_test = zscore.transform(X_test, list(subj_test))

        # Stage 2+3: per-band CSP features, fit on TRAIN fold only
        extractor = FixedBandCSPFeatures(
            bands=[tuple(b) for b in feat_cfg["bands"]],
            n_components=feat_cfg["n_components"],
            random_state=seed,
        )
        extractor.fit(X_train, y_train)
        F_train = extractor.transform(X_train)
        F_test = extractor.transform(X_test)

        # Feature scaling, fit on TRAIN fold only
        scaler = FeatureScaler().fit(F_train)
        F_train = scaler.transform(F_train)
        F_test = scaler.transform(F_test)

        # Stage 4: fixed threshold from TRAIN features only
        encoder = FixedThresholdEncoder(T=enc_cfg["T"])
        encoder.set_threshold_from_data(F_train)

        # Stage 5: baseline LIF SNN
        model = BaselineLIFNet(
            n_features=F_train.shape[1],
            n_hidden=snn_cfg["n_hidden"],
            n_classes=len(LABEL_ORDER),
            tau=snn_cfg["tau"],
            T=enc_cfg["T"],
            dropout=snn_cfg["dropout"],
        )
        cfg = TrainingConfig(
            n_epochs=train_cfg["n_epochs"],
            batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            max_grad_norm=train_cfg["max_grad_norm"],
            T=enc_cfg["T"],
            lambda_spike=train_cfg["lambda_spike"],
            device=device,
            random_state=seed + fold_idx,
        )
        trainer = BaselineTrainer(model, cfg, encoder=encoder)
        trainer.fit(F_train, y_train, verbose=False)

        # Stage 6: evaluate + SynOps
        eval_out = trainer.evaluate(F_test, y_test)
        y_pred = eval_out["predictions"]
        synops = count_synops(model, F_test[: min(32, len(F_test))], T=enc_cfg["T"], device=device, threshold=encoder.threshold)

        result = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            fold=fold_idx,
            synops_per_trial=synops["synops_per_trial"],
            spike_density=synops["spike_density"],
        )
        fold_results.append(result.to_dict())
        confusion_matrices.append(result.confusion_matrix)
        log.info(
            f"Fold {fold_idx + 1}: acc={result.accuracy:.4f} | kappa={result.kappa:.4f} "
            f"| f1_macro={result.f1_macro:.4f} | SynOps/trial={synops['synops_per_trial']:.1f}"
        )

    # -- Step 3: save results ------------------------------------------
    df = pd.DataFrame(fold_results)
    df.to_csv(output_dir / "fold_results.csv", index=False)

    summary = {
        "experiment": config["experiment"]["name"],
        "n_folds": len(fold_results),
        "n_subjects": n_ok,
        "n_trials_total": int(len(y_all)),
        "cv_type": ev_cfg["cv_type"],
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std()),
        "kappa_mean": float(df["kappa"].mean()),
        "kappa_std": float(df["kappa"].std()),
        "f1_macro_mean": float(df["f1_macro"].mean()),
        "f1_macro_std": float(df["f1_macro"].std()),
        "synops_per_trial_mean": float(df["synops_per_trial"].mean()),
        "spike_density_mean": float(df["spike_density"].mean()),
        "per_fold": fold_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(output_dir / "confusion_matrices.npy", np.array(confusion_matrices))

    log.info("=== BASELINE SNN RESULTS ===")
    log.info(f"Accuracy : {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
    log.info(f"Kappa    : {summary['kappa_mean']:.4f} +/- {summary['kappa_std']:.4f}")
    log.info(f"F1 Macro : {summary['f1_macro_mean']:.4f} +/- {summary['f1_macro_std']:.4f}")
    log.info(f"SynOps/trial: {summary['synops_per_trial_mean']:.1f}")
    log.info(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 baseline SNN CV runner")
    parser.add_argument("--config", default="config/phase2_config.yaml")
    parser.add_argument("--preprocess-config", default="config/phase1_config.yaml",
                        help="Phase 1 config used for Stage 1 preprocessing params")
    parser.add_argument("--subjects", default=None,
                        help="Comma-separated subject subset for smoke runs, e.g. S001,S002,S003")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    with open(args.preprocess_config) as f:
        preprocess_cfg = yaml.safe_load(f)

    subset = set(args.subjects.split(",")) if args.subjects else None
    run_baseline_cv(config, preprocess_cfg, subject_subset=subset)


if __name__ == "__main__":
    main()
