#!/usr/bin/env python
"""Phase 3 — Contribution 1: adaptive KNN-MI band selection.

Controlled checkpoint: the encoder (fixed threshold) and network (single-tau
LIF) are identical to Phase 2.  The only change is that each subject keeps
the top-3 bands from a 7-band filter bank, selected by KNN-MI on the
training fold.

The runner is checkpointed per fold.  A long run can be stopped and resumed
with ``--resume`` without retraining completed folds:

    python -u experiments/run_phase3.py --config config/phase3_config.yaml \
        --log-file results/phase3_c1_run.log
    # after interruption
    python -u experiments/run_phase3.py --config config/phase3_config.yaml \
        --log-file results/phase3_c1_run.log --resume

Usage (smoke):
    python experiments/run_phase3.py --subjects S001,S002,S003 --max-folds 1 \
        --epochs 2 --output-dir results/smoke/phase3_c1 --overwrite

Outputs (under ``config["results"]["output_dir"]``):
    checkpoint.json              # completed folds + run metadata
    folds/fold_{k}.json          # per-fold metrics + selected bands
    folds/fold_{k}_predictions.npz
    summary.json
    fold_results.csv
    confusion_matrices.npy
    predictions.npz
    selected_bands.json
    config_snapshot.yaml
    run_metadata.json
"""
import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv.splits import subject_mixed_kfold
from src.data.balancer import balance_classes
from src.data.cache import load_or_process_subject
from src.data.normalization import FeatureScaler, PerSubjectZScore
from src.features.adaptive_bands import AdaptiveBandCSPFeatures
from src.labels import LABEL_ORDER
from src.snn.encoder import FixedThresholdEncoder
from src.snn.evaluator import compute_metrics
from src.snn.lif_net import BaselineLIFNet
from src.snn.synops import count_synops
from src.snn.trainer import BaselineTrainer, TrainingConfig
from src.utils.seed import seed_everything

# ``force=True`` is required: some imported libraries call
# ``logging.basicConfig()`` at import time, which makes a plain
# ``basicConfig(level=INFO)`` a no-op and leaves the root logger at WARNING.
# Without the force, every ``log.info`` progress line is silently dropped.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

_CHECKPOINT_FILE = "checkpoint.json"
_FOLDS_DIR = "folds"
_FINAL_FILES = (
    "summary.json",
    "fold_results.csv",
    "confusion_matrices.npy",
    "predictions.npz",
    "selected_bands.json",
    "run_metadata.json",
    "config_snapshot.yaml",
)


def _format_duration(seconds: float) -> str:
    """Format seconds as a short human-readable duration."""
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


class _StatusBar:
    """Minimal terminal status bar (no external dependency).

    It is silent when stdout is not a TTY (e.g. redirected to a file), so
    log files stay clean.  Set ``enabled=True`` explicitly in tests.
    """

    def __init__(
        self,
        label: str,
        total: int,
        width: int = 28,
        stream=None,
        enabled: Optional[bool] = None,
    ):
        self.label = label
        self.total = max(int(total), 1)
        self.width = width
        self.stream = stream if stream is not None else sys.stdout
        if enabled is None:
            enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = enabled
        self.start = time.time()

    def update(self, n: int, suffix: str = "") -> None:
        n = max(0, min(int(n), self.total))
        if not self.enabled:
            # Non-interactive stream (background run, redirected output): the
            # bar cannot render, so log coarse progress instead of nothing.
            step = max(1, self.total // 10)
            if n == self.total or (n > 0 and n % step == 0):
                extra = f" | {suffix}" if suffix else ""
                log.info(
                    "%s: %d/%d (%.0f%%)%s",
                    self.label,
                    n,
                    self.total,
                    100.0 * n / self.total,
                    extra,
                )
            return
        frac = n / self.total
        filled = int(self.width * frac)
        bar = "=" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = (elapsed / n * (self.total - n)) if n > 0 else 0.0
        msg = f"\r{self.label} [{bar}] {n}/{self.total} ({100.0 * frac:5.1f}%)"
        if n > 0 and n < self.total:
            msg += f" ETA {_format_duration(eta)}"
        if suffix:
            msg += f" | {suffix}"
        self.stream.write(msg)
        self.stream.flush()

    def close(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


def _atomic_write_json(path: Path, payload: object) -> None:
    """Write JSON via a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    tmp.replace(path)


def _atomic_savez(path: Path, **arrays) -> None:
    """Write a compressed .npz via a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write a .npy via a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def _config_fingerprint(
    config: dict,
    effective_epochs: int,
    subject_subset: Optional[set] = None,
) -> str:
    """Fingerprint the scientific config so resume cannot mix runs."""
    relevant = {
        "experiment": config["experiment"],
        "data": config["data"],
        "balancing": config["balancing"],
        "features": config["features"],
        "encoder": config["encoder"],
        "snn": config["snn"],
        "training": {**config["training"], "n_epochs": int(effective_epochs)},
        "evaluation": config["evaluation"],
        "subject_subset": sorted(subject_subset) if subject_subset else None,
    }
    blob = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _read_checkpoint(checkpoint_file: Path, fingerprint: str) -> set:
    """Read completed folds, refusing a checkpoint from a different config."""
    if not checkpoint_file.exists():
        return set()
    with open(checkpoint_file, encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("config_fingerprint") != fingerprint:
        raise RuntimeError(
            "Checkpoint config does not match the current config. "
            "Use --overwrite to discard it, or restore the original config."
        )
    return {int(fold) for fold in payload.get("completed_folds", [])}


def _write_checkpoint(
    checkpoint_file: Path,
    fingerprint: str,
    effective_epochs: int,
    requested: List[int],
    completed: set,
    n_subjects: Optional[int] = None,
    n_trials: Optional[int] = None,
    feature_dim: Optional[int] = None,
) -> None:
    payload = {
        "config_fingerprint": fingerprint,
        "effective_epochs": int(effective_epochs),
        "requested_folds": sorted(requested),
        "completed_folds": sorted(completed),
        "n_subjects_loaded": n_subjects,
        "n_trials_total": n_trials,
        "feature_dim": feature_dim,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _atomic_write_json(checkpoint_file, payload)


def _prepare_checkpoint(
    output_dir: Path,
    checkpoint_file: Path,
    folds_dir: Path,
    fingerprint: str,
    resume: bool,
    overwrite: bool,
) -> set:
    """Prepare the output directory and return already-completed folds."""
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    if overwrite:
        if folds_dir.exists():
            shutil.rmtree(folds_dir)
        for name in (_CHECKPOINT_FILE, *_FINAL_FILES):
            path = output_dir / name
            if path.exists():
                path.unlink()
        return set()

    if checkpoint_file.exists() and not resume:
        raise RuntimeError(
            f"{checkpoint_file} already exists. Use --resume to continue it "
            "or --overwrite to start a fresh run."
        )
    return _read_checkpoint(checkpoint_file, fingerprint)


def _save_fold_checkpoint(
    folds_dir: Path,
    fold_idx: int,
    result,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: np.ndarray,
    selected_bands: Dict[str, List[str]],
    test_subject_bands: Dict[str, List[str]],
    feature_dim: int,
) -> None:
    payload = {
        "fold": int(fold_idx),
        "metrics": result.to_dict(),
        "confusion_matrix": result.confusion_matrix.tolist(),
        "feature_dim": int(feature_dim),
        "selected_bands": selected_bands,
        "test_subject_bands": test_subject_bands,
    }
    _atomic_write_json(folds_dir / f"fold_{fold_idx}.json", payload)
    _atomic_savez(
        folds_dir / f"fold_{fold_idx}_predictions.npz",
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        subjects=np.asarray(subjects),
        fold=np.full(len(y_true), fold_idx, dtype=np.int64),
    )


def _load_fold_checkpoint(folds_dir: Path, fold_idx: int):
    """Load one fold checkpoint; returns ``None`` if files are missing."""
    result_file = folds_dir / f"fold_{fold_idx}.json"
    prediction_file = folds_dir / f"fold_{fold_idx}_predictions.npz"
    if not result_file.exists() or not prediction_file.exists():
        return None

    with open(result_file, encoding="utf-8") as fh:
        payload = json.load(fh)
    required = {
        "fold",
        "metrics",
        "confusion_matrix",
        "feature_dim",
        "selected_bands",
        "test_subject_bands",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"fold {fold_idx} checkpoint missing keys: {sorted(missing)}")

    with np.load(prediction_file) as blob:
        arrays = {name: blob[name] for name in ("y_true", "y_pred", "subjects", "fold")}
    if len(arrays["y_true"]) != len(arrays["y_pred"]):
        raise ValueError(f"fold {fold_idx} checkpoint has mismatched predictions")
    return payload, arrays


def _restore_completed_folds(
    folds_dir: Path, completed: set, requested: List[int]
):
    """Validate all completed folds and load the ones requested now."""
    all_loaded = {}
    for fold_idx in sorted(completed):
        try:
            loaded = _load_fold_checkpoint(folds_dir, fold_idx)
            if loaded is None:
                raise FileNotFoundError("one or both checkpoint files are missing")
            all_loaded[fold_idx] = loaded
        except Exception as exc:
            log.warning("Discarding unreadable checkpoint for fold %d: %s", fold_idx, exc)
            completed.discard(fold_idx)
    return {fold: all_loaded[fold] for fold in requested if fold in all_loaded}


def _git_commit() -> str:
    """Return the current git commit, or ``"unknown"``."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_phase3_cv(
    config: dict,
    preprocess_cfg: dict,
    subject_subset=None,
    max_folds: Optional[int] = None,
    n_epochs: Optional[int] = None,
    resume: bool = False,
    overwrite: bool = False,
    fold_indices: Optional[List[int]] = None,
) -> None:
    """Run the C1 adaptive-band cross-validation with per-fold checkpoints."""
    output_dir = Path(config["results"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / _FOLDS_DIR
    checkpoint_file = output_dir / _CHECKPOINT_FILE
    cache_dir = Path(config["data"].get("cache_dir", "outputs/preprocessed"))
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)

    device = "cpu"
    if config["experiment"]["device"] == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = config["training"]
    effective_epochs = int(
        n_epochs if n_epochs is not None else train_cfg["n_epochs"]
    )
    n_splits = int(config["evaluation"]["cv_folds"])
    all_fold_indices = list(range(n_splits))
    if fold_indices is None:
        requested = all_fold_indices[:max_folds] if max_folds is not None else all_fold_indices
    else:
        requested = sorted({int(fold) for fold in fold_indices})
    invalid = [fold for fold in requested if fold not in all_fold_indices]
    if invalid:
        raise ValueError(f"fold indices {invalid} outside range 0..{n_splits - 1}")
    if not requested:
        raise ValueError("No folds requested")

    fingerprint = _config_fingerprint(config, effective_epochs, subject_subset)
    completed = _prepare_checkpoint(
        output_dir,
        checkpoint_file,
        folds_dir,
        fingerprint,
        resume=resume,
        overwrite=overwrite,
    )
    loaded_folds = _restore_completed_folds(folds_dir, completed, requested)

    log.info("=== Phase 3 C1: adaptive KNN-MI band selection ===")
    log.info("Device: %s", device)
    log.info(
        "Requested folds: %s | completed: %s",
        requested,
        sorted(completed),
    )
    if loaded_folds:
        log.info("Resuming completed folds: %s", sorted(loaded_folds))

    # -- Load + balance all subjects ------------------------------------
    excluded = set(config["data"]["excluded_subjects"])
    all_subjects = [
        f"S{i:03d}" for i in range(1, 110) if f"S{i:03d}" not in excluded
    ]
    if subject_subset:
        all_subjects = [s for s in all_subjects if s in subject_subset]

    bal_cfg = config["balancing"]
    X_parts, y_parts, subj_parts = [], [], []
    n_ok = 0
    for i, subj in enumerate(all_subjects, 1):
        try:
            X, y = load_or_process_subject(subj, preprocess_cfg, cache_dir)
            X, y, _ = balance_classes(
                X,
                y,
                strategy=bal_cfg["strategy"],
                random_state=bal_cfg["random_state"],
            )
            X_parts.append(X)
            y_parts.append(y)
            subj_parts.append([subj] * len(y))
            n_ok += 1
            log.info("[%d/%d] %s: %d balanced trials", i, len(all_subjects), subj, len(y))
        except Exception as exc:
            log.warning("Skipping %s: %s", subj, exc)

    if n_ok == 0:
        raise RuntimeError("No subjects loaded successfully")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    subjects_all = np.concatenate(subj_parts)
    # Each subject array is held separately above; once concatenated into the
    # full (n_trials, n_channels, n_timepoints) float32 block they would double
    # the resident trial memory for the rest of the run.  Release them now.
    del X_parts, y_parts, subj_parts
    log.info(
        "Loaded %d subjects | %d trials | class counts: %s",
        n_ok,
        len(y_all),
        dict(zip(*np.unique(y_all, return_counts=True))),
    )

    if completed and checkpoint_file.exists():
        with open(checkpoint_file, encoding="utf-8") as fh:
            previous = json.load(fh)
        prev_n = previous.get("n_subjects_loaded")
        prev_t = previous.get("n_trials_total")
        if prev_n is not None and int(prev_n) != n_ok:
            raise RuntimeError(
                f"Checkpoint was built with {prev_n} subjects but this run loaded "
                f"{n_ok}. Use --overwrite if the dataset changed."
            )
        if prev_t is not None and int(prev_t) != len(y_all):
            raise RuntimeError(
                f"Checkpoint was built with {prev_t} trials but this run loaded "
                f"{len(y_all)}. Use --overwrite if the dataset changed."
            )

    feat_cfg = config["features"]
    sel_cfg = feat_cfg.get("band_selection", {})
    band_bank = [tuple(band) for band in feat_cfg["band_bank"]]
    top_k = int(feat_cfg["top_k"])
    n_components = int(feat_cfg["n_components"])
    sfreq = float(config["data"].get("sfreq", 160.0))

    enc_cfg = config["encoder"]
    snn_cfg = config["snn"]
    ev_cfg = config["evaluation"]

    kfold = subject_mixed_kfold(n_splits=n_splits, random_state=seed)
    splits = list(kfold.split(X_all))

    fold_results = []
    confusion_matrices = []
    predictions = {"y_true": [], "y_pred": [], "subjects": [], "fold": []}
    selected_bands = {}
    bands_used_test = {}
    feature_dim = None
    resumed_folds = sorted(loaded_folds)

    fold_bar = _StatusBar("Folds", len(requested))
    done_count = len(loaded_folds)
    fold_bar.update(done_count, "restored")

    _write_checkpoint(
        checkpoint_file,
        fingerprint,
        effective_epochs,
        requested,
        completed,
        n_subjects=n_ok,
        n_trials=len(y_all),
    )

    for fold_idx in requested:
        if fold_idx in loaded_folds:
            payload, arrays = loaded_folds[fold_idx]
            fold_results.append(payload["metrics"])
            confusion_matrices.append(
                np.asarray(payload["confusion_matrix"], dtype=np.int64)
            )
            for key in predictions:
                predictions[key].append(arrays[key])
            selected_bands[f"fold_{fold_idx}"] = payload["selected_bands"]
            bands_used_test[f"fold_{fold_idx}"] = payload["test_subject_bands"]
            feature_dim = int(payload["feature_dim"])
            log.info("Fold %d restored from checkpoint; skipping.", fold_idx)
            fold_bar.update(done_count, f"fold {fold_idx} restored")
            continue

        train_idx, test_idx = splits[fold_idx]
        log.info(
            "--- Fold %d/%d (train %d | test %d) ---",
            fold_idx + 1,
            n_splits,
            len(train_idx),
            len(test_idx),
        )
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        subj_train, subj_test = subjects_all[train_idx], subjects_all[test_idx]

        # Stage 1 normalization: fit on training trials only.
        zscore = PerSubjectZScore().fit(X_train, list(subj_train))
        X_train = zscore.transform(X_train, list(subj_train))
        X_test = zscore.transform(X_test, list(subj_test))

        # Stage 2/3: full-bank CSP + per-subject top-k band selection.
        extractor = AdaptiveBandCSPFeatures(
            bands=band_bank,
            n_components=n_components,
            top_k=top_k,
            sfreq=sfreq,
            n_neighbors=int(sel_cfg.get("n_neighbors", 5)),
            min_trials=int(sel_cfg.get("min_trials_per_subject", 20)),
            random_state=seed,
        )
        extractor.fit(X_train, y_train, subj_train)
        F_train = extractor.transform(X_train, subj_train)
        F_test = extractor.transform(X_test, subj_test)
        fold_feature_dim = int(F_train.shape[1])
        if feature_dim is not None and feature_dim != fold_feature_dim:
            raise RuntimeError(
                f"Fold {fold_idx} produced {fold_feature_dim} features but restored "
                f"folds used {feature_dim}; check the band bank/config."
            )
        feature_dim = fold_feature_dim

        fold_selected_bands = extractor.get_selected_bands()
        fold_test_bands = {
            str(subj): extractor.selector_.get_bands(str(subj))
            for subj in np.unique(subj_test)
        }
        selected_bands[f"fold_{fold_idx}"] = fold_selected_bands
        bands_used_test[f"fold_{fold_idx}"] = fold_test_bands

        # Feature scaling: fit on training features only.
        scaler = FeatureScaler().fit(F_train)
        F_train = scaler.transform(F_train)
        F_test = scaler.transform(F_test)

        # Stage 4: fixed threshold from training features only.
        encoder = FixedThresholdEncoder(T=int(enc_cfg["T"]))
        encoder.set_threshold_from_data(F_train)

        # Stage 5: baseline single-tau LIF.
        model = BaselineLIFNet(
            n_features=F_train.shape[1],
            n_hidden=int(snn_cfg["n_hidden"]),
            n_classes=len(LABEL_ORDER),
            tau=float(snn_cfg["tau"]),
            T=int(enc_cfg["T"]),
            dropout=float(snn_cfg["dropout"]),
        )
        cfg = TrainingConfig(
            n_epochs=effective_epochs,
            batch_size=int(train_cfg["batch_size"]),
            learning_rate=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
            max_grad_norm=float(train_cfg["max_grad_norm"]),
            T=int(enc_cfg["T"]),
            lambda_spike=float(train_cfg.get("lambda_spike", 0.0)),
            device=device,
            random_state=seed + fold_idx,
        )
        trainer = BaselineTrainer(model, cfg, encoder=encoder)

        epoch_bar = _StatusBar(f"Fold {fold_idx} train", effective_epochs)
        trainer.fit(
            F_train,
            y_train,
            verbose=False,
            progress_callback=lambda epoch, record, bar=epoch_bar: bar.update(
                epoch + 1, f"CE={record['ce_loss']:.4f}"
            ),
        )
        epoch_bar.close()

        # Stage 6: evaluate + SynOps.
        eval_out = trainer.evaluate(F_test, y_test)
        y_pred = eval_out["predictions"]
        synops = count_synops(
            model,
            F_test[: min(32, len(F_test))],
            T=int(enc_cfg["T"]),
            device=device,
            threshold=encoder.threshold,
        )
        result = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            fold=fold_idx,
            synops_per_trial=synops["synops_per_trial"],
            spike_density=synops["spike_density"],
        )

        _save_fold_checkpoint(
            folds_dir,
            fold_idx,
            result,
            y_test,
            y_pred,
            subj_test,
            fold_selected_bands,
            fold_test_bands,
            feature_dim,
        )
        completed.add(fold_idx)
        _write_checkpoint(
            checkpoint_file,
            fingerprint,
            effective_epochs,
            requested,
            completed,
            n_subjects=n_ok,
            n_trials=len(y_all),
            feature_dim=feature_dim,
        )

        fold_results.append(result.to_dict())
        confusion_matrices.append(result.confusion_matrix)
        predictions["y_true"].append(y_test)
        predictions["y_pred"].append(y_pred)
        predictions["subjects"].append(subj_test)
        predictions["fold"].append(np.full(len(y_test), fold_idx, dtype=np.int64))

        done_count += 1
        fold_bar.update(
            done_count,
            f"fold {fold_idx} acc={result.accuracy:.3f}",
        )
        log.info(
            "Fold %d: acc=%.4f | kappa=%.4f | f1_macro=%.4f | "
            "SynOps/trial=%.1f | density=%.4f | features=%d",
            fold_idx + 1,
            result.accuracy,
            result.kappa,
            result.f1_macro,
            result.synops_per_trial,
            result.spike_density,
            F_train.shape[1],
        )

        # Release per-fold buffers before the next fold allocates its own
        # ~1.5 GB train/test copies.  ``y_test``/``subj_test`` stay alive
        # through the prediction lists appended above.
        del X_train, X_test, y_train, subj_train, F_train, F_test, extractor

    fold_bar.close()

    if not fold_results:
        raise RuntimeError("No folds completed")

    _write_checkpoint(
        checkpoint_file,
        fingerprint,
        effective_epochs,
        requested,
        completed,
        n_subjects=n_ok,
        n_trials=len(y_all),
        feature_dim=feature_dim,
    )

    order = sorted(range(len(fold_results)), key=lambda i: fold_results[i]["fold"])
    fold_results = [fold_results[i] for i in order]
    confusion_matrices = [confusion_matrices[i] for i in order]
    for key in predictions:
        predictions[key] = [predictions[key][i] for i in order]

    # -- Save final results ---------------------------------------------
    df = pd.DataFrame(fold_results)
    df.to_csv(output_dir / "fold_results.csv", index=False)

    def _mean_std(column: str) -> tuple:
        values = df[column]
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return mean, std

    accuracy_mean, accuracy_std = _mean_std("accuracy")
    kappa_mean, kappa_std = _mean_std("kappa")
    f1_mean, f1_std = _mean_std("f1_macro")

    all_folds_complete = set(all_fold_indices).issubset(completed)
    summary = {
        "experiment": config["experiment"]["name"],
        "n_folds": len(fold_results),
        "requested_folds": requested,
        "completed_folds": sorted(completed),
        "resumed_folds": resumed_folds,
        "all_folds_complete": all_folds_complete,
        "partial": not all_folds_complete,
        "effective_epochs": effective_epochs,
        "config_fingerprint": fingerprint,
        "n_subjects": n_ok,
        "n_trials_total": int(len(y_all)),
        "cv_type": ev_cfg["cv_type"],
        "band_bank": [list(b) for b in band_bank],
        "top_k": top_k,
        "feature_dim": feature_dim,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "kappa_mean": kappa_mean,
        "kappa_std": kappa_std,
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "synops_per_trial_mean": float(df["synops_per_trial"].mean()),
        "spike_density_mean": float(df["spike_density"].mean()),
        "per_fold": fold_results,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_save_npy(
        output_dir / "confusion_matrices.npy", np.array(confusion_matrices)
    )

    if config["results"].get("save_predictions", True):
        _atomic_savez(
            output_dir / "predictions.npz",
            y_true=np.concatenate(predictions["y_true"]),
            y_pred=np.concatenate(predictions["y_pred"]),
            subjects=np.concatenate(predictions["subjects"]),
            fold=np.concatenate(predictions["fold"]),
        )

    if config["results"].get("save_selected_bands", True):
        _atomic_write_json(
            output_dir / "selected_bands.json",
            {"per_fold": selected_bands, "test_subjects": bands_used_test},
        )

    if config["results"].get("save_config", True):
        with open(output_dir / "config_snapshot.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)

    metadata = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "device": device,
        "cv_type": ev_cfg["cv_type"],
        "n_subjects_loaded": n_ok,
        "n_trials": int(len(y_all)),
        "feature_dim": feature_dim,
        "effective_epochs": effective_epochs,
        "requested_folds": requested,
        "completed_folds": sorted(completed),
        "resumed_folds": resumed_folds,
        "all_folds_complete": all_folds_complete,
        "config_fingerprint": fingerprint,
        "git_commit": _git_commit(),
    }
    _atomic_write_json(output_dir / "run_metadata.json", metadata)

    log.info("=== PHASE 3 C1 RESULTS ===")
    log.info("Accuracy : %.4f +/- %.4f", accuracy_mean, accuracy_std)
    log.info("Kappa    : %.4f +/- %.4f", kappa_mean, kappa_std)
    log.info("F1 Macro : %.4f +/- %.4f", f1_mean, f1_std)
    log.info("SynOps/trial: %.1f", summary["synops_per_trial_mean"])
    log.info(
        "Folds complete: %s/%s%s",
        len(completed.intersection(all_fold_indices)),
        len(all_fold_indices),
        "" if all_folds_complete else " (partial)",
    )
    log.info("Results saved to: %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 C1 adaptive-band CV runner")
    parser.add_argument("--config", default="config/phase3_config.yaml")
    parser.add_argument(
        "--preprocess-config",
        default="config/phase1_config.yaml",
        help="Phase 1 config used for Stage 1 preprocessing parameters",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject subset for smoke runs, e.g. S001,S002,S003",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional limit on the number of folds (smoke runs)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override (smoke runs)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override (useful for smoke runs)",
    )
    parser.add_argument(
        "--fold-indices",
        default=None,
        help="Comma-separated 0-based fold indices, e.g. 0,1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a checkpointed run, skipping completed folds",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing checkpoint and start fresh",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Also write logs to this file (status bar stays on stdout)",
    )
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    with open(args.preprocess_config, encoding="utf-8") as fh:
        preprocess_cfg = yaml.safe_load(fh)
    if args.output_dir:
        config["results"]["output_dir"] = args.output_dir

    subset = set(args.subjects.split(",")) if args.subjects else None
    fold_indices = (
        [int(fold) for fold in args.fold_indices.split(",")]
        if args.fold_indices
        else None
    )
    run_phase3_cv(
        config,
        preprocess_cfg,
        subject_subset=subset,
        max_folds=args.max_folds,
        n_epochs=args.epochs,
        resume=args.resume,
        overwrite=args.overwrite,
        fold_indices=fold_indices,
    )


if __name__ == "__main__":
    main()
