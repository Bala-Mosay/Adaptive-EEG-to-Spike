from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.cv.splits import subject_mixed_kfold
from src.data.normalization import PerSubjectZScore
from src.spatial.rcsp import RegularizedCSP


def run_subject_mixed_cv(X: np.ndarray, y: np.ndarray, subjects: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run subject-mixed KFold CV ensuring normalization and RCSP fit only on train.

    X: (n_trials, n_channels, n_time)
    y: (n_trials,)
    subjects: list of subject IDs length n_trials
    cfg: configuration dict (expects rcsp.n_components and cv.n_splits)

    Returns dict with per-fold accuracy and trained models list.
    """
    n_splits = cfg.get("cv", {}).get("n_splits", 5)
    kf = subject_mixed_kfold(n_splits=n_splits, random_state=cfg.get("cv", {}).get("random_state", 42))

    if X.shape[0] < n_splits:
        raise ValueError(f"Not enough samples for {n_splits}-fold CV: n_samples={X.shape[0]}")

    fold_scores = []
    models = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subj_train = [subjects[i] for i in train_idx]
        subj_test = [subjects[i] for i in test_idx]

        # Per-subject z-score fit on training trials only
        normalizer = PerSubjectZScore()
        X_train_norm = normalizer.fit_transform(X_train, subj_train)
        X_test_norm = normalizer.transform(X_test, subj_test)

        # RCSP fit on training set only
        rcfg = cfg.get("rcsp", {})
        rcsp = RegularizedCSP(n_components=rcfg.get("n_components", 4))
        rcsp.fit(X_train_norm, y_train)
        X_train_feats = rcsp.transform(X_train_norm)
        X_test_feats = rcsp.transform(X_test_norm)

        # simple classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_feats, y_train)
        y_pred = clf.predict(X_test_feats)
        acc = accuracy_score(y_test, y_pred)
        fold_scores.append(acc)
        models.append({"normalizer": normalizer, "rcsp": rcsp, "clf": clf})

    return {"fold_scores": fold_scores, "models": models, "mean_accuracy": float(np.mean(fold_scores))}


def run_loso(X: np.ndarray, y: np.ndarray, subjects: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run Leave-One-Subject-Out evaluation.

    For each subject in `subjects` unique set, use that subject's trials as test
    and the rest as train. Returns per-subject accuracy and mean accuracy.
    """
    unique_subs = list(dict.fromkeys(subjects))
    if len(unique_subs) < 2:
        raise ValueError("LOSO requires at least 2 subjects' data to run")

    per_subj_scores = {}
    models = {}
    for subj in unique_subs:
        test_idx = [i for i, s in enumerate(subjects) if s == subj]
        train_idx = [i for i in range(len(subjects)) if i not in test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        subj_train = [subjects[i] for i in train_idx]
        subj_test = [subjects[i] for i in test_idx]

        normalizer = PerSubjectZScore()
        X_train_norm = normalizer.fit_transform(X_train, subj_train)
        X_test_norm = normalizer.transform(X_test, subj_test)

        rcfg = cfg.get("rcsp", {})
        rcsp = RegularizedCSP(n_components=rcfg.get("n_components", 4))
        rcsp.fit(X_train_norm, y_train)
        X_train_feats = rcsp.transform(X_train_norm)
        X_test_feats = rcsp.transform(X_test_norm)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_feats, y_train)
        y_pred = clf.predict(X_test_feats)
        acc = accuracy_score(y_test, y_pred)
        per_subj_scores[subj] = float(acc)
        models[subj] = {"normalizer": normalizer, "rcsp": rcsp, "clf": clf}

    mean_acc = float(np.mean(list(per_subj_scores.values())))
    return {"per_subject": per_subj_scores, "mean_accuracy": mean_acc, "models": models}
