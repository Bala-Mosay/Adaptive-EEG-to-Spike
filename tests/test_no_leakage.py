import sys
from pathlib import Path
# Ensure project root is on sys.path so `src` is importable when running tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.normalization import PerSubjectZScore
from src.spatial.rcsp import RegularizedCSP


def test_normalizer_and_rcsp_differ_between_folds():
    # create synthetic data: 20 trials, 4 channels, 100 timepoints, 2 subjects
    rng = np.random.RandomState(0)
    n_trials = 20
    X = rng.randn(n_trials, 4, 100)
    # ensure both folds contain both subjects so per-subject stats exist in each fold
    subjects = ["S001"] * 5 + ["S002"] * 5 + ["S001"] * 5 + ["S002"] * 5
    y = np.array([0, 1] * 10)

    # split into two folds manually
    idx1 = np.arange(0, 10)
    idx2 = np.arange(10, 20)

    # Fit normalizer on fold1 and fold2 separately
    norm1 = PerSubjectZScore().fit(X[idx1], [subjects[i] for i in idx1])
    norm2 = PerSubjectZScore().fit(X[idx2], [subjects[i] for i in idx2])

    # means for S001 should differ because fitted on different data
    m1 = norm1.means.get("S001")
    m2 = norm2.means.get("S001")
    assert m1 is not None
    assert m2 is not None
    assert not np.allclose(m1, m2), "Normalizer means should differ across folds"

    # RCSP: fit separately and ensure filters differ
    rcsp1 = RegularizedCSP(n_components=2).fit(X[idx1], y[idx1])
    rcsp2 = RegularizedCSP(n_components=2).fit(X[idx2], y[idx2])
    f1 = rcsp1.filters_.get('0')
    f2 = rcsp2.filters_.get('0')
    assert f1 is not None and f2 is not None
    assert not np.allclose(f1, f2), "RCSP filters should differ across folds"
