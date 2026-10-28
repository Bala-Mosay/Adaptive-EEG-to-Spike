import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.experiments.cv import run_subject_mixed_cv

# create synthetic data: 100 trials, 8 channels, 50 timepoints
rng = np.random.RandomState(0)
X = rng.randn(100, 8, 50)
# create labels (4 classes)
y = np.repeat(np.arange(4), 25)
# create subjects: 10 subjects repeated
subjects = [f"S{(i%10)+1:03d}" for i in range(100)]

cfg = {
    'cv': {'n_splits': 5, 'random_state': 0},
    'rcsp': {'n_components': 4}
}

res = run_subject_mixed_cv(X, y, subjects, cfg)
print('Fold scores:', res['fold_scores'])
print('Mean accuracy:', res['mean_accuracy'])
