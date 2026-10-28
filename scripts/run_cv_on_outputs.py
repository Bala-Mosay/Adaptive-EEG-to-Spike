import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import glob
from src.experiments.io import assemble_dataset
from src.experiments.cv import run_subject_mixed_cv
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--outputs-glob', default='outputs/*')
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    subject_dirs = [p for p in glob.glob(args.outputs_glob) if Path(p).is_dir()]
    X, y, subjects = assemble_dataset(subject_dirs)
    res = run_subject_mixed_cv(X, y, subjects, cfg)
    print('Fold scores:', res['fold_scores'])
    print('Mean accuracy:', res['mean_accuracy'])


if __name__ == '__main__':
    main()
