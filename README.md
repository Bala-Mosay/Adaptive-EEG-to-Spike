# Adaptive EEG-to-Spike Framework

Neuromorphic decoding of motor imagery from EEG using spiking neural networks (SNNs).

This repository implements an adaptive EEG-to-spike pipeline for 4-class motor-imagery
decoding (`left_hand`, `right_hand`, `feet`, `rest`) on the PhysioNet EEG Motor
Movement/Imagery dataset (EEGMMIDB). Phase 1 (signal conditioning + regularized CSP) and
a Phase 2 baseline SNN are complete. The Phase 3 contributions (adaptive band selection,
learnable spike threshold, dual-timescale PLIF network) are in progress.

## Status

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Signal conditioning, run-aware labels, regularized CSP, CV scaffold | Complete |
| Phase 2 | Fixed-band CSP, fixed-threshold encoder, single-tau LIF SNN, baseline CV | Complete (snapshot) |
| Phase 3 | Adaptive KNN-MI bands, learnable threshold, dual-branch PLIF | In progress |

## Pipeline

| Stage | Description | Implementation |
|---|---|---|
| 1 | Bandpass 0.5-40 Hz, average reference, ICA, epoching, AutoReject, bad-channel interpolation, per-subject z-score | `src/data/preprocessing.py` |
| 2 | Regularized CSP with Ledoit-Wolf shrinkage, One-vs-Rest for 4 classes | `src/spatial/rcsp.py` |
| 3 | Fixed-band CSP features (mu 8-12 Hz, beta 13-30 Hz); adaptive KNN-MI selection planned | `src/features/bandpass.py` |
| 4 | Fixed-threshold spike encoder, T=16 timesteps; learnable threshold planned | `src/snn/encoder.py` |
| 5 | Single-timescale LIF SNN; dual-branch PLIF planned | `src/snn/lif_net.py` |
| 6 | Metrics and SynOps energy proxy | `src/snn/evaluator.py`, `src/snn/synops.py` |

## Dataset

- **PhysioNet EEGMMIDB**: 109 subjects, 64 channels, 160 Hz.
- 103 subjects are used after excluding six subjects with recorded anomalies
  (`S038`, `S088`, `S089`, `S092`, `S100`, `S104`).
- Runs `R04`, `R06`, `R08`, `R10`, `R12`, `R14` are loaded. Label semantics are
  run-dependent: task-1 `T1/T2` = left/right hand; task-2 `T1` = both fists (excluded),
  `T2` = feet. See `src/data/events.py`.
- MNE downloads the EDF files automatically to `~/mne_data`; no EEG data is stored in
  this repository.

## Repository structure

```
config/                 Phase 1 and Phase 2 YAML configurations
src/
  data/                 Loading, run-aware events, preprocessing, balancing, normalization
  features/             Fixed-band CSP features
  spatial/              Regularized CSP
  snn/                  Encoder, LIF network, trainer, evaluator, SynOps
  cv/                   Cross-validation splits
  experiments/          Legacy CV and I/O helpers
experiments/            Phase 2 runners (baseline SNN, LogReg, per-subject diagnostics)
scripts/                Development/diagnostic scripts
tests/                  Unit and leakage-guard tests
notebooks/              Sanity-check notebook
results/                Phase 2 snapshot results
presentation/           Presentation sources
outputs/                Generated caches (gitignored)
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Preprocess one subject (Phase 1):

```bash
python -m src.pipeline --config config/phase1_config.yaml --subject S001
```

Run the Phase 2 baseline SNN with 5-fold subject-mixed CV:

```bash
python experiments/run_baseline.py --config config/phase2_config.yaml
```

Run the feature-bottleneck diagnostics:

```bash
python experiments/run_logreg_diagnostic.py
python experiments/run_per_subject_diagnostic.py
```

Preprocessed trials are cached per subject under `outputs/preprocessed/`. The cache is
regenerable and is not committed.

## Phase 2 snapshot results

Snapshot files: `results/baseline/` and `results/logreg_diagnostic/`.

Protocol: 5-fold **subject-mixed, trial-level** CV over 103 subjects. Trials from the
same subject can appear in both training and test folds. Per-subject z-score, CSP
filters, feature scaling, and the encoder threshold are fit on the training fold only.

| Model | Accuracy | Kappa | F1 macro |
|---|---:|---:|---:|
| Baseline SNN | 31.8% +/- 1.1 | 0.091 | 0.316 |
| Logistic regression (same features) | 34.6% +/- 0.8 | 0.128 | 0.345 |
| Logistic regression, within-subject | 42.0% +/- 10.6 | - | - |

Chance level is 25.0%. The balanced dataset contains 8,672 trials, approximately
84 trials per subject; this resolves the earlier N=84 question in the paper.

### Snapshot caveats

These are historical Phase 2 results and are **not** unseen-subject results:

- The protocol is **subject-mixed trial-level CV**, not leave-one-subject-out (LOSO).
  Every test fold contains trials from subjects the model trained on. The numbers
  therefore describe a calibrated within-subject/global-CSP setting, not cross-subject
  generalization.
- `S091` has an all-zero preprocessed cache and is included in this snapshot as-run.
  The per-subject diagnostic skips it, so it reports 102 subjects.
- The SynOps counter in `src/snn/synops.py` is known to undercount synaptic operations
  (it omits the per-layer neuron count). The reported 0.72 SynOps/trial is not used for
  final energy claims.
- Phase 1 preprocessing (ICA, AutoReject, bad-channel detection) runs per subject before
  CV; it does not use labels, but it is transductive across folds.
- The T=16 encoder repeats one static feature vector per trial, so the network does not
  yet receive time-resolved EEG.
- LOSO is not implemented yet: `PerSubjectZScore` requires the held-out subject to have
  been seen during fitting.

## Configuration

All hyperparameters live in `config/phase1_config.yaml` and
`config/phase2_config.yaml`. Key Phase 2 settings:

```yaml
features.bands: [[mu, 8.0, 12.0], [beta, 13.0, 30.0]]
features.n_components: 4
encoder.T: 16
snn.n_hidden: 64
snn.tau: 20.0
snn.dropout: 0.5
training.n_epochs: 100
training.max_grad_norm: 5.0
evaluation.cv_folds: 5
```

## Tests

```bash
pytest tests/ -v
```

The suite covers signal conditioning, RCSP, run-aware labels, encoding, the LIF network,
SynOps, training determinism, and leakage guards.

## Methodology deviations

Documented in `README`/status files and in code:

- Average EEG referencing before ICA.
- AutoReject instead of a fixed +/-100 uV peak rejection threshold.
- Automatic bad-channel detection and spherical-spline interpolation.
- Standard montage assignment for interpolation.
- Run-aware event labelling; task-2 "both fists" trials excluded.
- Per-subject class balancing (`cap_to_min`) before CV.
- Fixed-threshold encoder uses the 75th percentile of absolute training-feature values.

## Roadmap

1. Fix cache validation and exclude or re-download the corrupted `S091` subject.
2. Correct the SynOps counter and add an analytic unit test.
3. Add subject-disjoint LOSO with subject-independent normalization.
4. Seed Torch/NumPy globally, save predictions/checkpoints, and report multiple seeds.
5. Move Stage 1 fitting inside the CV loop (or quantify the transductive effect).
6. Implement Phase 3: adaptive KNN-MI bands, learnable threshold, dual-branch PLIF.
7. Update all reported numbers and remove the "cross-subject" label where it does not apply.

## References and paper

- `Survey_Paper.pdf` is the author's survey and is included in this repository.
- The third-party reference papers in `refence paper/` are kept locally and are not
  distributed with this repository.

## Author

Bala Mosay J ([@Bala-Mosay](https://github.com/Bala-Mosay))
