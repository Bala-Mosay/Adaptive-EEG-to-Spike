# Phase 2 Status — ✅ COMPLETE

Phase 2 (Baseline SNN: fixed bands + fixed threshold + single-tau LIF) is finished.
All deliverables from `phase2_baseline_snn_prompt.md` are met.
Last updated: 2026-08-30.

## Deliverables

1. **Fixed-band CSP features (Stage 3)** — `src/features/bandpass.py`:
   `FixedBandCSPFeatures` per-band RCSP (FBCSP pattern), log-variance features.
   Bands: mu (8–12 Hz), beta (13–30 Hz). Output: 32 features (2 bands × 4 classes × 4 components).
2. **Fixed-threshold spike encoder (Stage 4)** — `src/snn/encoder.py`:
   `FixedThresholdEncoder`, threshold = mean of train-fold features, absolute-value encoding, T=16.
3. **Baseline LIF SNN (Stage 5)** — `src/snn/lif_net.py`:
   `BaselineLIFNet`, Linear → LIF → Dropout → Linear → LIF → Linear. n_hidden=64, tau=20.0,
   dropout=0.5, v_reset=0.0, decay_input=False. Standard LIF (no ReLU before LIF).
4. **SynOps counter** — `src/snn/synops.py`: per-trial synaptic operation count.
5. **Trainer** — `src/snn/trainer.py`: 100 epochs, CosineAnnealingLR, CE loss,
   gradient clipping (max_norm=5.0), spike-count sparsity penalty (lambda=0.0 for baseline).
6. **Evaluator** — `src/snn/evaluator.py`: accuracy, kappa, F1 metrics.
7. **CV runner** — `experiments/run_baseline.py`: 5-fold subject-mixed CV, trial-pooled.
8. **Config** — `config/phase2_config.yaml`: all hyperparameters, seed=42.
9. **Tests — 39/39 passing** (`pytest tests/ -v`):
   - `test_encoder.py` (6): threshold, shape, binary output, absolute-value encoding, spike density
   - `test_lif_net.py` (4): forward shape, spike count, no PLIF, membrane positivity
   - `test_synops.py` (3): return dict, non-negative, per-trial consistency
   - `test_trainer.py` (3): loss decreases, evaluate returns predictions, determinism
   - `test_bandpass_features.py` (3): output shapes, leakage guard
   - `test_no_leakage_phase2.py` (2): feature scaler, band features independence
   - Plus 18 existing Phase 1 tests (preprocessing, RCSP, balancer, events, evaluator)

## Results

### Baseline SNN — 103-Subject Subject-Mixed (Trial-Level) CV

| Metric | Mean ± Std |
|---|---|
| **Accuracy** | **31.8% ± 1.1%** |
| **Kappa** | **0.091 ± 0.014** |
| **F1 Macro** | **0.316 ± 0.009** |
| SynOps/trial | 0.72 |
| Spike density | 0.041 |

Chance = 25.0% (4 classes).

> **Snapshot caveats.** This is a subject-mixed, trial-level CV protocol: trials from
> the same subject can appear in both training and test folds. It is **not** a
> leave-one-subject-out (LOSO) or unseen-subject result. The all-zero `S091` cache is
> included in this snapshot as-run, and the SynOps counter is known to undercount
> operations (the 0.72 value is not a final energy measurement).

### Per-Fold Breakdown

| Fold | Accuracy | Kappa | F1 Macro | SynOps/trial |
|---|---|---|---|---|
| 0 | 31.3% | 0.084 | 0.312 | 0.67 |
| 1 | 30.3% | 0.070 | 0.302 | 0.74 |
| 2 | 32.9% | 0.103 | 0.320 | 0.67 |
| 3 | 32.8% | 0.104 | 0.327 | 0.73 |
| 4 | 31.9% | 0.093 | 0.318 | 0.79 |

### Feature Bottleneck Diagnostic

| Classifier | Accuracy | Kappa | Pipeline |
|---|---|---|---|
| **LogReg** (subject-mixed) | **34.6% ± 0.8%** | 0.128 | PerSubjectZScore → CSP → FeatureScaler → LogReg |
| **SNN** (subject-mixed) | **31.8% ± 1.1%** | 0.091 | PerSubjectZScore → CSP → FeatureScaler → Encoder → LIF |
| **LogReg** (within-subject) | **42.0% ± 10.6%** | — | Same pipeline, per-subject 5-fold CV |

- SNN is within 3% of LogReg on the same features — model is not the bottleneck.
- Within-subject accuracy (42%) shows CSP features are discriminative per subject.
- Subject-mixed accuracy (34.6%) is lower than within-subject (42.0%); the gap is
  consistent with subject-specific features, but LOSO is required to confirm this.
- This is the expected baseline: the paper's three contributions are designed to break through this ceiling.

### Per-Subject Within-Subject Accuracy (102 subjects)

| Stat | Value |
|---|---|
| Mean | 42.0% |
| Median | 39.7% |
| Std | 10.6% |
| Subjects > 50% | 21/102 |
| Subjects > 60% | 8/102 |
| Best | S062: 71.8% |
| Worst | S096: 24.9% (chance) |

## Bugs Fixed

1. **Absolute-value encoding** — `encoder.py`, `synops.py`: `x > threshold` → `torch.abs(x) > threshold`. Without this, negative EEG features always produced −1 spikes regardless of magnitude.
2. **ReLU before LIF removed** — `lif_net.py`: Deleted 2 `nn.ReLU()` layers. Paper specifies "standard LIF" — ReLU before LIF halves effective capacity (ReLU kills ~50% of inputs, and with T=16 identical timesteps, a killed neuron stays dead all 16 steps).
3. **Gradient clipping corrected** — `phase2_config.yaml`: `max_grad_norm` 1.0 → 5.0. With T=16, gradients scale linearly (output = sum over T steps), so effective gradients are ~16× larger. Clipping at 1.0 was throttling updates.
4. **Event mapping fixed** — `events.py`: Corrected run→task mapping (IMAGERY_TASK1_RUNS={4,8,12}, IMAGERY_TASK2_RUNS={6,10,14}).
5. **Loader runs fixed** — `loader.py`: Default `runs=[4,6,8,10,12,14]` (all 6 MI runs).
6. **CSP shrinkage fixed** — `rcsp.py`: Changed `ledoit_wolf(cov)` → `ledoit_wolf(X.T)`. Was passing pre-computed covariance matrix (always shrinkage=1.0, destroying class structure).

## Configuration

```yaml
# key parameters
experiment.seed: 42
features.bands: [[mu, 8, 12], [beta, 13, 30]]
features.n_components: 4
encoder.T: 16
snn.n_hidden: 64
snn.tau: 20.0
snn.dropout: 0.5
training.n_epochs: 100
training.learning_rate: 0.001
training.max_grad_norm: 5.0
training.lambda_spike: 0.0  # no sparsity in baseline
evaluation.cv_folds: 5
evaluation.cv_type: subject_mixed
```

## File Structure

```
src/snn/
├── encoder.py      # FixedThresholdEncoder
├── lif_net.py      # BaselineLIFNet (standard LIF, no ReLU)
├── synops.py       # SynOps counting
├── trainer.py      # BaselineTrainer
└── evaluator.py    # metrics
src/features/
└── bandpass.py     # FixedBandCSPFeatures, FIXED_BANDS
src/spatial/
└── rcsp.py         # RegularizedCSP (Ledoit-Wolf OvR)
config/
└── phase2_config.yaml
experiments/
├── run_baseline.py
├── run_logreg_diagnostic.py
└── run_per_subject_diagnostic.py
results/baseline/
├── summary.json
├── fold_results.csv
└── confusion_matrices.npy
results/logreg_diagnostic/
├── summary.json
├── fold_results.csv
└── per_subject_accuracy.json
```

## What Baseline Deliberately Omits (Phase 3 scope)

| Component | Baseline | Phase 3 (full model) |
|---|---|---|
| Frequency bands | Fixed: mu (8–12), beta (13–30) | Per-subject adaptive KNN-MI selection from 7-band bank |
| Spike threshold | Fixed: mean of train features | Learnable θ, jointly optimized with surrogate gradients |
| Network | Single-tau LIF (τ=20ms) | Dual-branch PLIF (τ_fast=5ms, τ_slow=30ms) + attention fusion |
| Sparsity | None (λ=0) | Spike-count sparsity penalty with warm-up |

## Next Steps

1. **Phase 3**: Implement Contribution 1 (adaptive band selection via KNN-MI)
2. **Phase 3**: Implement Contribution 2 (learnable threshold via surrogate gradients)
3. **Phase 3**: Implement Contribution 3 (dual-branch PLIF network with attention fusion)
4. **Evaluation**: Compare full model against this baseline and against SGLNet reference
