# AGENTS.md — Adaptive EEG-to-Spike Framework

Place this file at the **root of your repository**. GitHub Copilot (Chat,
Agent Mode, and Coding Agent), and most other coding agents, will read it
automatically on every task in this repo — you won't need to re-paste
project context each time.

---

## Who you are on this project

You are acting as a research-engineering collaborator on a final-year
neuromorphic BCI project. Prioritize **correctness and reproducibility over
speed**. This is going toward a thesis and potentially a publication —
silent bugs (especially data leakage) are worse than slow progress.

## Project summary

Building an **Adaptive EEG-to-Spike Framework** for energy-efficient motor
imagery (MI) decoding on spiking neural networks, evaluated on the
PhysioNet EEG Motor Movement/Imagery dataset (103 usable subjects, 64
channels, 160 Hz, 4-class: left_hand / right_hand / feet / rest).

Six-stage pipeline, built incrementally phase by phase:

1. **Signal conditioning** — bandpass filter, ICA, epoch, reject, baseline,
   per-subject z-score.
2. **Spatial filtering** — Regularized CSP (Ledoit-Wolf shrinkage),
   One-vs-Rest for 4 classes.
3. **Adaptive band selection** *(novel contribution)* — per-subject
   KNN-Mutual-Information ranking over a 7-band filter bank (4–32 Hz),
   top-3 bands kept.
4. **Trainable spike encoding** *(novel contribution)* — threshold θ as a
   learnable parameter, jointly optimized with a spike-count sparsity
   penalty via surrogate gradients.
5. **Multi-timescale PLIF network** *(novel contribution)* — dual-branch
   Parametric LIF (fast τ≈5ms / slow τ≈30ms), fused via learned attention.
6. **Classification** — spike-vote over T=16 timesteps; combined
   MSE + CE + sparsity + attention-entropy loss.

Only build the stage(s) explicitly requested in a given task. Do not
implement later stages "ahead of time" even if it seems convenient — each
stage's interface should be agreed via its own task prompt.

## Non-negotiable engineering rules

1. **No data leakage, ever.** Any parameter estimated from data — z-score
   stats, CSP filters, band rankings, learned thresholds — must be fit
   **only on the training fold** inside cross-validation. Never fit on the
   full dataset before splitting. If you're unsure whether something
   leaks, treat it as if it does and ask.
2. **Config-driven, not hardcoded.** All hyperparameters live in YAML
   config files under `config/`, never hardcoded in logic.
3. **sklearn-compatible where possible.** Preprocessing/feature-extraction
   steps that fit on training data should be `BaseEstimator` +
   `TransformerMixin` classes, so they compose safely inside a `Pipeline`
   and can't accidentally be fit out of order.
4. **Deterministic.** Every stochastic step (ICA init, fold shuffling,
   weight init) takes and respects a seed from config.
5. **Typed and documented.** Type hints and docstrings on every public
   function/class. No unexplained magic numbers — cite the paper section
   (e.g. `# Eq. 8, Kraskov KNN-MI estimator`) when a constant comes from
   the methodology.
6. **Test what could silently break.** Every stage needs at least one test
   that would fail if leakage were reintroduced or a shape/units mistake
   crept in (e.g. assert fold-to-fold fitted parameters differ).
7. **Don't guess on ambiguity.** If the paper's methodology is
   underspecified for an implementation detail (this happens — e.g.
   whether CSP applies per sub-band or broadband), stop and ask rather
   than picking silently. Leave a `# TODO(ask): ...` comment either way.

## Code style

- Python 3.10+, `numpy`/`scipy`/`scikit-learn`/`mne` for signal processing,
  PyTorch + `spikingjelly` for the SNN stages (3 onward).
- Format with `black`, lint with `ruff`.
- Prefer explicit, readable code over clever one-liners — this codebase
  will be read by examiners, not just run.

## Project structure (grows phase by phase)

```
project/
├── config/            # one YAML per phase/stage
├── src/
│   ├── data/           # loading, preprocessing, normalization
│   ├── spatial/         # CSP
│   ├── bands/           # KNN-MI band selection (Stage 3)
│   ├── encoding/         # learnable spike encoder (Stage 4)
│   ├── snn/               # PLIF dual-branch network (Stage 5)
│   ├── cv/                 # fold generators (subject-mixed, LOSO)
│   └── pipeline.py          # wires stages together
├── tests/
├── notebooks/               # sanity checks / visualizations only, no logic
├── requirements.txt
└── README.md
```

## When starting any new task in this repo

1. Re-read the relevant paper section before writing code (I'll usually
   quote it in the task prompt, but check `docs/paper_summary.md` if
   present).
2. Confirm which stage(s) are in scope — do not spill into adjacent stages.
3. Confirm how this stage's output will be consumed by the next one (shape,
   dtype, on-disk format) before finalizing an interface.
4. Flag any deviation from the paper's stated methodology explicitly in
   the README and in code comments — don't let deviations go undocumented.