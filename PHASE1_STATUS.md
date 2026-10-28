# Phase 1 Status — ✅ COMPLETE

Phase 1 (Stage 1 Signal Conditioning + Stage 2 Regularized CSP) is finished.
All acceptance criteria from `phase1_data_pipeline_prompt.md` are met.
Last updated: 2026-08-24.

## ✅ Deliverables

1. **Data loading** — `src/data/loader.py`: PhysioNet EEGBCI via MNE;
   `load_subject_raw_with_events()` extracts run-aware events before concatenation
2. **Event extraction** — `src/data/events.py`: run-aware label mapping
   (`annotation_map_for_run`, `build_run_aware_events`); legacy global mapper deprecated
3. **Signal conditioning (Stage 1)** — `src/data/preprocessing.py`:
   bandpass 0.5–40 Hz (4th-order Butterworth), average reference, ICA (n=20, EOG heuristic),
   4 s epoching, bad-channel detection + interpolation, AutoReject trial cleaning,
   baseline correction (−200–0 ms)
4. **Per-subject z-score normalization** — `src/data/normalization.py` (sklearn fit/transform,
   fit on train fold only)
5. **Regularized CSP (Stage 2)** — `src/spatial/rcsp.py`: Ledoit-Wolf shrinkage, OvR 4-class,
   sklearn-compatible, fit on train fold only
6. **CV scaffold** — `src/cv/splits.py`: subject-mixed 5-fold + LOSO generators
7. **Pipeline runner** — `src/pipeline.py` (CLI: `python -m src.pipeline --config ... --subject ...`)
8. **Config** — `config/phase1_config.yaml` (all hyperparameters, seed, excluded subjects)
9. **Experiment runners** — `src/experiments/cv.py`, `src/experiments/io.py`
10. **Tests — 10/10 passing** (`pytest tests/`):
    - `test_preprocessing.py` (filter attenuation, epoch counts, AutoReject log)
    - `test_rcsp.py` (fit/transform shapes, fold-to-fold filter difference)
    - `test_no_leakage.py` (leakage guard)
    - `test_events.py` (run-aware label regression test)
11. **Sanity notebook** — `notebooks/01_sanity_check.ipynb` fully executed on S001
    (plot saved to `notebooks/sanity_check_plot.png`)
12. **README** — complete: data acquisition, run instructions, config reference,
    documented deviations, open questions

## ✅ Acceptance Criteria (from phase1_data_pipeline_prompt.md)

| Criterion | Status |
|---|---|
| `python -m src.pipeline --config config/phase1_config.yaml --subject S001` runs end-to-end | ✅ |
| Leakage-guard test (fold params differ) | ✅ `test_no_leakage.py` |
| README with data/run/config/deviations/open questions | ✅ |
| Type hints + docstrings on public functions/classes | ✅ |
| Sanity notebook: raw vs. preprocessed epochs, per-class trial counts | ✅ executed |

## ✅ Verification (S001, run-aware 4-class labels)

- Trial counts: left_hand 8 · right_hand 7 · feet 15 · rest 42 → 72 trials
- ICA removed 1 EOG component; 6 bad channels interpolated (F7, F8, PO8, O1, O2, Iz)
- AutoReject: 0 trials dropped after repair (72/72 retained)

## ⚠️ Carry-Forward Items for Phase 2 (documented, not blocking)

1. **N=84 discrepancy** — paper Section 5.3 says N=84 trials/subject; actual imagery runs
   yield ~30 MI + ~42 rest trials. Reconcile before Stage 3 (does rest count toward N?)
2. **CSP broadband vs. per-sub-band** — open question for Stage 3 (FBCSP pattern);
   flagged in README Section 5, Question 2
3. **Documented deviations** — AutoReject (vs. fixed ±100 µV), average referencing,
   bad-channel interpolation, montage standardization (README Section 4)
