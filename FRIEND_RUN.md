# Running Phase 3 C1 on a second machine

This machine is only running the **Phase 3 C1** experiment (adaptive KNN-MI
band selection). It resumes the existing checkpoint, so it does not redo
folds 0-2.

## 1. Important: AMD GPU expectation

The project uses PyTorch. On **Windows**, an AMD GPU is not a CUDA device, so
PyTorch will run on the **CPU**. That is expected and fine:

- The network is tiny (`48 -> 64 -> 64 -> 4`, `T=16`), so GPU acceleration
  gives little or no benefit for the SNN.
- Most of the runtime is 7-band filtering + CSP feature extraction, which is
  CPU/MNE work.
- The validated, reproducible path is CPU.

`config/phase3_config.yaml` uses `device: auto`: CUDA if available, otherwise
CPU. Do **not** install CUDA for an AMD card.

If the machine runs **Linux** and has a working ROCm PyTorch build, `device:
auto` may expose the GPU as `cuda`. That path is optional and untested here;
use it only after `python scripts/verify_setup.py` reports `CUDA: True`.

## 2. Get the code

```powershell
git clone https://github.com/Bala-Mosay/Adaptive-EEG-to-Spike.git
cd Adaptive-EEG-to-Spike
```

The repo contains the code **and** the partial checkpoint
(`results/phase3_c1/`, folds 0-2). It does **not** contain the 2.3 GB
preprocessed dataset (gitignored).

## 3. Transfer the preprocessed data

The checkpoint expects exactly **103 subjects / 8672 trials**. Copy the
`outputs/preprocessed` folder from the original machine.

### Option A (recommended): copy the cache

On the original machine, create the archive:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pack_preprocessed.ps1
```

Copy `eeg_preprocessed_103.zip` (~2.3 GB) to the other machine and extract it
so the layout is:

```
Adaptive-EEG-to-Spike/
  outputs/
    preprocessed/
      S001.npz
      ...
      S109.npz
```

Only 103 of the 109 subjects are used; six are excluded in the config.

### Option B: re-download and preprocess

MNE can download the raw PhysioNet EEGMMIDB data automatically, but full
preprocessing (ICA + AutoReject) is slow and may produce different trial
counts. If the counts differ, `--resume` will refuse the checkpoint and a
fresh run (`-Fresh`) is required. Option A is strongly preferred.

## 4. Install (Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

This creates `.venv`, installs `requirements.txt` + `pytest`, verifies
imports/cache count/checkpoint, and runs the test suite.

To skip tests:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -SkipTests
```

## 5. Run / resume

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase3_resume.ps1
```

Leave the console window open. You will see:

- subject-loading lines for ~2-5 minutes,
- a per-epoch status bar inside each fold,
- a fold status bar, e.g. `Folds [==========----------------] 2/5 ...`.

Folds 0-2 are restored from the checkpoint and skipped; only folds 3 and 4
train. After each fold, progress is saved automatically.

Start over from scratch (discards the checkpoint):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase3_resume.ps1 -Fresh
```

## 6. Check progress

```powershell
Get-Content results\phase3_c1\checkpoint.json
```

`completed_folds` is the source of truth, e.g. `[0, 1, 2]`.

Final outputs appear under `results/phase3_c1/`:

- `summary.json` - 5-fold mean/std metrics
- `fold_results.csv` - per-fold metrics
- `selected_bands.json` - bands chosen per subject/fold
- `predictions.npz`
- `confusion_matrices.npy`

## 7. Verify the machine at any time

```powershell
python scripts\verify_setup.py
python scripts\verify_setup.py --full-cache-check   # slower, reads all caches
```

## 8. Linux + ROCm (optional)

```bash
bash scripts/setup_linux.sh
bash scripts/run_phase3_resume.sh
```

For ROCm, install the ROCm PyTorch wheel first (see
<https://pytorch.org/get-started/locally/>), then confirm with
`python scripts/verify_setup.py`.

## Troubleshooting

- **`Checkpoint config does not match`** - the config or subject subset
  changed. Restore the original config, or start fresh with `-Fresh`.
- **`Checkpoint was built with 103 subjects but this run loaded ...`** - the
  cache folder is incomplete/different. Copy the full 103-file cache set.
- **No status bar** - the bar only renders in an interactive console. The
  run still works when output is redirected; check `checkpoint.json`.
- **`torch` install is slow** - the CPU wheel is large; first install can take
  several minutes.
