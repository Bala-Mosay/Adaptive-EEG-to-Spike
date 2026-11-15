"""Check that a second machine is ready to run Phase 3 C1.

Usage:
    python scripts/verify_setup.py
    python scripts/verify_setup.py --full-cache-check   # loads all 103 caches
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-cache-check",
        action="store_true",
        help="Load and validate every cache (slower, reads ~2.3 GB)",
    )
    args = parser.parse_args()

    print(f"Python : {sys.version.split()[0]}")

    try:
        import torch

        print(f"torch  : {torch.__version__}")
        print(f"CUDA   : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU    : {torch.cuda.get_device_name(0)}")
        else:
            print("Device : CPU")
            print(
                "         (expected on Windows + AMD; the SNN is small and "
                "preprocessing/CSP dominate, so CPU is the validated path)"
            )
    except Exception as exc:
        print(f"ERROR importing torch: {exc}")
        return 1

    for name in ("mne", "numpy", "sklearn", "pandas", "yaml", "autoreject", "spikingjelly"):
        try:
            module = __import__(name)
            print(f"{name:<14}: {getattr(module, '__version__', 'ok')}")
        except Exception as exc:
            print(f"ERROR importing {name}: {exc}")
            return 1

    cache_dir = ROOT / "outputs" / "preprocessed"
    if not cache_dir.exists():
        print(f"ERROR: missing cache directory: {cache_dir}")
        print("Copy the 103 preprocessed .npz files there (see FRIEND_RUN.md).")
        return 1

    caches = sorted(cache_dir.glob("S*.npz"))
    print(f"caches        : {len(caches)} .npz files")
    if len(caches) < 103:
        print("WARNING: expected 103 caches; the runner will skip missing subjects.")

    if args.full_cache_check:
        import numpy as np

        from src.data.cache import validate_preprocessed

        bad = []
        for path in caches:
            try:
                with np.load(path) as blob:
                    validate_preprocessed(blob["X"], blob["y"], path.stem)
            except Exception as exc:
                bad.append((path.name, str(exc)))
        print(f"cache validity: {len(caches) - len(bad)} valid, {len(bad)} bad")
        for name, err in bad[:10]:
            print(f"  {name}: {err}")
        if bad:
            return 1

    checkpoint = ROOT / "results" / "phase3_c1" / "checkpoint.json"
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        print(f"checkpoint    : folds {payload.get('completed_folds')}")
        print(
            "checkpoint    : "
            f"{payload.get('n_subjects_loaded')} subjects / "
            f"{payload.get('n_trials_total')} trials"
        )
    else:
        print("checkpoint    : none yet (run starts from fold 0)")

    print("")
    print("Setup looks OK. Start/resume with:")
    print(r"  powershell -ExecutionPolicy Bypass -File scripts\run_phase3_resume.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
