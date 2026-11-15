#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== Adaptive EEG-to-Spike: Phase 3 setup (Linux) ==="
echo "Repo root: $(pwd)"

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
print(f"Using Python {sys.version.split()[0]}")
PY

if [ ! -d ".venv" ]; then
    echo "Creating .venv"
    "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install pytest

echo
echo "=== Verifying environment and data ==="
python scripts/verify_setup.py

if [ "${SKIP_TESTS:-0}" != "1" ]; then
    echo
    echo "=== Running test suite ==="
    python -m pytest tests/ -q
fi

echo
echo "Setup complete."
echo "Start or resume with:"
echo "  bash scripts/run_phase3_resume.sh"
