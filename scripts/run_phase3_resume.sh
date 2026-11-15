#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv not found. Run scripts/setup_linux.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f "results/phase3_c1/checkpoint.json" ]; then
    echo "No checkpoint found; the run will start from fold 0."
elif [ "${1:-}" != "--fresh" ]; then
    echo "=== Existing checkpoint ==="
    cat results/phase3_c1/checkpoint.json
    echo
fi

if [ "${1:-}" = "--fresh" ]; then
    python -u experiments/run_phase3.py \
        --config config/phase3_config.yaml \
        --log-file results/phase3_c1_run.log \
        --overwrite
else
    python -u experiments/run_phase3.py \
        --config config/phase3_config.yaml \
        --log-file results/phase3_c1_run.log \
        --resume
fi
