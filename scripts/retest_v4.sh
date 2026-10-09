#!/usr/bin/env bash
# Run inside an allocated GPU and the working Python 3.11 / PyTorch cu128 image.
# This suite is independent of RealMotion, AV2 and pretrained checkpoints.
set -euo pipefail

if [[ $# -gt 1 || "${1:-}" == --* ]]; then
  echo 'Usage: bash scripts/retest_v4.sh [/path/to/working-v1]' >&2
  echo 'Overrides: PMR_PYTHON, PMR_V4_OUT, PMR_V4_MAX_SECONDS (default: 3600).' >&2
  exit 2
fi

PMR_V4_STARTED="$(date +%s)"
PMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMR_ROOT"
PMR_V4_MAX_SECONDS="${PMR_V4_MAX_SECONDS:-3600}"
[[ "$PMR_V4_MAX_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo 'PMR_V4_MAX_SECONDS must be a positive integer.' >&2; exit 2;
}

# Prefer an explicitly chosen interpreter, then the previous successful environment.
if [[ $# -eq 1 ]]; then
  PMR_OLD_ROOT="$(cd -- "$1" && pwd)"
  if [[ -z "${PMR_PYTHON:-}" ]]; then
    for PMR_CANDIDATE in "$PMR_OLD_ROOT/source_852f57d/.venv/bin/python" "$PMR_OLD_ROOT/.venv/bin/python"; do
      if [[ -x "$PMR_CANDIDATE" ]]; then
        PMR_PYTHON="$PMR_CANDIDATE"
        break
      fi
    done
  fi
fi
PMR_PYTHON="${PMR_PYTHON:-python}"
PMR_PYTHON="$(command -v -- "$PMR_PYTHON")" || {
  echo 'Python not found. Set PMR_PYTHON to the working environment interpreter.' >&2; exit 2;
}
[[ -x "$PMR_PYTHON" ]] || { echo "Python is not executable: $PMR_PYTHON" >&2; exit 2; }

PMR_V4_OUT="${PMR_V4_OUT:-$PMR_ROOT/runs/state_memory_v4_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-local}_$$}"
for PMR_PATH in "$PMR_V4_OUT" "$PMR_V4_OUT.log" "$PMR_V4_OUT.smoke" "$PMR_V4_OUT.tar.gz" "$PMR_V4_OUT.tar.gz.sha256"; do
  [[ ! -e "$PMR_PATH" ]] || { echo "Use a fresh PMR_V4_OUT; already exists: $PMR_PATH" >&2; exit 2; }
done
mkdir -p "$(dirname -- "$PMR_V4_OUT")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PMR_ROOT/runs/matplotlib-cache}"
export MPLBACKEND=Agg
# Required by deterministic CUDA matrix multiplication; set before importing torch.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

pmr_v4_exit() {
  PMR_STATUS=$?
  if [[ "$PMR_STATUS" -ne 0 ]]; then
    echo "RETEST_V4_FAILED: exit=$PMR_STATUS; inspect $PMR_V4_OUT and $PMR_V4_OUT.log" >&2
    echo 'Partial outputs are retained. No resume is supported; rerun with a fresh output directory.' >&2
  fi
}
trap pmr_v4_exit EXIT

pmr_v4_remaining() {
  PMR_V4_REMAINING=$((PMR_V4_MAX_SECONDS - $(date +%s) + PMR_V4_STARTED))
  if [[ "$PMR_V4_REMAINING" -le 0 ]]; then
    echo 'V4 wrapper wall-time budget exhausted before the next stage.' >&2
    return 3
  fi
}

pmr_v4_run() {
  echo "V4 source: $PMR_ROOT"
  echo "V4 output: $PMR_V4_OUT"
  echo "Python: $PMR_PYTHON"
  echo "Wall-time budget: $PMR_V4_MAX_SECONDS seconds; three seeds run serially on one GPU."
  echo 'This script does not install packages or replace PyTorch.'
  "$PMR_PYTHON" - <<'PY'
import sys
import numpy
import matplotlib
import pytest
import torch

if sys.version_info < (3, 11):
    raise SystemExit('Python >= 3.11 is required.')
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('NumPy:', numpy.__version__, 'Matplotlib:', matplotlib.__version__, 'pytest:', pytest.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Run inside an allocated GPU job; no CPU fallback is performed.')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(8, device='cuda')
if float((x * x).sum().item()) != 8.0:
    raise SystemExit('CUDA arithmetic check failed.')
torch.cuda.synchronize()
PY
  pmr_v4_remaining
  "$PMR_PYTHON" -m pytest -q tests/test_state_memory_*.py

  pmr_v4_remaining
  PMR_SMOKE_SECONDS=$PMR_V4_REMAINING
  if [[ "$PMR_SMOKE_SECONDS" -gt 120 ]]; then PMR_SMOKE_SECONDS=120; fi
  "$PMR_PYTHON" -m pmr.state_memory \
    --device cuda --out "$PMR_V4_OUT.smoke" --smoke \
    --steps 3 --train-episodes 32 --dev-episodes 8 --eval-episodes 8 \
    --seeds 42 --train-length 24 --eval-lengths 24 \
    --models window,retrieval,gru,slot,analytic \
    --hidden-dim 64 --batch-size 64 --window 4 --memory-slots 8 \
    --max-seconds "$PMR_SMOKE_SECONDS"

  pmr_v4_remaining
  echo "Full suite remaining budget: $PMR_V4_REMAINING seconds."
  "$PMR_PYTHON" -m pmr.state_memory \
    --device cuda --out "$PMR_V4_OUT" \
    --steps 1500 --train-episodes 4096 --dev-episodes 512 --eval-episodes 512 \
    --seeds 42,43,44 --train-length 24 --eval-lengths 24,48,96 \
    --models window,retrieval,gru,slot,analytic \
    --hidden-dim 64 --batch-size 64 --window 4 --memory-slots 8 \
    --max-seconds "$PMR_V4_REMAINING"
  # A zero process exit alone is insufficient: require the full-suite status contract.
  "$PMR_PYTHON" - "$PMR_V4_OUT" <<'PY'
import json
import sys
from pathlib import Path

stats = json.loads((Path(sys.argv[1]) / 'run_stats.json').read_text())
if stats.get('status') != 'complete' or stats.get('smoke_only') is not False:
    raise SystemExit('Full V4 run is incomplete or marked as smoke; refusing to package it.')
PY
}

pmr_v4_run 2>&1 | tee "$PMR_V4_OUT.log"
"$PMR_PYTHON" scripts/package_results.py --run "$PMR_V4_OUT" --log "$PMR_V4_OUT.log"
echo "RETEST_V4_COMPLETE: $PMR_V4_OUT.tar.gz"
echo "SHA256: $PMR_V4_OUT.tar.gz.sha256"
