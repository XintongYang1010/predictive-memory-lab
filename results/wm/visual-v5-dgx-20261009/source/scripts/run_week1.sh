#!/usr/bin/env bash
set -euo pipefail
PMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMR_ROOT"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/run_week1.sh /path/to/av2 [extra pmr run arguments]' >&2
  exit 2
fi
DATA_ROOT="$1"
shift
PMR_PYTHON="${PMR_PYTHON:-$PMR_ROOT/.venv/bin/python}"
[[ -x "$PMR_PYTHON" ]] || { echo 'Run bash scripts/setup.sh --av2 first' >&2; exit 2; }
"$PMR_PYTHON" -m pmr doctor --device cuda --strict
"$PMR_PYTHON" -m pytest -q
if compgen -G "$DATA_ROOT/train/*.pt" > /dev/null; then
  PROCESSED="$DATA_ROOT"
elif compgen -G "$DATA_ROOT/realmotion_processed/train/*.pt" > /dev/null; then
  PROCESSED="$DATA_ROOT/realmotion_processed"
else
  PROCESSED="${PMR_PROCESSED_ROOT:-$PMR_ROOT/data/realmotion_processed}"
  "$PMR_PYTHON" scripts/preprocess.py --raw "$DATA_ROOT" --out "$PROCESSED"
fi
MODEL_PATH="${CHECKPOINT:-$PMR_ROOT/checkpoints/realmotion.pth}"
if [[ ! -f "$MODEL_PATH" ]]; then
  "$PMR_PYTHON" scripts/download_checkpoint.py --out "$MODEL_PATH"
fi
exec "$PMR_PYTHON" -m pmr run --data-root "$PROCESSED" --checkpoint "$MODEL_PATH" --trust-inputs --device cuda --out runs/week1_v3 "$@"
