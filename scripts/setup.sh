#!/usr/bin/env bash
set -euo pipefail
PMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMR_ROOT"
CPU=0
AV2=0
for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    --av2) AV2=1 ;;
    *) echo "Usage: bash scripts/setup.sh [--cpu] [--av2]" >&2; exit 2 ;;
  esac
done
PMR_PYTHON="${PYTHON:-python3}"
"$PMR_PYTHON" - "$CPU" <<'PY'
import sys
if sys.version_info < (3,11):
    raise SystemExit('Python >=3.11 required')
if sys.argv[1]=='0' and sys.version_info[:2]!=(3,11):
    raise SystemExit('Target image must use Python 3.11. Use --cpu for local smoke on 3.11/3.12.')
PY
if [[ ! -x .venv/bin/python ]]; then
  "$PMR_PYTHON" -m venv --system-site-packages .venv
fi
PMR_PYTHON="$PMR_ROOT/.venv/bin/python"
if ! "$PMR_PYTHON" - "$CPU" <<'PY'
import sys
try: import torch
except ImportError: raise SystemExit(1)
assert torch.__version__.split('+')[0]=='2.9.1'
if sys.argv[1]=='0': assert torch.version.cuda=='12.8'
PY
then
  INDEX=https://download.pytorch.org/whl/cu128
  if [[ "$CPU" == 1 ]]; then INDEX=https://download.pytorch.org/whl/cpu; fi
  "$PMR_PYTHON" -m pip install torch==2.9.1 --index-url "$INDEX"
fi
REQ=requirements.txt
if [[ "$AV2" == 1 ]]; then REQ=requirements-av2.txt; fi
"$PMR_PYTHON" -m pip install -r "$REQ"
"$PMR_PYTHON" -m pip install --no-deps -e .
"$PMR_PYTHON" -m pmr setup
if [[ "$CPU" == 1 ]]; then
  "$PMR_PYTHON" -m pmr doctor --device cpu
else
  "$PMR_PYTHON" -m pmr doctor --device cuda --strict
fi
echo 'Ready. Activate with: source .venv/bin/activate'
