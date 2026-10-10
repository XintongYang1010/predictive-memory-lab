#!/usr/bin/env bash
# Run inside the same GPU allocation and Python/CUDA image used for V1.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/retest_v2.sh /path/to/working-v1 [extra pmr run arguments]' >&2
  exit 2
fi
PMR_OLD_ROOT="$(cd -- "$1" && pwd)"
shift
PMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMR_ROOT"
export PMR_PYTHON="${PMR_PYTHON:-$PMR_OLD_ROOT/.venv/bin/python}"
export PATH="$PMR_OLD_ROOT/bin:$PATH"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PMR_ROOT/runs/matplotlib-cache}"
export CHECKPOINT="${CHECKPOINT:-$PMR_OLD_ROOT/checkpoints/realmotion.pth}"
PMR_DATA_ROOT="${PMR_DATA_ROOT:-$PMR_OLD_ROOT/data/av2_week1_processed}"
PMR_UPSTREAM="${PMR_UPSTREAM:-$PMR_OLD_ROOT/external/RealMotion}"
PMR_V2_OUT="${PMR_V2_OUT:-$PMR_ROOT/runs/week1_v2_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ -x "$PMR_PYTHON" ]] || { echo "Missing working Python: $PMR_PYTHON" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing existing checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -d "$PMR_UPSTREAM" ]] || { echo "Missing pinned upstream: $PMR_UPSTREAM" >&2; exit 2; }
compgen -G "$PMR_DATA_ROOT/train/*.pt" > /dev/null || { echo "Missing processed train data: $PMR_DATA_ROOT" >&2; exit 2; }
compgen -G "$PMR_DATA_ROOT/val/*.pt" > /dev/null || { echo "Missing processed val data: $PMR_DATA_ROOT" >&2; exit 2; }
echo "V2 source: $PMR_ROOT"
echo "V2 output: $PMR_V2_OUT"
"$PMR_PYTHON" -m pmr setup --source "$PMR_UPSTREAM"
"$PMR_PYTHON" -m pmr doctor --device cuda --strict
"$PMR_PYTHON" -m pmr smoke --device cuda --out "${PMR_V2_OUT}.smoke"
exec bash scripts/run_week1.sh "$PMR_DATA_ROOT" --out "$PMR_V2_OUT" "$@"
