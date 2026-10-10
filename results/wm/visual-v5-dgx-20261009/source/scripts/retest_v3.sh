#!/usr/bin/env bash
# Inside an allocated GPU and the working Python 3.11 / cu128 image.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/retest_v3.sh /path/to/working-v1 [extra pmr run arguments]' >&2
  exit 2
fi
PMR_OLD_ROOT="$(cd -- "$1" && pwd)"
shift
PMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMR_ROOT"
# The completed jobs used source_852f57d, with older flat layouts as fallback.
if [[ -z "${PMR_PYTHON:-}" ]]; then
  for PMR_CANDIDATE in "$PMR_OLD_ROOT/source_852f57d/.venv/bin/python" "$PMR_OLD_ROOT/.venv/bin/python"; do
    if [[ -x "$PMR_CANDIDATE" ]]; then PMR_PYTHON="$PMR_CANDIDATE"; break; fi
  done
fi
if [[ -z "${PMR_UPSTREAM:-}" ]]; then
  for PMR_CANDIDATE in "$PMR_OLD_ROOT/source_852f57d/external/RealMotion" "$PMR_OLD_ROOT/external/RealMotion"; do
    if [[ -d "$PMR_CANDIDATE" ]]; then PMR_UPSTREAM="$PMR_CANDIDATE"; break; fi
  done
fi
export PMR_PYTHON="${PMR_PYTHON:-}"
export PATH="$PMR_OLD_ROOT/bin:$PATH"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PMR_ROOT/runs/matplotlib-cache}"
export CHECKPOINT="${CHECKPOINT:-$PMR_OLD_ROOT/checkpoints/realmotion.pth}"
PMR_DATA_ROOT="${PMR_DATA_ROOT:-$PMR_OLD_ROOT/data/av2_week1_processed}"
PMR_V3_OUT="${PMR_V3_OUT:-$PMR_ROOT/runs/week1_v3_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-local}}"
[[ -x "$PMR_PYTHON" ]] || { echo 'Set PMR_PYTHON to the working environment python.' >&2; exit 2; }
[[ -d "${PMR_UPSTREAM:-}" ]] || { echo 'Set PMR_UPSTREAM to the pinned RealMotion checkout.' >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 2; }
compgen -G "$PMR_DATA_ROOT/train/*.pt" > /dev/null || { echo "Missing train data: $PMR_DATA_ROOT" >&2; exit 2; }
compgen -G "$PMR_DATA_ROOT/val/*.pt" > /dev/null || { echo "Missing val data: $PMR_DATA_ROOT" >&2; exit 2; }
[[ ! -e "$PMR_V3_OUT" && ! -e "$PMR_V3_OUT.log" ]] || { echo 'Use a fresh PMR_V3_OUT.' >&2; exit 2; }
# Output and shared-row protocol are fixed by this wrapper.
for PMR_ARG in "$@"; do
  case "$PMR_ARG" in
    --out|--out=*|--rounds|--rounds=*|--feature-set|--feature-set=*|--resume)
      echo 'This wrapper fixes the three-arm, one-round protocol; use python -m pmr run for other protocols.' >&2; exit 2 ;;
  esac
done
mkdir -p "$(dirname -- "$PMR_V3_OUT")"
pmr_retest() {
  echo "V3 source: $PMR_ROOT"
  echo "V3 output: $PMR_V3_OUT"
  echo "Python: $PMR_PYTHON"
  "$PMR_PYTHON" -m pmr setup --source "$PMR_UPSTREAM"
  "$PMR_PYTHON" -m pmr doctor --device cuda --strict
  "$PMR_PYTHON" -m pytest -q
  "$PMR_PYTHON" -m pmr smoke --device cuda --ablation --steps 10 --out "${PMR_V3_OUT}.smoke"
  "$PMR_PYTHON" -m pmr run --data-root "$PMR_DATA_ROOT" --checkpoint "$CHECKPOINT" \
    --trust-inputs --device cuda --ablation --rounds 1 --out "$PMR_V3_OUT" "$@"
  "$PMR_PYTHON" -m pmr audit-selection --run "$PMR_V3_OUT"
}
pmr_retest "$@" 2>&1 | tee "$PMR_V3_OUT.log"
"$PMR_PYTHON" scripts/package_results.py --run "$PMR_V3_OUT" --log "$PMR_V3_OUT.log"
echo "RETEST_V3_COMPLETE: $PMR_V3_OUT.tar.gz"
