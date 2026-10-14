#!/usr/bin/env bash
# V6.1 BF-Delta: run one stage inside an existing GPU allocation.
#
#   PMR_V6_DATA=/abs/memory_maze_9x9 PMR_V6_CACHE=/abs/cache bash scripts/run_v6.sh S0 RUN
#   bash scripts/run_v6.sh S1 RUN                         # profile + overfit (<= 2 GPUh)
#   PMR_V6_PREREG_COMMIT=<hash> bash scripts/run_v6.sh S2 RUN   # then S3, S4 (first decision point)
#   bash scripts/run_v6.sh SMOKE RUN                      # CPU, synthetic data, plumbing only
#   bash scripts/run_v6.sh REPORT RUN
#
# Environment:
#   PMR_PYTHON            interpreter with torch (default: python)
#   PMR_V6_DATA           directory with Memory Maze NPZ files (train/ and eval/ somewhere below)
#   PMR_V6_CACHE          token cache directory (created by S0)
#   PMR_V6_GPUS           comma list of visible GPU indices (default: all from nvidia-smi)
#   PMR_V6_PROCS_PER_GPU  concurrent jobs per GPU (default 4; the arms are small)
#   PMR_V6_PREREG_COMMIT  commit whose pmr/predictive_memory_v6/prereg.py equals the working copy (S2+)
#   PMR_V6_MAX_GPUH       lower the stage cap (never above prereg.BUDGET_GPUH)
#   PMR_V6_FINAL_ACK=1    only after the user signs off the checks before FINAL
#   PMR_VSC_GT_JSON       ground truth for FINAL VSC scoring (scripts/vsc/export_gt.py)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:?stage}"
OUT="${2:?run root}"
PY="${PMR_PYTHON:-python}"
mkdir -p "$OUT"
LOG="$OUT/console_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ).log"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
{
  echo "[run_v6] stage=$STAGE out=$OUT host=$(hostname) commit=$(git rev-parse HEAD 2>/dev/null || echo none)"
  "$PY" - <<'PY'
import sys
try:
    import torch
    print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
          "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
except ImportError:
    print("python", sys.version.split()[0], "torch MISSING")
PY
} | tee -a "$LOG"
if [[ -z "${PMR_V6_GPUS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  PMR_V6_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
ARGS=(--out "$OUT" --gpus "${PMR_V6_GPUS:-}" --procs-per-gpu "${PMR_V6_PROCS_PER_GPU:-4}")
[[ -n "${PMR_V6_CACHE:-}" ]] && ARGS+=(--cache "$PMR_V6_CACHE")
[[ -n "${PMR_V6_DATA:-}" ]] && ARGS+=(--data "$PMR_V6_DATA")
[[ -n "${PMR_V6_PREREG_COMMIT:-}" ]] && ARGS+=(--prereg-commit "$PMR_V6_PREREG_COMMIT")
[[ -n "${PMR_V6_MAX_GPUH:-}" ]] && ARGS+=(--max-gpuh "$PMR_V6_MAX_GPUH")
[[ "${PMR_V6_FINAL_ACK:-0}" == "1" ]] && ARGS+=(--final-ack)
[[ -n "${PMR_VSC_GT_JSON:-}" ]] && ARGS+=(--vsc-gt-json "$PMR_VSC_GT_JSON")
case "$STAGE" in
  S0)
    : "${PMR_V6_DATA:?set PMR_V6_DATA}"; : "${PMR_V6_CACHE:?set PMR_V6_CACHE}"
    "$PY" -m pmr.predictive_memory_v6 s0 --data "$PMR_V6_DATA" --cache "$PMR_V6_CACHE" --out "$OUT" 2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
    ;;
  REPORT)
    "$PY" -m pmr.predictive_memory_v6 report --out "$OUT" 2>&1 | tee -a "$LOG"
    ;;
  SMOKE|S1|S2|S3|S4|S5|S6|FINAL)
    set +e
    "$PY" -m pmr.predictive_memory_v6 run --stage "$STAGE" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    CODE=${PIPESTATUS[0]}
    set -e
    "$PY" -m pmr.predictive_memory_v6 report --out "$OUT" 2>&1 | tee -a "$LOG"
    exit "$CODE"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
