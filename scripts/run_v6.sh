#!/usr/bin/env bash
# V6 predictive memory: run one stage inside an existing GPU allocation.
#
#   bash scripts/run_v6.sh S0    <out_root>      # CPU gates (+ data prep if paths are set)
#   bash scripts/run_v6.sh S1    <out_root>      # smoke/overfit/profile/numerics (<= 2 GPUh)
#   PMR_V6_PREREG_COMMIT=<hash> bash scripts/run_v6.sh S2 <out_root>
#   bash scripts/run_v6.sh SMOKE <out_root>      # CPU end-to-end software check
#
# Environment (all optional):
#   PMR_PYTHON              interpreter with torch 2.9.1 (default: python)
#   PMR_V6_GPUS             comma list of visible GPU indices (default: all from nvidia-smi)
#   PMR_V6_PROCS_PER_GPU    concurrent jobs per GPU (default 4)
#   PMR_V6_CACHE            data cache root (L1 tokens, real tokens)
#   PMR_V6_PREREG_COMMIT    commit hash of the frozen docs/PREREGISTRATION_V6.md (required for S2+)
#   PMR_V6_MAX_GPUH         override the stage cap downward (never above prereg.BUDGET_GPUH)
#   PMR_V6_NUSC_META        nuScenes v1.0-trainval metadata dir (S0 data prep)
#   PMR_V6_MOT_ROOTS        space separated MOT17/train MOT20/train dirs (S0 data prep)
#   PMR_V6_MINIWORLD        Miniworld checkout at b28988d (S0 L1 smoke)
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAGE="${1:?stage}"
OUT="${2:?out_root}"
PY="${PMR_PYTHON:-python}"
mkdir -p "$OUT"
LOG="$OUT/console_${STAGE}_$(date -u +%Y%m%dT%H%M%SZ).log"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1
{
  echo "[run_v6] stage=$STAGE out=$OUT host=$(hostname) commit=$(git rev-parse HEAD 2>/dev/null || echo none)"
  "$PY" - <<'PY'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
} | tee -a "$LOG"
if [[ -z "${PMR_V6_GPUS:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  PMR_V6_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
GPUS="${PMR_V6_GPUS:-}"
PPG="${PMR_V6_PROCS_PER_GPU:-4}"
ARGS=()
[[ -n "${PMR_V6_CACHE:-}" ]] && ARGS+=(--cache "$PMR_V6_CACHE")
case "$STAGE" in
  S0)
    S0ARGS=(--out "$OUT")
    [[ -n "${PMR_V6_CACHE:-}" ]] && S0ARGS+=(--cache "$PMR_V6_CACHE")
    [[ -n "${PMR_V6_NUSC_META:-}" ]] && S0ARGS+=(--nuscenes-meta "$PMR_V6_NUSC_META")
    # shellcheck disable=SC2206
    [[ -n "${PMR_V6_MOT_ROOTS:-}" ]] && S0ARGS+=(--mot-roots ${PMR_V6_MOT_ROOTS})
    [[ -n "${PMR_V6_MINIWORLD:-}" ]] && S0ARGS+=(--miniworld "$PMR_V6_MINIWORLD")
    "$PY" -m pmr.predictive_memory_v6 s0 "${S0ARGS[@]}" 2>&1 | tee -a "$LOG"
    ;;
  SMOKE|S1|S2|S3|S4|S5|S6|S7|FINAL)
    [[ -n "${PMR_V6_PREREG_COMMIT:-}" ]] && ARGS+=(--prereg-commit "$PMR_V6_PREREG_COMMIT")
    [[ -n "${PMR_V6_MAX_GPUH:-}" ]] && ARGS+=(--max-gpuh "$PMR_V6_MAX_GPUH")
    set +e
    "$PY" -m pmr.predictive_memory_v6 run --stage "$STAGE" --out "$OUT" --gpus "$GPUS" \
      --procs-per-gpu "$PPG" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    CODE=${PIPESTATUS[0]}
    set -e
    "$PY" -m pmr.predictive_memory_v6 report --out "$OUT" 2>&1 | tee -a "$LOG"
    exit "$CODE"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
