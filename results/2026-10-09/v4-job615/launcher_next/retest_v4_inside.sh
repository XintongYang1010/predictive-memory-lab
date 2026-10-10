#!/bin/bash
set -euo pipefail
: "${PMR_JOB_ID:?} ${PMR_NEW_ROOT:?} ${PMR_OLD_ROOT:?} ${PMR_SOURCE:?} ${PMR_V4_OUT:?}"
exec > "$PMR_NEW_ROOT/logs/retest_v4_inside_${PMR_JOB_ID}.log" 2>&1
export XDG_CACHE_HOME="$PMR_NEW_ROOT/.cache"
export MPLCONFIGDIR="$PMR_NEW_ROOT/.cache/matplotlib"
export PATH="$PMR_OLD_ROOT/bin:$PATH"
export PMR_V4_MAX_SECONDS=3600
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
cd "$PMR_SOURCE"
printf 'source_commit=%s job=%s allocated_host_gpu=%s output=%s\n' "$(git rev-parse HEAD)" "$PMR_JOB_ID" "$PMR_ALLOCATED_GPU" "$PMR_V4_OUT"
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
PMR_CONTAINER_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
[[ "$PMR_CONTAINER_GPU_UUID" == "${PMR_ALLOCATED_GPU_UUID:?}" ]] || { echo 'Container GPU differs from scheduler allocation; stopping before tests' >&2; exit 2; }
bash scripts/retest_v4.sh "$PMR_OLD_ROOT"
