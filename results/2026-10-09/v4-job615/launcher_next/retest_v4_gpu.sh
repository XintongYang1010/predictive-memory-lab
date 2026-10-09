#!/bin/bash
#SBATCH --job-name=yxt_pmr_retest_v4
#SBATCH --partition=gpu_m
#SBATCH --time=01:05:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
set -euo pipefail

PMR_NEW_ROOT=/share/home/s2026316084/yxt_pmr_v4_20261009
PMR_OLD_ROOT=/share/home/s2026316084/yxt_pmr_20261002
PMR_SOURCE="$PMR_NEW_ROOT/source_e66bdd3"
PMR_ARCHIVE="$PMR_NEW_ROOT/pmr_v4_source_e66bdd3.tar.gz"
mkdir -p "$PMR_NEW_ROOT/logs"
exec > "$PMR_NEW_ROOT/logs/retest_v4_gpu_${SLURM_JOB_ID}.log" 2>&1

hostname
nvidia-smi
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory --format=csv || true
PMR_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
# Slurm's device ordinal is remapped; pass its sole visible UUID to Docker.
[[ "$PMR_GPU_UUID" =~ ^GPU-[[:xdigit:]-]+$ ]] || { echo 'Expected exactly one allocated GPU UUID' >&2; exit 2; }
ps -eo user,pid,pcpu,pmem,comm --sort=-pcpu | head -n 20 || true
free -m
df -h "$PMR_NEW_ROOT"
printf '%s  %s\n' d650dfddc1f877baa5dd4eba526fa53ef42097c2775b58788752ead2c099b08f "$PMR_ARCHIVE" | sha256sum -c -
test ! -e "$PMR_SOURCE"
mkdir "$PMR_SOURCE"
tar -xzf "$PMR_ARCHIVE" -C "$PMR_SOURCE"
test "$("$PMR_OLD_ROOT/bin/git" -C "$PMR_SOURCE" rev-parse HEAD)" = e66bdd368e1895f62bde788f3075edcf32180415
PMR_V4_OUT="$PMR_SOURCE/runs/state_memory_v4_$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}"
printf 'job=%s host=%s visible_gpu=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-}" "$PMR_SOURCE"
printf 'command=bash scripts/retest_v4.sh %s\noutput=%s\nlog=%s.log\npython=%s/source_852f57d/.venv/bin/python\nenvironment=Python 3.11 / PyTorch 2.9.1+cu128 image; old project venv\ngpu_count=1 cpus=4 container_memory=40g experiment_budget=3600s scheduler_limit=01:05:00\n' "$PMR_OLD_ROOT" "$PMR_V4_OUT" "$PMR_V4_OUT" "$PMR_OLD_ROOT"

PMR_CONTAINER_NAME="yxt-pmr-retest-v4-${SLURM_JOB_ID}"
cleanup() { paracloud-socker rm --force --name "$PMR_CONTAINER_NAME" || true; }
trap cleanup EXIT
paracloud-socker --epilog --insecure run \
  --cpus=4 --memory=40g --shm-size=2g \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" -e PMR_JOB_ID="$SLURM_JOB_ID" \
  -e PMR_NEW_ROOT="$PMR_NEW_ROOT" -e PMR_OLD_ROOT="$PMR_OLD_ROOT" \
  -e PMR_SOURCE="$PMR_SOURCE" -e PMR_V4_OUT="$PMR_V4_OUT" \
  -e PMR_ALLOCATED_GPU="$CUDA_VISIBLE_DEVICES" -e PMR_ALLOCATED_GPU_UUID="$PMR_GPU_UUID" \
  --gpus="\"device=$PMR_GPU_UUID\"" \
  --workdir "$PMR_SOURCE" -v "$PMR_OLD_ROOT:$PMR_OLD_ROOT" -v "$PMR_NEW_ROOT:$PMR_NEW_ROOT" \
  --user "$(id -u):$(id -g)" --name "$PMR_CONTAINER_NAME" \
  10.0.0.1:88/cadmin/pytorch@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385 \
  bash "$PMR_NEW_ROOT/retest_v4_inside.sh"
