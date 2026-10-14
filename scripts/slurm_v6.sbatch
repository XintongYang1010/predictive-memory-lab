#!/usr/bin/env bash
# Template only: fill partition/account/QOS from the cluster's real `sinfo` /
# `sacctmgr` output (docs/CODEX_V6_HANDOFF.md section 1).  Never guess them.
# 8 x A100 or 4 x H800 (prereg hard cap 160 GPUh, ledger in $OUT/budget_ledger.json).
#SBATCH --job-name=pmr-v6.1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
##SBATCH --partition=FILL_ME
##SBATCH --account=FILL_ME
set -euo pipefail
: "${STAGE:?export STAGE=S1|S2|...}"
: "${OUT:?export OUT=/abs/path/runs/v6_1_<id>}"
cd "${SLURM_SUBMIT_DIR}"
echo "job ${SLURM_JOB_ID} on $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# Inside the allocation the visible GPUs are renumbered 0..n-1; pass those indices.
export PMR_V6_GPUS="$(${PMR_PYTHON:-python} -c 'import torch; print(",".join(map(str, range(torch.cuda.device_count()))))')"
bash scripts/run_v6.sh "$STAGE" "$OUT"
