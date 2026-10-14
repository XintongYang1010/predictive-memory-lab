#!/usr/bin/env bash
# Template only: fill partition/account/QOS from the cluster's real `sinfo` /
# `sacctmgr` output (see docs/CODEX_DGX_HANDOFF.md section 1).  Never guess them.
#SBATCH --job-name=pmr-v6
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
: "${OUT:?export OUT=/abs/path/runs/v6_<id>}"
cd "${SLURM_SUBMIT_DIR}"
echo "job ${SLURM_JOB_ID} on $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
# Inside the allocation the visible GPUs are renumbered 0..n-1; pass those indices.
export PMR_V6_GPUS="$(python - <<'PY'
import torch; print(",".join(str(i) for i in range(torch.cuda.device_count())))
PY
)"
bash scripts/run_v6.sh "$STAGE" "$OUT"
