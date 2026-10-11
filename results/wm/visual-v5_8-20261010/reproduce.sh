#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root at the source commit in registration_publication.json.
# Restore dependency paths from the V5.5 and V5.6 GitHub Releases first.
# A fresh output root is mandatory. Do not overwrite archived results.
out_root=${1:?Supply a new, nonexistent output directory}
if [[ -e "$out_root" ]]; then
  echo 'Output directory already exists; choose a fresh path.' >&2
  exit 2
fi
mkdir -p "$out_root"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python3 scripts/v5_8_spatial_diagnosis.py run \
  --scene execution/visual-v5_5-20261010/diagnostic-20261010-v55 \
  --costs execution/visual-v5_6-20261010/correspondence/results/costs_and_predictions.npz \
  --baseline-report execution/visual-v5_6-20261010/correspondence/results/report.json \
  --plan scripts/v5_8_spatial_plan.json \
  --publication execution/visual-v5_8-20261010/registration_publication.json \
  --out "$out_root/spatial"
