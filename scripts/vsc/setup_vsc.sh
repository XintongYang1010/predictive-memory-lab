#!/usr/bin/env bash
# One-time VSC setup in a SEPARATE environment (Cambrian-S pins transformers
# 4.37/4.39, which must not touch the V6 torch env).
#
#   CAMBRIAN_S_DIR=/abs/cambrian-s HF_HOME=/abs/hf bash scripts/vsc/setup_vsc.sh
#
# 1. clone cambrian-s at the audited commit and apply the logging patch;
# 2. download only the 10-minute VSI-SUPER-Count split (13.5 GB) + parquet,
#    never the 311 GB snapshot lmms-eval would fetch on its own;
# 3. build repeat1/2/5, phase-shift and reversed variants (ffmpeg);
# 4. export the ground truth used only by FINAL scoring.
# Creating the Python env follows the cambrian-s README (not guessed here):
# set PMR_VSC_PYTHON to that env's interpreter before running VSC stages.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CAMBRIAN_S_DIR:?set CAMBRIAN_S_DIR}"
: "${HF_HOME:?set HF_HOME}"
PY="${PMR_VSC_PYTHON:-python}"
if [[ ! -d "$CAMBRIAN_S_DIR/.git" ]]; then
  git clone https://github.com/cambrian-mllm/cambrian-s "$CAMBRIAN_S_DIR"
fi
git -C "$CAMBRIAN_S_DIR" checkout 058b1b4c96846a229d1c9bb69da6ecf5ceee4a6b
"$PY" "$ROOT/scripts/vsc/patch_cambrians.py" "$CAMBRIAN_S_DIR"
TMP="${PMR_VSC_TMP:-$HF_HOME/vsc_download}"
mkdir -p "$TMP" "$HF_HOME/cambrians_vsc"
huggingface-cli download nyu-visionx/VSI-SUPER-Count --repo-type dataset \
  --include "10mins.zip" "*.parquet" --local-dir "$TMP"
if [[ ! -d "$HF_HOME/cambrians_vsc/10mins" ]]; then
  unzip -q "$TMP/10mins.zip" -d "$HF_HOME/cambrians_vsc/"
fi
"$PY" "$ROOT/scripts/vsc/make_variants.py" --hf-home "$HF_HOME"
"$PY" "$ROOT/scripts/vsc/export_gt.py" --parquet-dir "$TMP" --out "$HF_HOME/vsc_gt_10mins.json"
echo "VSC ready: export PMR_VSC_GT_JSON=$HF_HOME/vsc_gt_10mins.json"
