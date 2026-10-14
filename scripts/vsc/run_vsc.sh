#!/usr/bin/env bash
# One logged Cambrian-S-7B-LFP VSC pass (official flags; called by vsc_infer jobs).
#   PMR_VSC_DUMP=dir CAMBRIANS_VSC_CACHE_NAME=cambrians_vsc_repeat1 bash scripts/vsc/run_vsc.sh cambrians_vsc_10mins
# 60-minute or repeat5 streams keep all KV (~15 GB) plus ~16 GB weights: use 80 GB cards.
set -euo pipefail
TASK="${1:-cambrians_vsc_10mins}"
: "${CAMBRIAN_S_DIR:?set CAMBRIAN_S_DIR}"
: "${PMR_VSC_DUMP:?set PMR_VSC_DUMP}"
export DECORD_EOF_RETRY_MAX=20480
cd "$CAMBRIAN_S_DIR/lmms-eval"
if [[ -n "${PMR_VSC_PYTHON:-}" ]]; then export PATH="$(dirname "$PMR_VSC_PYTHON"):$PATH"; fi
NP="${PMR_VSC_NUM_PROCESSES:-$(python -c 'import torch;print(torch.cuda.device_count())')}"
bash evaluate_all_in_one.sh --model cambrians_vsc --benchmark "$TASK" --num_processes "$NP" --num_frames -1 \
  --pretrained nyu-visionx/Cambrian-S-7B-LFP --miv_token_len 64 --si_token_len 729 \
  --sensory_window_size 128 --enable_visual_feature_caching True --surprise_threshold 0.39
