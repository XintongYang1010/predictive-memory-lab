#!/usr/bin/env bash
# Clone the Apache-2.0 Miniworld fork used by FloWM at the pinned commit and
# install its rendering deps into a separate venv that can still see the main
# environment's torch (needed by generate_videos.py to save *_block_info.pt).
#   bash scripts/setup_v6_miniworld.sh <cache_dir>
#   export PMR_MINIWORLD_PYTHON=<cache_dir>/miniworld-venv/bin/python
set -euo pipefail
CACHE="${1:?cache dir}"
PY="${PMR_PYTHON:-python}"
COMMIT=b28988d5cd9d0d261a53b8cfa4f162e7fe14853e
mkdir -p "$CACHE"
if [[ ! -d "$CACHE/Miniworld/.git" ]]; then
  git clone https://github.com/hlillemark/Miniworld.git "$CACHE/Miniworld"
fi
git -C "$CACHE/Miniworld" fetch origin master
git -C "$CACHE/Miniworld" checkout "$COMMIT"
grep -q "store-block-info" "$CACHE/Miniworld/scripts/generate_videos.py"
"$PY" -m venv --system-site-packages "$CACHE/miniworld-venv"
"$CACHE/miniworld-venv/bin/python" -m pip install -e "$CACHE/Miniworld[dataset]"
cp "$CACHE/Miniworld/license" "$CACHE/Miniworld_LICENSE_Apache-2.0.txt"
echo "export PMR_MINIWORLD_PYTHON=$CACHE/miniworld-venv/bin/python"
echo "Headless rendering: MINIWORLD_HEADLESS=1 (Mesa llvmpipe); alternatives PYOPENGL_PLATFORM=egl or xvfb-run."
