#!/usr/bin/env bash
# Prepare pinned assets and a private environment without altering shared torch.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: PMR_BASE_PYTHON=/path/to/python bash scripts/prepare_visual_v5.sh /absolute/work/root [nyu|dgx]' >&2
  exit 2
fi
work_root=$1
backend=${2:-nyu}
[[ "$work_root" = /* ]] || { echo 'Use an absolute work root' >&2; exit 2; }
[[ "$backend" = nyu || "$backend" = dgx ]] || { echo 'Unknown backend' >&2; exit 2; }
mkdir -p "$work_root/assets" "$work_root/logs" "$work_root/cache"
export PIP_CACHE_DIR="$work_root/cache/pip"
export MPLCONFIGDIR="$work_root/cache/matplotlib"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
base_python=${PMR_BASE_PYTHON:-python}
if [[ "$backend" = dgx ]]; then
  # This venv has its own torch and dependencies. A shared interpreter is only
  # used to create it; no installation occurs in the shared environment.
  if [[ ! -e "$work_root/.venv" ]]; then "$base_python" -m venv "$work_root/.venv"; fi
  py="$work_root/.venv/bin/python"
  "$py" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
else
  py="$base_python"
fi
"$py" -c 'import sys,torch; print("Python:",sys.executable,"PyTorch:",torch.__version__,"CUDA:",torch.version.cuda); assert torch.version.cuda'
"$py" -m pip install -r requirements-visual-memory.txt
ijepa_dir="$work_root/assets/ijepa"
if [[ ! -e "$ijepa_dir" ]]; then git clone https://github.com/facebookresearch/ijepa.git "$ijepa_dir"; fi
[[ -z "$(git -C "$ijepa_dir" status --porcelain)" ]] || { echo 'I-JEPA checkout has local changes; inspect first' >&2; exit 1; }
git -C "$ijepa_dir" checkout --detach 52c1ae95d05f743e000e8f10a1f3a79b10cff048
checkpoint="$work_root/assets/IN1K-vit.h.14-300e.pth.tar"
if [[ ! -s "$checkpoint" ]]; then
  curl --fail --location --retry 3 --continue-at - \
    'https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar' -o "$checkpoint.partial"
  mv "$checkpoint.partial" "$checkpoint"
fi
"$py" -m pip freeze > "$work_root/logs/pip-freeze.txt"
"$py" -m unittest discover -s tests/visual_memory -v
"$py" -m pmr.visual_memory preview --out "$work_root/previews"
"$py" - "$work_root" "$backend" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); ck=root/'assets/IN1K-vit.h.14-300e.pth.tar'
h=hashlib.sha256()
with ck.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
(root/'preparation.json').write_text(json.dumps(dict(status='complete',backend=sys.argv[2],python=sys.executable,checkpoint_sha256=h.hexdigest(),checkpoint_bytes=ck.stat().st_size,completed_unix=time.time()),indent=2)+'\n')
PY
printf 'Prepared %s using %s. Inspect previews before GPU execution.\n' "$backend" "$py"
