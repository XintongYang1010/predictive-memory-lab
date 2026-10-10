#!/usr/bin/env bash
# Run on a Cloud Bursting CPU compute node, before allocating an A100.
set -euo pipefail
if [[ $# -ne 1 ]]; then echo 'Usage: bash scripts/prepare_visual_v5.sh /absolute/burst/work/root' >&2; exit 2; fi
work_root=$1
[[ "$work_root" = /* ]] || { echo 'Use an absolute work root on the Burst compute filesystem' >&2; exit 2; }
mkdir -p "$work_root/assets"
python -c 'import torch; print("Existing PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)'
python -m pip install -r requirements-visual-memory.txt
ijepa_dir="$work_root/assets/ijepa"
if [[ ! -e "$ijepa_dir" ]]; then
  git clone https://github.com/facebookresearch/ijepa.git "$ijepa_dir"
fi
[[ -z "$(git -C "$ijepa_dir" status --porcelain)" ]] || { echo 'I-JEPA checkout has local changes; inspect first' >&2; exit 1; }
git -C "$ijepa_dir" checkout --detach 52c1ae95d05f743e000e8f10a1f3a79b10cff048
checkpoint="$work_root/assets/IN1K-vit.h.14-300e.pth.tar"
if [[ ! -s "$checkpoint" ]]; then
  curl --fail --location --retry 3 --continue-at - \
    'https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar' -o "$checkpoint.partial"
  mv "$checkpoint.partial" "$checkpoint"
fi
python -m unittest discover -s tests/visual_memory -v
python -m pmr.visual_memory preview --out "$work_root/previews"
echo 'Assets and CPU checks prepared. Use launch_visual_v5.py from the Burst compute filesystem.'
