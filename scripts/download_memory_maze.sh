#!/usr/bin/env bash
# Download and unpack the Memory Maze offline dataset (MIT, Pasukonis et al. 2022).
#
#   bash scripts/download_memory_maze.sh /abs/data/memory_maze [9x9]
#
# Source (memory-maze README @4030901): Google Drive folder
#   https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB
# ~100 GB per maze size, 30k NPZ trajectories (29k train / 1k eval).
# The folder layout (archives vs loose files) is not documented, so this
# script first LISTS the folder and only downloads entries whose name contains
# the requested size.  If gdown cannot list it (quota / robots), use rclone with
# a Drive remote:  rclone lsf  then  rclone copy --include "*9x9*"  and point
# PMR_V6_DATA at the unpacked directory.  Needs outbound access to
# drive.google.com and *.googleusercontent.com.
set -euo pipefail
DEST="${1:?destination directory}"
SIZE="${2:-9x9}"
URL="https://drive.google.com/drive/folders/1RcnkTZVwEHnAQeEuw7X8Y1RPSmrFLDFB"
PY="${PMR_PYTHON:-python}"
mkdir -p "$DEST/raw"
"$PY" -m pip install --quiet "gdown==5.2.0"
cd "$DEST/raw"
echo "[mm] listing $URL"
"$PY" - "$URL" "$SIZE" <<'PY'
import sys, gdown
url, size = sys.argv[1], sys.argv[2]
files = gdown.download_folder(url, skip_download=True, quiet=True, remaining_ok=True)
sel = [f for f in files if size in f.path]
print(f"[mm] {len(files)} entries, {len(sel)} match {size}")
for f in files:
    print("   ", "*" if f in sel else " ", f.path)
if not sel:
    sys.exit("no entry matches; inspect the listing above and download manually")
import os
for f in sel:
    os.makedirs(os.path.dirname(f.path) or ".", exist_ok=True)
    gdown.download(id=f.id, output=f.path, quiet=False, resume=True)
PY
echo "[mm] unpacking archives"
find . -type f \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.zip" \) -print0 |
  while IFS= read -r -d '' a; do
    case "$a" in
      *.zip) unzip -q -n "$a" -d "$DEST" ;;
      *) tar -xf "$a" -C "$DEST" ;;
    esac
  done
N=$(find "$DEST" -name "*.npz" | wc -l)
echo "[mm] $N npz files under $DEST (expect ~30000 for one size)"
