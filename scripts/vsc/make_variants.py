#!/usr/bin/env python3
"""Build VSC-Repeat variants next to the official cache (ffmpeg).

repeat{1,2,5}: ``-stream_loop N -c copy`` exactly as supersanity @296c8c3.
phase:   original + a re-encoded copy that drops the first 12 frames (0.5 s),
         so frames are near- but not exact duplicates.
reverse: original + its 1 FPS frames in reverse order, each held 1 s.
File names are unchanged so the official annotations stay valid.
"""
import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def run(cmd):
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    ap.add_argument("--split", default="10mins")
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = ap.parse_args()
    src = Path(a.hf_home) / "cambrians_vsc" / a.split
    vids = sorted(src.glob("*.mp4"))
    if not vids:
        raise SystemExit(f"no videos in {src}")
    for n in (1, 2, 5):
        dst = Path(a.hf_home) / f"cambrians_vsc_repeat{n}" / a.split
        dst.mkdir(parents=True, exist_ok=True)
        for v in vids:
            if not (dst / v.name).exists():
                run([a.ffmpeg, "-y", "-v", "error", "-stream_loop", str(n), "-i", str(v), "-c", "copy", str(dst / v.name)])
    for kind in ("phase", "reverse"):
        dst = Path(a.hf_home) / f"cambrians_vsc_{kind}" / a.split
        dst.mkdir(parents=True, exist_ok=True)
        for v in vids:
            out = dst / v.name
            if out.exists():
                continue
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "second.mp4"
                if kind == "phase":
                    run([a.ffmpeg, "-y", "-v", "error", "-i", str(v), "-vf", "trim=start_frame=12,setpts=PTS-STARTPTS",
                         "-an", "-c:v", "libx264", "-crf", "18", "-r", "24", str(tmp)])
                else:
                    fr = Path(td) / "f"
                    fr.mkdir()
                    run([a.ffmpeg, "-y", "-v", "error", "-i", str(v), "-vf", "fps=1", str(fr / "%06d.png")])
                    frames = sorted(fr.glob("*.png"))[::-1]
                    lst = Path(td) / "list.txt"
                    lst.write_text("".join(f"file '{f}'\nduration 1\n" for f in frames) + f"file '{frames[-1]}'\n")
                    run([a.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "fps=24",
                         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)])
                first = Path(td) / "first.mp4"
                run([a.ffmpeg, "-y", "-v", "error", "-i", str(v), "-an", "-c:v", "libx264", "-crf", "18", "-r", "24",
                     "-pix_fmt", "yuv420p", str(first)])
                cat = Path(td) / "cat.txt"
                cat.write_text(f"file '{first}'\nfile '{tmp}'\n")
                run([a.ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(cat), "-c", "copy", str(out)])
    print("variants ready under", a.hf_home)


if __name__ == "__main__":
    main()
