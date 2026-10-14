#!/usr/bin/env python3
"""Export {video key: answer} for the 10-minute VSC split (FINAL scoring only).
The key matches the dump file name written by the logging patch:
video_path with '/' replaced by '_'."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="10mins")
    a = ap.parse_args()
    df = pd.concat([pd.read_parquet(p) for p in sorted(Path(a.parquet_dir).rglob("*.parquet"))])
    df = df[df["split"] == a.split]
    gt = {str(r.video_path).replace("/", "_"): float(r.answer) for r in df.itertuples()}
    Path(a.out).write_text(json.dumps(gt, indent=2))
    print(len(gt), "videos ->", a.out)


if __name__ == "__main__":
    main()
