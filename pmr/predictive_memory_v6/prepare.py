"""Memory Maze NPZ -> token cache (numpy only).

  python -m pmr.predictive_memory_v6 prepare --data DIR --cache DIR [--splits train dev] [--procs 32]

The test split is only built when explicitly requested (FINAL, AGENTS.md
rule 7).  Every cache directory holds memory-mapped ``.npy`` arrays plus
``ids.json`` and ``meta.json`` (axis convention, visibility parameters,
source file list hash).
"""
from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import maze as M
from . import prereg

EVAL_DIR_WORDS = ("eval", "test")


def discover(data_root):
    """All trajectories with their split, sorted by sha1(basename)."""
    root = Path(data_root)
    files = sorted(root.rglob("*.npz"))
    out = []
    for f in files:
        rel = f.relative_to(root)
        is_eval = any(w in part.lower() for part in rel.parts[:-1] for w in EVAL_DIR_WORDS)
        out.append({"path": str(f), "id": str(rel), "basename": f.name,
                    "split": M.split_of(f.name, is_eval, prereg.DATA["dev_mod"])})
    out.sort(key=lambda r: M.sha1_int(r["basename"]))
    return out


def select(files, split):
    rows = [r for r in files if r["split"] == split]
    if split == "train":
        rows = rows[: prereg.DATA["train_max_episodes"]]
    return rows


def calibrate(files, n_axis=20, n_vis=50, procs=8):
    """S0: axis convention, data unit checks, privileged colour calibration of
    the visibility rule.  Train files only."""
    train = [r for r in files if r["split"] == "train"]
    eps = [M.load_labels(r["path"]) for r in train[:n_axis]]
    conv, errs = M.fit_axis_convention(eps)
    checks = [M.check_episode(e) for e in eps]
    G = eps[0]["targets_pos"].shape[1] if eps else 3
    agree = {}
    grid = [(prereg.DATA["fov_deg"], np.inf), (prereg.DATA["fov_deg"], 8.0), (prereg.DATA["fov_deg"], 6.0),
            (prereg.DATA["fov_deg"] - 10, np.inf)]
    stats = {g: [0, 0, 0, 0] for g in grid}    # agree, total, tp, pos_img
    for r in train[:n_vis]:
        ep = M.load_labels(r["path"])
        try:
            img = M.load_images(r["path"])
        except KeyError:
            continue
        det = M.color_detect(img, G)
        lay = ep["maze_layout"][0]
        for (fov, md) in grid:
            vis = M.object_visibility(lay, ep["agent_pos"], ep["agent_dir"], ep["targets_pos"][0],
                                      ep["targets_vec"], conv, fov, md)
            s = stats[(fov, md)]
            s[0] += int((vis == det).sum())
            s[1] += vis.size
            s[2] += int((vis & det).sum())
            s[3] += int(det.sum())
    for (fov, md), s in stats.items():
        if s[1]:
            agree[f"fov{fov:g}_max{md:g}"] = {"agreement": s[0] / s[1], "image_recall": s[2] / max(1, s[3])}
    best = max(agree, key=lambda k: agree[k]["agreement"]) if agree else None
    meta = {"axis_convention": conv, "axis_errors": errs,
            "axis_ok": errs[conv] <= prereg.GATES["axis_fit_max_err_cells"],
            "data_checks": checks, "visibility_calibration": agree, "visibility_choice": best,
            "visibility_ok": bool(best and agree[best]["agreement"] >= prereg.GATES["s0_vis_agreement_min"])}
    if best:
        fov, md = best[3:].split("_max")
        meta["fov_deg"] = float(fov)
        meta["max_dist"] = float(md)
    else:
        meta["fov_deg"], meta["max_dist"] = prereg.DATA["fov_deg"], float("inf")
    return meta


def _work(args):
    path, conv, fov, md = args
    try:
        ep = M.load_labels(path)
        return M.episode_tokens(ep, conv, fov, md), None
    except Exception as exc:  # recorded, never silently dropped
        return None, f"{path}: {exc!r}"


def build_split(rows, out_dir, calib, procs=8):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists() and json.loads((out / "meta.json").read_text()).get("complete"):
        return json.loads((out / "meta.json").read_text())
    first = M.episode_tokens(M.load_labels(rows[0]["path"]), calib["axis_convention"],
                             calib["fov_deg"], calib["max_dist"])
    N = len(rows)
    spec = {"y": np.float16, "o": np.bool_, "pose": np.float32, "act": np.uint8, "key": np.int32,
            "vis": np.bool_, "tvec": np.float32, "pos": np.float32, "dir": np.float32,
            "layout": np.uint8, "tpos": np.float32}
    arrs = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=(N,) + first[k].shape)
            for k, dt in spec.items()}
    failures, ok = [], np.zeros(N, dtype=bool)
    jobs = [(r["path"], calib["axis_convention"], calib["fov_deg"], calib["max_dist"]) for r in rows]
    with Pool(procs) as pool:
        for i, (tok, err) in enumerate(pool.imap(_work, jobs, chunksize=4)):
            if err:
                failures.append(err)
                continue
            for k in spec:
                arrs[k][i] = tok[k]
            ok[i] = True
    for a in arrs.values():
        a.flush()
    np.save(out / "ok.npy", ok)
    meta = {"n": N, "n_ok": int(ok.sum()), "failures": failures[:50], "n_failures": len(failures),
            "S": int(first["layout"].shape[0]), "G": int(first["vis"].shape[1]),
            "D": int(first["y"].shape[1]), "calibration": {k: calib[k] for k in
                                                           ("axis_convention", "fov_deg", "max_dist")},
            "ids_sha1": hashlib.sha1("\n".join(r["id"] for r in rows).encode()).hexdigest(),
            "complete": True}
    (out / "ids.json").write_text(json.dumps([r["id"] for r in rows]))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def prepare(data_root, cache_root, splits=("train", "dev"), procs=8, maze_name=None):
    files = discover(data_root)
    if not files:
        raise SystemExit(f"no .npz files under {data_root}")
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    cal_f = cache / "calibration.json"
    if cal_f.exists():
        calib = json.loads(cal_f.read_text())
    else:
        calib = calibrate(files, procs=procs)
        cal_f.write_text(json.dumps(calib, indent=2, default=str))
    counts = {s: sum(r["split"] == s for r in files) for s in ("train", "dev", "test")}
    res = {"counts_found": counts, "calibration": {k: calib[k] for k in
                                                   ("axis_convention", "axis_ok", "visibility_choice",
                                                    "visibility_ok")}}
    for s in splits:
        rows = select(files, s)
        if not rows:
            res[s] = {"n": 0}
            continue
        res[s] = {k: v for k, v in build_split(rows, cache / s, calib, procs).items() if k != "failures"}
    (cache / "prepare.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def default_procs():
    return max(1, min(32, (os.cpu_count() or 2) - 1))
