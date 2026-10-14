"""R stage: token-level validation on public amodal track annotations (numpy).

Only annotation files are used -- no images, no detector, no encoder.

nuScenes (primary; metadata tables only, CC BY-NC-SA 4.0 per the paper -- save
the licence text with the download).  2 Hz keyframes, global-frame boxes that
are kept even when an object has no lidar/radar points.
    hidden_strict  num_lidar_pts == 0 and num_radar_pts == 0
    hidden_loose   visibility_token == "1" (0-40 %)
The official tracking eval drops point-less boxes and interpolates; V6 keeps
ALL boxes ("All-boxes setup" of Offline Tracking with Object Permanence,
arXiv 2310.01288).  Keyframes are never interpolated to 10/20 Hz.

MOT17 / MOT20 (secondary; train GT only).  gt.txt rows
``frame,id,left,top,w,h,flag,class,visibility``; keep flag == 1 and class == 1
(pedestrian); hidden = visibility < 0.1 (Detecting Invisible People,
arXiv 2012.08419).  Static-camera sequences only (MOT17-05/10/11/13 move).

CAVEAT (pre-registered): hidden-period positions are annotator inferences and
nearly linear, so linear interpolation is a ceiling by construction; report
gap closure against it, never a win over constant velocity.

Splits: nuScenes scenes are split by a stable hash of the scene name
(80/10/10); MOT uses leave-one-sequence-out folds.
"""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import Geometry

NUSCENES_CATEGORIES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.bicycle",
                       "vehicle.motorcycle", "human.pedestrian.adult", "human.pedestrian.child")
MOT_STATIC = ("MOT17-02", "MOT17-04", "MOT17-09", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05")
MAX_OBJECTS = 8
NUSC_SCALE, MOT_SCALE = 50.0, 500.0


def _hash_split(name, fractions=(0.8, 0.1, 0.1)):
    h = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little") / 2 ** 32
    return "train" if h < fractions[0] else ("dev" if h < fractions[0] + fractions[1] else "test")


def _runs(mask):
    """[(start, end)) runs of True."""
    out, t, T = [], 0, len(mask)
    while t < T:
        if mask[t]:
            s = t
            while t < T and mask[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def has_event(exists, hidden, min_before=2):
    """A hidden run preceded by >= min_before visible steps and followed by a visible step."""
    vis = exists & ~hidden
    for s, e in _runs(exists & hidden):
        if s >= min_before and vis[s - min_before:s].all() and e < len(vis) and vis[e]:
            return True
    return False


def _choose_subsets(tracks, rng, max_objects=MAX_OBJECTS):
    """Partition track ids into episodes of <= max_objects, event tracks first."""
    ev = [k for k, v in tracks.items() if v["event"]]
    other = [k for k, v in tracks.items() if not v["event"]]
    rng.shuffle(ev)
    rng.shuffle(other)
    groups = []
    while ev:
        g = ev[:max_objects // 2]
        ev = ev[max_objects // 2:]
        fill = max_objects - len(g)
        g += other[:fill]
        other = other[fill:]
        groups.append(g)
    return groups


def _pack(tracks_by_group, T, app_dim, rng, extra=None):
    eps = []
    for g in tracks_by_group:
        N = MAX_OBJECTS
        pos = np.zeros((T, N, 2))
        ex = np.zeros((T, N), bool)
        hid = np.zeros((T, N), bool)
        size = np.zeros((T, N, 2))
        for j, tr in enumerate(g):
            pos[:, j], ex[:, j], hid[:, j] = tr["pos"], tr["exists"], tr["hidden"]
            if "size" in tr:
                size[:, j] = tr["size"]
        app = rng.standard_normal((N, app_dim))
        app /= np.linalg.norm(app, axis=-1, keepdims=True)
        eps.append({"pos": pos, "exists": ex, "vis": ex & ~hid, "size": size, "app": app.astype(np.float32)})
    return eps


# -------------------------------------------------------------- nuScenes
def _load_table(root, name):
    return json.loads((Path(root) / f"{name}.json").read_text())


def prepare_nuscenes(meta_root, cache, hidden_rule="strict", app_dim=16, seed=0, T=38):
    """Parse nuScenes metadata tables (v1.0-trainval) into token episodes."""
    root = Path(meta_root)
    samples = {s["token"]: s for s in _load_table(root, "sample")}
    scenes = _load_table(root, "scene")
    cat = {c["token"]: c["name"] for c in _load_table(root, "category")}
    inst_cat = {i["token"]: cat[i["category_token"]] for i in _load_table(root, "instance")}
    vis_level = {v["token"]: v.get("level", v["token"]) for v in _load_table(root, "visibility")}
    ego = {e["token"]: e for e in _load_table(root, "ego_pose")}
    sd_by_sample = {}
    for sd in _load_table(root, "sample_data"):
        if sd.get("is_key_frame") and "LIDAR_TOP" in sd.get("filename", ""):
            sd_by_sample[sd["sample_token"]] = sd["ego_pose_token"]
    ann_by_sample = {}
    for a in _load_table(root, "sample_annotation"):
        ann_by_sample.setdefault(a["sample_token"], []).append(a)
    rng = np.random.default_rng(seed)
    out = {"train": [], "dev": [], "test": []}
    stats = {"scenes": 0, "event_tracks": 0, "hidden_run_lengths": []}
    for sc in scenes:
        order, tok = [], sc["first_sample_token"]
        while tok:
            order.append(tok)
            tok = samples[tok]["next"]
        L = len(order)
        if L < T:
            continue
        order = order[:T]
        times = np.array([samples[s]["timestamp"] for s in order], dtype=np.float64) * 1e-6
        ego_xy = np.array([ego[sd_by_sample[s]]["translation"][:2] for s in order]) if all(
            s in sd_by_sample for s in order) else np.zeros((T, 2))
        center = ego_xy.mean(0)
        tracks = {}
        for t, s in enumerate(order):
            for a in ann_by_sample.get(s, []):
                if not any(inst_cat.get(a["instance_token"], "").startswith(c) for c in NUSCENES_CATEGORIES):
                    continue
                tr = tracks.setdefault(a["instance_token"], {"pos": np.zeros((T, 2)), "exists": np.zeros(T, bool),
                                                             "hidden": np.zeros(T, bool), "size": np.zeros((T, 2))})
                tr["pos"][t] = np.array(a["translation"][:2]) - center
                tr["exists"][t] = True
                tr["size"][t] = a["size"][:2]
                if hidden_rule == "strict":
                    tr["hidden"][t] = a["num_lidar_pts"] == 0 and a["num_radar_pts"] == 0
                else:
                    tr["hidden"][t] = str(vis_level.get(a["visibility_token"], a["visibility_token"])) in ("1", "v0-40")
        for tr in tracks.values():
            tr["event"] = has_event(tr["exists"], tr["hidden"])
            stats["hidden_run_lengths"] += [e - s for s, e in _runs(tr["exists"] & tr["hidden"])]
            stats["event_tracks"] += int(tr["event"])
        groups = [[tracks[k] for k in g] for g in _choose_subsets(tracks, rng)]
        eps = _pack(groups, T, app_dim, rng)
        split = _hash_split(sc["name"])
        dt = np.ones(T)
        dt[1:] = np.diff(times) * 2.0           # 2 Hz keyframes -> dt ~ 1
        u = np.zeros((T, 2))
        u[1:] = ego_xy[1:] - ego_xy[:-1]
        for e in eps:
            e.update(dt=dt, u=u, ego=ego_xy - center, scene=sc["name"])
            out[split].append(e)
        stats["scenes"] += 1
    return _save(cache, "nuscenes", out, stats, Geometry("plane", scale=NUSC_SCALE))


def pseudo_occlusion(ep, rng, min_len=3, max_len=12, context=2):
    """R-b: hide a random interval of a sensor-supported visible segment."""
    ep = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()}
    T, N = ep["vis"].shape
    for j in range(N):
        for s, e in _runs(ep["vis"][:, j]):
            if e - s >= min_len + 2 * context:
                L = int(rng.integers(min_len, min(max_len, e - s - 2 * context) + 1))
                a = int(rng.integers(s + context, e - context - L + 1))
                ep["vis"][a:a + L, j] = False
                break
    return ep


# ------------------------------------------------------------------- MOT
def _read_seqinfo(seq_dir):
    cp = configparser.ConfigParser()
    cp.read(Path(seq_dir) / "seqinfo.ini")
    s = cp["Sequence"]
    return {"name": s.get("name"), "fps": float(s.get("frameRate")), "w": float(s.get("imWidth")),
            "h": float(s.get("imHeight")), "length": int(s.get("seqLength"))}


def read_mot_gt(path):
    rows = np.loadtxt(path, delimiter=",", ndmin=2)
    keep = (rows[:, 6] == 1) & (rows[:, 7] == 1)
    return rows[keep]


def prepare_mot(roots, cache, target_fps=5.0, T=64, stride=32, app_dim=16, seed=0, sequences=MOT_STATIC):
    """roots: list of MOT17/train and MOT20/train directories."""
    rng = np.random.default_rng(seed)
    seqs = {}
    for root in roots:
        for d in sorted(Path(root).iterdir()):
            base = "-".join(d.name.split("-")[:2])            # MOT17-02-FRCNN -> MOT17-02
            if base in sequences and base not in seqs and (d / "gt" / "gt.txt").exists():
                seqs[base] = d
    out, stats = {}, {"sequences": {}, "hidden_run_lengths": []}
    for base, d in seqs.items():
        info = _read_seqinfo(d)
        step = max(1, int(round(info["fps"] / target_fps)))
        gt = read_mot_gt(d / "gt" / "gt.txt")
        frames = np.arange(1, info["length"] + 1, step)
        F = len(frames)
        fidx = {int(f): k for k, f in enumerate(frames)}
        tracks = {}
        for fr, tid, left, top, w, h, flag, cls, visr in gt:
            k = fidx.get(int(fr))
            if k is None:
                continue
            tr = tracks.setdefault(int(tid), {"pos": np.zeros((F, 2)), "exists": np.zeros(F, bool),
                                              "hidden": np.zeros(F, bool), "size": np.zeros((F, 2))})
            tr["pos"][k] = [left + w / 2 - info["w"] / 2, top + h / 2 - info["h"] / 2]
            tr["exists"][k] = True
            tr["hidden"][k] = visr < 0.1
            tr["size"][k] = [w, h]
        eps = []
        for s0 in range(0, max(1, F - T + 1), stride):
            win = {}
            for k, tr in tracks.items():
                ex = tr["exists"][s0:s0 + T]
                if ex.sum() < 3:
                    continue
                w = {key: v[s0:s0 + T] for key, v in tr.items()}
                w["event"] = has_event(w["exists"], w["hidden"])
                stats["hidden_run_lengths"] += [e - s for s, e in _runs(w["exists"] & w["hidden"])]
                win[k] = w
            groups = [[win[k] for k in g] for g in _choose_subsets(win, rng)]
            for e in _pack(groups, T, app_dim, rng):
                e.update(dt=np.full(T, step / info["fps"] * 2.0), u=np.zeros((T, 2)),
                         ego=np.zeros((T, 2)), scene=f"{base}@{s0}")
                eps.append(e)
        out[base] = eps
        stats["sequences"][base] = {"episodes": len(eps), "stride_frames": step, "fps": info["fps"]}
    return _save(cache, "mot", out, stats, Geometry("plane", scale=MOT_SCALE))


def mot_fold(fold: int, sequences=MOT_STATIC):
    seqs = list(sequences)
    test = seqs[fold % len(seqs)]
    dev = seqs[(fold + 1) % len(seqs)]
    return {"test": [test], "dev": [dev], "train": [s for s in seqs if s not in (test, dev)]}


# ------------------------------------------------------------ save/load
def _save(cache, source, groups, stats, geom: Geometry):
    dst = Path(cache) / "real" / source
    dst.mkdir(parents=True, exist_ok=True)
    for name, eps in groups.items():
        if not eps:
            continue
        arr = {k: np.stack([e[k] for e in eps]) for k in ("pos", "exists", "vis", "size", "app", "dt", "u", "ego")}
        arr["scene"] = np.array([e["scene"] for e in eps])
        np.savez_compressed(dst / f"{name}.npz", **arr)
    hl = np.array(stats.pop("hidden_run_lengths", []), dtype=int)
    stats["hidden_run_histogram"] = {int(k): int(v) for k, v in zip(*np.unique(hl, return_counts=True))} if len(hl) else {}
    stats["groups"] = {k: len(v) for k, v in groups.items()}
    stats["geometry"] = geom.to_dict()
    (dst / "manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


_CACHE = {}


def load_real(spec, n, T, seed, cache):
    """spec: real_nuscenes_<train|dev|test>[_pseudo] or real_mot_<fold>_<train|dev|test>."""
    name = spec.split(":")[0]
    parts = name.split("_")
    source = parts[1]
    dst = Path(cache) / "real" / source
    geom = Geometry.from_dict(json.loads((dst / "manifest.json").read_text())["geometry"])
    pseudo = parts[-1] == "pseudo"
    if source == "nuscenes":
        groups = [parts[2]]
    elif source == "mot":
        groups = mot_fold(int(parts[2]))[parts[3]]
    else:
        raise ValueError(spec)
    arrs = []
    for g in groups:
        key = (str(dst), g)
        if key not in _CACHE:
            _CACHE[key] = dict(np.load(dst / f"{g}.npz"))
        arrs.append(_CACHE[key])
    d = {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}
    rng = np.random.default_rng(seed)
    E, L = d["vis"].shape[:2]
    T = min(T, L)
    idx = np.arange(E) if (n is None or n <= 0 or n >= E) else rng.choice(E, n, replace=False)
    start = rng.integers(0, L - T + 1, len(idx))
    sl = lambda a: np.stack([a[e, s:s + T] for e, s in zip(idx, start)])
    out = {"pos": sl(d["pos"]), "exists": sl(d["exists"]), "vis": sl(d["vis"]), "size": sl(d["size"]),
           "dt": sl(d["dt"]), "u": sl(d["u"]), "ego": sl(d["ego"]), "app": d["app"][idx]}
    if pseudo:
        eps = [pseudo_occlusion({k: out[k][i] for k in ("vis",)}, rng) for i in range(len(idx))]
        out["vis"] = np.stack([e["vis"] for e in eps]) & out["exists"]
    B, _, N = out["vis"].shape
    out["obs"] = np.where(out["vis"][..., None], out["pos"], 0.0)
    out["pos_unwrapped"] = out["pos"]
    out["change_step"] = np.zeros((B, T, N), dtype=np.int8)
    out["reveal_changed"] = np.zeros((B, T, N), bool)
    out["episode_id"] = np.array([f"{name}-{e}-{s}" for e, s in zip(idx, start)])
    out["meta"] = {"world": "R", "spec": spec, "T": T, "seed": int(seed), "frame": "allo",
                   "geometry": geom.to_dict(), "config": {"visibility": "annotated", "n_objects": N,
                                                          "obs_noise": 0.0, "vel_max": 1.0}}
    return out
