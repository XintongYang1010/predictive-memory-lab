"""Evaluation conditions, L1 perturbations and transition tensors (numpy only).

A *transition* t predicts frame j = t + 1 of a frame sequence from frame j-1:

  L0/L1  z_in = [pose_{j-1}, pose_j, onehot(action_j), o_{j-1} * y_{j-1}, o_{j-1}]
         value y_j, mask o_j, oracle key = (cell, octant) of pose_j
  R      z_in = [x_{j-1}, onehot(action_j)], value x_j (no mask)

The pose of the frame being predicted is an input (ground-truth odometry):
the memory stores *what is where*, never integrates the path.

Conditions (frame sequences, t_s = 400):
  STATIC   A[0:1001]                       unspliced; STAT window, probes, FA
  CONT     A[0:800]                        reference
  EXACT    A[0:400] + A[0:400]             familiar: protect
  NEWPATH  A[0:400] + A[600:1000]          same world, new start: protect what was seen
  CHANGE   A[0:400] + B[600:1000]          different world: reset conflicting keys
  MOVE     A[0:1001], one hidden object moved at t_m (L1 only)
The teleport transition into frame t_s is neither written nor scored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import maze as M
from . import prereg

CONDITIONS = ("STATIC", "CONT", "EXACT", "NEWPATH", "CHANGE", "MOVE")
N_ACT = 6


def _seed(*parts):
    return M.sha1_int("/".join(str(p) for p in parts)) % (2 ** 63)


class Store:
    """Memory-mapped token cache of one split."""

    KEYS = ("y", "o", "pose", "act", "key", "vis", "tvec", "pos", "dir", "layout", "tpos")

    def __init__(self, cache, split):
        d = Path(cache) / split
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        self.ids = json.loads((d / "ids.json").read_text())
        self.ok = np.load(d / "ok.npy")
        self.a = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in self.KEYS}
        self.S, self.G, self.D = self.meta["S"], self.meta["G"], self.meta["D"]
        self.conv = self.meta["calibration"]["axis_convention"]
        self.fov = float(self.meta["calibration"]["fov_deg"])
        self.max_dist = float(self.meta["calibration"]["max_dist"])
        self.order = np.flatnonzero(self.ok)
        feat = d / "rfeat.npy"
        self.rfeat = np.load(feat, mmap_mode="r") if feat.exists() else None

    def __len__(self):
        return len(self.order)

    def episode(self, i):
        e = {k: np.asarray(self.a[k][i]) for k in self.KEYS}
        e["y"] = e["y"].astype(np.float32)
        if self.rfeat is not None:
            e["x"] = np.asarray(self.rfeat[i], dtype=np.float32)
        e["eid"] = self.ids[i]
        return e


# ------------------------------------------------------------- planning
def plan(store, cond, n_items, salt="dev"):
    """Deterministic item specs; pairs A with the next episode for CHANGE."""
    order = list(store.order)
    specs = []
    if cond == "MOVE":
        for i in order:
            mv = _plan_move(store, i, salt)
            if mv is not None:
                specs.append({"cond": cond, "A": int(i), **mv})
            if len(specs) >= n_items:
                break
        return specs
    for k, i in enumerate(order[:n_items]):
        s = {"cond": cond, "A": int(i)}
        if cond == "CHANGE":
            s["B"] = int(order[(k + 1) % len(order)])
        specs.append(s)
    return specs


def _plan_move(store, i, salt):
    cfg = prereg.L1
    vis = np.asarray(store.a["vis"][i])
    lay = np.asarray(store.a["layout"][i])
    tpos = np.asarray(store.a["tpos"][i])
    rng = np.random.default_rng(_seed("move", store.ids[i], salt))
    lo, hi = cfg["move_window"]
    cands = []
    for g in range(vis.shape[1]):
        for tm in range(lo, hi + 1):
            if not vis[tm - cfg["move_hidden_before"]:tm].any() and not vis[tm:tm + cfg["move_hidden_after"]].any():
                cands.append((g, tm))
    if not cands:
        return None
    g, tm = cands[rng.integers(len(cands))]
    D = M.bfs_all(lay)
    S = lay.shape[0]
    c0 = np.clip(M.cell_of(tpos[g]), 0, S - 1)
    src = c0[1] * S + c0[0]
    free = [s for s in range(S * S) if np.isfinite(D[src, s]) and D[src, s] >= cfg["move_min_bfs"]]
    occupied = {int(np.floor(p[1])) * S + int(np.floor(p[0])) for p in tpos}
    free = [s for s in free if s not in occupied]
    if not free:
        return None
    dst = free[rng.integers(len(free))]
    new = np.array([dst % S + 0.5, dst // S + 0.5])
    return {"g": int(g), "tm": int(tm), "new": new.tolist()}


# ----------------------------------------------------------- perturbation
def apply_move(ep, spec, conv, fov, max_dist):
    """Move object g to ``new`` from frame tm on; recompute its egocentric
    vector, visibility and value/mask entries.  Returns reveal frame or None."""
    g, tm, new = spec["g"], spec["tm"], np.asarray(spec["new"])
    pos, d, lay = ep["pos"].astype(np.float64), ep["dir"].astype(np.float64), ep["layout"]
    tv = M.egocentric(new[None] - pos[tm:], d[tm:], conv)
    vis = M.object_visibility(lay, pos[tm:], d[tm:], new[None], tv[:, None], conv, fov, max_dist)[:, 0]
    ep["tvec"] = ep["tvec"].copy()
    ep["vis"] = ep["vis"].copy()
    ep["tvec"][tm:, g] = tv
    ep["vis"][tm:, g] = vis
    b = M.N_GRID + 3 * g
    ep["y"] = ep["y"].copy()
    ep["o"] = ep["o"].copy()
    ep["y"][tm:, b] = vis
    ep["y"][tm:, b + 1:b + 3] = np.where(vis[:, None], tv, 0.0)
    ep["o"][tm:, b + 1:b + 3] = vis[:, None]
    seen = np.flatnonzero(vis)
    return int(tm + seen[0]) if len(seen) else None


def perturb_l1(ep, rng, sigma_vec, S, conv, fov, flicker_g=None):
    """Observation-space noise (L1).  ``sigma_vec`` [G,2] raw-unit noise s.d.
    of the object-vector dims; pose noise only enters the inputs."""
    cfg = prereg.L1
    T, G = ep["vis"].shape
    y, o = ep["y"].copy(), ep["o"].copy()
    vis = ep["vis"].copy()
    drop = vis & (rng.random((T, G)) < cfg["vis_dropout"])
    fp = (~vis) & (rng.random((T, G)) < cfg["vis_false_pos"])
    for g in range(G):
        b = M.N_GRID + 3 * g
        v = vis[:, g] & ~drop[:, g]
        noise = rng.normal(0.0, 1.0, (T, 2)) * sigma_vec[g]
        vec = ep["tvec"][:, g] + noise
        if flicker_g is not None and g == flicker_g:
            vec = vec + rng.normal(0.0, 1.0, (T, 2)) * np.sqrt(cfg["flicker_var"]) * sigma_vec[g] \
                / np.sqrt(cfg["obs_noise_frac"] / (1 - cfg["obs_noise_frac"]))
        ang = rng.uniform(-np.deg2rad(fov) / 2, np.deg2rad(fov) / 2, T)
        dist = rng.uniform(0.5, 4.0, T)
        i_lat, i_fwd, _ = M.AXIS_CANDIDATES[conv]
        junk = np.zeros((T, 2))
        junk[:, i_lat], junk[:, i_fwd] = dist * np.sin(ang), dist * np.cos(ang)
        f = fp[:, g]
        shown = v | f
        y[:, b] = shown
        y[:, b + 1:b + 3] = np.where(v[:, None], vec, np.where(f[:, None], junk, 0.0))
        o[:, b + 1:b + 3] = shown[:, None]
    pose = ep["pose"].copy()
    pose[:, :2] += rng.normal(0.0, cfg["pose_noise_cells"] * 2.0 / S, (T, 2))
    a = np.arctan2(pose[:, 3], pose[:, 2]) + rng.normal(0.0, cfg["dir_noise_rad"], T)
    pose[:, 2], pose[:, 3] = np.cos(a), np.sin(a)
    out = dict(ep)
    out.update(y=y.astype(np.float32), o=o, pose_in=pose, shown=y[:, M.N_GRID::3] > 0.5)
    return out


# ------------------------------------------------------------ transitions
def _onehot(a):
    out = np.zeros(a.shape + (N_ACT,), dtype=np.float32)
    np.put_along_axis(out, a[..., None].astype(np.int64), 1.0, -1)
    return out


def frames_to_transitions(fr, kind):
    """fr: dict of frame arrays [L, ...] (already concatenated).  Returns the
    transition arrays [L-1, ...]."""
    if kind in ("r", "rpose"):
        x = fr["x"]
        zin = np.concatenate([x[:-1], _onehot(fr["act"][1:])], -1)
        y = x[1:]
        o = np.ones_like(y, dtype=bool)
    else:
        pose_in = fr.get("pose_in", fr["pose"])
        yo = fr["y"] * fr["o"]
        zin = np.concatenate([pose_in[:-1], pose_in[1:], _onehot(fr["act"][1:]), yo[:-1],
                              fr["o"][:-1].astype(np.float32)], -1)
        y, o = fr["y"][1:], fr["o"][1:]
    out = {"zin": zin.astype(np.float32), "y": y.astype(np.float32), "o": o.astype(bool),
           "key": fr["key"][1:].astype(np.int64)}
    if kind == "rpose":
        out["zkey"] = np.concatenate([fr["pose"][:-1], fr["pose"][1:], _onehot(fr["act"][1:])], -1)
    return out


def _concat(segs):
    keys = segs[0].keys()
    return {k: np.concatenate([s[k] for s in segs], 0) for k in keys}


FRAME_KEYS = ("y", "o", "pose", "act", "key", "vis", "tvec", "pos", "dir")


def _frames(ep, sl, extra=()):
    d = {k: ep[k][sl] for k in FRAME_KEYS + tuple(extra) if k in ep}
    return d


def build_item(store, spec, track, rng_salt="dev", sigma_vec=None, sbar_inj=0.0, item_uid=None,
               want_revisits=True):
    """Numpy arrays for one evaluation item.

    track: dict(kind in {'l0','l1','r','rpose'}).  Returns transitions plus
    scoring metadata: w (write flag), ts, tm/tr/g (MOVE), flicker, vis/tvec of
    targets, eps (residual-space noise for T0/T1)."""
    kind = track["kind"]
    cond = spec["cond"]
    uid = item_uid or f"{cond}/{store.ids[spec['A']]}/{spec.get('B', '')}/{rng_salt}"
    A = store.episode(spec["A"])
    extra = ("x",) if kind in ("r", "rpose") else ()
    ts = prereg.DATA["splice_ts"]
    meta = {"uid": uid, "cond": cond, "A": store.ids[spec["A"]], "ts": None}
    if kind == "l1":
        G = store.G
        rng0 = np.random.default_rng(_seed("flicker", store.ids[spec["A"]]))
        meta["flicker"] = int(rng0.integers(G))
    move_tr = None
    if cond == "MOVE":
        move_tr = apply_move(A, spec, store.conv, store.fov, store.max_dist)
        meta.update(g=spec["g"], tm=spec["tm"], tr=move_tr)

    def seg(ep, sl, segno):
        fr = _frames(ep, sl, extra)
        if kind == "l1":
            rng = np.random.default_rng(_seed("l1", uid, segno))
            p = perturb_l1({**fr, "tvec": fr["tvec"]}, rng, sigma_vec, store.S, store.conv, store.fov,
                           meta.get("flicker"))
            fr.update(y=p["y"], o=p["o"], pose_in=p["pose_in"])
            fr["shown"] = p["shown"]
        return fr

    if cond in ("STATIC", "MOVE"):
        segs = [seg(A, slice(0, 1001), 0)]
    elif cond == "CONT":
        segs = [seg(A, slice(0, 800), 0)]
    elif cond == "EXACT":
        segs = [seg(A, slice(0, ts), 0), seg(A, slice(0, ts), 1)]
    elif cond == "NEWPATH":
        segs = [seg(A, slice(0, ts), 0), seg(A, slice(600, 1000), 1)]
    elif cond == "CHANGE":
        B = store.episode(spec["B"])
        meta["B"] = store.ids[spec["B"]]
        segs = [seg(A, slice(0, ts), 0), seg(B, slice(600, 1000), 1)]
    else:
        raise ValueError(cond)
    fr = _concat(segs)
    tr = frames_to_transitions(fr, kind)
    L1 = len(tr["y"])
    w = np.ones(L1, dtype=np.float32)
    if len(segs) == 2:
        w[ts - 1] = 0.0           # teleport into frame ts: no write, no score
        meta["ts"] = ts - 1       # transition index of the splice
    tr["w"] = w
    tr["vis"] = fr["vis"][1:]
    tr["tvec"] = fr["tvec"][1:]
    if "shown" in fr:
        tr["shown"] = fr["shown"][1:]
    tr["pos"] = fr["pos"][1:]
    tr["dir"] = fr["dir"][1:]
    if sbar_inj > 0:
        rng = np.random.default_rng(_seed("eps", uid))
        tr["eps"] = rng.standard_normal(tr["y"].shape).astype(np.float32)
    if cond == "STATIC" and want_revisits:
        meta["revisits"] = M.revisit_events(A["pos"], A["dir"], A["layout"], prereg.DATA["revisit_min_gap"],
                                            prereg.DATA["revisit_min_bfs"], prereg.DATA["view_revisit_cos"])
    tr["meta"] = meta
    return tr


def stack_items(items):
    keys = [k for k in items[0] if k != "meta"]
    out = {k: np.stack([it[k] for it in items]) for k in keys}
    out["meta"] = [it["meta"] for it in items]
    return out


def train_windows(store, rng, batch, kind, length=1000, sigma_vec=None):
    """Random unspliced train windows for fitting (f_theta, phi, gates)."""
    items = []
    for _ in range(batch):
        i = int(store.order[rng.integers(len(store.order))])
        uid = f"train/{store.ids[i]}/{rng.integers(2**31)}"
        tr = build_item(store, {"cond": "STATIC", "A": i}, {"kind": kind}, sigma_vec=sigma_vec, item_uid=uid,
                        want_revisits=False)
        if length < len(tr["y"]):
            s = int(rng.integers(0, len(tr["y"]) - length + 1))
            tr = {k: (v[s:s + length] if k != "meta" else v) for k, v in tr.items()}
        items.append(tr)
    return stack_items(items)


def train_spliced(store, rng, batch, kind, sigma_vec=None):
    """A2+ training mixture EXACT : NEWPATH : CHANGE : unspliced = 1:1:1:1,
    all from the train split (dev/test never used for fitting)."""
    items = []
    n = len(store.order)
    for b in range(batch):
        cond = ("EXACT", "NEWPATH", "CHANGE", "CONT")[b % 4]
        i = int(store.order[rng.integers(n)])
        spec = {"cond": cond, "A": i}
        if cond == "CHANGE":
            spec["B"] = int(store.order[rng.integers(n)])
        items.append(build_item(store, spec, {"kind": kind}, sigma_vec=sigma_vec,
                                item_uid=f"trainsplice/{b}/{rng.integers(2**31)}"))
    return stack_items(items)
