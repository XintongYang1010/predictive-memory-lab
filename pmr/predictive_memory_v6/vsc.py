"""VSC-Repeat: memory-referenced familiarity at count aggregation (CPU only).

Input: per-video logs written by the patched Cambrian-S VSC loop
(scripts/vsc/patch_cambrians.py): per frame the memoryless surprise s[t],
the pooled SigLIP2 feature d_pool[t] and the LFP prediction p_pool[t]
(1152-d), and per segment its frame range and the decoded count.  The
official boundaries, LFP head and 0.39 threshold are never changed; every
memory variant is evaluated offline from one 7B pass.

Aggregation rule shared by every veto arm: a closing segment judged familiar
is merged into the most similar existing slot (count = max(old, new)); only
an unfamiliar segment appends a new slot; the answer is the sum of slots.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import bf
from . import metrics as Me
from . import prereg

D_WHITE, D_KEY, WINDOW = 256, 128, 128
ARMS = ("M0", "A8", "A4", "ABS", "M1", "A9", "A9x10", "M3")


def load_dump(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["segments"] = [tuple(map(int, s)) for s in d["seg_frames"]]
    d["counts"] = [float(c) for c in d["seg_counts"]]
    d["name"] = Path(path).stem
    return d


# ------------------------------------------------------------ calibration
def fit_whitening(dumps, dim=D_WHITE):
    X = np.concatenate([d["d_pool"].astype(np.float64) for d in dumps])
    mu = X.mean(0)
    C = np.cov((X - mu).T)
    ev, U = np.linalg.eigh(C)
    ev, U = ev[::-1][:dim], U[:, ::-1][:, :dim]
    W = (U / np.sqrt(np.maximum(ev, 1e-12))).T
    return mu, W


def features(d, cal):
    x = (d["d_pool"].astype(np.float64) - cal["mu"]) @ cal["W"].T
    p = d["p_pool"].astype(np.float64)
    xh = np.where(np.isfinite(p).all(1, keepdims=True), (np.nan_to_num(p) - cal["mu"]) @ cal["W"].T, 0.0)
    r = (x - xh) / cal["sigma"]
    r[~np.isfinite(d["p_pool"]).all(1)] = 0.0
    xk = np.vstack([x[:1], x[:-1]])
    k = xk @ cal["Q"].T
    k = k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-9)
    return x, r, k


def bf_stream(r, k, sbar, pi, arm="A8"):
    """Delayed-write BF memory over one stream: frame t reads the state that
    holds frames <= t - WINDOW - 1, then writes frame t - WINDOW."""
    T, D = r.shape
    S = np.zeros((1, D, k.shape[1]))
    n = np.full((1, k.shape[1]), sbar / (1 - sbar))
    ell = np.zeros(T)
    dstat = np.zeros(T)
    e2 = np.zeros(T)
    lp = bf.logit(pi)
    one = np.ones((1, D))
    for t in range(T):
        m = (S[0] @ k[t])
        e = r[t] - m
        c = max(float(n[0] @ (k[t] ** 2)), 1e-12)
        v = sbar * (1 + 1 / c)
        ell[t] = np.clip(e @ e / (2 * v) - r[t] @ r[t] / 2 + 0.5 * D * math.log(v), -50, 50)
        dstat[t] = (e @ e - r[t] @ r[t]) / 2
        e2[t] = e @ e / D
        tw = t - WINDOW
        if tw >= 0:
            S, n, _ = bf.bf_step(S, n, k[tw][None], r[tw][None], one, np.ones(1), sbar=sbar,
                                 logit_pi=lp, arm=arm)
    return {"ell": ell, "d": dstat, "e2": e2, "S": S[0]}


def ml2_stream(streams, grid):
    best, lls = None, {}
    for sb in grid:
        tot = 0.0
        for r, k in streams:
            S = np.zeros((1, r.shape[1], k.shape[1]))
            n = np.full((1, k.shape[1]), sb / (1 - sb))
            for t in range(len(r)):
                tw = t - WINDOW
                if tw < 0:
                    continue
                m = S[0] @ k[tw]
                c = max(float(n[0] @ (k[tw] ** 2)), 1e-12)
                v = sb * (1 + 1 / c)
                e = r[tw] - m
                tot += -0.5 * (e @ e / v + r.shape[1] * math.log(2 * math.pi * v))
                S, n, _ = bf.bf_step(S, n, k[tw][None], r[tw][None], np.ones((1, r.shape[1])), np.ones(1),
                                     sbar=sb, logit_pi=bf.logit(1e-3), arm="A6")
        lls[float(sb)] = tot
    best = max(lls, key=lls.get)
    return best, lls


# --------------------------------------------------------- familiarity
def frame_scores(d, cal, use_ell):
    """Per-frame familiarity (smaller = more familiar) for every arm; content
    inside the 128-frame sensory window is never used to judge familiarity."""
    x, r, k = features(d, cal)
    T = len(x)
    out = {a: np.zeros(T) for a in ARMS if a not in ("M0", "M1")}
    st = bf_stream(r, k, cal["sbar"], cal["pi"])
    out["A8"] = st["ell"] if use_ell else st["d"]
    out["A4"] = st["e2"]
    out["ABS"] = -np.abs(st["d"])
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    for name, cap in (("A9", 128), ("A9x10", 1280)):
        bank, ages = [], []
        thr_admit = cal.get("knn_admit", 0.9)
        for t in range(T):
            out[name][t] = -(max(0.0, float(np.max(np.asarray(bank) @ xn[t]))) if bank else 0.0)
            tw = t - WINDOW
            if tw >= 0:
                sim = float(np.max(np.asarray(bank) @ xn[tw])) if bank else -1.0
                if sim < thr_admit:                     # novelty admission
                    bank.append(xn[tw])
                    if len(bank) > cap:
                        bank.pop(0)                     # evict oldest
    G = xn @ xn.T
    for t in range(T):
        tw = t - WINDOW
        out["M3"][t] = -(float(G[t, :tw + 1].max()) if tw >= 0 else 0.0)
    return out, x, st


def segment_fam(d, scores, x, slots_state=None):
    fam = {a: [] for a in ARMS}
    mus = []
    for (a, b) in d["segments"]:
        a, b = max(0, a), max(a + 1, b)
        mu = x[a:b].mean(0)
        mus.append(mu)
        for arm, sc in scores.items():
            fam[arm].append(float(np.mean(sc[a:b])))
        fam["M0"].append(np.inf)
    # M1: equal-byte slot memory of segment means, LRU (127 x (256 + 2))
    slots = []
    for j, mu in enumerate(mus):
        if slots:
            S = np.asarray([s for s in slots])
            cs = (S @ mu) / (np.linalg.norm(S, axis=1) * np.linalg.norm(mu) + 1e-9)
            fam["M1"].append(-float(cs.max()))
        else:
            fam["M1"].append(0.0)
        slots.append(mu)
        if len(slots) > 127:
            slots.pop(0)
    return fam, mus


def aggregate(mus, counts, fam, thr):
    slots = []                                  # [mu, count, n]
    vetoed = 0
    for mu, c, f in zip(mus, counts, fam):
        if slots and f < thr:
            i = max(range(len(slots)), key=lambda i: _cos(mu, slots[i][0]))
            s = slots[i]
            s[1] = max(s[1], c)
            s[0] = (s[0] * s[2] + mu) / (s[2] + 1)
            s[2] += 1
            vetoed += 1
        else:
            slots.append([mu, c, 1])
    return sum(s[1] for s in slots), vetoed


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def calibrate(calib_dumps, seed=0):
    rng = np.random.default_rng(seed)
    mu, W = fit_whitening(calib_dumps)
    cal = {"mu": mu, "W": W, "sigma": np.ones(D_WHITE), "Q": rng.standard_normal((D_KEY, D_WHITE)) / math.sqrt(D_WHITE),
           "pi": prereg.CALIB["pi_vsc"]}
    res = []
    for d in calib_dumps:
        x = (d["d_pool"].astype(np.float64) - mu) @ W.T
        p = d["p_pool"].astype(np.float64)
        ok = np.isfinite(p).all(1)
        xh = (np.nan_to_num(p) - mu) @ W.T
        res.append((x - xh)[ok])
    cal["sigma"] = np.concatenate(res).std(0) + 1e-6
    streams = []
    for d in calib_dumps:
        _, r, k = features(d, cal)
        streams.append((r, k))
    cal["sbar"], cal["sbar_ll"] = ml2_stream(streams, prereg.CALIB["sbar_grid"])
    # key overlap (F1) decides between the full log-ratio and the equal-variance statistic
    ov = []
    for r, k in streams:
        T = len(k)
        for _ in range(200):
            t, u = rng.integers(T, size=2)
            if abs(int(t) - int(u)) >= WINDOW:
                ov.append(abs(float(k[t] @ k[u])))
    cal["key_overlap"] = float(np.mean(ov)) if ov else float("nan")
    cal["use_ell"] = bool(cal["key_overlap"] < prereg.H6["key_overlap_max"])
    fams = {a: [] for a in ARMS}
    for d in calib_dumps:
        sc, x, _ = frame_scores(d, cal, cal["use_ell"])
        fam, _ = segment_fam(d, sc, x)
        for a in ARMS:
            fams[a].extend(fam[a][1:])
    rate = prereg.H6["false_familiar_calib"]
    cal["thr"] = {a: (float(np.quantile(fams[a], rate)) if a != "M0" and fams[a] else -np.inf) for a in ARMS}
    return cal


def save_cal(cal, path):
    np.savez(path, **{k: np.asarray(v) for k, v in cal.items() if not isinstance(v, dict)},
             thr=json.dumps(cal["thr"]), sbar_ll=json.dumps(cal["sbar_ll"]))


def load_cal(path):
    z = np.load(path, allow_pickle=False)
    cal = {k: z[k] for k in z.files}
    cal["thr"] = json.loads(str(cal["thr"]))
    for k in ("sbar", "pi", "key_overlap"):
        cal[k] = float(cal[k])
    cal["use_ell"] = bool(cal["use_ell"])
    return cal


def evaluate_variant(dumps, gts, cal, base_len=None):
    """Per-video rows: count per arm, MRA, vetoes, per-frame familiarity."""
    rows = []
    for d in dumps:
        gt = gts[d["name"]]
        sc, x, _ = frame_scores(d, cal, cal["use_ell"])
        fam, mus = segment_fam(d, sc, x)
        row = {"video": d["name"], "gt": gt, "n_segments": len(mus)}
        for a in ARMS:
            pred, vet = aggregate(mus, d["counts"], fam[a], cal["thr"][a])
            row[f"{a}|pred"] = pred
            row[f"{a}|mra"] = Me.mra(pred, gt)
            row[f"{a}|veto"] = vet / max(1, len(mus))
        if base_len:
            T = len(x)
            copy = np.minimum(np.arange(T) // max(1, base_len[d["name"]]), 99) + 1
            lab = copy >= 2
            if lab.any() and (~lab).any():
                for a in ("A8", "A9", "A9x10", "M3"):
                    row[f"{a}|fam_auroc"] = Me.auroc(-sc[a], lab)
        rows.append(row)
    return rows
