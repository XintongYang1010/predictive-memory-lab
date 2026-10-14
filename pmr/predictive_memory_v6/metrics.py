"""BJP (budget-matched joint persistence) protocol metrics -- numpy only.

All errors are in world units; on the torus they are shortest differences.
Predictions are PRE-WRITE: ``pred[:, t]`` is the belief about step ``t``
before the observation at ``t`` is used.  NaN predictions are never scored.

Definitions (docs/V6_PREDICTIVE_MEMORY.md, section "Metrics"):

* evolution ratio  ER = sum <x_hat - x_last, x - x_last> / sum |x - x_last|^2
  over hidden steps with non-zero true displacement.  0 = pause, 1 = correct
  evolution.  On the torus only displacements with |d|_inf <= period/4 are
  used (larger ones are ambiguous modulo the period).
* normalized hidden error  NHE = RMSE(x_hat, x) / RMSE(x_last, x)  (copy-last
  reference, as in arXiv 2610.07355).
* forgetting      history-pair separation error (V4 definition).
* correction      error at reveal (pre-write), +1/+3/+10 steps, never-recovered
  rate and success-only latency (V4 definitions, threshold in world units).
* non-interference change-pair displacement of the other objects.
* extrapolation   hidden RMSE / NHE binned by steps since last seen.
* MemoryGain      blackout error with memory reset at blackout start minus
  error of the intact memory (R2M-Bench control-pair logic).
* gap closure     (ADE_pause - ADE_model) / (ADE_pause - ADE_interp).
* MRA             VSI-Bench mean relative accuracy, 10 hard-coded thresholds.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Geometry

EXTRAP_BINS = ((1, 8), (9, 32), (33, 64), (65, 192), (193, 576), (577, 10 ** 9))
MRA_THRESHOLDS = np.arange(10) * 0.05 + 0.5  # [0.50, 0.55, ..., 0.95]


# ------------------------------------------------------------------ masks
def tracking_masks(data: dict) -> dict:
    """Per-step bookkeeping derived from visibility (never from predictions)."""
    vis = data["vis"].astype(bool)
    exists = data.get("exists", np.ones_like(vis)).astype(bool)
    B, T, N = vis.shape
    seen_before = np.zeros_like(vis)
    since = np.zeros((B, T, N), dtype=np.int64)
    last_t = np.full((B, T, N), -1, dtype=np.int64)
    run = np.zeros((B, T, N), dtype=np.int64)          # visible run length ending at last sighting
    changed = np.zeros_like(vis)
    cs = data.get("change_step")
    cur_last = np.full((B, N), -1)
    cur_run = np.zeros((B, N), dtype=np.int64)
    run_at_last = np.zeros((B, N), dtype=np.int64)
    seen = np.zeros((B, N), dtype=bool)
    pend = np.zeros((B, N), dtype=bool)
    for t in range(T):
        seen_before[:, t] = seen
        last_t[:, t] = cur_last
        since[:, t] = np.where(seen, t - cur_last, 0)
        run[:, t] = run_at_last
        if cs is not None:
            pend |= cs[:, t] != 0
        changed[:, t] = pend & seen
        v = vis[:, t] & exists[:, t]
        cur_run = np.where(v, cur_run + 1, 0)
        run_at_last = np.where(v, cur_run, run_at_last)
        cur_last = np.where(v, t, cur_last)
        seen |= v
        pend &= ~v
    hidden = exists & ~vis & seen_before
    prev_vis = np.zeros_like(vis)
    prev_vis[:, 1:] = vis[:, :-1]
    reveal = vis & exists & seen_before & ~prev_vis
    return {"vis": vis, "exists": exists, "seen_before": seen_before, "hidden": hidden,
            "since": since, "last_t": last_t, "identifiable": run >= 2,
            "changed": changed, "reveal": reveal}


def _gather_last(arr, last_t):
    """arr [B,T,N,2] -> value at each (b,t,i)'s last sighting."""
    B, T, N = last_t.shape
    idx = np.clip(last_t, 0, T - 1)
    b = np.arange(B)[:, None, None]
    n = np.arange(N)[None, None, :]
    return arr[b, idx, n]


# --------------------------------------------------------------- scoring
def score_episodes(data: dict, pred: np.ndarray, geom: Geometry, *, training_seed=None,
                   recovery_threshold=None, er_cutoff=None) -> list[dict]:
    """Per-episode sufficient statistics for the hidden/correction metrics."""
    m = tracking_masks(data)
    pos = data["pos"]
    unwrapped = data.get("pos_unwrapped", pos)
    B, T, N = m["vis"].shape
    thr = recovery_threshold if recovery_threshold is not None else 0.01 * geom.unit
    cutoff = er_cutoff if er_cutoff is not None else (geom.period / 4 if geom.kind == "torus" else math.inf)
    ok = ~np.isnan(pred[..., 0])
    err = geom.diff(np.nan_to_num(pred), pos)                    # [B,T,N,2]
    err2 = (err ** 2).sum(-1)
    x_last = _gather_last(pos, m["last_t"])
    copy_err2 = (geom.diff(x_last, pos) ** 2).sum(-1)
    if geom.kind == "torus":
        d_true = unwrapped - _gather_last(unwrapped, m["last_t"])
    else:
        d_true = pos - x_last
    d_pred = d_true + err
    moved = (np.abs(d_true).max(-1) > 1e-9) & (np.abs(d_true).max(-1) <= cutoff)
    base = m["hidden"] & ok
    er_mask = base & moved & ~m["changed"] & m["identifiable"]
    er_mask_all = base & moved
    nhe_mask = base & ~m["changed"] & (copy_err2 > 1e-18)
    rows = []
    ids = data.get("episode_id", np.array([str(b) for b in range(B)]))
    for b in range(B):
        r = {"episode_id": str(ids[b]), "training_seed": training_seed}
        mk = er_mask[b]
        r["er_num"] = float((d_pred[b][mk] * d_true[b][mk]).sum())
        r["er_den"] = float((d_true[b][mk] ** 2).sum())
        mk = er_mask_all[b]
        r["er_all_num"] = float((d_pred[b][mk] * d_true[b][mk]).sum())
        r["er_all_den"] = float((d_true[b][mk] ** 2).sum())
        mk = nhe_mask[b]
        r["nhe_num"], r["nhe_den"] = float(err2[b][mk].sum()), float(copy_err2[b][mk].sum())
        r["hidden_count"] = int(base[b].sum())
        r["hidden_sse"] = float(err2[b][base[b]].sum())
        r["hidden_rmse"] = math.sqrt(r["hidden_sse"] / r["hidden_count"]) if r["hidden_count"] else None
        for lo, hi in EXTRAP_BINS:
            mk = base[b] & (m["since"][b] >= lo) & (m["since"][b] <= hi)
            mk_n = mk & ~m["changed"][b] & (copy_err2[b] > 1e-18)
            r[f"bin{lo}_sse"], r[f"bin{lo}_n"] = float(err2[b][mk].sum()), int(mk.sum())
            r[f"bin{lo}_nhe_num"], r[f"bin{lo}_nhe_den"] = float(err2[b][mk_n].sum()), float(copy_err2[b][mk_n].sum())
        vis_ok = m["vis"][b] & m["seen_before"][b] & ok[b]
        r["visible_count"] = int(vis_ok.sum())
        r["visible_sse"] = float(err2[b][vis_ok].sum())
        # reveal / correction events (hidden run of >= 2 steps)
        rev = []
        for t, i in zip(*np.nonzero(m["reveal"][b] & ok[b])):
            if m["since"][b, t, i] < 2:
                continue
            e = np.sqrt(err2[b, t:min(T, t + 11), i])
            ev = {"t": int(t), "obj": int(i), "gap": int(m["since"][b, t, i]),
                  "changed": bool(m["changed"][b, t, i]), "reveal_err": float(e[0]),
                  "post1": float(e[1]) if len(e) > 1 else None,
                  "post3": float(e[3]) if len(e) > 3 else None,
                  "post10": float(e[10]) if len(e) > 10 else None}
            rec = None
            if len(e) == 11:
                good = e <= thr
                hits = np.nonzero(good[:-1] & good[1:])[0]
                rec = int(hits[0]) if len(hits) else None
                ev["recovered"] = rec is not None
                ev["latency"] = rec
            rev.append(ev)
        r["reveals"] = rev
        rows.append(r)
    return rows


def _ratio(rows, num, den, sqrt=False):
    n = sum(r[num] for r in rows)
    d = sum(r[den] for r in rows)
    if d <= 0:
        return None
    v = n / d
    return math.sqrt(v) if sqrt else v


def summarize(rows: list[dict]) -> dict:
    """Pooled (ratio-of-sums) summaries; event-level means for corrections."""
    out = {"episodes": len(rows)}
    out["er"] = _ratio(rows, "er_num", "er_den")
    out["er_all"] = _ratio(rows, "er_all_num", "er_all_den")
    out["nhe"] = _ratio(rows, "nhe_num", "nhe_den", sqrt=True)
    hc = sum(r["hidden_count"] for r in rows)
    out["hidden_rmse"] = math.sqrt(sum(r["hidden_sse"] for r in rows) / hc) if hc else None
    vc = sum(r["visible_count"] for r in rows)
    out["visible_rmse"] = math.sqrt(sum(r["visible_sse"] for r in rows) / vc) if vc else None
    for lo, _ in EXTRAP_BINS:
        n = sum(r[f"bin{lo}_n"] for r in rows)
        out[f"bin{lo}_rmse"] = math.sqrt(sum(r[f"bin{lo}_sse"] for r in rows) / n) if n else None
        out[f"bin{lo}_nhe"] = _ratio(rows, f"bin{lo}_nhe_num", f"bin{lo}_nhe_den", sqrt=True)
        out[f"bin{lo}_n"] = n
    ev = [e for r in rows for e in r["reveals"]]
    for group, sel in (("pred", [e for e in ev if not e["changed"]]),
                       ("unexp", [e for e in ev if e["changed"]])):
        out[f"reveal_{group}_n"] = len(sel)
        for k in ("reveal_err", "post1", "post3", "post10"):
            vals = [e[k] for e in sel if e.get(k) is not None]
            out[f"{k}_{group}"] = float(np.mean(vals)) if vals else None
        judged = [e for e in sel if "recovered" in e]
        out[f"never_recovered_{group}"] = (float(np.mean([not e["recovered"] for e in judged]))
                                           if judged else None)
        lat = [e["latency"] for e in judged if e["recovered"]]
        out[f"latency_success_only_{group}"] = float(np.mean(lat)) if lat else None
    return out


# ------------------------------------------------------------- bootstrap
def bootstrap(rows, stat, draws=1000, seed=0, rows_b=None):
    """Percentile CI over episodes.  With ``rows_b`` the statistic is the paired
    difference stat(rows) - stat(rows_b); episodes must align by episode_id."""
    rows = list(rows)
    if rows_b is not None:
        ib = {r["episode_id"]: r for r in rows_b}
        if set(ib) != {r["episode_id"] for r in rows}:
            raise ValueError("paired bootstrap needs identical episode ids")
        rows_b = [ib[r["episode_id"]] for r in rows]
    rng = np.random.default_rng(seed)
    n = len(rows)
    point = stat(rows) if rows_b is None else _diff(stat, rows, rows_b)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        a = [rows[j] for j in idx]
        v = stat(a) if rows_b is None else _diff(stat, a, [rows_b[j] for j in idx])
        if v is not None:
            vals.append(v)
    ci = np.quantile(vals, [0.025, 0.975]).tolist() if vals else [None, None]
    return {"point": point, "ci95": ci, "draws": draws, "valid_draws": len(vals)}


def _diff(stat, a, b):
    x, y = stat(a), stat(b)
    return None if x is None or y is None else x - y


def stat_er(rows):
    return _ratio(rows, "er_num", "er_den")


def stat_nhe(rows):
    return _ratio(rows, "nhe_num", "nhe_den", sqrt=True)


def stat_bin_nhe(lo):
    return lambda rows: _ratio(rows, f"bin{lo}_nhe_num", f"bin{lo}_nhe_den", sqrt=True)


# ------------------------------------------------------------------ pairs
def history_pair_scores(data_a, data_b, pred_a, pred_b, geom: Geometry, last=10, training_seed=None):
    """Forgetting: |(pred_A - pred_B) - (true_A - true_B)| for the probe object
    over the final ``last`` steps (it is hidden there in both worlds)."""
    i = int(data_a.get("probe_object", 0))
    T = data_a["pos"].shape[1]
    sl = slice(T - last, T)
    true_sep = geom.diff(data_a["pos"][:, sl, i], data_b["pos"][:, sl, i])
    pred_sep = geom.diff(pred_a[:, sl, i], pred_b[:, sl, i])
    err = np.sqrt((geom.diff(pred_sep, true_sep) ** 2).sum(-1)).mean(-1)
    rows = []
    for b in range(len(err)):
        rows.append({"episode_id": str(data_a["episode_id"][b]), "training_seed": training_seed,
                     "history_pair_error": float(err[b]),
                     "pred_separation": float(np.sqrt((pred_sep[b] ** 2).sum(-1)).mean()),
                     "true_separation": float(np.sqrt((true_sep[b] ** 2).sum(-1)).mean())})
    return rows


def interference_scores(data_a, data_b, pred_a, pred_b, geom: Geometry, window=10,
                        key_cos=None, predicted_delta=None, training_seed=None):
    """Non-interference after the probe object's correction.

    For every other object: mean displacement between the two worlds' beliefs
    over ``[reveal, reveal+window]`` steps, together with the size of the probe
    correction (reveal error difference) and, when supplied, the key cosine and
    the lemma-4 predicted displacement at reveal+1.
    """
    i = int(data_a.get("probe_object", 0))
    valid, rev = data_a["pair_valid"], data_a["probe_reveal"]
    B, T, N, _ = data_a["pos"].shape
    rows = []
    for b in range(B):
        if not valid[b]:
            continue
        r0, r1 = int(rev[b]), min(T, int(rev[b]) + window + 1)
        corr = float(np.sqrt((geom.diff(data_a["pos"][b, r0, i], pred_a[b, r0, i]) ** 2).sum()))
        for j in range(N):
            if j == i:
                continue
            d = np.sqrt((geom.diff(pred_a[b, r0 + 1:r1, j], pred_b[b, r0 + 1:r1, j]) ** 2).sum(-1))
            row = {"episode_id": f"{data_a['episode_id'][b]}:{j}", "training_seed": training_seed,
                   "displacement": float(d.mean()) if len(d) else 0.0,
                   "displacement_next": float(d[0]) if len(d) else 0.0,
                   "probe_correction": corr}
            if key_cos is not None:
                row["key_cos"] = float(key_cos[b, i, j])
            if predicted_delta is not None:
                row["predicted_next"] = float(predicted_delta[b, j])
            rows.append(row)
    return rows


def interference_summary(rows):
    if not rows:
        return {}
    disp = np.array([r["displacement"] for r in rows])
    corr = np.array([r["probe_correction"] for r in rows])
    out = {"pairs": len(rows), "mean_displacement": float(disp.mean()),
           "relative_to_correction": float(disp.mean() / max(corr.mean(), 1e-12))}
    if "predicted_next" in rows[0]:
        x = np.array([r["predicted_next"] for r in rows])
        y = np.array([r["displacement_next"] for r in rows])
        out["lemma4_slope"] = float((x * y).sum() / max((x * x).sum(), 1e-24))
    return out


def memory_gain(data, pred, pred_reset, geom: Geometry, start, length):
    """Blackout MemoryGain per episode (positive = intact memory helps)."""
    sl = slice(start, start + length)
    e = np.sqrt((geom.diff(pred[:, sl], data["pos"][:, sl]) ** 2).sum(-1)).mean((1, 2))
    e0 = np.sqrt((geom.diff(pred_reset[:, sl], data["pos"][:, sl]) ** 2).sum(-1)).mean((1, 2))
    return e0 - e


def gap_closure(ade_pause, ade_model, ade_interp):
    den = ade_pause - ade_interp
    return None if abs(den) < 1e-12 else (ade_pause - ade_model) / den


# ------------------------------------------------------ counting/surprise
def mra(pred, gt):
    """VSI-Bench mean relative accuracy (thinking-in-space utils.py L87-94).

    Ten hard-coded thresholds 0.50..0.95, non-strict ``<=``.  The original has
    no guard for gt == 0; here gt == 0 scores 1.0 iff pred == 0.
    """
    pred, gt = float(pred), float(gt)
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    rel = abs(pred - gt) / gt
    return float((rel <= 1 - MRA_THRESHOLDS).mean())


def auroc(scores, labels):
    """Rank-based AUROC with average ranks for ties (labels: 1 = positive)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def f1_at(scores, labels, thr):
    s = np.asarray(scores) > thr
    y = np.asarray(labels).astype(bool)
    tp, fp, fn = int((s & y).sum()), int((s & ~y).sum()), int((~s & y).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def best_f1_threshold(scores, labels):
    s = np.unique(np.asarray(scores, dtype=np.float64))
    if len(s) == 0:
        return 0.0, 0.0
    cands = np.concatenate([[s[0] - 1e-9], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-9]])
    f = [f1_at(scores, labels, c) for c in cands]
    k = int(np.argmax(f))
    return float(cands[k]), float(f[k])
