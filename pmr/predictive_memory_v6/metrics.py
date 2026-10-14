"""Metrics and statistics for V6.1 (numpy only).

Per-item window sums are produced by ``evaluate`` as rows
``{"uid", "cond", "<arm>|<window>|num", "<arm>|<window>|den", ...}``; every
statistic below is a ratio of sums over items, and every interval is a paired
episode bootstrap (same resampled items for all arms).
"""
from __future__ import annotations

import numpy as np

MRA_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


# ----------------------------------------------------------- window sums
def window_mask(T, lo, hi):
    m = np.zeros(T, dtype=bool)
    m[max(0, lo):max(0, min(T, hi))] = True
    return m


def mpe_sums(r, m, o, w, mask):
    """r, m, o [T,D]; w, mask [T] -> (sum |o (r - m)|^2, sum D_t) over
    written transitions inside mask (pre-write prediction m)."""
    sel = (w > 0) & mask
    e = (r - m) * o
    return float((e[sel] ** 2).sum()), float(o[sel].sum())


def ratio(rows, key):
    num = sum(r.get(f"{key}|num", 0.0) for r in rows)
    den = sum(r.get(f"{key}|den", 0.0) for r in rows)
    return None if den <= 0 else num / den


def nu_c(rows, arm, win="POST", ref="A6", oracle="OC"):
    a, b, c = ratio(rows, f"{ref}|{win}"), ratio(rows, f"{arm}|{win}"), ratio(rows, f"{oracle}|{win}")
    if None in (a, b, c) or abs(a - c) < 1e-12:
        return None
    return (a - b) / (a - c)


def nu_r(rows, arm, win="POST", ref="A6", oracle="OC"):
    a, b, c = ratio(rows, f"{ref}|{win}"), ratio(rows, f"{arm}|{win}"), ratio(rows, f"{oracle}|{win}")
    if None in (a, b, c) or abs(c - a) < 1e-12:
        return None
    return (c - b) / (c - a)


def frontier(points):
    """Piecewise-linear nu_R as a function of nu_C through the A1 points
    (sorted by nu_C); outside the range the nearest endpoint is used."""
    pts = sorted((x, y) for x, y in points if x is not None and y is not None)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])

    def F(x):
        if len(xs) == 0:
            return None
        return float(np.interp(x, xs, ys))
    return F


def dominated(point, others):
    x, y = point
    for (a, b) in others:
        if a is None or b is None:
            continue
        if a >= x and b >= y and (a > x or b > y):
            return True
    return False


# ------------------------------------------------------------- bootstrap
def bootstrap(rows, stat, draws=1000, seed=0):
    """Percentile CI of stat(rows) resampling items (rows are paired across
    arms because one row holds all arms of an item)."""
    rows = list(rows)
    point = stat(rows)
    rng = np.random.default_rng(seed)
    n = len(rows)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        v = stat([rows[j] for j in idx])
        if v is not None:
            vals.append(v)
    ci = np.quantile(vals, [0.025, 0.975]).tolist() if vals else [None, None]
    return {"point": point, "ci95": ci, "draws": draws, "valid_draws": len(vals)}


def passes(boot, threshold, direction=">="):
    """Pre-registered rule: point meets threshold and the CI excludes 0 on the
    correct side (for a lower-bound claim the CI lower end must be > 0 when the
    threshold is >= 0; for a margin claim like ``>= -0.1`` the CI lower end
    must be > threshold)."""
    p, (lo, hi) = boot["point"], boot["ci95"]
    if p is None or lo is None:
        return None
    if direction == ">=":
        return bool(p >= threshold and lo > min(0.0, threshold))
    return bool(p <= threshold and hi < max(0.0, threshold))


# ---------------------------------------------------------------- AUROC
def auroc(scores, labels):
    """Rank-based AUROC with average ranks for ties (labels: 1 = positive)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def threshold_at_rate(scores_neg, rate):
    """Smallest threshold whose exceedance rate on negatives is <= rate."""
    s = np.sort(np.asarray(scores_neg, dtype=np.float64))
    if len(s) == 0:
        return np.inf
    k = int(np.floor((1 - rate) * len(s)))
    return float(s[min(k, len(s) - 1)]) if rate > 0 else float(s[-1] + 1e-9)


# ------------------------------------------------------------------ MRA
def mra(pred, gt):
    """VSI-Bench mean relative accuracy; gt == 0 scores 1 iff pred == 0."""
    pred, gt = float(pred), float(gt)
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    rel = abs(pred - gt) / gt
    return float((rel <= 1 - MRA_THRESHOLDS + 1e-12).mean())


def mean_mra(preds, gts):
    return 100.0 * float(np.mean([mra(p, g) for p, g in zip(preds, gts)]))


# --------------------------------------------------------------- probes
def probe_wall_accuracy(logits, layout, seen=None):
    pred = logits > 0
    ok = pred == (layout > 0.5)
    out = {"acc_all": float(ok.mean())}
    if seen is not None and seen.any():
        out["acc_seen"] = float(ok[seen].mean())
    return out


def visage_bins(visage, edges=(-1, 0, 5, 10, 50, 200, 1000)):
    b = np.digitize(visage, edges[1:], right=True)
    return b
