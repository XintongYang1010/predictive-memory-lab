"""Independent BR-0.1 residual-bank reader, data contract, and audit helpers."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("EXACT", "NEWPATH", "CHANGE")
D, DK, FEATURE_DIM = 34, 648, 69
RECENT = 4
CAPACITIES = (4, 16, 64)
ANCHORS = (400, 464, 528, 592)
HORIZON = 6
SEED = 17
WINDOWS = {"POST": (400, 600), "DET": (400, 420)}
CONTROLLER_RESERVATION = 16384


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    actual_hash = sha256(path)
    if manifest.get("frozen_npz_sha256") != actual_hash:
        raise ValueError("frozen NPZ does not match the manifest hash")
    with np.load(path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    for cond in CONDITIONS:
        r, o, w, key, gate = (arrays[f"{cond}__{name}"] for name in ("R", "O", "W", "key", "gate"))
        if r.shape != (32, 799, D) or o.shape != r.shape or w.shape != r.shape[:2] or key.shape != w.shape or gate.shape != (*w.shape, 2):
            raise ValueError("BR-0.1 requires the original 32 x 799 frozen streams")
        for name, arr in (("R", r), ("O", o), ("W", w), ("gate", gate)):
            if arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise ValueError(f"invalid {cond} {name}")
        if key.dtype != np.int64 or np.any((key < 0) | (key >= DK)):
            raise ValueError("invalid oracle key")
        if not np.isin(o, [0, 1]).all() or not np.isin(w, [0, 1]).all():
            raise ValueError("masks must be binary")
        if not all("/dev/" in uid for uid in manifest["uids"][cond]):
            raise ValueError("only the frozen dev streams are allowed")
    inventory = {"input_sha256": actual_hash, "input_bytes": Path(path).stat().st_size,
                 "available_fields": sorted(arrays),
                 "missing_complete_l0_fields": [x for x in ("Z", "Y", "action", "pose", "RGB")
                                                if not all(f"{c}__{x}" in arrays for c in CONDITIONS)],
                 "claim": "L0-derived frozen-residual diagnostic; not complete L0 or visual evaluation",
                 "streams_available": 96, "steps_per_stream": 799,
                 "manifest_only_Z_EPS_hashes_are_not_arrays": True}
    return arrays, manifest, inventory


def source_split(manifest):
    ids = manifest["ordered_a_ids"]
    pairs = manifest["original_b_pairs"]
    if len(ids) != 32 or len(pairs) != 32 or any(p[0] != a for p, a in zip(pairs, ids)):
        raise ValueError("unexpected original source order")
    train, dev, excluded = list(range(23)), list(range(24, 32)), [23]
    train_sources = set(sum((pairs[i] for i in train), []))
    dev_sources = set(sum((pairs[i] for i in dev), []))
    if train_sources & dev_sources:
        raise ValueError("train/dev A or B sources overlap")
    return {"train_indices": train, "dev_indices": dev, "excluded_indices": excluded,
            "train_sources": sorted(train_sources), "dev_sources": sorted(dev_sources),
            "train_streams": len(train) * 3, "dev_streams": len(dev) * 3,
            "source_disjoint": True, "boundary_purge_reason": "CHANGE adjacent A/B source sharing"}


def representation(residual, observed, key):
    return np.concatenate([residual * observed, observed, np.array([key / 647.], np.float32)]).astype(np.float32)


def item_key(entry):
    return int(round(float(entry.representation[-1]) * 647))


def read(entries, query_key, query_time=None):
    """Predict before observing a target; returns actual per-coordinate weights."""
    entries = tuple(entries)
    if query_time is not None and any(e.timestamp >= query_time for e in entries):
        raise ValueError("reader received a current/future item")
    weights = np.zeros((len(entries), D), np.float64)
    if not entries:
        return np.zeros(D, np.float64), weights
    observed = np.stack([e.representation[D:2 * D] for e in entries]).astype(np.float64)
    observed *= np.array([item_key(e) == int(query_key) for e in entries])[:, None]
    denom = observed.sum(axis=0)
    np.divide(observed, denom, out=weights, where=denom > 0)
    values = np.stack([e.representation[:D] for e in entries]).astype(np.float64)
    return (values * weights).sum(axis=0), weights


def window_sums(predictions, residuals, observed, writes):
    e = (predictions.astype(np.float64) - residuals.astype(np.float64)) ** 2
    masks = observed.astype(np.float64) * writes[..., None]
    return {name: {"num": float((e[lo:hi] * masks[lo:hi]).sum()),
                   "den": float(masks[lo:hi].sum())}
            for name, (lo, hi) in WINDOWS.items()}


def matrix_predictions(arrays, condition, arm, sbar=.85):
    """Reference-only recurrence; changes none of the original experiment code."""
    r, o, w, key, gates = (arrays[f"{condition}__{x}"] for x in ("R", "O", "W", "key", "gate"))
    state = np.zeros((len(r), DK, D), np.float32)
    n = np.full((len(r), DK), sbar / (1 - sbar), np.float32)
    pred = np.zeros_like(r)
    rows = np.arange(len(r))
    for t in range(r.shape[1]):
        k = key[:, t]
        old = state[rows, k].copy()
        pred[:, t] = old
        if arm == "A1":
            state[rows, k] += (.1 * w[:, t])[:, None] * o[:, t] * (r[:, t] - old)
        elif arm == "A6":
            beta = 1 / (1 + n[rows, k])
            state[rows, k] += (beta * w[:, t])[:, None] * o[:, t] * (r[:, t] - old)
            n[rows, k] += w[:, t]
        elif arm == "A2":
            a, b = gates[:, t, 0], gates[:, t, 1]
            shrink = 1 - w[:, t] + w[:, t] * a
            state *= shrink[:, None, None]
            state[rows, k] += (w[:, t] * b)[:, None] * o[:, t] * (r[:, t] - a[:, None] * old)
        else:
            raise ValueError(arm)
    return pred


def audit_references(arrays, manifest, reference_path):
    if sha256(reference_path) != manifest["source_hashes"]["reference_rows"]:
        raise ValueError("reference rows hash mismatch")
    refs = {}
    with gzip.open(reference_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            refs[(row["cond"], row["A"])] = row
    predictions, checks = {}, 0
    for condition in CONDITIONS:
        for arm in ("A1", "A2", "A6"):
            pred = matrix_predictions(arrays, condition, arm)
            predictions[(condition, arm)] = pred
            for i, ident in enumerate(manifest["ordered_a_ids"]):
                sums = window_sums(pred[i], arrays[f"{condition}__R"][i], arrays[f"{condition}__O"][i], arrays[f"{condition}__W"][i])
                archived = refs[(condition, ident)]
                name = "A1@0.1" if arm == "A1" else arm
                for window, values in sums.items():
                    for part, actual in values.items():
                        expected = archived[f"{name}|{window}|{part}"]
                        ok = actual == expected if part == "den" else np.isclose(actual, expected, rtol=1e-5, atol=1e-3)
                        if not ok:
                            raise ValueError(f"reference mismatch {condition}/{i}/{name}/{window}/{part}: {actual} vs {expected}")
                        checks += 1
    return predictions, {"passed": True, "checks": checks, "arms": ["A1", "A2", "A6"],
                         "reference_rows_sha256": sha256(reference_path),
                         "rtol": 1e-5, "atol": 1e-3, "denominators_exact": True}


def paired_summary(rows, comparisons, seed=SEED, draws=1000):
    """Resample original A clusters; repeated conditions stay together."""
    out = {}
    for condition in (*CONDITIONS, "RETENTION"):
        wanted = ("EXACT", "NEWPATH") if condition == "RETENTION" else (condition,)
        sub = [x for x in rows if x["condition"] in wanted]
        for window in WINDOWS:
            label = f"{condition}/{window}"
            methods = sorted({x["method"] for x in sub})
            scores = {}
            ids = sorted({x["source_index"] for x in sub})
            for method in methods:
                num, den = [], []
                for i in ids:
                    m = [x[window] for x in sub if x["source_index"] == i and x["method"] == method]
                    num.append(sum(x["num"] for x in m)); den.append(sum(x["den"] for x in m))
                scores[method] = (np.asarray(num), np.asarray(den))
            rng = np.random.default_rng(seed)
            sample = rng.integers(0, len(ids), (draws, len(ids)))
            metrics = {m: float(n.sum() / d.sum()) for m, (n, d) in scores.items()}
            differences = {}
            for baseline, other in comparisons:
                if baseline not in scores or other not in scores:
                    continue
                bn, bd = scores[baseline]; on, od = scores[other]
                delta = bn[sample].sum(1) / bd[sample].sum(1) - on[sample].sum(1) / od[sample].sum(1)
                differences[f"{baseline}_minus_{other}"] = {"mean": metrics[baseline] - metrics[other],
                                                             "ci95": np.quantile(delta, [.025, .975]).tolist()}
            out[label] = {"mpe": metrics, "paired_differences": differences, "source_clusters": len(ids)}
    return out
