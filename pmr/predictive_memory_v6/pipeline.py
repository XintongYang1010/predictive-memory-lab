"""Shared torch plumbing: devices, pools, residuals, keys and arm execution."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import bf
from . import data as Dm
from . import models as Mo
from . import prereg


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_t(x, dev, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), device=dev, dtype=dtype)


class Timer:
    def __init__(self, max_seconds):
        self.t0 = time.time()
        self.max = max_seconds

    def over(self):
        return self.max is not None and time.time() - self.t0 > self.max

    def elapsed(self):
        return time.time() - self.t0


# ------------------------------------------------------------------ track
def track_kind(track):
    return {"T0d": "l0", "T0": "l0", "T1": "l0", "T2": "l1", "T3": "r", "T3pose": "rpose"}[track["name"]]


def oracle_dk(store):
    return store.S * store.S * 8


def batch_tensors(st, dev):
    """numpy stacked items -> torch tensors on dev."""
    out = {"Z": to_t(st["zin"], dev), "Y": to_t(st["y"], dev), "O": to_t(st["o"], dev),
           "W": to_t(st["w"], dev), "KEY": torch.as_tensor(st["key"], device=dev, dtype=torch.long)}
    if "eps" in st:
        out["EPS"] = to_t(st["eps"], dev)
    if "zkey" in st:
        out["ZK"] = to_t(st["zkey"], dev)
    return out


def residuals(bt, slow, sbar_inj):
    with torch.no_grad():
        R = slow.residual(bt["Z"], bt["Y"], bt["O"])
    if sbar_inj > 0:
        eps = bt["EPS"] if "EPS" in bt else torch.randn_like(R)
        R = (math.sqrt(1 - sbar_inj) * R + math.sqrt(sbar_inj) * eps) * bt["O"]
    return R


def keys(bt, keynet, dk_oracle=None):
    if keynet is None:
        return F.one_hot(bt["KEY"], dk_oracle).float()
    z = bt["ZK"] if "ZK" in bt else bt["Z"]
    return keynet(z)


# ------------------------------------------------------------------ pools
def build_pool(store, n, kind, rng, sigma_vec=None, length=1000):
    """Unspliced train sequences kept in memory (numpy)."""
    idx = store.order[: n] if n < len(store.order) else store.order
    items = [Dm.build_item(store, {"cond": "STATIC", "A": int(i)}, {"kind": kind}, sigma_vec=sigma_vec,
                           item_uid=f"pool/{store.ids[int(i)]}", want_revisits=False) for i in idx]
    st = Dm.stack_items(items)
    if length < st["y"].shape[1]:
        st = {k: (v[:, :length] if k != "meta" else v) for k, v in st.items()}
    return st


def pool_batch(pool, rng, b, dev):
    n = pool["y"].shape[0]
    idx = rng.integers(0, n, b)
    st = {k: v[idx] for k, v in pool.items() if k != "meta" and k in ("zin", "y", "o", "w", "key", "zkey")}
    return batch_tensors(st, dev)


# ------------------------------------------------------------------- arms
def arm_list(track):
    a1 = [("A1", b) for b in prereg.CALIB["a1_beta0"]]
    a1bf = [("A1BF", b) for b in prereg.CALIB["a1_beta0"]]
    a6l = [("A6L", l) for l in prereg.CALIB["a6_lambda"]]
    base = [("A0", None), ("A6", None), ("A7", None), ("A8", None), ("A8g", None), ("OC", None)]
    return base + a1 + a1bf + a6l + [("A2", None), ("A2P", None), ("A3", None), ("A4", None)]


def arm_name(arm, p):
    if arm in ("A1", "A1BF"):
        return f"{arm}@{p:g}"
    if arm == "A6L":
        return f"A6L@{p:g}"
    return arm


def run_arm(arm, p, K, R, O, W, sbar, pi, gates, Z=None, slow_cos=None, reset=None, state_at=None):
    """Returns dict with M (pre-write prediction) and diagnostics."""
    kw = dict(sbar=sbar, pi=pi, collect=("m", "ell", "g", "tau", "c", "beta", "alpha"), state_at=state_at)
    if arm == "A0":
        return {"m": torch.zeros_like(R), "ell": None, "g": None, "states": {}}
    if arm == "A1":
        return bf.scan("A1", K, R, O, W, beta0=p, **kw)
    if arm == "A1BF":
        return bf.scan("A1BF", K, R, O, W, beta0=p, **kw)
    if arm == "A6L":
        return bf.scan("A6L", K, R, O, W, lam=p, **kw)
    if arm == "OC":
        return bf.scan("OC", K, R, O, W, reset=reset, **kw)
    if arm in ("A6", "A7", "A8", "A8g"):
        return bf.scan(arm, K, R, O, W, **kw)
    if arm == "A2":
        g = gates["A2"](Z)
        return bf.scan("A2", K, R, O, W, gate=g, **kw)
    if arm == "A2P":
        return bf.scan("A2P", K, R, O, W, gate=gates["A2P"], **kw)
    if arm == "A3":
        sg = gates["A3"]
        beta = torch.sigmoid(sg.a * slow_cos + sg.b)
        return bf.scan("A3", K, R, O, W, gate=beta, **kw)
    if arm == "A4":
        sg = gates["A4"]
        return bf.scan("A4", K, R, O, W, gate=(sg.a, sg.b), **kw)
    raise ValueError(arm)


def slow_surprise(bt, slow):
    """Cambrian-S style memoryless surprise 1 - cos(o*yhat, o*y), raw units."""
    with torch.no_grad():
        yh = slow(bt["Z"]) * bt["O"]
    y = bt["Y"] * bt["O"]
    return 1.0 - F.cosine_similarity(yh, y, dim=-1, eps=1e-6)


# ------------------------------------------------------------- artifacts
def load_slow(path, dev):
    m, extra = Mo.load(path, dev)
    return m, extra


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_jsonable))


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return str(x)
