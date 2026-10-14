"""World registry, train pools and numpy -> torch conversion.

World strings select a generator plus overrides, e.g. ``l0_sched``,
``l0_ctrl:rho=0.5,n_objects=8``.  Synthetic train data are drawn from a pool
of fresh episodes that is regenerated every ``refresh`` steps (an effectively
unbounded stream); dev/test sets are fixed seeds in separate namespaces.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from . import worlds_l0
from .geometry import Geometry
from .worlds_l0 import L0Config

TRAIN_T = 64

L0_WORLDS = {
    # FloWM MNIST-World spec, natural hide/reveal from the moving window
    "l0_main": dict(visibility="window"),
    # static camera, per-object occlusion schedules (clean H1/H2/H4 world)
    "l0_sched": dict(visibility="schedule"),
    # schedule world with unexpected hidden changes (H3/H5)
    "l0_ctrl": dict(visibility="schedule", p_change=0.5),
    # window world with a small hidden-change hazard
    "l0_window_ctrl": dict(visibility="window", p_change=0.02),
    # lights-off world for MemoryGain
    "l0_blackout": dict(visibility="blackout", blackout_start=8, blackout_len=32),
    # per-episode velocity rotation/damping (H6)
    "l0_drift": dict(visibility="schedule", continuous=True, omega_max=0.15, gamma_max=0.03),
    # counting tours (H5): static objects, k identical copies
    "l0_tour_static": dict(visibility="tour", static_objects=True),
    "l0_tour_dynamic": dict(visibility="tour"),
}

INT_KEYS = {"n_objects", "world", "window", "vel_max", "ego_max", "appearance_dim", "copies",
            "blackout_start", "blackout_len", "tour_speed", "tour_row"}
FLOAT_KEYS = {"rho", "obs_noise", "p_change", "teleport_frac", "omega_max", "gamma_max", "hidden_scale"}
BOOL_KEYS = {"static_objects", "continuous", "allow_zero_velocity"}


def parse_world(spec: str):
    name, _, rest = spec.partition(":")
    over = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        if k in INT_KEYS:
            over[k] = int(v)
        elif k in FLOAT_KEYS:
            over[k] = float(v)
        elif k in BOOL_KEYS:
            over[k] = v.lower() in ("1", "true", "yes")
        elif k in ("blind_after", "frame"):
            over[k] = v
        else:
            raise ValueError(f"unknown world override {k!r}")
    return name, over


def stable_seed(*parts) -> int:
    h = hashlib.sha256("/".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "little")


def l0_config(spec: str, T: int = TRAIN_T, scale_hidden=True) -> tuple[L0Config, dict]:
    name, over = parse_world(spec)
    if name not in L0_WORLDS:
        raise ValueError(f"unknown L0 world {name!r}")
    extra = {k: over.pop(k) for k in ("blind_after", "frame") if k in over}
    kw = dict(L0_WORLDS[name])
    kw.update(over)
    cfg = L0Config(**kw)
    if scale_hidden and cfg.visibility == "schedule" and "hidden_scale" not in over:
        cfg = replace(cfg, hidden_scale=max(1.0, T / TRAIN_T))
    return cfg, extra


def make_l0(spec: str, n: int, T: int, seed: int, scale_hidden=True) -> dict:
    cfg, extra = l0_config(spec, T, scale_hidden)
    if cfg.visibility == "tour":
        data = worlds_l0.generate(cfg, n, T, seed)
    else:
        data = worlds_l0.generate(cfg, n, T, seed)
    if "blind_after" in extra:  # FloWM protocol: context then fully unobserved rollout
        k = int(extra["blind_after"])
        data["vis"][:, k:] = False
        data["obs"][:, k:] = 0.0
        data["reveal_changed"][:, k:] = False
    if extra.get("frame") == "ego":
        data = worlds_l0.to_ego_frame(data)
    data["meta"]["spec"] = spec
    return data


def make_world(spec: str, n: int, T: int, seed: int, cache=None) -> dict:
    """Dispatch on world family.  L1 and real worlds read prepared caches."""
    if spec.startswith("l0_"):
        return make_l0(spec, n, T, seed)
    if spec.startswith("l1"):
        from .worlds_l1 import load_l1
        return load_l1(spec, n, T, seed, cache)
    if spec.startswith("real_"):
        from .worlds_real import load_real
        return load_real(spec, n, T, seed, cache)
    raise ValueError(spec)


def geometry_of(data) -> Geometry:
    return Geometry.from_dict(data["meta"]["geometry"])


def n_objects_of(data) -> int:
    return data["vis"].shape[2]


def concat(datas):
    out = {}
    for k in datas[0]:
        if k == "meta":
            out[k] = datas[0][k]
        elif isinstance(datas[0][k], np.ndarray):
            out[k] = np.concatenate([d[k] for d in datas], 0)
        else:
            out[k] = datas[0][k]
    return out


# ----------------------------------------------------------------- torch
def to_tbatch(data: dict, device, dtype=None):
    import torch
    from .memories import TBatch
    dtype = dtype or torch.float32
    geom = geometry_of(data)
    frame = data["meta"].get("frame", "allo")
    vis = data["vis"].astype(bool)
    if frame == "ego":
        # ego observations are window-relative world units on the torus
        obs = np.where(vis[..., None], data["obs"] / geom.period, 0.0)
    else:
        obs = np.where(vis[..., None], geom.normalize(data["obs"]), 0.0)
    if geom.kind == "torus":
        obs = np.mod(obs, 1.0)
    u = geom.normalize_vector(data["u"]) if geom.kind != "box" else data["u"] / geom.unit
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)
    fov, window = None, 1.0
    cfg = data["meta"].get("config", {})
    if geom.kind == "torus" and cfg.get("visibility") in ("window", "tour"):
        fov, window = t(data["ego"] / geom.period), cfg["window"] / cfg["world"]
    drift = t(data["drift"]) if "drift" in data else None
    return TBatch(app=t(data["app"]), obs=t(obs), vis=t(vis.astype(np.float32)), u=t(u),
                  dt=t(data["dt"]), kind=geom.kind, frame=frame, fov=fov, window=window, drift=drift)


class TrainPool:
    """Pool of fresh training episodes, regenerated every ``refresh`` steps."""

    def __init__(self, spec, T=TRAIN_T, size=16384, refresh=5000, seed=0, cache=None):
        self.spec, self.T, self.size, self.refresh, self.seed, self.cache = spec, T, size, refresh, seed, cache
        self.generation, self.rng = -1, np.random.default_rng(seed)
        self.data = None

    def _regen(self, step):
        g = step // self.refresh
        if g != self.generation:
            chunks, per = [], 2048
            for c in range(0, self.size, per):
                s = stable_seed("train", self.spec, self.seed, g, c)
                chunks.append(make_world(self.spec, min(per, self.size - c), self.T, s, self.cache))
            self.data = concat(chunks)
            self.generation = g

    def sample(self, step, batch):
        self._regen(step)
        idx = self.rng.integers(0, len(self.data["vis"]), batch)
        return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in self.data.items()}
