"""PMW-L0: object-token worlds following the public FloWM MNIST-World spec.

Spec (FloWM, arXiv 2601.01075, Table 8 "dynamic po" + App. F.1): 50x50 torus
("edges roll"), 32x32 egocentric window, 5 objects, constant integer
velocities in [-2, 2] per axis, and every step a random integer window shift
bounded by +-10.  The official generator is not public; the choices below fill
the unstated details and are recorded in every dataset manifest:

=====================  ==================================================
unstated item          V6 choice
=====================  ==================================================
velocity (0,0)         allowed; 25 combinations uniform
self-motion            uniform over the 21x21 integers in [-10, 10]^2
initial positions      uniform on the 50x50 integer grid
visibility             object centre inside the 32x32 window
observation noise      none in the main world; sigma swept in L0-ctrl
appearance             random unit vectors, pairwise cosine = rho
digit size / overlap   N/A for tokens (pixels are never rendered)
=====================  ==================================================

Observations never contain velocity; identity is carried only by the
appearance code.  Hidden ground truth is returned for evaluation and is never
an input.  Numpy only; generation is cheap enough to run on CPU workers.

Visibility modes
----------------
``window``    FloWM-style moving window (natural hide/reveal events).
``schedule``  per-object alternating visible/hidden runs, static camera.
``blackout``  everything visible, then everything hidden ("lights off"), then
              visible again.
``tour``      static camera path that sweeps the torus; ``copies`` tiles one
              tour's token stream k times (VSC-Repeat analogue).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace

import numpy as np

from .geometry import Geometry

NONE, VELOCITY, TELEPORT = 0, 1, 2


@dataclass
class L0Config:
    world: int = 50
    window: int = 32
    n_objects: int = 5
    vel_max: int = 2
    allow_zero_velocity: bool = True
    ego_max: int = 10
    visibility: str = "window"
    obs_noise: float = 0.0
    appearance_dim: int = 16
    rho: float = 0.0
    vis_run: tuple = (3, 8)
    hid_run: tuple = (4, 24)
    hidden_scale: float = 1.0
    p_change: float = 0.0
    teleport_frac: float = 0.5
    continuous: bool = False
    omega_max: float = 0.0
    gamma_max: float = 0.0
    blackout_start: int = 8
    blackout_len: int = 32
    tour_speed: int = 4
    tour_row: int = 16
    static_objects: bool = False
    copies: int = 1
    unstated_choices: dict = field(default_factory=lambda: {
        "zero_velocity": "allowed, 25 combinations uniform",
        "self_motion": "uniform integer in [-10,10]^2 per step",
        "initial_positions": "uniform integer grid",
        "visibility": "object centre inside window",
        "noise": "none unless obs_noise > 0",
        "appearance": "random unit vectors with pairwise cosine rho",
        "digit_size_overlap": "N/A for tokens",
    })

    def __post_init__(self):
        if self.visibility not in ("window", "schedule", "blackout", "tour"):
            raise ValueError(f"unknown visibility mode {self.visibility!r}")
        if self.n_objects < 1 or self.world < 2 or not 0 < self.window <= self.world:
            raise ValueError("invalid world/window/object count")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1)")

    def geometry(self) -> Geometry:
        return Geometry("torus", period=float(self.world))

    def to_dict(self):
        d = asdict(self)
        d["vis_run"], d["hid_run"] = list(self.vis_run), list(self.hid_run)
        return d


# --------------------------------------------------------------- appearance
def appearance_codes(rng, batch, n, dim, rho):
    """Unit vectors with pairwise cosine exactly ``rho`` when n + 1 <= dim."""
    out = np.empty((batch, n, dim), dtype=np.float64)
    for b in range(batch):
        if n + 1 <= dim:
            q, _ = np.linalg.qr(rng.standard_normal((dim, n + 1)))
            common, private = q[:, 0], q[:, 1:].T
        else:  # more objects than dimensions: approximate (capacity sweep)
            common = rng.standard_normal(dim)
            common /= np.linalg.norm(common)
            private = rng.standard_normal((n, dim))
            private /= np.linalg.norm(private, axis=1, keepdims=True)
        a = np.sqrt(rho) * common[None] + np.sqrt(1.0 - rho) * private
        out[b] = a / np.linalg.norm(a, axis=1, keepdims=True)
    return out.astype(np.float32)


def _sample_velocity(rng, shape, cfg):
    if cfg.static_objects:
        return np.zeros(shape + (2,))
    if cfg.continuous:
        return rng.uniform(-cfg.vel_max, cfg.vel_max, shape + (2,))
    v = rng.integers(-cfg.vel_max, cfg.vel_max + 1, shape + (2,)).astype(np.float64)
    if not cfg.allow_zero_velocity:
        bad = np.all(v == 0, axis=-1)
        while bad.any():
            v[bad] = rng.integers(-cfg.vel_max, cfg.vel_max + 1, (int(bad.sum()), 2))
            bad = np.all(v == 0, axis=-1)
    return v


def _different_velocity(rng, old, cfg):
    while True:
        new = _sample_velocity(rng, (), replace(cfg, static_objects=False))
        if np.abs(new - old).sum() > 1e-9:
            return new


# --------------------------------------------------------------- schedules
def _runs_schedule(rng, T, cfg):
    """Alternating visible/hidden runs, first visible run lasts >= 2 steps."""
    vis = np.zeros(T, dtype=bool)
    t, visible, first = 0, True, True
    while t < T:
        if visible:
            n = int(rng.integers(cfg.vis_run[0], cfg.vis_run[1] + 1))
            n = max(n, 2) if first else n
            first = False
        else:
            lo = max(1, int(round(cfg.hid_run[0] * cfg.hidden_scale)))
            hi = max(lo, int(round(cfg.hid_run[1] * cfg.hidden_scale)))
            n = int(rng.integers(lo, hi + 1))
        vis[t:t + n] = visible
        t += n
        visible = not visible
    return vis


def hidden_intervals(vis_row):
    """[(start, end)) hidden runs that start after a visible step."""
    out, t, T = [], 0, len(vis_row)
    while t < T:
        if not vis_row[t] and t > 0 and vis_row[t - 1]:
            s = t
            while t < T and not vis_row[t]:
                t += 1
            out.append((s, t))
        else:
            t += 1
    return out


def _tour_origins(T, cfg):
    W = cfg.world
    s = np.arange(T)
    ox = (cfg.tour_speed * s) % W
    oy = (cfg.tour_row * ((cfg.tour_speed * s) // W)) % W
    return np.stack([ox, oy], -1).astype(np.float64)


# --------------------------------------------------------------- simulation
def _simulate(pos0, vel0, T, cfg, vis_fixed, ego, changes, change_values, drift, rng):
    """Roll the world forward.  ``vis_fixed`` is None in window mode."""
    B, N, _ = pos0.shape
    W = float(cfg.world)
    pos = np.zeros((B, T, N, 2))
    pos_u = np.zeros((B, T, N, 2))
    vel = np.zeros((B, T, N, 2))
    vis = np.zeros((B, T, N), dtype=bool)
    change_step = np.zeros((B, T, N), dtype=np.int8)
    p, pu, v = pos0.copy(), pos0.copy(), vel0.copy()
    omega, gamma = drift[:, 0], drift[:, 1]
    c, s = np.cos(omega), np.sin(omega)
    rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2) * (1 - gamma)[:, None, None]
    for t in range(T):
        pos[:, t], pos_u[:, t] = p, pu
        if vis_fixed is None:
            rel = np.mod(p - ego[:, t, None, :], W)
            vis[:, t] = np.all(rel < cfg.window, axis=-1)
        else:
            vis[:, t] = vis_fixed[:, t]
        if t == T - 1:
            vel[:, t] = v
            break
        # changes scheduled for the transition into t+1
        if changes is not None:
            code = changes[:, t + 1]
            vm = code == VELOCITY
            if vm.any():
                v[vm] = change_values[:, t + 1][vm]
            change_step[:, t + 1] = code
        elif vis_fixed is None and cfg.p_change > 0:
            hazard = (~vis[:, t]) & (rng.random((B, N)) < cfg.p_change)
            for b, i in zip(*np.nonzero(hazard)):
                if rng.random() < cfg.teleport_frac:
                    change_step[b, t + 1] = TELEPORT
                else:
                    v[b, i] = _different_velocity(rng, v[b, i], cfg)
                    change_step[b, t + 1] = VELOCITY
        vel[:, t] = v
        step = v.copy()
        pn, pun = p + step, pu + step
        tele = change_step[:, t + 1] == TELEPORT
        if tele.any():
            if changes is not None:
                target = change_values[:, t + 1][tele]
            else:
                target = rng.integers(0, cfg.world, (int(tele.sum()), 2)).astype(np.float64)
                if cfg.continuous:
                    target = rng.uniform(0, W, (int(tele.sum()), 2))
            jump = target - np.mod(p[tele], W)
            pn[tele] = p[tele] + jump
            pun[tele] = pu[tele] + jump
        p, pu = np.mod(pn, W), pun
        if cfg.continuous and (np.any(omega != 0) or np.any(gamma != 0)):
            v = np.einsum("bij,bnj->bni", rot, v)
    return pos, pos_u, vel, vis, change_step


def _schedule_changes(rng, vis, vel_init, cfg):
    """Unexpected changes inside hidden runs (schedule/blackout modes)."""
    B, T, N = vis.shape
    codes = np.zeros((B, T, N), dtype=np.int8)
    values = np.zeros((B, T, N, 2))
    if cfg.p_change <= 0:
        return codes, values
    for b in range(B):
        for i in range(N):
            for s, e in hidden_intervals(vis[b, :, i]):
                if e - s < 3 or e >= T or rng.random() >= cfg.p_change:
                    continue
                c = int(rng.integers(s + 1, e))  # hidden at c-1 and c
                if rng.random() < cfg.teleport_frac:
                    codes[b, c, i] = TELEPORT
                    values[b, c, i] = (rng.uniform(0, cfg.world, 2) if cfg.continuous
                                       else rng.integers(0, cfg.world, 2))
                else:
                    codes[b, c, i] = VELOCITY
                    values[b, c, i] = _different_velocity(rng, vel_init[b, i], cfg)
    return codes, values


def reveal_labels(vis, change_step):
    """At each reveal step, whether an unexpected change occurred while hidden."""
    B, T, N = vis.shape
    out = np.zeros((B, T, N), dtype=bool)
    pending = np.zeros((B, N), dtype=bool)
    for t in range(T):
        pending |= change_step[:, t] != NONE
        reveal = vis[:, t] & (np.zeros((B, N), bool) if t == 0 else ~vis[:, t - 1])
        out[:, t] = reveal & pending
        pending &= ~vis[:, t]
    return out


def _package(cfg, seed, app, pos, pos_u, vel, vis, ego, change_step, drift, rng, T, extra=None):
    B, _, N, _ = pos.shape
    W = float(cfg.world)
    obs = pos.copy()
    if cfg.obs_noise > 0:
        obs = np.mod(obs + rng.normal(0, cfg.obs_noise, obs.shape), W)
    obs = np.where(vis[..., None], obs, 0.0)
    u = np.zeros((B, T, 2))
    u[:, 1:] = ego[:, 1:] - ego[:, :-1]
    u = u - W * np.round(u / W)
    data = {
        "app": app,
        "pos": pos, "pos_unwrapped": pos_u, "vel": vel,
        "vis": vis, "exists": np.ones_like(vis),
        "obs": obs, "ego": ego, "u": u, "dt": np.ones((B, T)),
        "change_step": change_step,
        "reveal_changed": reveal_labels(vis, change_step),
        "drift": drift,
        "episode_id": np.array([f"l0-{seed}-{b}" for b in range(B)]),
        "meta": {"world": "L0", "config": cfg.to_dict(), "seed": int(seed), "T": int(T),
                 "geometry": cfg.geometry().to_dict(), "frame": "allo"},
    }
    if extra:
        data.update(extra)
    return data


def generate(cfg: L0Config, n_episodes: int, T: int, seed: int) -> dict:
    """Generate ``n_episodes`` independent episodes of length ``T``."""
    if T < 2 or n_episodes < 1:
        raise ValueError("need T >= 2 and at least one episode")
    rng = np.random.default_rng(seed)
    B, N, W = n_episodes, cfg.n_objects, cfg.world
    app = appearance_codes(rng, B, N, cfg.appearance_dim, cfg.rho)
    if cfg.continuous:
        pos0 = rng.uniform(0, W, (B, N, 2))
    else:
        pos0 = rng.integers(0, W, (B, N, 2)).astype(np.float64)
    vel0 = _sample_velocity(rng, (B, N), cfg)
    drift = np.stack([rng.uniform(-cfg.omega_max, cfg.omega_max, B),
                      rng.uniform(0.0, cfg.gamma_max, B)], -1) if cfg.continuous else np.zeros((B, 2))

    vis_fixed, changes, values = None, None, None
    ego = np.zeros((B, T, 2))
    if cfg.visibility == "window":
        ego[:, 0] = rng.integers(0, W, (B, 2))
        for t in range(1, T):
            ego[:, t] = np.mod(ego[:, t - 1] + rng.integers(-cfg.ego_max, cfg.ego_max + 1, (B, 2)), W)
    elif cfg.visibility == "schedule":
        vis_fixed = np.stack([np.stack([_runs_schedule(rng, T, cfg) for _ in range(N)], -1)
                              for _ in range(B)])
        changes, values = _schedule_changes(rng, vis_fixed, vel0, cfg)
    elif cfg.visibility == "blackout":
        vis_fixed = np.ones((B, T, N), dtype=bool)
        vis_fixed[:, cfg.blackout_start:cfg.blackout_start + cfg.blackout_len] = False
        changes, values = _schedule_changes(rng, vis_fixed, vel0, cfg)
    elif cfg.visibility == "tour":
        L = T // cfg.copies
        if L * cfg.copies != T:
            raise ValueError("tour length T must be divisible by copies")
        ego[:] = np.tile(_tour_origins(L, cfg), (cfg.copies, 1))[None]
    pos, pos_u, vel, vis, change_step = _simulate(
        pos0, vel0, T, cfg, vis_fixed, ego, changes, values, drift, rng)
    # With static objects and a tiled camera path the k copies are exactly
    # identical token streams (the VSC-Repeat construction); with moving
    # objects the tour simply repeats without a seam in the world state.
    return _package(cfg, seed, app, pos, pos_u, vel, vis, ego, change_step, drift, rng, T)


# ------------------------------------------------------------------ pairs
def history_pair(cfg: L0Config, n_episodes: int, T: int, seed: int, early: int = 4):
    """Two worlds with identical inputs after ``early`` steps.

    Object 0 is visible only for t < early, then hidden to the end; in world B
    its initial position and velocity are resampled.  All other objects and
    all schedules are shared, so any difference in the final belief about
    object 0 must come from memory of the early history.
    """
    cfg = replace(cfg, visibility="schedule", p_change=0.0)
    rng = np.random.default_rng(seed)
    B, N, W = n_episodes, cfg.n_objects, cfg.world
    app = appearance_codes(rng, B, N, cfg.appearance_dim, cfg.rho)
    pos0 = rng.integers(0, W, (B, N, 2)).astype(np.float64)
    vel0 = _sample_velocity(rng, (B, N), replace(cfg, allow_zero_velocity=False))
    vis = np.stack([np.stack([_runs_schedule(rng, T, cfg) for _ in range(N)], -1) for _ in range(B)])
    vis[:, :early, 0], vis[:, early:, 0] = True, False
    pos0_b, vel0_b = pos0.copy(), vel0.copy()
    pos0_b[:, 0] = rng.integers(0, W, (B, 2))
    for b in range(B):
        vel0_b[b, 0] = _different_velocity(rng, vel0[b, 0], replace(cfg, allow_zero_velocity=False))
    ego, drift = np.zeros((B, T, 2)), np.zeros((B, 2))
    out = []
    for p0, v0, tag in ((pos0, vel0, "A"), (pos0_b, vel0_b, "B")):
        pos, pos_u, vel, vis_out, cs = _simulate(p0, v0, T, cfg, vis, ego, None, None, drift, rng)
        d = _package(cfg, seed, app, pos, pos_u, vel, vis_out, ego, cs, drift, rng, T,
                     extra={"pair_kind": "history", "pair_side": tag, "probe_object": 0})
        d["episode_id"] = np.array([f"l0h-{seed}-{b}" for b in range(B)])
        out.append(d)
    return out[0], out[1]


def change_pair(cfg: L0Config, n_episodes: int, T: int, seed: int, min_hidden: int = 6):
    """Worlds that differ only by one hidden velocity change of object 0.

    World A has the change, world B does not; schedules, appearances and all
    other trajectories are identical.  ``pair_valid`` marks episodes where
    object 0 had a long enough hidden run that ends before T.
    """
    cfg = replace(cfg, visibility="schedule", p_change=0.0)
    rng = np.random.default_rng(seed)
    B, N, W = n_episodes, cfg.n_objects, cfg.world
    app = appearance_codes(rng, B, N, cfg.appearance_dim, cfg.rho)
    pos0 = rng.integers(0, W, (B, N, 2)).astype(np.float64)
    vel0 = _sample_velocity(rng, (B, N), cfg)
    vis = np.stack([np.stack([_runs_schedule(rng, T, cfg) for _ in range(N)], -1) for _ in range(B)])
    codes = np.zeros((B, T, N), dtype=np.int8)
    values = np.zeros((B, T, N, 2))
    valid = np.zeros(B, dtype=bool)
    reveal = np.full(B, -1)
    for b in range(B):
        for s, e in hidden_intervals(vis[b, :, 0]):
            if e - s >= min_hidden and e < T:
                c = s + (e - s) // 2
                codes[b, c, 0] = VELOCITY
                values[b, c, 0] = _different_velocity(rng, vel0[b, 0], cfg)
                valid[b], reveal[b] = True, e
                break
    ego, drift = np.zeros((B, T, 2)), np.zeros((B, 2))
    out = []
    for ch, tag in ((codes, "A"), (np.zeros_like(codes), "B")):
        pos, pos_u, vel, vis_out, cs = _simulate(pos0, vel0, T, cfg, vis, ego, ch, values, drift, rng)
        d = _package(cfg, seed, app, pos, pos_u, vel, vis_out, ego, cs, drift, rng, T,
                     extra={"pair_kind": "change", "pair_side": tag, "probe_object": 0,
                            "pair_valid": valid, "probe_reveal": reveal})
        d["episode_id"] = np.array([f"l0c-{seed}-{b}" for b in range(B)])
        out.append(d)
    return out[0], out[1]


def to_ego_frame(data: dict) -> dict:
    """Express observations in the egocentric window frame (positions - origin).

    Ground-truth ``pos`` stays allocentric for scoring; ``obs_frame`` records
    the choice.  Models then need the action ``u`` to transport memory.
    """
    out = dict(data)
    W = data["meta"]["config"]["world"]
    obs = np.mod(data["obs"] - data["ego"][:, :, None, :], W)
    out["obs"] = np.where(data["vis"][..., None], obs, 0.0)
    out["meta"] = dict(data["meta"], frame="ego")
    return out


def slice_batch(data: dict, idx) -> dict:
    """Select episodes; leaves ``meta`` untouched."""
    out = {}
    for k, v in data.items():
        if k == "meta" or not isinstance(v, np.ndarray):
            out[k] = v
        else:
            out[k] = v[idx]
    return out
