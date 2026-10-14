"""L1: Dynamic Block World states regenerated with the public Miniworld fork.

Source: github.com/hlillemark/Miniworld @ b28988d (Apache-2.0, master only;
``--store-block-info`` does not exist on ``flowm_release``).  The FloWM data
were generated with policy ``biased_walk_v2 --forward-prob 0.90`` (README and
generate_videos.py L65-77), room size 16 (reachable block cells 1..15),
cardinal grid velocities of magnitude 1, wall bounces only (overlap allowed),
60 deg vertical FoV.  Pitfalls handled here:

* ``--box-and-ball`` drops the balls from block_info -> never used;
* the number of blocks is drawn by ``random.randint`` that ``--seed`` does not
  seed -> the wrapper samples it and passes ``--num-blocks`` explicitly;
* palette ``grey`` vs ``gray`` differs from the released data -> irrelevant for
  tokens, recorded in the manifest.

block_info contains no visibility; it is computed from ``block_pos_agent``
(x forward, z right): in front, |bearing| <= 30 deg (square render => 60 deg
horizontal), and the centre ray is not blocked by a nearer block's 0.7 x 0.7
footprint (axis aligned: 90-degree turns with --heading-zero).  ``depth_check``
compares this rule with the rendered depth on sample episodes (S0 gate 95%).

Token geometry is a reflecting box with walls at 1 and 15 (fold readout).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .geometry import Geometry

MINIWORLD_COMMIT = "b28988d5cd9d0d261a53b8cfa4f162e7fe14853e"
LO, HI = 1.0, 15.0
HALF_FOOT = 0.35
HALF_FOV = np.deg2rad(30.0)
N_MAX = 10
SPLITS = {"train": 10000, "dev": 1000, "test": 1000, "peekaboo": 1000, "backforth": 1000}
POLICY = {"train": "biased_walk_v2", "dev": "biased_walk_v2", "test": "biased_walk_v2",
          "peekaboo": "peekaboo_motion", "backforth": "back_and_forth"}


def geometry() -> Geometry:
    return Geometry("box", lo=LO, hi=HI)


def command(out_prefix, seed, num_blocks, policy, steps=500, size=64):
    python = os.environ.get("PMR_MINIWORLD_PYTHON", sys.executable)
    cmd = [python, "-m", "scripts.generate_videos",
           "--env-name", "MiniWorld-MovingBlockWorld-v0", "--out-prefix", str(out_prefix),
           "--seed", str(seed), "--num-blocks", str(num_blocks),
           "--turn-step-deg", "90", "--forward-step", "1.0", "--heading-zero",
           "--grid-mode", "--grid-vel-min", "-1", "--grid-vel-max", "1", "--grid-cardinal-only",
           "--no-time-limit",
           "--render-width", str(size), "--render-height", str(size),
           "--obs-width", str(size), "--obs-height", str(size),
           "--steps", str(steps), "--room-size", "16", "--block-size-xy", "0.7", "--block-height", "1.5",
           "--agent-box-allow-overlap", "--box-allow-overlap", "--ensure-base-palette",
           "--policy", policy, "--cam-fov-y", "60", "--store-block-info"]
    if policy == "biased_walk_v2":
        cmd += ["--forward-prob", "0.90"]
    if policy == "back_and_forth":
        cmd += ["--segment-len", "10"]
    return cmd


def _gen_one(args):
    miniworld_dir, prefix, seed, nb, policy, steps = args
    if Path(str(prefix) + "_block_info.pt").exists():
        return 0
    env = dict(os.environ, MINIWORLD_HEADLESS="1")
    r = subprocess.run(command(prefix, seed, nb, policy, steps), cwd=miniworld_dir, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        Path(str(prefix) + "_error.txt").write_text(r.stdout[-4000:] + "\n" + r.stderr[-4000:])
    return r.returncode


def generate(miniworld_dir, cache, split, n=None, procs=48, steps=500, seed_base=20261013):
    """Render ``n`` episodes of one split with a process pool (CPU only)."""
    n = SPLITS[split] if n is None else n
    out = Path(cache) / "l1" / "raw" / split
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed_base + list(SPLITS).index(split))
    jobs = []
    for i in range(n):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        nb = int(rng.integers(6, 11))
        jobs.append((str(miniworld_dir), out / f"ep_{i:05d}", seed, nb, POLICY[split], steps))
    with ProcessPoolExecutor(procs) as ex:
        codes = list(ex.map(_gen_one, jobs, chunksize=4))
    manifest = {"split": split, "episodes": n, "failures": int(sum(c != 0 for c in codes)),
                "policy": POLICY[split], "steps": steps, "miniworld_commit": MINIWORLD_COMMIT,
                "seeds": [j[2] for j in jobs], "num_blocks": [j[3] for j in jobs],
                "palette_note": "master uses 'grey'; flowm_release 'gray' is silently filtered"}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return manifest


# --------------------------------------------------------------- tokens
def _ray_blocked(target, others):
    """Does the segment 0 -> target cross any square (half size HALF_FOOT) centred at others?"""
    if len(others) == 0:
        return False
    d = target
    lo = others - HALF_FOOT
    hi = others + HALF_FOOT
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(d) > 1e-12, 1.0 / d, np.inf)
        t1, t2 = lo * inv, hi * inv
    tmin = np.where(np.abs(d) > 1e-12, np.minimum(t1, t2), np.where((lo <= 0) & (hi >= 0), -np.inf, np.inf))
    tmax = np.where(np.abs(d) > 1e-12, np.maximum(t1, t2), np.where((lo <= 0) & (hi >= 0), np.inf, -np.inf))
    enter = tmin.max(-1)
    leave = tmax.min(-1)
    return bool(np.any((enter <= leave) & (leave >= 0) & (enter <= 1.0)))


def visibility(block_pos_agent, exists=None):
    """[T,N,2] agent-frame centres (x forward, z right) -> [T,N] visible."""
    T, N, _ = block_pos_agent.shape
    vis = np.zeros((T, N), dtype=bool)
    for t in range(T):
        p = block_pos_agent[t]
        dist = np.linalg.norm(p, axis=-1)
        for i in range(N):
            if exists is not None and not exists[t, i]:
                continue
            x, z = p[i]
            if x <= 1e-6 or abs(np.arctan2(z, x)) > HALF_FOV:
                continue
            others = [j for j in range(N) if j != i and dist[j] < dist[i] and (exists is None or exists[t, j])]
            vis[t, i] = not _ray_blocked(p[i], p[others])
    return vis


def tokens_from_block_info(info: dict, app_dim=16, seed=0) -> dict:
    """block_info dict (numpy arrays) -> padded token episode."""
    pos3 = np.asarray(info["block_pos_world"], dtype=np.float64)          # [T,N,3]
    T, N, _ = pos3.shape
    if N > N_MAX:
        raise ValueError("more blocks than N_MAX")
    pos = pos3[..., [0, 2]]
    agent = np.asarray(info["agent_pos_world"], dtype=np.float64)[:, [0, 2]]
    pa = np.asarray(info["block_pos_agent"], dtype=np.float64)
    vis = visibility(pa)
    rng = np.random.default_rng(seed)
    app = rng.standard_normal((N_MAX, app_dim))
    app /= np.linalg.norm(app, axis=-1, keepdims=True)
    P = np.zeros((T, N_MAX, 2))
    P[:, :N] = pos
    V = np.zeros((T, N_MAX), dtype=bool)
    V[:, :N] = vis
    E = np.zeros((T, N_MAX), dtype=bool)
    E[:, :N] = True
    u = np.zeros((T, 2))
    u[1:] = agent[1:] - agent[:-1]
    return {"app": app.astype(np.float32), "pos": P, "vis": V, "exists": E,
            "obs": np.where(V[..., None], P, 0.0), "ego": agent, "u": u,
            "agent_dir": np.asarray(info["agent_dir"], dtype=np.float64), "n_blocks": N}


def convert_split(cache, split):
    """raw/*_block_info.pt -> tokens/<split>.npz (needs torch to unpickle)."""
    import torch
    raw = Path(cache) / "l1" / "raw" / split
    eps = []
    for f in sorted(raw.glob("ep_*_block_info.pt")):
        info = torch.load(f, map_location="cpu", weights_only=False)
        info = {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in info.items()}
        eps.append(tokens_from_block_info(info, seed=int(f.name[3:8])))
    if not eps:
        raise FileNotFoundError(f"no block_info files in {raw}")
    T = min(len(e["vis"]) for e in eps)
    out = {k: np.stack([e[k][:T] for e in eps]) for k in ("pos", "vis", "exists", "obs", "ego", "u", "agent_dir")}
    out["app"] = np.stack([e["app"] for e in eps])
    out["n_blocks"] = np.array([e["n_blocks"] for e in eps])
    dst = Path(cache) / "l1" / "tokens"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / f"{split}.npz", **out)
    return {"split": split, "episodes": len(eps), "T": T,
            "visible_fraction": float(out["vis"][out["exists"]].mean())}


def check_dynamics(cache, split="dev"):
    """Verify the fold model on generated states: positions stay in [1,15] and
    the per-step motion equals fold(unfolded linear motion) for every block."""
    d = np.load(Path(cache) / "l1" / "tokens" / f"{split}.npz")
    pos, ex = d["pos"], d["exists"]
    inside = bool(np.all((pos[ex] >= LO - 1e-6) & (pos[ex] <= HI + 1e-6)))
    from .geometry import fold
    ok, total = 0, 0
    B, T, N, _ = pos.shape
    for b in range(B):
        for i in range(int(d["n_blocks"][b])):
            p = pos[b, :, i]
            v0 = p[1] - p[0]
            if np.abs(v0).sum() == 0:
                continue
            # unfold: find the unfolded start consistent with p[0] and first move
            pred = fold(p[0][None] + v0[None] * np.arange(T)[:, None], LO, HI)
            total += 1
            ok += int(np.allclose(pred, p, atol=1e-4))
    return {"inside_box": inside, "fold_consistent_fraction": ok / max(total, 1), "checked": total}


def depth_check(cache, split="dev", episodes=50):
    """Agreement between the centre-ray rule and rendered depth (S0 gate 0.95).

    For each block at each step: project its centre column (square image,
    60 deg FoV); compare rendered depth on the horizon band with the block's
    forward distance.  Visible <=> depth within the block footprint."""
    import torch
    raw = Path(cache) / "l1" / "raw" / split
    agree = total = 0
    for f in sorted(raw.glob("ep_*_block_info.pt"))[:episodes]:
        info = torch.load(f, map_location="cpu", weights_only=False)
        depth = torch.load(str(f).replace("_block_info.pt", "_depth.pt"), map_location="cpu").numpy()[..., 0]
        pa = np.asarray(info["block_pos_agent"], dtype=np.float64)
        vis = visibility(pa)
        T, H, W = depth.shape
        band = depth[:, H // 2 - 2:H // 2 + 3]
        for t in range(T):
            for i in range(pa.shape[1]):
                x, z = pa[t, i]
                if x <= 0.5 or abs(np.arctan2(z, x)) > HALF_FOV * 0.9:
                    continue
                col = int(round((W - 1) / 2 * (1 + (z / x) / np.tan(HALF_FOV))))
                if not 0 <= col < W:
                    continue
                seen = bool(np.min(np.abs(band[t, :, col] - x)) <= HALF_FOOT + 0.2)
                agree += int(seen == vis[t, i])
                total += 1
    return {"agreement": agree / max(total, 1), "checked": total}


_CACHE = {}


def load_l1(spec, n, T, seed, cache):
    """spec 'l1_<split>'; random crops of length T (full episodes if T >= length)."""
    split = spec.split(":")[0][3:] or "train"
    key = (str(cache), split)
    if key not in _CACHE:
        f = Path(cache) / "l1" / "tokens" / f"{split}.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing: run `python -m pmr.predictive_memory_v6 prepare-l1`")
        _CACHE[key] = dict(np.load(f))
    d = _CACHE[key]
    rng = np.random.default_rng(seed)
    E, L = d["vis"].shape[:2]
    T = min(T, L)
    idx = rng.integers(0, E, n)
    start = rng.integers(0, L - T + 1, n)
    sl = lambda a: np.stack([a[e, s:s + T] for e, s in zip(idx, start)])
    B = n
    out = {"pos": sl(d["pos"]), "vis": sl(d["vis"]), "exists": sl(d["exists"]), "obs": sl(d["obs"]),
           "ego": sl(d["ego"]), "u": sl(d["u"]), "dt": np.ones((B, T)),
           "app": d["app"][idx], "change_step": np.zeros((B, T, N_MAX), dtype=np.int8),
           "episode_id": np.array([f"l1-{split}-{e}-{s}" for e, s in zip(idx, start)])}
    out["pos_unwrapped"] = out["pos"]
    out["reveal_changed"] = np.zeros_like(out["vis"])
    out["meta"] = {"world": "L1", "spec": spec, "T": T, "seed": int(seed), "frame": "allo",
                   "geometry": geometry().to_dict(),
                   "config": {"visibility": "fov", "n_objects": N_MAX, "vel_max": 1.0, "obs_noise": 0.0}}
    return out
