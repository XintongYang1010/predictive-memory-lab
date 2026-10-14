"""Synthetic Memory-Maze-format trajectories for unit tests and the CPU SMOKE
stage only.  Same NPZ keys, shapes and conventions as the offline dataset
(``right_fwd`` egocentric axes, 1001 frames).  Never used by a science stage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import maze as M


def random_maze(S, rng, loop_frac=0.25):
    lay = np.zeros((S, S), dtype=np.uint8)
    start = (rng.integers(0, S // 2) * 2, rng.integers(0, S // 2) * 2)
    stack = [start]
    lay[start[1], start[0]] = 1
    while stack:
        x, y = stack[-1]
        nb = [(x + dx, y + dy, dx, dy) for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2))
              if 0 <= x + dx < S and 0 <= y + dy < S and lay[y + dy, x + dx] == 0]
        if not nb:
            stack.pop()
            continue
        nx, ny, dx, dy = nb[rng.integers(len(nb))]
        lay[y + dy // 2, x + dx // 2] = 1
        lay[ny, nx] = 1
        stack.append((nx, ny))
    walls = np.argwhere(lay == 0)
    rng.shuffle(walls)
    for (yy, xx) in walls[: int(loop_frac * len(walls))]:
        lay[yy, xx] = 1
    return lay


def _bfs_path(lay, a, b):
    S = lay.shape[0]
    prev = {a: None}
    q = [a]
    while q:
        cur = q.pop(0)
        if cur == b:
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, y + dy)
            if 0 <= n[0] < S and 0 <= n[1] < S and lay[n[1], n[0]] == 1 and n not in prev:
                prev[n] = cur
                q.append(n)
    if b not in prev:
        return [a]
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def _render(lay, pos, d, tpos, G, rng_tex, size=64):
    """Crude first-person stand-in: background texture keyed by (cell, dir);
    visible objects drawn as colour blobs at their bearing."""
    S = lay.shape[0]
    c = np.clip(M.cell_of(pos), 0, S - 1)
    q = int(M.quantized_dir(d))
    seed = int((c[1] * S + c[0]) * 4 + q)
    img = rng_tex[seed % len(rng_tex)].copy()
    vec = M.egocentric(tpos - pos[None], d[None], "right_fwd")
    vis = M.object_visibility(lay, pos[None], d[None], tpos, vec[None], "right_fwd")[0]
    for g in range(G):
        if not vis[g]:
            continue
        ang = M.ego_angle(vec[g], "right_fwd")
        dist = max(np.linalg.norm(vec[g]), 0.3)
        cx = int(size / 2 + ang / np.deg2rad(40) * size / 2)
        r = int(np.clip(6 / dist, 1, 10))
        col = (M.TARGET_COLORS[g] * 220).astype(np.uint8)
        img[max(0, 32 - r):32 + r, max(0, cx - r):min(size, cx + r)] = col
    img[:2] = img[-2:] = 128
    img[:, :2] = img[:, -2:] = 128
    return img


def make_episode(seed, S=9, G=3, T=1001, images=True):
    rng = np.random.default_rng(seed)
    lay = random_maze(S, rng)
    free = [tuple(v[::-1]) for v in np.argwhere(lay == 1)]
    idx = rng.choice(len(free), size=G + 1, replace=False)
    start = free[idx[0]]
    tpos = np.array([[free[i][0] + 0.5, free[i][1] + 0.5] for i in idx[1:]])
    pos = np.array(start, dtype=np.float64) + 0.5
    ang = rng.uniform(0, 2 * np.pi)
    P, Dn, A = [], [], []
    goal = start
    path = [start]
    tex = [np.repeat(np.random.default_rng(1000 + k).integers(30, 90, (64, 64, 1)), 3, -1).astype(np.uint8)
           for k in range(97)]
    for t in range(T):
        d = np.array([np.cos(ang), np.sin(ang)])
        P.append(pos.copy())
        Dn.append(d)
        cur = tuple(np.clip(M.cell_of(pos), 0, S - 1))
        if cur == goal or len(path) <= 1:
            goal = free[rng.integers(len(free))]
            path = _bfs_path(lay, cur, goal)
        while len(path) > 1 and path[0] == cur:
            path = path[1:]
        wp = np.array(path[0], dtype=np.float64) + 0.5
        want = np.arctan2(wp[1] - pos[1], wp[0] - pos[0])
        diff = (want - ang + np.pi) % (2 * np.pi) - np.pi
        if abs(diff) > 0.5:
            a = 2 if diff > 0 else 3
        elif abs(diff) > 0.15:
            a = 4 if diff > 0 else 5
        else:
            a = 1
        if rng.random() < 0.15:
            a = int(rng.integers(6))
        A.append(a)
        turn = {0: 0, 1: 0, 2: 0.35, 3: -0.35, 4: 0.2, 5: -0.2}[a]
        speed = {0: 0, 1: 0.22, 2: 0, 3: 0, 4: 0.15, 5: 0.15}[a] * rng.uniform(0.7, 1.3)
        ang = ang + turn
        nd = np.array([np.cos(ang), np.sin(ang)])
        newp = pos + speed * nd
        c = M.cell_of(newp)
        if 0 <= c[0] < S and 0 <= c[1] < S and lay[c[1], c[0]] == 1:
            pos = newp
    P, Dn = np.array(P), np.array(Dn)
    act = np.zeros((T, 6), dtype=np.float32)
    act[np.arange(1, T), np.array(A[:-1])] = 1.0
    tvec = M.egocentric(tpos[None] - P[:, None], Dn[:, None], "right_fwd")
    ep = {
        "action": act, "reward": np.zeros(T, np.float32),
        "maze_layout": np.repeat(lay[None], T, 0),
        "agent_pos": P.astype(np.float64), "agent_dir": Dn.astype(np.float64),
        "targets_pos": np.repeat(tpos[None], T, 0), "targets_vec": tvec,
        "target_pos": np.repeat(tpos[None, 0], T, 0), "target_vec": tvec[:, 0],
        "target_color": np.repeat(M.TARGET_COLORS[None, 0], T, 0),
    }
    if images:
        ep["image"] = np.stack([_render(lay, P[t], Dn[t], tpos, G, tex) for t in range(T)])
    return ep


def write_fake_dataset(root, n_train=40, n_eval=8, S=9, G=3, T=1001, images=True, seed=0):
    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "eval").mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_train + n_eval):
        sub = "train" if i < n_train else "eval"
        p = root / sub / f"fake-{seed:03d}-{i:05d}.npz"
        if not p.exists():
            np.savez_compressed(p, **make_episode(seed * 100000 + i, S, G, T, images))
        paths.append(p)
    return paths
