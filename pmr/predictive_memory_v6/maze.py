"""Memory Maze offline trajectories -> ground-truth object tokens (numpy only).

Facts checked against memory-maze @4030901 and dm_control (see
docs/V6_BF_DELTA.md section "Data"):

* one NPZ per trajectory, every key has a leading time axis of length 1001;
  ``action[t]`` is the one-hot action that *led to* frame ``t`` (``action[0]``
  is undefined); ``maze_layout`` is 1 = corridor, 0 = wall, outer walls removed,
  bottom-left at ``[0, 0]``, indexed ``layout[int(y), int(x)]``;
* ``agent_pos`` is in grid units, ``agent_dir`` a unit vector such that going
  forward increases ``agent_pos`` along it; ``targets_vec`` is the object offset
  rotated into the agent frame (grid units).  Which component is "forward" and
  the sign of "right" are fitted once in S0 (``fit_axis_convention``) and stored
  in the cache metadata;
* the egocentric camera has ``fovy = 80`` degrees and square images.

The memory never has to do path integration: the ground-truth pose is an
*input* (Shell Game lesson), never a value.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np

LABEL_KEYS = ("action", "reward", "maze_layout", "agent_pos", "agent_dir", "targets_pos", "targets_vec")
T_FRAMES = 1001
GRID_F, GRID_L = 5, 5                      # forward rows 0..4, lateral cols -2..+2
N_GRID = GRID_F * GRID_L
VISAGE_BINS = (-1, 0, 5, 10, 50, 200, 1000)
# memory_maze/maze.py TARGET_COLORS (rgb in [0, 1]); 9x9 uses the first 3, 15x15 the first 6.
TARGET_COLORS = np.array([[170, 38, 30], [99, 170, 88], [39, 140, 217],
                          [93, 105, 199], [220, 193, 59], [220, 128, 107]], dtype=np.float64) / 220.0


# ------------------------------------------------------------------ files
def load_labels(path, keys=LABEL_KEYS):
    """Read label arrays only; NPZ members are decompressed lazily, so the
    ``image`` member is never touched here."""
    with np.load(path) as f:
        ep = {k: np.asarray(f[k]) for k in keys}
    return ep


def load_images(path):
    with np.load(path) as f:
        return np.asarray(f["image"])


def sha1_int(name: str) -> int:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16)


def split_of(basename: str, is_eval_file: bool, dev_mod: int = 29) -> str:
    if is_eval_file:
        return "test"
    return "dev" if sha1_int(basename) % dev_mod == 0 else "train"


# --------------------------------------------------------------- geometry
def right_of(direction, sign=1.0):
    """Unit vector to the agent's right (sign=+1: rotate dir by -90 degrees)."""
    d = np.asarray(direction, dtype=np.float64)
    return sign * np.stack([d[..., 1], -d[..., 0]], -1)


AXIS_CANDIDATES = {"right_fwd": (0, 1, 1.0), "left_fwd": (0, 1, -1.0),
                   "fwd_right": (1, 0, 1.0), "fwd_left": (1, 0, -1.0)}


def egocentric(delta, direction, convention="right_fwd"):
    """World offset(s) [..., 2] -> egocentric vector in the dataset convention."""
    i_lat, i_fwd, sign = AXIS_CANDIDATES[convention]
    d = np.asarray(direction, dtype=np.float64)
    fwd = (delta * d).sum(-1)
    lat = (delta * right_of(d, sign)).sum(-1)
    out = np.empty(np.broadcast_shapes(np.shape(fwd), np.shape(lat)) + (2,))
    out[..., i_lat] = lat
    out[..., i_fwd] = fwd
    return out


def fit_axis_convention(eps):
    """Pick the convention that reproduces ``targets_vec`` from global
    positions; returns (name, mean abs error in cells per candidate)."""
    errs = {}
    for name in AXIS_CANDIDATES:
        e = []
        for ep in eps:
            delta = ep["targets_pos"] - ep["agent_pos"][:, None, :]
            pred = egocentric(delta, ep["agent_dir"][:, None, :], name)
            e.append(np.abs(pred - ep["targets_vec"]).ravel())
        errs[name] = float(np.mean(np.concatenate(e)))
    best = min(errs, key=errs.get)
    return best, errs


def fwd_index(convention):
    return AXIS_CANDIDATES[convention][1]


def ego_angle(vec, convention="right_fwd"):
    i_lat, i_fwd, _ = AXIS_CANDIDATES[convention]
    return np.arctan2(vec[..., i_lat], vec[..., i_fwd])


def cell_of(pos):
    return np.floor(pos).astype(np.int64)


def wall_at(layout, cx, cy):
    """1 where the cell is a wall or outside the maze."""
    S = layout.shape[0]
    cx = np.asarray(cx)
    cy = np.asarray(cy)
    inside = (cx >= 0) & (cx < S) & (cy >= 0) & (cy < S)
    out = np.ones(np.broadcast_shapes(cx.shape, cy.shape), dtype=bool)
    out[inside] = layout[cy[inside], cx[inside]] == 0
    return out


def line_of_sight(layout, p, q, step=0.02):
    """Vectorised LOS: segment p->q [..., 2] is blocked if a sample point lies
    in a wall cell other than q's own cell.  ``step`` is in cells; this is the
    definition used everywhere in V6.1 (tests compare it with exact DDA)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p, q = np.broadcast_arrays(p, q)
    L = np.linalg.norm(q - p, axis=-1)
    n = max(2, int(np.ceil(np.max(L) / step)) + 1) if L.size else 2
    s = np.linspace(0.0, 1.0, n)
    pts = p[..., None, :] + (q - p)[..., None, :] * s[:, None]
    cx = np.floor(pts[..., 0]).astype(np.int64)
    cy = np.floor(pts[..., 1]).astype(np.int64)
    qx = np.floor(q[..., 0]).astype(np.int64)[..., None]
    qy = np.floor(q[..., 1]).astype(np.int64)[..., None]
    own = (cx == qx) & (cy == qy)
    blocked = wall_at(layout, cx, cy) & ~own
    return ~blocked.any(-1)


def quantized_dir(direction):
    """0:+x, 1:+y, 2:-x, 3:-y (closest axis)."""
    d = np.asarray(direction, dtype=np.float64)
    ang = np.arctan2(d[..., 1], d[..., 0])
    return np.mod(np.round(ang / (np.pi / 2)).astype(np.int64), 4)


_AXES = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int64)


def local_cells(pos, direction):
    """Cell coordinates [T,25,2] of the 5x5 forward grid."""
    pos = np.asarray(pos, dtype=np.float64)
    q = quantized_dir(direction)
    F = _AXES[q]
    R = np.stack([F[:, 1], -F[:, 0]], -1)
    c = cell_of(pos)
    f = np.arange(GRID_F)
    l = np.arange(GRID_L) - GRID_L // 2
    ff, ll = np.meshgrid(f, l, indexing="ij")
    ff, ll = ff.ravel(), ll.ravel()
    return c[:, None, :] + ff[None, :, None] * F[:, None, :] + ll[None, :, None] * R[:, None, :]


def seen_maps(pos, direction, grid_mask, S):
    """Cumulative 'cell has been observed' maps [T,S,S] from the grid masks."""
    cells = local_cells(pos, direction)
    T = len(cells)
    seen = np.zeros((T, S, S), dtype=bool)
    cur = np.zeros((S, S), dtype=bool)
    for t in range(T):
        cc = cells[t][grid_mask[t]]
        ok = (cc[:, 0] >= 0) & (cc[:, 0] < S) & (cc[:, 1] >= 0) & (cc[:, 1] < S)
        cc = cc[ok]
        cur[cc[:, 1], cc[:, 0]] = True
        seen[t] = cur
    return seen


def local_grid(layout, pos, direction):
    """5x5 forward grid in the agent's quantised frame.

    Returns (value [T,25] float {0 wall/out, 1 corridor}, mask [T,25] bool LOS
    from the continuous position to each cell centre)."""
    pos = np.asarray(pos, dtype=np.float64)
    q = quantized_dir(direction)
    F = _AXES[q]                              # [T,2]
    R = np.stack([F[:, 1], -F[:, 0]], -1)     # right of F
    c = cell_of(pos)                          # [T,2]
    f = np.arange(GRID_F)
    l = np.arange(GRID_L) - GRID_L // 2
    ff, ll = np.meshgrid(f, l, indexing="ij")
    ff, ll = ff.ravel(), ll.ravel()           # [25]
    cells = c[:, None, :] + ff[None, :, None] * F[:, None, :] + ll[None, :, None] * R[:, None, :]
    val = (~wall_at(layout, cells[..., 0], cells[..., 1])).astype(np.float32)
    centres = cells.astype(np.float64) + 0.5
    mask = line_of_sight(layout, pos[:, None, :], centres)
    return val, mask


def object_visibility(layout, pos, direction, tpos, tvec, convention, fov_deg=80.0, max_dist=np.inf):
    """vis[T,G]: inside the horizontal FOV, unobstructed, within max_dist."""
    ang = ego_angle(tvec, convention)
    in_fov = np.abs(ang) < np.deg2rad(fov_deg) / 2
    dist = np.linalg.norm(tvec, axis=-1)
    T, G = tvec.shape[:2]
    los = line_of_sight(layout, np.broadcast_to(pos[:, None, :], (T, G, 2)),
                        np.broadcast_to(tpos[None], (T, G, 2)))
    return in_fov & los & (dist <= max_dist)


def octant(direction):
    d = np.asarray(direction, dtype=np.float64)
    a = np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi)
    return np.minimum((a / (np.pi / 4)).astype(np.int64), 7)


def oracle_key(pos, direction, S):
    c = cell_of(pos)
    c = np.clip(c, 0, S - 1)
    return ((c[..., 1] * S + c[..., 0]) * 8 + octant(direction)).astype(np.int64)


def visage(vis):
    """Steps since the object was last visible (-1 = never seen so far)."""
    T, G = vis.shape
    out = np.full((T, G), -1, dtype=np.int64)
    last = np.full(G, -1, dtype=np.int64)
    for t in range(T):
        last = np.where(vis[t], t, last)
        out[t] = np.where(last >= 0, t - last, -1)
    return out


def bfs_all(layout):
    """All-pairs grid distances between corridor cells (inf for walls)."""
    S = layout.shape[0]
    D = np.full((S * S, S * S), np.inf)
    for s in range(S * S):
        sy, sx = divmod(s, S)
        if layout[sy, sx] == 0:
            continue
        D[s, s] = 0
        dq = deque([(sx, sy)])
        while dq:
            x, y = dq.popleft()
            d0 = D[s, y * S + x]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < S and 0 <= ny < S and layout[ny, nx] == 1 and np.isinf(D[s, ny * S + nx]):
                    D[s, ny * S + nx] = d0 + 1
                    dq.append((nx, ny))
    return D


def revisit_events(pos, direction, layout, min_gap=50, min_bfs=2, view_cos=0.7):
    """Frames where the agent re-enters a cell it left >= min_gap frames ago
    after having been > min_bfs cells away.  Returns list of
    (frame, last_frame_before, is_view_revisit)."""
    S = layout.shape[0]
    c = np.clip(cell_of(pos), 0, S - 1)
    idx = c[:, 1] * S + c[:, 0]
    D = bfs_all(layout)
    events = []
    last_seen = {}
    T = len(idx)
    for t in range(T):
        cur = idx[t]
        if t > 0 and cur != idx[t - 1] and cur in last_seen:
            lf = last_seen[cur]
            if t - lf >= min_gap:
                far = np.nanmax(np.where(np.isinf(D[cur, idx[lf + 1:t]]), np.nan, D[cur, idx[lf + 1:t]])) \
                    if t - lf > 1 else 0
                if np.isfinite(far) and far > min_bfs:
                    cosv = float(np.dot(direction[t], direction[lf]))
                    events.append((t, int(lf), bool(cosv > view_cos)))
        last_seen[cur] = t
    return events


# ----------------------------------------------------------------- tokens
def value_dim(G):
    return N_GRID + 3 * G


def object_slots(G):
    """Indices of the (vis, x, y) value dims of each object."""
    return [np.array([N_GRID + 3 * g, N_GRID + 3 * g + 1, N_GRID + 3 * g + 2]) for g in range(G)]


def build_values(layout, pos, direction, vis, tvec):
    """Value vector y[T,D] and observation mask o[T,D] (D = 25 + 3G).

    Grid cells: corridor indicator, observed iff LOS.  Object g: [vis
    (always observed), vis * targets_vec (observed iff visible)]."""
    T, G = vis.shape
    gv, gm = local_grid(layout, pos, direction)
    y = np.zeros((T, value_dim(G)), dtype=np.float32)
    o = np.zeros((T, value_dim(G)), dtype=bool)
    y[:, :N_GRID] = gv
    o[:, :N_GRID] = gm
    for g in range(G):
        b = N_GRID + 3 * g
        y[:, b] = vis[:, g]
        o[:, b] = True
        y[:, b + 1:b + 3] = np.where(vis[:, g, None], tvec[:, g], 0.0)
        o[:, b + 1:b + 3] = vis[:, g, None]
    return y, o


def pose_features(pos, direction, S):
    return np.concatenate([pos / S * 2.0 - 1.0, direction], -1).astype(np.float32)


def episode_tokens(ep, convention, fov_deg=80.0, max_dist=np.inf):
    """All per-frame arrays for one trajectory (numpy)."""
    layout = ep["maze_layout"][0].astype(np.uint8)
    S = layout.shape[0]
    pos = ep["agent_pos"].astype(np.float64)
    direction = ep["agent_dir"].astype(np.float64)
    tpos = ep["targets_pos"][0].astype(np.float64)
    tvec = ep["targets_vec"].astype(np.float64)
    vis = object_visibility(layout, pos, direction, tpos, tvec, convention, fov_deg, max_dist)
    y, o = build_values(layout, pos, direction, vis, tvec)
    act = np.argmax(ep["action"], -1).astype(np.uint8)
    act[0] = 0
    return {
        "y": y, "o": o, "pose": pose_features(pos, direction, S), "act": act,
        "key": oracle_key(pos, direction, S).astype(np.int32),
        "vis": vis, "tvec": tvec.astype(np.float32), "pos": pos.astype(np.float32),
        "dir": direction.astype(np.float32), "layout": layout, "tpos": tpos.astype(np.float32),
    }


def check_episode(ep):
    """S0 data unit checks (1)-(4), (6); (5) is the axis fit.  Returns dict of
    booleans and measured values (never raises)."""
    out = {}
    T = len(ep["agent_pos"])
    out["len_1001"] = all(len(v) == T_FRAMES for v in ep.values())
    lay = ep["maze_layout"]
    out["layout_static"] = bool((lay == lay[0]).all())
    out["targets_static"] = bool(np.allclose(ep["targets_pos"], ep["targets_pos"][0], atol=1e-4))
    c = np.clip(cell_of(ep["agent_pos"]), 0, lay.shape[1] - 1)
    on = lay[0][c[:, 1], c[:, 0]] == 1
    out["agent_on_corridor_frac"] = float(on.mean())
    dpos = np.diff(ep["agent_pos"], axis=0)
    act = np.argmax(ep["action"][1:], -1)
    along = (dpos * ep["agent_dir"][:-1]).sum(-1)
    fwd = np.isin(act, (1, 4, 5))
    out["forward_moves_along_dir"] = float(along[fwd].mean()) if fwd.any() else float("nan")
    out["noop_moves_along_dir"] = float(along[act == 0].mean()) if (act == 0).any() else float("nan")
    return out


def color_detect(images, G, border=2, tol=0.06, min_pixels=3):
    """Privileged image check used only for the S0 visibility calibration:
    count pixels whose chromaticity matches each target colour."""
    im = images[:, border:-border, border:-border].astype(np.float64)
    s = im.sum(-1, keepdims=True)
    chrom = im / np.maximum(s, 1e-6)
    bright = s[..., 0] > 60
    out = np.zeros((len(images), G), dtype=bool)
    for g in range(G):
        c = TARGET_COLORS[g] / TARGET_COLORS[g].sum()
        d = np.abs(chrom - c).sum(-1)
        out[:, g] = ((d < tol) & bright).sum((1, 2)) >= min_pixels
    return out
