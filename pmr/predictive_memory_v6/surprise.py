"""Surprise, novelty and counting analogues of Cambrian-S (numpy).

Cambrian-S (arXiv 2511.04670; cambrian-mllm/cambrian-s @ 058b1b4):
* a Linear-GELU-Linear head predicts the next frame's patch features; at
  inference surprise = 1 - mean cosine(pred, actual), frame 0 scores 1, and
  the head is frozen (``torch.inference_mode``);
* VSC counting cuts a segment whenever surprise > threshold (frame > 0) and at
  the last frame, answers each segment separately, and returns ``sum()``.

Analogues on object tokens:
* ``lfp_analog``   memoryless next-frame predictor on rasterized tokens
                   (torch model in ``lfp.py``; segmentation/counting here);
* ``nn_novelty``   Surprise-Forcing-style distance to the nearest stored
                   descriptor in a bounded bank (static, memory-referenced);
* memory residuals from ``memories`` (static_delta vs ours): surprise relative
  to what the memory says the world is NOW.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry


def view_coords(data, geom: Geometry):
    """Observed coordinates in [0,1) of the agent's view (window or world)."""
    cfg = data["meta"].get("config", {})
    obs = data["obs"]
    if geom.kind == "torus" and cfg.get("visibility") in ("window", "tour"):
        rel = np.mod(obs - data["ego"][:, :, None, :], geom.period)
        return rel / cfg["window"]
    if geom.kind == "torus":
        return np.mod(obs / geom.period, 1.0)
    z = geom.normalize(obs)
    return np.clip(z, 0.0, 1.0 - 1e-9) if geom.kind == "box" else (np.tanh(z) + 1) / 2


def rasterize(data, geom: Geometry, grid=8):
    """Tokens -> patch features [B,T,grid*grid, A+2] (appearance sum + offset)."""
    vis = data["vis"].astype(bool)
    B, T, N = vis.shape
    A = data["app"].shape[-1]
    z = np.clip(view_coords(data, geom), 0.0, 1.0 - 1e-9)
    cell = np.floor(z * grid).astype(int)
    off = z * grid - cell
    idx = cell[..., 0] * grid + cell[..., 1]
    F = np.zeros((B, T, grid * grid, A + 2), dtype=np.float32)
    feat = np.concatenate([np.broadcast_to(data["app"][:, None], (B, T, N, A)), off], -1)
    for i in range(N):
        m = vis[:, :, i]
        b_idx, t_idx = np.nonzero(m)
        np.add.at(F, (b_idx, t_idx, idx[b_idx, t_idx, i]), feat[b_idx, t_idx, i])
    return F


def frame_surprise(pred_feat, feat, eps=1e-6):
    """1 - mean cosine over patches active in prediction or target; frame 0 -> 1.

    ``pred_feat[:, t]`` is the prediction OF frame t (made at t-1)."""
    B, T = feat.shape[:2]
    out = np.ones((B, T))
    pn = np.linalg.norm(pred_feat, axis=-1)
    fn = np.linalg.norm(feat, axis=-1)
    cos = (pred_feat * feat).sum(-1) / np.maximum(pn * fn, eps)
    active = (pn > 1e-3) | (fn > 1e-3)
    both_zero = ~active
    cos = np.where(both_zero, 1.0, cos)
    cnt = active.sum(-1)
    val = 1.0 - np.where(cnt > 0, (cos * active).sum(-1) / np.maximum(cnt, 1), 1.0)
    out[:, 1:] = val[:, 1:]
    return out


def segment_count(surprise, vis, thr):
    """VSC: cut when surprise > thr (t > 0); count unique visible objects per
    segment (segment-level oracle) and sum over segments."""
    B, T, N = vis.shape
    out = np.zeros(B)
    for b in range(B):
        seen = np.zeros(N, bool)
        total = 0
        for t in range(T):
            if t > 0 and surprise[b, t] > thr:
                total += int(seen.sum())
                seen[:] = False
            seen |= vis[b, t]
        out[b] = total + int(seen.sum())
    return out


def novelty_count(novelty, vis, thr=0.5):
    """Count visible tokens whose memory novelty exceeds ``thr``."""
    return ((novelty > thr) & vis).sum((1, 2)).astype(float)


def true_count(vis):
    return vis.any(1).sum(-1).astype(float)


def nn_novelty(data, geom: Geometry, use_position=False, bank=64, pos_weight=1.0, admit_thr=None):
    """Distance of each visible token's descriptor to the nearest descriptor
    already in a FIFO bank (pre-insertion); inf for an empty bank.

    Descriptor = appearance (and optionally a position embedding).  With
    ``admit_thr`` only descriptors whose novelty exceeds it are admitted to the
    bank (Surprise-Forcing-style novelty-gated admission); otherwise every
    visible token is inserted and the oldest entries are evicted."""
    vis = data["vis"].astype(bool)
    B, T, N = vis.shape
    desc = data["app"][:, None].repeat(T, 1).astype(np.float64)             # [B,T,N,A]
    if use_position:
        z = view_coords(data, geom) if geom.kind != "torus" else np.mod(data["obs"] / geom.period, 1.0)
        ang = 2 * np.pi * z
        pe = np.concatenate([np.cos(ang), np.sin(ang)], -1) * pos_weight
        desc = np.concatenate([desc, pe], -1)
    out = np.zeros((B, T, N))
    for b in range(B):
        store = []
        for t in range(T):
            for i in np.nonzero(vis[b, t])[0]:
                d = desc[b, t, i]
                out[b, t, i] = min((np.linalg.norm(d - s) for s in store), default=np.inf)
            for i in np.nonzero(vis[b, t])[0]:
                if admit_thr is None or out[b, t, i] > admit_thr:
                    store.append(desc[b, t, i])
            store = store[-bank:]
    return out


def reveal_events(data, scores_by_name: dict, min_gap=2):
    """Collect (label, score per method) at reveal steps.  Frame-level scores
    ([B,T]) are broadcast to all objects revealed in that frame."""
    from .metrics import tracking_masks
    m = tracking_masks(data)
    sel = m["reveal"] & (m["since"] >= min_gap)
    labels = m["changed"][sel]
    out = {"label": labels.astype(int)}
    for name, s in scores_by_name.items():
        s = np.asarray(s)
        if s.ndim == 2:
            s = np.broadcast_to(s[:, :, None], sel.shape)
        out[name] = s[sel]
    return out
