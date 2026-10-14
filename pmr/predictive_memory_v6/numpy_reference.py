"""Float64 numpy reference of the associative ("ours") family.

Mirrors ``memories.AssocMemory`` equation by equation with the same parameter
names, so that (a) the algebra (lemmas 1-4, TTT equivalence, fold Jacobian)
is testable without torch, and (b) the torch implementation can be checked
against it on the cluster (tests/predictive_memory_v6/test_torch_models.py).
No scipy: ``expm`` uses scaling-and-squaring with a Taylor series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PI, TAU, P, Q, Z = 0, 1, slice(2, 4), slice(4, 6), slice(6, 8)
D_V = 8
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])


def expm(A, terms=20):
    """Matrix exponential for (batched) small matrices."""
    A = np.asarray(A, dtype=np.float64)
    norm = np.abs(A).sum(-1).max() if A.size else 0.0
    s = max(0, int(math.ceil(math.log2(max(norm, 1e-16)))) + 1) if norm > 0.5 else 0
    X = A / (2 ** s)
    eye = np.broadcast_to(np.eye(A.shape[-1]), A.shape)
    out, term = eye.copy(), eye.copy()
    for k in range(1, terms):
        term = term @ X / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class AssocParams:
    """Parameter set with the torch defaults (see AssocMemory.__init__)."""
    transition: str = "ours"           # static | ours | vo
    learned_gain: bool = True
    frame_form: bool = False
    beta_fixed: float | None = None
    B_shear: np.ndarray = field(default_factory=lambda: np.eye(2))
    w_p: float = 0.0
    w_q: float = 0.0
    d_q: np.ndarray = field(default_factory=lambda: np.full(2, math.log(math.expm1(1e-3))))
    w_z: float = 0.0
    d_z: np.ndarray = field(default_factory=lambda: np.full(2, math.log(math.expm1(1e-2))))
    T_ego: np.ndarray = field(default_factory=lambda: np.eye(2))
    vo_freqs: tuple = (2 * math.pi / 256, 2 * math.pi / 64)
    K_p: np.ndarray = field(default_factory=lambda: np.eye(2))
    K_q: np.ndarray = field(default_factory=lambda: 0.5 * np.eye(2))
    K_z: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    w_s: float = 0.0
    w_b: float = 3.0
    c0: float = math.log(math.expm1(1e-2))
    c1: float = math.log(math.expm1(1e-3))
    W_key: np.ndarray | None = None     # [heads*d_k, app_dim]
    heads: int = 4


def structured_dyn(p: AssocParams):
    L = np.zeros((D_V, D_V))
    L[TAU, PI] = 1.0
    if p.transition == "static":
        return L
    if p.transition == "vo":
        f1, f2 = p.vo_freqs
        L[2, 4], L[4, 2], L[3, 5], L[5, 3] = f1, -f1, f1, -f1
        L[6:8, 6:8] = f2 * J2
        return L
    L[P, P] = p.w_p * J2
    L[P, Q] = p.B_shear
    L[Q, Q] = p.w_q * J2 - np.diag(softplus(p.d_q))
    L[Z, Z] = p.w_z * J2 - np.diag(softplus(p.d_z))
    return L


def keys(p: AssocParams, app):
    B, N, _ = app.shape
    k = app @ p.W_key.T
    k = k.reshape(B, N, p.heads, -1)
    k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-6)
    return k.reshape(B, N, -1) / math.sqrt(p.heads)


def readout_h(z, kind):
    if kind == "torus":
        return np.mod(z, 1.0)
    if kind == "box":
        return 1.0 - np.abs(np.mod(z, 2.0) - 1.0)
    return z


def fold_jac(z, kind):
    if kind != "box":
        return np.ones_like(z)
    return np.where(np.mod(z, 2.0) < 1.0, 1.0, -1.0)


def obs_diff(y, yh, kind):
    d = y - yh
    if kind == "torus":
        d = d - np.round(d)
    return d


def rollout(p: AssocParams, app, obs, vis, u, dt, kind="torus", frame="allo", return_internals=False):
    """Pre-write predictions [B,T,N,2] (+ internals) for normalized inputs."""
    B, T, N, _ = obs.shape
    k = keys(p, app)
    dK = k.shape[-1]
    S = np.zeros((B, D_V, dK))
    Pm = np.broadcast_to(np.eye(D_V), (B, D_V, D_V)).copy()
    Ld = structured_dyn(p)
    preds = np.zeros((B, T, N, 2))
    internals = {"r": np.zeros((B, T, N, D_V)), "write": np.zeros((B, T, N, D_V)),
                 "surprise": np.zeros((B, T, N)), "beta": np.zeros((B, T, N))}
    if p.learned_gain:
        Kp, Kq, Kz = p.K_p, p.K_q, p.K_z
    else:
        Kp, Kq, Kz = np.eye(2), np.zeros((2, 2)), np.zeros((2, 2))
    for t in range(T):
        if t > 0:
            A = dt[:, t, None, None] * Ld[None]
            if frame == "ego":
                A = A.copy()
                A[:, P, PI] = -(u[:, t] @ p.T_ego.T)
            K = expm(A)
            if p.frame_form:
                Pm = K @ Pm
            else:
                S = K @ S
        r = np.einsum("bvw,bwk,bnk->bnv", Pm, S, k) if p.frame_form else np.einsum("bvk,bnk->bnv", S, k)
        p_hat = r[..., P]
        pred = readout_h(p_hat, kind)
        preds[:, t] = pred
        e_pos = fold_jac(p_hat, kind) * obs_diff(obs[:, t], pred, kind)
        pi_c = np.clip(r[..., PI], 0, 1)
        nu = 1 - pi_c
        tau_c = np.maximum(r[..., TAU], 0)
        s = np.linalg.norm(e_pos, axis=-1) / (softplus(p.c0) + 1e-4 + softplus(p.c1) * tau_c)
        beta = np.full_like(s, p.beta_fixed) if p.beta_fixed is not None else nu + (1 - nu) * sigmoid(p.w_s * s + p.w_b)
        g_p = pi_c[..., None] * (e_pos @ Kp.T) + nu[..., None] * e_pos
        g_q = pi_c[..., None] * (e_pos @ Kq.T)
        g_z = pi_c[..., None] * (e_pos @ Kz.T)
        book = np.stack([1 - r[..., PI], -r[..., TAU]], -1)
        dv = np.concatenate([book, beta[..., None] * np.concatenate([g_p, g_q, g_z], -1)], -1)
        dv = dv * vis[:, t, :, None]
        if p.frame_form:
            dv = np.linalg.solve(Pm, dv.transpose(0, 2, 1)).transpose(0, 2, 1)
        S = S + np.einsum("bnv,bnk->bvk", dv, k)
        internals["r"][:, t], internals["write"][:, t] = r, dv
        internals["surprise"][:, t], internals["beta"][:, t] = s * vis[:, t], beta
    return (preds, internals) if return_internals else preds


def ideal_params(app_dim=16, heads=4, d_k=32, transition="ours"):
    """Hand-set parameters that make 'ours' an exact alpha-beta filter on
    noise-free constant-velocity worlds with orthonormal appearance codes:
    exact shear B = I, no damping, K_p = I, K_q = I (dt = 1), beta = 1."""
    W = np.zeros((heads * d_k, app_dim))
    for h in range(heads):
        W[h * d_k:h * d_k + app_dim, :] = np.eye(app_dim)
    return AssocParams(transition=transition, W_key=W, heads=heads, K_q=np.eye(2),
                       d_q=np.full(2, -40.0), beta_fixed=1.0)
