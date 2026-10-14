"""BF-Delta and every comparison arm as one backend-agnostic recurrence.

All functions accept numpy arrays or torch tensors (``xp`` is inferred), so
the exact same code is unit-tested in float64 numpy on CPU and run in float32
torch on GPU.  Shapes: S [B,D,dk], n [B,dk], k [B,dk] (unit norm), r [B,D]
standardized residual (0 where unobserved), o [B,D] observation mask (float),
w [B] write flag (0 on the splice/teleport transition).

Generative model (docs/V6_BF_DELTA.md): r = M k + eps, eps ~ N(0, sbar I),
prior M k ~ N(0, (1 - sbar) I) i.e. prior precision n0 = sbar / (1 - sbar) in
noise units, and with per-write hazard pi the value along k is redrawn from
the prior.  One exact step of the posterior mean, collapsed to one Gaussian:

  m = S k ; e = r - m ; c = n.(k*k) ; tau = 1 / (1 + c)
  v = sbar (1 + 1/c) ; ell = |e|^2/(2v) - |r|^2/2 + (D/2) log v
  gamma = sigmoid(ell + logit pi)                    (no gradient)
  A8  (local):  S += [(1-gamma) tau e + gamma (rho r - m)] k^T
                n += (k*k) * (1 - gamma (n - n0))
  A8g (global): S = (1-gamma) S (I - tau k k^T) + [(1-gamma) tau + gamma rho] r k^T
                n = (1-gamma) n + gamma n0 + k*k
with rho = 1 - sbar.  A8g is Gated DeltaNet with decay 1-gamma and write tau.
"""
from __future__ import annotations

import math

import numpy as np

CLOSED_FORM = ("A0", "A1", "A1BF", "A6", "A6L", "A7", "A8", "A8g", "OC")
LEARNED = ("A2", "A2P", "A3", "A4")


def _xp(a):
    if isinstance(a, np.ndarray):
        return np
    import torch
    return torch


def _sum(x, axis):
    return x.sum(axis)


def sigmoid(xp, x):
    return 1.0 / (1.0 + xp.exp(-x))


def logit(p):
    return math.log(p) - math.log1p(-p)


def init_state(xp, like, B, D, dk, sbar):
    n0 = sbar / (1.0 - sbar)
    if xp is np:
        return np.zeros((B, D, dk), dtype=like.dtype), np.full((B, dk), n0, dtype=like.dtype)
    return (xp.zeros((B, D, dk), dtype=like.dtype, device=like.device),
            xp.full((B, dk), n0, dtype=like.dtype, device=like.device))


def _outer(xp, u, k):
    return xp.einsum("bd,bk->bdk", u, k)


def bf_step(S, n, k, r, o, w, *, sbar, logit_pi, arm, beta0=None, lam=None, gate=None, ell_clip=50.0):
    """One write.  Returns (S, n, info) with the *pre-write* prediction m.

    gate: A2 -> (alpha [B], beta [B]); A3 -> beta [B]; A4 -> (a, b) scalars;
          A2P -> callable(features [B,3]) -> (alpha, beta, s) each [B].
    """
    xp = _xp(S)
    n0 = sbar / (1.0 - sbar)
    rho = 1.0 - sbar
    kk = k * k
    m = xp.einsum("bdk,bk->bd", S, k) * o
    e = r - m
    Dt = xp.clip(_sum(o, -1), 1.0, None)
    if arm in ("A1BF",):
        c = xp.clip(xp.zeros_like(Dt) + (1.0 / beta0 - 1.0), 1e-6, None)
        tau = xp.zeros_like(Dt) + beta0
    else:
        c = _sum(n * kk, -1)
        tau = 1.0 / (1.0 + c)
    vb = sbar * (1.0 + 1.0 / xp.clip(c, 1e-12, None))
    ell = _sum(e * e, -1) / (2.0 * vb) - _sum(r * r, -1) / 2.0 + 0.5 * Dt * xp.log(vb)
    ell = xp.clip(ell, -ell_clip, ell_clip)
    if arm in ("A1BF", "A7", "A8", "A8g"):
        g = sigmoid(xp, ell + logit_pi)
        if xp is not np:
            g = g.detach()
    else:
        g = xp.zeros_like(ell)
    G, T, W = g[:, None], tau[:, None], w[:, None]
    info = {"m": m, "e": e, "ell": ell, "g": g, "tau": tau, "c": c, "Dt": Dt}

    if arm == "A0":
        return S, n, info
    if arm == "A1":
        S = S + _outer(xp, W * beta0 * o * e, k)
    elif arm in ("A6", "OC"):
        S = S + _outer(xp, W * T * o * e, k)
        n = n + W * kk
    elif arm == "A6L":
        S = S + _outer(xp, W * T * o * e, k)
        n = W * (lam * n + (1.0 - lam) * n0 + kk) + (1.0 - W) * n
    elif arm in ("A8", "A1BF"):
        u = o * ((1.0 - G) * T * e + G * (rho * r - m))
        S = S + _outer(xp, W * u, k)
        if arm == "A8":
            n = n + W * kk * (1.0 - G * (n - n0))
    elif arm == "A8g":
        Snew = (1.0 - G)[..., None] * (S + _outer(xp, o * T * e, k)) + _outer(xp, G * rho * r * o, k)
        S = W[..., None] * Snew + (1.0 - W)[..., None] * S
        n = W * ((1.0 - G) * n + G * n0 + kk) + (1.0 - W) * n
    elif arm == "A7":
        S = S + _outer(xp, W * G * o * e, k)
        n = n + W * kk * (1.0 - G * (n - n0))
    elif arm == "A2":
        alpha, beta = gate
        a, b = alpha[:, None], beta[:, None]
        Snew = a[..., None] * S + _outer(xp, b * o * (r - a * m), k)
        S = W[..., None] * Snew + (1.0 - W)[..., None] * S
    elif arm == "A3":
        S = S + _outer(xp, W * gate[:, None] * o * e, k)
    elif arm == "A4":
        a, b = gate
        beta = sigmoid(xp, a * _sum(e * e, -1) / Dt + b)
        info["beta"] = beta
        S = S + _outer(xp, W * beta[:, None] * o * e, k)
    elif arm == "A2P":
        feats = xp.stack([_sum(e * e, -1) / Dt, _sum(r * r, -1) / Dt, xp.log(xp.clip(c, 1e-12, None))], -1)
        alpha, beta, s = gate(feats)
        a, b, s_ = alpha[:, None], beta[:, None], s[:, None]
        Snew = a[..., None] * S + _outer(xp, b * o * (s_ * r - a * m), k)
        S = W[..., None] * Snew + (1.0 - W)[..., None] * S
        n = W * (a * n + (1.0 - a) * n0 + kk) + (1.0 - W) * n
        info["alpha"] = alpha
    else:
        raise ValueError(arm)
    return S, n, info


def scan(arm, K, R, O, W, *, sbar, pi=1e-3, beta0=None, lam=None, gate=None, reset=None,
         collect=("m", "ell", "g", "tau", "c"), state_at=None, ell_clip=50.0, S0=None, n0=None):
    """Run one arm over a batch of sequences.

    K [B,T,dk], R [B,T,D], O [B,T,D], W [B,T].  ``gate`` for A2/A3 is a
    per-step array ([B,T,2] / [B,T]); for A4 a scalar pair; for A2P a callable.
    ``reset`` [B,T] (Oracle C): zero the state *before* step t.
    ``state_at``: list of step indices whose *pre-write* state S is returned.
    """
    xp = _xp(R)
    B, T, D = R.shape
    dk = K.shape[-1]
    if S0 is None:
        S, n = init_state(xp, R, B, D, dk, sbar)
    else:
        S, n = S0, n0
    lp = logit(pi)
    nz = sbar / (1.0 - sbar)
    outs = {c: [] for c in collect}
    states = {}
    want = set(state_at or [])
    for t in range(T):
        if reset is not None:
            rs = reset[:, t]
            S = S * (1.0 - rs)[:, None, None]
            n = n * (1.0 - rs)[:, None] + rs[:, None] * nz
        if t in want:
            states[t] = S
        if arm == "A2":
            gt = (gate[:, t, 0], gate[:, t, 1])
        elif arm == "A3":
            gt = gate[:, t]
        else:
            gt = gate
        S, n, info = bf_step(S, n, K[:, t], R[:, t], O[:, t], W[:, t], sbar=sbar, logit_pi=lp, arm=arm,
                             beta0=beta0, lam=lam, gate=gt, ell_clip=ell_clip)
        for c in collect:
            if c in info:
                outs[c].append(info[c])
    res = {c: xp.stack(v, 1) for c, v in outs.items() if v}
    res["S_final"], res["n_final"] = S, n
    res["states"] = states
    return res


# -------------------------------------------------- A6 chunked (training phi)
def a6_counts(K, W, sbar):
    """c_t = n0 + sum_k N_t[k] k_t[k]^2 with N_t the exclusive prefix count."""
    xp = _xp(K)
    kk = K * K
    wk = kk * W[..., None]
    if xp is np:
        cum = np.cumsum(wk, 1) - wk
    else:
        cum = xp.cumsum(wk, 1) - wk
    n0 = sbar / (1.0 - sbar)
    return n0 * _sum(kk, -1) + _sum(cum * kk, -1)


def a6_chunk(K, R, O, W, sbar, chunk=64):
    """Exact chunkwise A6 (gamma = 0): per-row delta rule with beta[t,d] =
    w_t tau_t o[t,d], tau from the causal prefix counts (no gradient through
    tau).  Returns the pre-write predictions M [B,T,D] (masked by O).

    Within a chunk with start state S0: (I + tril(diag(beta) K K^T, -1)) U =
    beta (V - K S0^T); M = K S0^T + tril(K K^T, -1) U; S0 += U^T K."""
    xp = _xp(R)
    B, T, D = R.shape
    dk = K.shape[-1]
    c = a6_counts(K, W, sbar)
    tau = 1.0 / (1.0 + c)
    if xp is not np:
        tau = tau.detach()
    beta = (W * tau)[..., None] * O                       # [B,T,D]
    if xp is np:
        S0 = np.zeros((B, D, dk), dtype=R.dtype)
    else:
        S0 = xp.zeros((B, D, dk), dtype=R.dtype, device=R.device)
    outs = []
    for s in range(0, T, chunk):
        Kc, Rc, Bc = K[:, s:s + chunk], R[:, s:s + chunk], beta[:, s:s + chunk]
        C = Kc.shape[1]
        Gm = xp.einsum("bik,bjk->bij", Kc, Kc)
        if xp is np:
            low = np.tril(np.ones((C, C), dtype=R.dtype), -1)
            eye = np.eye(C, dtype=R.dtype)
        else:
            low = xp.tril(xp.ones((C, C), dtype=R.dtype, device=R.device), -1)
            eye = xp.eye(C, dtype=R.dtype, device=R.device)
        Gl = Gm * low
        bT = xp.einsum("bid->bdi", Bc)                     # [B,D,C]
        P = xp.einsum("bik,bdk->bdi", Kc, S0)              # [B,D,C]
        A = bT[..., :, None] * Gl[:, None]                 # [B,D,C,C]
        rhs = bT * (xp.einsum("bid->bdi", Rc) - P)
        Mat = eye + A
        if xp is np:
            U = np.linalg.solve(Mat, rhs[..., None])[..., 0]
        else:
            U = xp.linalg.solve_triangular(Mat, rhs[..., None], upper=False, unitriangular=True)[..., 0]
        Mc = P + xp.einsum("bij,bdj->bdi", Gl, U)
        outs.append(xp.einsum("bdi->bid", Mc))
        S0 = S0 + xp.einsum("bdi,bik->bdk", U, Kc)
    M = xp.concatenate(outs, 1) if xp is np else xp.cat(outs, 1)
    return M * O


# --------------------------------------------------------------- ML-II
def ml2_loglik(K, R, O, W, sbar, mask=None):
    """Sequential log evidence of A6 (gamma = 0) at one sbar: sum over written
    steps of log N(r_t; m_t, v_t I) on observed dims, v_t = sbar (1 + 1/c_t)."""
    xp = _xp(R)
    res = scan("A6", K, R, O, W, sbar=sbar, collect=("m", "c", "Dt"))
    e = (R - res["m"]) * O
    v = sbar * (1.0 + 1.0 / res["c"])
    Dt = _sum(O, -1)
    ll = -0.5 * (_sum(e * e, -1) / v + Dt * xp.log(2 * math.pi * v))
    wm = W if mask is None else W * mask
    return float(_sum(ll * wm, None) if xp is np else (ll * wm).sum())


def ml2_select(K, R, O, W, grid):
    lls = {float(s): ml2_loglik(K, R, O, W, float(s)) for s in grid}
    best = max(lls, key=lls.get)
    return best, lls


# ----------------------------------------------------------- references
def nassar_reference(x, sbar, pi):
    """Independent scalar reference for U5: one key, one dim, Gaussian change
    distribution N(0, 1 - sbar) and noise sbar.  Returns per-step learning
    rate beta_t, change posterior gamma_t and pre-observation mean."""
    n0 = sbar / (1.0 - sbar)
    rho = 1.0 - sbar
    mu, cnt = 0.0, n0
    out = []
    for xt in x:
        var_pred = sbar + sbar / cnt          # predictive variance under "no change"
        lp_mem = -0.5 * ((xt - mu) ** 2 / var_pred + math.log(2 * math.pi * var_pred))
        lp_prior = -0.5 * (xt ** 2 + math.log(2 * math.pi))
        lo = max(-50.0, min(50.0, lp_prior - lp_mem))
        gamma = 1.0 / (1.0 + math.exp(-(lo + logit(pi))))
        tau = 1.0 / (1.0 + cnt)
        new_mu = (1 - gamma) * (mu + tau * (xt - mu)) + gamma * rho * xt
        beta = gamma + (1 - gamma) * tau
        out.append((beta, gamma, mu))
        mu = new_mu
        cnt = cnt + 1 - gamma * (cnt - n0)
    return out
