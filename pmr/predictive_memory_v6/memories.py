"""Memory variants for V6 (torch).  One interface, many state-update rules.

Every model exposes ``rollout(batch) -> dict`` with PRE-WRITE predictions for
all objects at every step (``pred[:, t]`` is the belief about step t before the
token at t is written).  Variants differ only in how the persistent state is
transported between steps and written; inputs, losses and evaluation are
shared.  Persistent per-episode state is reported by ``state_scalars``.

Associative ("ours") family  -- state S in R^{d_v x d_K}, d_v = 8 value channels
``[pi, tau, p_x, p_y, q_x, q_y, z1, z2]`` (presence, age, position, velocity,
two free channels), d_K = H * d_k (4 heads x 32, concatenated unit keys / sqrt H).

    predict   S <- K S,  K = expm(dt L_dyn + L_ego(u))         (value side)
    compare   r = S k,  y_hat = h(p_hat),  e = [1-pi_hat, -tau_hat, J (y (-) y_hat)]
    correct   S <- S + [e_book ; beta * G_dyn J e_pos] k^T

``static_delta`` (L_dyn = 0), ``vo_rotation`` (fixed rotations, no shear),
``tpa_readout`` (static memory + generic time-aware MLP readout), ``ours``
(structured learned generator), ``ours_GCt`` (G = C^T: plain SGD step, no
learned value-space gain), ``ours_dense`` (unconstrained generator),
``ours_frame`` (lemma-3 frame form, numerically equal to ``ours``),
``ours_ttt`` / ``oracle_L`` / ``amortized_dL`` (H6, velocity block adapted).

Key-side / value-space-delta family: ``gdn``, ``gdn_neg`` (beta in [0,2]),
``kdn_diag`` (per-dimension key decay), ``m2rnn`` (value-side tanh transition).
Others: ``fifo`` (bounded event retrieval), ``gru`` (global recurrent state),
``slot_known_id`` (per-object GRU, structural reference), ``lact`` (LaCT fast
weights, official update rule, dot-product inner loss), ``flowm_token``
(FloWM-style velocity-channel grid, structural reference, torus only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

PI, TAU, P, Q, Z = 0, 1, slice(2, 4), slice(4, 6), slice(6, 8)
D_V = 8
J2 = torch.tensor([[0.0, -1.0], [1.0, 0.0]])


# --------------------------------------------------------------- batches
@dataclass
class TBatch:
    app: torch.Tensor           # [B,N,A] appearance/identity code
    obs: torch.Tensor           # [B,T,N,2] normalized observed coordinate
    vis: torch.Tensor           # [B,T,N] float 0/1
    u: torch.Tensor             # [B,T,2] normalized ego shift (zeros in allo frame)
    dt: torch.Tensor            # [B,T]
    kind: str                   # torus | box | plane
    frame: str = "allo"
    fov: Optional[torch.Tensor] = None     # [B,T,2] window origin (normalized), torus window mode
    window: float = 1.0                    # window size / world size
    drift: Optional[torch.Tensor] = None   # [B,2] (omega, gamma) for oracle_L

    @property
    def shape(self):
        return self.obs.shape[:3]


def h_readout(z, kind):
    if kind == "torus":
        return torch.remainder(z, 1.0)
    if kind == "box":
        return 1.0 - (torch.remainder(z, 2.0) - 1.0).abs()
    return z


def fold_jacobian(z, kind):
    if kind != "box":
        return torch.ones_like(z)
    return torch.where(torch.remainder(z, 2.0) < 1.0, 1.0, -1.0).to(z.dtype)


def obs_diff(y, y_hat, kind):
    d = y - y_hat
    if kind == "torus":
        d = d - torch.round(d)
    return d


def pos_embed(y, kind):
    if kind == "torus":
        a = 2 * math.pi * y
        return torch.cat([torch.cos(a), torch.sin(a)], -1)
    return y


def pos_decode(z, kind):
    if kind == "torus":
        c, s = z[..., :2], z[..., 2:4]
        return torch.remainder(torch.atan2(s, c) / (2 * math.pi), 1.0)
    return z


def emb_dim(kind):
    return 4 if kind == "torus" else 2


def inv_softplus(x):
    return math.log(math.expm1(x))


# ------------------------------------------------------------------ base
class Memory(nn.Module):
    name = "base"
    has_presence = False

    def init_state(self, b: TBatch):
        raise NotImplementedError

    def advance(self, state, b: TBatch, t):
        return state

    def read(self, state, b: TBatch, t):
        """-> dict(pred [B,N,2] readout coords, optional presence/unfolded)."""
        raise NotImplementedError

    def write(self, state, b: TBatch, t, rd):
        """-> (state, surprise [B,N])"""
        raise NotImplementedError

    def state_scalars(self, n_objects: int) -> int:
        raise NotImplementedError

    def rollout(self, b: TBatch, reset_at=None, keep=()):
        B, T, N = b.shape
        state = self.init_state(b)
        preds, pres, surp = [], [], []
        kept = {k: [] for k in keep}
        for t in range(T):
            if reset_at is not None and t == reset_at:
                state = self.init_state(b)
            if t > 0:
                state = self.advance(state, b, t)
            rd = self.read(state, b, t)
            state, s = self.write(state, b, t, rd)
            preds.append(rd["pred"])
            pres.append(rd.get("presence"))
            surp.append(s)
            for k in keep:
                kept[k].append(rd.get(k))
        out = {"pred": torch.stack(preds, 1), "surprise": torch.stack(surp, 1)}
        out["presence"] = torch.stack(pres, 1) if pres[0] is not None else None
        for k in keep:
            out[k] = torch.stack(kept[k], 1) if kept[k][0] is not None else None
        return out


def unit_keys(proj: nn.Linear, app, heads):
    B, N, _ = app.shape
    k = proj(app).view(B, N, heads, -1)
    k = F.normalize(k, dim=-1, eps=1e-6)
    return k.reshape(B, N, -1) / math.sqrt(heads)


# ===================================================== associative family
class AssocMemory(Memory):
    """predict (value side) -> compare -> correct.  See module docstring."""

    has_presence = True
    TRANSITIONS = ("static", "ours", "dense", "vo")

    def __init__(self, app_dim=16, heads=4, d_k=32, transition="ours", learned_gain=True,
                 tpa=False, frame_form=False, adapt=None, vo_freqs=(2 * math.pi / 256, 2 * math.pi / 64),
                 beta_fixed=None, name="ours"):
        super().__init__()
        if transition not in self.TRANSITIONS:
            raise ValueError(transition)
        self.name, self.heads, self.d_K = name, heads, heads * d_k
        self.transition, self.learned_gain, self.tpa = transition, learned_gain, tpa
        self.frame_form, self.adapt, self.beta_fixed = frame_form, adapt, beta_fixed
        self.key = nn.Linear(app_dim, heads * d_k, bias=False)
        # generator parameters (structured)
        self.B_shear = nn.Parameter(torch.eye(2))
        self.w_p = nn.Parameter(torch.zeros(()))
        self.w_q = nn.Parameter(torch.zeros(()))
        self.d_q = nn.Parameter(torch.full((2,), inv_softplus(1e-3)))
        self.w_z = nn.Parameter(torch.zeros(()))
        self.d_z = nn.Parameter(torch.full((2,), inv_softplus(1e-2)))
        self.T_ego = nn.Parameter(torch.eye(2))
        if transition == "dense":
            self.M_dense = nn.Parameter(self._structured_dyn()[2:, 2:].detach().clone())
        self.register_buffer("vo_freqs", torch.tensor(vo_freqs, dtype=torch.float32))
        # value-space gain on the position residual (rows p, q, z)
        self.K_p = nn.Parameter(torch.eye(2))
        self.K_q = nn.Parameter(0.5 * torch.eye(2))
        self.K_z = nn.Parameter(torch.zeros(2, 2))
        # write gate beta = nu + (1 - nu) sigmoid(w_s s + w_b)
        self.w_s = nn.Parameter(torch.zeros(()))
        self.w_b = nn.Parameter(torch.tensor(3.0))
        self.c0 = nn.Parameter(torch.tensor(inv_softplus(1e-2)))
        self.c1 = nn.Parameter(torch.tensor(inv_softplus(1e-3)))
        if tpa:
            self.tpa_mlp = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                         nn.Linear(64, 2))
            nn.init.zeros_(self.tpa_mlp[-1].weight)
            nn.init.zeros_(self.tpa_mlp[-1].bias)
        if adapt == "amortized":
            self.amort = nn.GRUCell(4, 16)
            self.amort_out = nn.Linear(16, 2)
            nn.init.zeros_(self.amort_out.weight)
            nn.init.zeros_(self.amort_out.bias)
        self.ttt_lr, self.ttt_buffer = 0.3, 8

    # ---------------------------------------------------------- generator
    def _structured_dyn(self, delta=None):
        """L_dyn [8,8] (or [B,8,8] when an episode-level velocity delta is given)."""
        dev, dt_ = self.B_shear.device, self.B_shear.dtype
        J = J2.to(dev, dt_)
        L = torch.zeros(D_V, D_V, device=dev, dtype=dt_)
        L[TAU, PI] = 1.0
        if self.transition == "static":
            Lq = torch.zeros(2, 2, device=dev, dtype=dt_)
            Lpq = torch.zeros(2, 2, device=dev, dtype=dt_)
            Lpp = torch.zeros(2, 2, device=dev, dtype=dt_)
            Lz = torch.zeros(2, 2, device=dev, dtype=dt_)
        elif self.transition == "vo":
            f1, f2 = self.vo_freqs[0], self.vo_freqs[1]
            # VO-RoPE-style fixed rotation of the (p, q) pairs and of z; no shear.
            L[2, 4], L[4, 2], L[3, 5], L[5, 3] = f1, -f1, f1, -f1
            L[6:8, 6:8] = f2 * J
            return L
        else:
            Lpp = self.w_p * J
            Lpq = self.B_shear
            Lq = self.w_q * J - torch.diag(F.softplus(self.d_q))
            Lz = self.w_z * J - torch.diag(F.softplus(self.d_z))
        L[P, P], L[P, Q], L[Q, Q], L[Z, Z] = Lpp, Lpq, Lq, Lz
        if self.transition == "dense" and hasattr(self, "M_dense"):
            L[2:, 2:] = self.M_dense
        if delta is not None:  # [B,2] = (d_omega, d_gamma) on the velocity block
            L = L.unsqueeze(0).repeat(delta.shape[0], 1, 1).to(delta.dtype)
            dq = delta[:, 0, None, None] * J - delta[:, 1, None, None] * torch.eye(2, device=dev, dtype=dt_)
            L[:, Q, Q] = L[:, Q, Q] + dq
        return L

    def _oracle_dyn(self, drift):
        """Exact generator for v_{t+1} = (1-g) R(w) v_t, x_{t+1} = x_t + v_t."""
        Bn = drift.shape[0]
        dev, dt_ = drift.device, self.B_shear.dtype
        drift = drift.to(dt_)
        J = J2.to(dev, dt_)
        eye = torch.eye(2, device=dev, dtype=dt_)
        Lq = torch.log1p(-drift[:, 1])[:, None, None] * eye + drift[:, 0, None, None] * J
        # x += v over one step needs B * phi(Lq) = I with phi(M) = sum M^k/(k+1)!
        aug = torch.zeros(Bn, 4, 4, device=dev, dtype=dt_)
        aug[:, :2, :2] = Lq
        aug[:, :2, 2:] = eye
        phi = torch.linalg.matrix_exp(aug)[:, :2, 2:]
        L = self._structured_dyn().unsqueeze(0).repeat(Bn, 1, 1)
        L[:, P, P] = 0.0
        L[:, Q, Q] = Lq
        L[:, P, Q] = torch.linalg.inv(phi)
        return L

    def generator(self, b: TBatch, t, delta=None):
        if self.adapt == "oracle":
            Ld = self._oracle_dyn(b.drift)
        else:
            Ld = self._structured_dyn(delta)
        dt = b.dt[:, t]
        A = (dt[:, None, None] * Ld) if Ld.dim() == 3 else dt[:, None, None] * Ld.unsqueeze(0)
        if b.frame == "ego":
            ego = torch.zeros_like(A)
            ego[:, P, PI] = -(b.u[:, t] @ self.T_ego.T)
            A = A + ego
        return A

    # ------------------------------------------------------------- state
    def init_state(self, b: TBatch):
        B, T, N = b.shape
        z = dict(device=b.obs.device, dtype=b.obs.dtype)
        st = {"S": torch.zeros(B, D_V, self.d_K, **z), "k": unit_keys(self.key, b.app, self.heads)}
        if self.frame_form:
            st["P"] = torch.eye(D_V, **z).expand(B, D_V, D_V).clone()
        if self.adapt in ("ttt", "amortized"):
            st["delta"] = torch.zeros(B, 2, **z)
        if self.adapt == "amortized":
            st["h"] = torch.zeros(B, 16, **z)
        if self.adapt == "ttt":
            R = self.ttt_buffer
            st["last_r"] = torch.zeros(B, N, D_V, **z)    # readout at last sighting
            st["last_t"] = torch.full((B, N), -1, device=b.obs.device, dtype=torch.long)
            st["buf_m"] = torch.zeros(B, R, D_V, **z)     # recent reveal events
            st["buf_y"] = torch.zeros(B, R, 2, **z)
            st["buf_tau"] = torch.zeros(B, R, **z)        # hidden duration (time)
            st["buf_valid"] = torch.zeros(B, R, **z)
            st["buf_ptr"] = torch.zeros(B, dtype=torch.long, device=b.obs.device)
        return st

    def advance(self, st, b, t):
        K = torch.linalg.matrix_exp(self.generator(b, t, st.get("delta")))
        st = dict(st)
        if self.frame_form:
            st["P"] = K @ st["P"]
        else:
            st["S"] = K @ st["S"]
        return st

    def memory_readout(self, st):
        """r = S k for all objects -> [B,N,8]."""
        if self.frame_form:
            return torch.einsum("bvw,bwk,bnk->bnv", st["P"], st["S"], st["k"])
        return torch.einsum("bvk,bnk->bnv", st["S"], st["k"])

    def read(self, st, b, t):
        r = self.memory_readout(st)
        p_hat = r[..., P]
        if self.tpa:
            feat = torch.cat([r[..., Q], r[..., TAU:TAU + 1], r[..., PI:PI + 1]], -1)
            p_hat = p_hat + self.tpa_mlp(feat)
        return {"pred": h_readout(p_hat, b.kind), "unfolded": p_hat, "r": r,
                "presence": r[..., PI]}

    def gains(self):
        dev = self.K_p.device
        if not self.learned_gain:
            return torch.eye(2, device=dev), torch.zeros(2, 2, device=dev), torch.zeros(2, 2, device=dev)
        return self.K_p, self.K_q, self.K_z

    def write(self, st, b, t, rd):
        y, vis = b.obs[:, t], b.vis[:, t]
        r, p_hat = rd["r"], rd["unfolded"]
        J = fold_jacobian(p_hat, b.kind)
        e_pos = J * obs_diff(y, rd["pred"], b.kind)                  # [B,N,2]
        pi_c = r[..., PI].clamp(0.0, 1.0)
        nu = 1.0 - pi_c
        tau_c = r[..., TAU].clamp(min=0.0)
        s = e_pos.norm(dim=-1) / (F.softplus(self.c0) + 1e-4 + F.softplus(self.c1) * tau_c)
        if self.beta_fixed is not None:
            beta = torch.full_like(s, float(self.beta_fixed))
        else:
            beta = nu + (1 - nu) * torch.sigmoid(self.w_s * s + self.w_b)
        Kp, Kq, Kz = self.gains()
        # novelty-mixed gain: brand-new keys are overwritten (G_init = C^T rows),
        # tracked keys use the learned value-space gain (alpha-beta filter).
        g_p = pi_c[..., None] * (e_pos @ Kp.T) + nu[..., None] * e_pos
        g_q = pi_c[..., None] * (e_pos @ Kq.T)
        g_z = pi_c[..., None] * (e_pos @ Kz.T)
        book = torch.stack([1.0 - r[..., PI], -r[..., TAU]], -1)    # exact bookkeeping
        dv = torch.cat([book, beta[..., None] * torch.cat([g_p, g_q, g_z], -1)], -1)  # [B,N,8]
        dv = dv * vis[..., None]
        st = dict(st)
        if self.frame_form:
            dv = torch.linalg.solve(st["P"], dv.transpose(1, 2)).transpose(1, 2)
        st["S"] = st["S"] + torch.einsum("bnv,bnk->bvk", dv, st["k"])
        rd["write"] = dv
        rd["beta"] = beta
        if self.adapt == "amortized":
            cnt = vis.sum(1, keepdim=True)
            feat = torch.cat([(e_pos * vis[..., None]).sum(1) / cnt.clamp(min=1),
                              (s * vis).sum(1, keepdim=True) / cnt.clamp(min=1), cnt / vis.shape[1]], -1)
            st["h"] = self.amort(feat, st["h"])
            st["delta"] = self.amort_out(st["h"])
        if self.adapt == "ttt" and not self.training:
            # H6: test-time adaptation only; training uses the plain 'ours' rule
            st = self._ttt_step(st, b, t, rd)
        return st, s * vis

    # ---------------------------------------------- H6: reveal-error TTT
    def _ttt_step(self, st, b, t, rd):
        """One SGD step on the episode's velocity-block delta, using a ring buffer
        of the last R reveal events: m (readout at last sighting), hidden time,
        and the revealed observation.  Gradients never reach outer parameters."""
        B, _, N = b.shape
        R = self.ttt_buffer
        vis = b.vis[:, t] > 0.5
        gap = t - st["last_t"]
        reveal = vis & (st["last_t"] >= 0) & (gap >= 2)
        if reveal.any():
            csum = torch.cumsum(b.dt, 1)                                         # [B,T]
            lt = st["last_t"].clamp(min=0)
            elapsed = csum[:, t, None] - torch.gather(csum, 1, lt)               # [B,N]
            order = torch.cumsum(reveal.long(), 1) - 1
            slot = (st["buf_ptr"][:, None] + order) % R
            bi = torch.arange(B, device=vis.device)[:, None].expand(B, N)
            for key, val in (("buf_m", st["last_r"].detach()), ("buf_y", b.obs[:, t].detach()),
                             ("buf_tau", elapsed), ("buf_valid", torch.ones_like(elapsed))):
                buf = st[key].clone()
                buf[bi[reveal], slot[reveal]] = val[reveal]
                st[key] = buf
            st["buf_ptr"] = (st["buf_ptr"] + reveal.long().sum(1)) % R
            active = reveal.any(1).float()
            with torch.enable_grad():
                delta = st["delta"].detach().clone().requires_grad_(True)
                L = self._structured_dyn(delta)                                  # [B,8,8]
                A = st["buf_tau"][..., None, None] * L[:, None]                    # [B,R,8,8]
                r = torch.einsum("brvw,brw->brv", torch.linalg.matrix_exp(A), st["buf_m"])
                err = obs_diff(st["buf_y"], h_readout(r[..., P], b.kind), b.kind).pow(2).sum(-1)
                loss = (err * st["buf_valid"] * active[:, None]).sum()
                g, = torch.autograd.grad(loss, delta)
            st["delta"] = (delta - self.ttt_lr * g).detach()
        post = self.memory_readout(st)
        st["last_r"] = torch.where(vis[..., None], post.detach(), st["last_r"])
        st["last_t"] = torch.where(vis, torch.full_like(st["last_t"], t), st["last_t"])
        return st

    def state_scalars(self, n_objects):
        n = D_V * self.d_K
        if self.frame_form:
            n += D_V * D_V
        if self.adapt == "amortized":
            n += 16 + 2
        if self.adapt == "ttt":
            n += 2 + n_objects * (D_V + 1) + self.ttt_buffer * (D_V + 4)
        return n


# ===================================== key-side / value-space-delta family
class KeySideMemory(Memory):
    """GDN, GDN-neg, KDN-diag and M2RNN with a shared token encoder/readout.

    GDN (arXiv 2412.06464 Eq. 10, FLA naive_recurrent_gated_delta_rule):
        S_t = S_{t-1}(alpha (I - beta k k^T)) + beta v k^T = alpha S + beta (v - alpha S k) k^T
    with alpha = exp(g), g = -exp(A_log) softplus(a + dt_bias) * dt and
    beta = sigmoid (x2 for negative eigenvalues).  All visible tokens of one
    step are written simultaneously against the same decayed state.
    """

    has_presence = True

    def __init__(self, app_dim=16, heads=4, d_k=32, kind_name="gdn", geom="torus", name=None):
        super().__init__()
        self.name = name or kind_name
        self.variant, self.heads, self.d_K = kind_name, heads, heads * d_k
        e = emb_dim(geom)
        self.key = nn.Linear(app_dim, heads * d_k, bias=False)
        self.val = nn.Linear(e + app_dim, D_V)
        self.beta = nn.Linear(e + app_dim, 1)
        self.read_pos = nn.Linear(D_V, e)
        self.read_pres = nn.Linear(D_V, 1)
        A = torch.empty(heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt0 = torch.exp(torch.empty(heads).uniform_(math.log(1e-3), math.log(0.1)))
        self.dt_bias = nn.Parameter(dt0 + torch.log(-torch.expm1(-dt0)))  # inv_softplus(dt0)
        self.a0 = nn.Parameter(torch.zeros(heads))
        if kind_name == "kdn_diag":
            self.lam = nn.Parameter(torch.full((heads * d_k,), 4.0))
        if kind_name == "m2rnn":
            self.W_m = nn.Parameter(torch.eye(D_V))

    def init_state(self, b):
        B = b.shape[0]
        return {"S": torch.zeros(B, D_V, self.d_K, device=b.obs.device, dtype=b.obs.dtype),
                "k": unit_keys(self.key, b.app, self.heads)}

    def _decay(self, b, t):
        g = -torch.exp(self.A_log) * F.softplus(self.a0 + self.dt_bias)      # [H]
        g = g[None, :] * b.dt[:, t, None]                                    # [B,H]
        return torch.exp(g).repeat_interleave(self.d_K // self.heads, dim=1)  # [B,d_K]

    def advance(self, st, b, t):
        st = dict(st)
        if self.variant in ("gdn", "gdn_neg"):
            st["S"] = st["S"] * self._decay(b, t)[:, None, :]
        elif self.variant == "kdn_diag":
            lam = torch.sigmoid(self.lam)[None, :] ** b.dt[:, t, None]
            st["S"] = st["S"] * lam[:, None, :]
        elif self.variant == "m2rnn":
            st["S"] = torch.tanh(torch.einsum("vw,bwk->bvk", self.W_m, st["S"]))
        return st

    def read(self, st, b, t):
        r = torch.einsum("bvk,bnk->bnv", st["S"], st["k"])
        z = self.read_pos(r)
        if b.kind == "torus":
            pred = pos_decode(z, "torus")
        else:
            pred = h_readout(z, b.kind)
        return {"pred": pred, "r": r, "presence": torch.sigmoid(self.read_pres(r))[..., 0]}

    def write(self, st, b, t, rd):
        y, vis = b.obs[:, t], b.vis[:, t]
        x = torch.cat([pos_embed(y, b.kind), b.app], -1)
        v = self.val(x)
        beta = torch.sigmoid(self.beta(x))[..., 0]
        if self.variant == "gdn_neg":
            beta = 2.0 * beta
        st = dict(st)
        if self.variant == "m2rnn":
            dv = v
        else:
            dv = beta[..., None] * (v - rd["r"])
        dv = dv * vis[..., None]
        st["S"] = st["S"] + torch.einsum("bnv,bnk->bvk", dv, st["k"])
        err = obs_diff(y, rd["pred"], b.kind).norm(dim=-1)
        return st, err * vis

    def state_scalars(self, n_objects):
        return D_V * self.d_K


# ================================================================= others
class FIFOMemory(Memory):
    """Bounded FIFO of observed object events + learned attention reader."""

    def __init__(self, app_dim=16, geom="torus", budget=1024, name="fifo"):
        super().__init__()
        self.name = name
        self.e = emb_dim(geom)
        self.slot = app_dim + 2 + 1 + 1        # appearance, position, time, valid
        self.M = budget // self.slot
        self.q = nn.Linear(app_dim, 32, bias=False)
        self.k = nn.Linear(app_dim, 32, bias=False)
        self.w_age = nn.Parameter(torch.tensor(-0.05))
        self.out = nn.Sequential(nn.Linear(self.e + 1, 64), nn.SiLU(), nn.Linear(64, self.e))

    def init_state(self, b):
        B = b.shape[0]
        z = dict(device=b.obs.device, dtype=b.obs.dtype)
        A = b.app.shape[-1]
        return {"a": torch.zeros(B, self.M, A, **z), "y": torch.zeros(B, self.M, 2, **z),
                "tw": torch.zeros(B, self.M, **z), "valid": torch.zeros(B, self.M, **z),
                "ptr": torch.zeros(B, dtype=torch.long, device=b.obs.device), "clock": torch.zeros(B, **z)}

    def advance(self, st, b, t):
        st = dict(st)
        st["clock"] = st["clock"] + b.dt[:, t]
        return st

    def read(self, st, b, t):
        age = (st["clock"][:, None] - st["tw"])
        score = torch.einsum("bnd,bmd->bnm", self.q(b.app), self.k(st["a"])) / math.sqrt(32)
        score = score + self.w_age * age[:, None, :]
        score = score.masked_fill(st["valid"][:, None, :] < 0.5, -1e9)
        w = torch.softmax(score, -1) * (st["valid"].sum(-1) > 0).to(score.dtype)[:, None, None]
        feat = torch.cat([pos_embed(st["y"], b.kind), age[..., None] / 64.0], -1)
        ret = torch.einsum("bnm,bmf->bnf", w, feat)
        z = ret[..., :self.e] + self.out(ret)
        pred = pos_decode(z, b.kind) if b.kind == "torus" else h_readout(z, b.kind)
        return {"pred": pred}

    def write(self, st, b, t, rd):
        B, _, N = b.shape
        vis = b.vis[:, t] > 0.5
        st = {k: v.clone() for k, v in st.items()}
        order = torch.cumsum(vis.long(), 1) - 1                      # slot offset per visible token
        idx = (st["ptr"][:, None] + order) % self.M
        bi = torch.arange(B, device=idx.device)[:, None].expand(B, N)
        sel = vis
        st["a"][bi[sel], idx[sel]] = b.app[sel]
        st["y"][bi[sel], idx[sel]] = b.obs[:, t][sel]
        st["tw"][bi[sel], idx[sel]] = st["clock"][:, None].expand(B, N)[sel]
        st["valid"][bi[sel], idx[sel]] = 1.0
        st["ptr"] = (st["ptr"] + vis.long().sum(1)) % self.M
        err = obs_diff(b.obs[:, t], rd["pred"], b.kind).norm(dim=-1)
        return st, err * b.vis[:, t]

    def state_scalars(self, n_objects):
        return self.M * self.slot


class GRUMemory(Memory):
    """Global recurrent state; objects enter in index order (identity given)."""

    def __init__(self, n_objects, geom="torus", hidden=1024, name="gru"):
        super().__init__()
        self.name, self.N, self.hid = name, n_objects, hidden
        self.e = emb_dim(geom)
        self.inp = nn.Linear(n_objects * (1 + self.e) + 4, 256)
        self.cell = nn.GRUCell(256, hidden)
        self.out = nn.Linear(hidden, n_objects * self.e)

    def init_state(self, b):
        return {"h": torch.zeros(b.shape[0], self.hid, device=b.obs.device, dtype=b.obs.dtype)}

    def read(self, st, b, t):
        z = self.out(st["h"]).view(-1, self.N, self.e)
        pred = pos_decode(z, b.kind) if b.kind == "torus" else h_readout(z, b.kind)
        return {"pred": pred}

    def write(self, st, b, t, rd):
        vis = b.vis[:, t]
        x = torch.cat([vis[..., None], pos_embed(b.obs[:, t], b.kind) * vis[..., None]], -1).flatten(1)
        x = torch.cat([x, b.u[:, t], b.dt[:, t, None], torch.ones_like(b.dt[:, t, None])], -1)
        h = self.cell(F.silu(self.inp(x)), st["h"])
        err = obs_diff(b.obs[:, t], rd["pred"], b.kind).norm(dim=-1)
        return {"h": h}, err * vis

    def state_scalars(self, n_objects):
        return self.hid


class SlotMemory(Memory):
    """Per-object GRU with known identity (structural reference, not budget-matched)."""

    def __init__(self, geom="torus", hidden=64, name="slot_known_id"):
        super().__init__()
        self.name, self.hid = name, hidden
        self.e = emb_dim(geom)
        self.cell = nn.GRUCell(1 + self.e + 3, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Linear(64, self.e))

    def init_state(self, b):
        B, _, N = b.shape
        return {"h": torch.zeros(B * N, self.hid, device=b.obs.device, dtype=b.obs.dtype)}

    def read(self, st, b, t):
        B, _, N = b.shape
        z = self.out(st["h"]).view(B, N, self.e)
        pred = pos_decode(z, b.kind) if b.kind == "torus" else h_readout(z, b.kind)
        return {"pred": pred}

    def write(self, st, b, t, rd):
        B, _, N = b.shape
        vis = b.vis[:, t]
        x = torch.cat([vis[..., None], pos_embed(b.obs[:, t], b.kind) * vis[..., None],
                       b.u[:, t, None].expand(B, N, 2), b.dt[:, t, None, None].expand(B, N, 1)], -1)
        h = self.cell(x.reshape(B * N, -1), st["h"])
        err = obs_diff(b.obs[:, t], rd["pred"], b.kind).norm(dim=-1)
        return {"h": h}, err * vis

    def state_scalars(self, n_objects):
        return n_objects * self.hid


def _silu_grad(x):
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


class LaCTMemory(Memory):
    """LaCT fast-weight memory (arXiv 2505.23884; a1600012888/LaCT @ a648340,
    minimal_implementations/causal_lact_with_sliding_window_attn.py L84-211).

    Per head f(x) = w1 (silu(w0 x) * (w2 x)); inner loss -v^T f(k) (dot product,
    NOT MSE); apply-then-update with one chunk = the tokens of one time step;
    per-token learning rates softplus(Linear(x) + inv_softplus(1e-2));
    after each update every row is L2-renormalized to its INITIAL norm.  Muon is
    off (the reference enables it only when a chunk has > 2*head_dim tokens) and
    momentum is off so that the fast weights fit the 1024-scalar state budget.
    The sliding-window attention branch is removed for the same reason.
    """

    def __init__(self, app_dim=16, geom="torus", heads=4, d_k=16, d_v=8, d_h=6, name="lact"):
        super().__init__()
        self.name, self.H, self.dk, self.dv, self.dh = name, heads, d_k, d_v, d_h
        self.e = emb_dim(geom)
        self.key = nn.Linear(app_dim, heads * d_k, bias=False)
        self.val = nn.Linear(self.e + app_dim, heads * d_v)
        self.lr = nn.Linear(self.e + app_dim, heads * 3)
        self.w0 = nn.Parameter(torch.randn(heads, d_h, d_k) / math.sqrt(d_k))
        self.w2 = nn.Parameter(torch.randn(heads, d_h, d_k) / math.sqrt(d_k))
        self.w1 = nn.Parameter(torch.randn(heads, d_v, d_h) / math.sqrt(d_h))
        self.out = nn.Linear(heads * d_v, self.e)
        self.base_lr = inv_softplus(1e-2)

    def init_state(self, b):
        B = b.shape[0]
        k = F.normalize(self.key(b.app).view(B, -1, self.H, self.dk), dim=-1).transpose(1, 2)  # [B,H,N,dk]
        st = {"w0": self.w0.expand(B, -1, -1, -1).clone(), "w1": self.w1.expand(B, -1, -1, -1).clone(),
              "w2": self.w2.expand(B, -1, -1, -1).clone(), "k": k}
        st["n0"] = st["w0"].norm(dim=-1, keepdim=True)
        st["n1"] = st["w1"].norm(dim=-1, keepdim=True)
        st["n2"] = st["w2"].norm(dim=-1, keepdim=True)
        return st

    def read(self, st, b, t):
        k = st["k"]
        hid = F.silu(torch.einsum("bhdk,bhnk->bhnd", st["w0"], k)) * torch.einsum("bhdk,bhnk->bhnd", st["w2"], k)
        o = torch.einsum("bhvd,bhnd->bhnv", st["w1"], hid)
        B, H, N, _ = o.shape
        z = self.out(o.transpose(1, 2).reshape(B, N, H * self.dv))
        pred = pos_decode(z, b.kind) if b.kind == "torus" else h_readout(z, b.kind)
        return {"pred": pred}

    def write(self, st, b, t, rd):
        B, _, N = b.shape
        vis = b.vis[:, t]
        x = torch.cat([pos_embed(b.obs[:, t], b.kind), b.app], -1)
        v = self.val(x).view(B, N, self.H, self.dv).transpose(1, 2)              # [B,H,N,dv]
        lr = F.softplus(self.lr(x) + self.base_lr).view(B, N, self.H, 3).permute(0, 2, 1, 3)
        lr = lr * vis[:, None, :, None]
        k = st["k"]
        gb = torch.einsum("bhdk,bhnk->bhnd", st["w0"], k)
        hb = torch.einsum("bhdk,bhnk->bhnd", st["w2"], k)
        hidden = F.silu(gb) * hb
        dhidden = torch.einsum("bhvd,bhnv->bhnd", st["w1"], v)
        dhb = dhidden * F.silu(gb)
        dgb = dhidden * hb * _silu_grad(gb)
        dw1 = torch.einsum("bhnv,bhnd->bhvd", v, hidden * lr[..., 1:2])
        dw0 = torch.einsum("bhnd,bhnk->bhdk", dgb, k * lr[..., 0:1])
        dw2 = torch.einsum("bhnd,bhnk->bhdk", dhb, k * lr[..., 2:3])
        st = dict(st)
        for name, dw, n in (("w0", dw0, "n0"), ("w1", dw1, "n1"), ("w2", dw2, "n2")):
            w = st[name] + dw
            st[name] = w / (w.norm(dim=-1, keepdim=True) + 1e-5) * st[n]
        err = obs_diff(b.obs[:, t], rd["pred"], b.kind).norm(dim=-1)
        return st, err * vis

    def state_scalars(self, n_objects):
        return self.H * (2 * self.dh * self.dk + self.dv * self.dh)


class FlowMToken(Memory):
    """FloWM-style structural reference on the torus (address-side transport).

    Hidden grid per velocity channel nu in {-2..2}^2, rolled by nu every step
    (minus the ego shift in the ego frame); inside the field of view the grid is
    overwritten by a shared 3x3 circular conv of [state, encoded observation];
    readout = softmax over cells of max_nu <q(app), h_nu(x)>, circular mean.
    Not budget matched (25 * C * 50 * 50 scalars); reported as a reference.
    """

    def __init__(self, app_dim=16, grid=50, channels=4, vmax=2, name="flowm_token"):
        super().__init__()
        self.name, self.G, self.C = name, grid, channels
        self.vel = [(vx, vy) for vx in range(-vmax, vmax + 1) for vy in range(-vmax, vmax + 1)]
        self.enc = nn.Linear(app_dim, channels)
        self.conv = nn.Conv2d(2 * channels, channels, 3, padding=1, padding_mode="circular")
        self.q = nn.Linear(app_dim, channels)
        self.temp = nn.Parameter(torch.tensor(2.0))

    def init_state(self, b):
        if b.kind != "torus":
            raise ValueError("flowm_token is defined on the torus only")
        B = b.shape[0]
        return {"h": torch.zeros(B, len(self.vel), self.C, self.G, self.G, device=b.obs.device,
                                 dtype=b.obs.dtype)}

    def advance(self, st, b, t):
        h = st["h"]
        shift = torch.round(b.u[:, t] * self.G).long() if b.frame == "ego" else None
        out = []
        for j, (vx, vy) in enumerate(self.vel):
            hj = h[:, j]
            if shift is None or not torch.any(shift != 0):
                out.append(torch.roll(hj, shifts=(vx, vy), dims=(2, 3)))
            else:
                out.append(torch.stack([torch.roll(hj[i], shifts=(vx - int(shift[i, 0]), vy - int(shift[i, 1])),
                                                   dims=(1, 2)) for i in range(hj.shape[0])]))
        return {"h": torch.stack(out, 1)}

    def _cells(self, y):
        return torch.remainder(torch.floor(y * self.G), self.G).long()

    def read(self, st, b, t):
        q = self.q(b.app)                                                   # [B,N,C]
        score = torch.einsum("bnc,bvcxy->bnvxy", q, st["h"]).amax(2)       # [B,N,G,G]
        w = torch.softmax(F.softplus(self.temp) * score.flatten(2), -1).view_as(score)
        ang = 2 * math.pi * (torch.arange(self.G, device=q.device, dtype=q.dtype) + 0.5) / self.G
        wx, wy = w.sum(3), w.sum(2)
        px = torch.atan2((wx * ang.sin()).sum(-1), (wx * ang.cos()).sum(-1))
        py = torch.atan2((wy * ang.sin()).sum(-1), (wy * ang.cos()).sum(-1))
        pred = torch.remainder(torch.stack([px, py], -1) / (2 * math.pi), 1.0)
        return {"pred": pred}

    def write(self, st, b, t, rd):
        B, _, N = b.shape
        vis = b.vis[:, t]
        cells = self._cells(b.obs[:, t])
        o = torch.zeros(B, self.C, self.G, self.G, device=b.obs.device, dtype=b.obs.dtype)
        enc = self.enc(b.app) * vis[..., None]
        bi = torch.arange(B, device=o.device)[:, None].expand(B, N)
        for c in range(self.C):
            o[:, c].index_put_((bi.reshape(-1), cells[..., 0].reshape(-1), cells[..., 1].reshape(-1)),
                               enc[..., c].reshape(-1), accumulate=True)
        if b.fov is not None:
            org = self._cells(b.fov[:, t])                                   # [B,2]
            ar = torch.arange(self.G, device=o.device)
            in_x = torch.remainder(ar[None] - org[:, :1], self.G) < round(b.window * self.G)
            in_y = torch.remainder(ar[None] - org[:, 1:], self.G) < round(b.window * self.G)
            m = (in_x[:, :, None] & in_y[:, None, :]).to(o.dtype)[:, None]  # [B,1,G,G]
        else:
            m = (o.abs().sum(1, keepdim=True) > 0).to(o.dtype)
            m = F.max_pool2d(F.pad(m, (1, 1, 1, 1), mode="circular"), 3, 1)
        h = st["h"]
        inp = torch.cat([h.flatten(0, 1), o[:, None].expand(-1, h.shape[1], -1, -1, -1).flatten(0, 1)], 1)
        new = torch.tanh(self.conv(inp)).view_as(h)
        h = h * (1 - m[:, None]) + new * m[:, None]
        err = obs_diff(b.obs[:, t], rd["pred"], b.kind).norm(dim=-1)
        return {"h": h}, err * vis

    def state_scalars(self, n_objects):
        return len(self.vel) * self.C * self.G * self.G


# ================================================================ factory
VARIANTS = {
    # name: (family, kwargs)
    "static_delta": ("assoc", dict(transition="static")),
    "ours": ("assoc", dict(transition="ours")),
    "ours_GCt": ("assoc", dict(transition="ours", learned_gain=False)),
    "ours_dense": ("assoc", dict(transition="dense")),
    "ours_frame": ("assoc", dict(transition="ours", frame_form=True)),
    "vo_rotation": ("assoc", dict(transition="vo")),
    "tpa_readout": ("assoc", dict(transition="static", tpa=True)),
    "ours_ttt": ("assoc", dict(transition="ours", adapt="ttt")),
    "oracle_L": ("assoc", dict(transition="ours", adapt="oracle")),
    "amortized_dL": ("assoc", dict(transition="ours", adapt="amortized")),
    "gdn": ("key", dict(kind_name="gdn")),
    "gdn_neg": ("key", dict(kind_name="gdn_neg")),
    "kdn_diag": ("key", dict(kind_name="kdn_diag")),
    "m2rnn": ("key", dict(kind_name="m2rnn")),
    "fifo": ("fifo", {}),
    "gru": ("gru", {}),
    "slot_known_id": ("slot", {}),
    "lact": ("lact", {}),
    "flowm_token": ("flowm", {}),
}

BUDGET_MATCHED = ("static_delta", "ours", "ours_GCt", "ours_dense", "vo_rotation", "tpa_readout",
                  "gdn", "gdn_neg", "kdn_diag", "m2rnn", "fifo", "gru", "lact")


def build(name, *, n_objects, geom, app_dim=16, grid=50):
    family, kw = VARIANTS[name]
    if family == "assoc":
        return AssocMemory(app_dim=app_dim, name=name, **kw)
    if family == "key":
        return KeySideMemory(app_dim=app_dim, geom=geom, name=name, **kw)
    if family == "fifo":
        return FIFOMemory(app_dim=app_dim, geom=geom, name=name)
    if family == "gru":
        return GRUMemory(n_objects, geom=geom, name=name)
    if family == "slot":
        return SlotMemory(geom=geom, name=name)
    if family == "lact":
        return LaCTMemory(app_dim=app_dim, geom=geom, name=name)
    if family == "flowm":
        return FlowMToken(app_dim=app_dim, grid=grid, name=name)
    raise ValueError(name)


# ================================================================== loss
def prequential_loss(out, b: TBatch, seen_before):
    """Mean squared pre-write error on visible tokens of already-seen objects.

    Hidden ground truth never enters: hidden objects are penalized only when
    they reappear (``vis``), so in-between evolution must live in the state.
    """
    m = b.vis * seen_before
    d = obs_diff(b.obs, out["pred"], b.kind)
    return (d.pow(2).sum(-1) * m).sum() / m.sum().clamp(min=1.0)


def seen_before_mask(vis):
    """[B,T,N] 1 where the object was visible at some earlier step."""
    c = torch.cumsum(vis, 1)
    return ((c - vis) > 0).float()
