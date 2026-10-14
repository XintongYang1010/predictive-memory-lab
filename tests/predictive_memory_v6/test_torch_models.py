"""Torch-side checks (skipped where torch is unavailable; run in S0 on the cluster)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from pmr.predictive_memory_v6 import memories as M
    from pmr.predictive_memory_v6 import numpy_reference as R
    from pmr.predictive_memory_v6.data import make_world, to_tbatch, geometry_of

ALL = ["static_delta", "ours", "ours_GCt", "ours_dense", "ours_frame", "vo_rotation", "tpa_readout", "ours_ttt",
       "amortized_dL", "gdn", "gdn_neg", "kdn_diag", "m2rnn", "fifo", "gru", "slot_known_id", "lact", "flowm_token"]


def _copy_params(model, p):
    g = lambda t: t.detach().double().cpu().numpy()
    p.W_key = g(model.key.weight)
    p.heads = model.heads
    p.B_shear, p.T_ego = g(model.B_shear), g(model.T_ego)
    p.w_p, p.w_q, p.w_z = float(model.w_p), float(model.w_q), float(model.w_z)
    p.d_q, p.d_z = g(model.d_q), g(model.d_z)
    p.vo_freqs = tuple(g(model.vo_freqs).tolist())
    p.K_p, p.K_q, p.K_z = g(model.K_p), g(model.K_q), g(model.K_z)
    p.w_s, p.w_b, p.c0, p.c1 = float(model.w_s), float(model.w_b), float(model.c0), float(model.c1)
    return p


def fla_naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale):
    """Transcription of fla/ops/gated_delta_rule/naive.py (MIT License,
    Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li) @ 1fe52b5."""
    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float64), [q, k, v, beta, g])
    B, H, T, K = k.shape
    V = v.shape[-1]
    o = torch.zeros(B, H, T, V, dtype=torch.float64)
    h = torch.zeros(B, H, K, V, dtype=torch.float64)
    q = q * scale
    for i in range(T):
        b_q, b_k, b_v = q[:, :, i], k[:, :, i], v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    return o, h


@unittest.skipIf(torch is None, "torch not installed")
class TestTorchModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)

    def _batch(self, spec="l0_sched", n=4, T=12, dtype=None):
        d = make_world(spec, n, T, 3)
        return d, to_tbatch(d, "cpu", dtype or torch.float32)

    def test_all_variants_forward_backward(self):
        for name in ALL:
            spec = "l0_drift" if name in ("amortized_dL",) else "l0_sched"
            d, b = self._batch(spec)
            m = M.build(name, n_objects=5, geom="torus")
            m.train()
            out = m.rollout(b)
            self.assertEqual(tuple(out["pred"].shape), (4, 12, 5, 2), name)
            loss = M.prequential_loss(out, b, M.seen_before_mask(b.vis))
            self.assertTrue(torch.isfinite(loss), name)
            loss.backward()
            grads = [p.grad for p in m.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads), name)
            self.assertTrue(((out["pred"] >= 0) & (out["pred"] < 1)).all(), name)
        m = M.build("oracle_L", n_objects=5, geom="torus")
        d, b = self._batch("l0_drift")
        self.assertTrue(torch.isfinite(m.rollout(b)["pred"]).all())

    def test_ttt_eval_mode_runs(self):
        m = M.build("ours_ttt", n_objects=5, geom="torus").eval()
        d, b = self._batch("l0_drift", T=40)
        with torch.no_grad():
            out = m.rollout(b)
        self.assertTrue(torch.isfinite(out["pred"]).all())

    def test_box_and_plane(self):
        for kind in ("box", "plane"):
            d, b = self._batch()
            b.kind = kind
            for name in ("ours", "gdn_neg", "gru", "lact", "fifo"):
                m = M.build(name, n_objects=5, geom=kind)
                out = m.rollout(b)
                self.assertTrue(torch.isfinite(out["pred"]).all(), (kind, name))

    def test_budget(self):
        for name in M.BUDGET_MATCHED:
            m = M.build(name, n_objects=5, geom="torus")
            self.assertLessEqual(m.state_scalars(5), 1024, name)

    def test_matches_numpy_reference(self):
        d, b = self._batch(n=6, T=24, dtype=torch.float64)
        dego = make_world("l0_main:frame=ego", 6, 24, 4)
        bego = to_tbatch(dego, "cpu", torch.float64)
        for name, tr, frame_form in (("ours", "ours", False), ("static_delta", "static", False),
                                     ("vo_rotation", "vo", False), ("ours_frame", "ours", True)):
            m = M.build(name, n_objects=5, geom="torus").double()
            with torch.no_grad():
                for p_ in m.parameters():
                    p_.add_(0.05 * torch.randn_like(p_))
            p = _copy_params(m, R.AssocParams(transition=tr, frame_form=frame_form))
            for dd, bb, frame in ((d, b, "allo"), (dego, bego, "ego")):
                with torch.no_grad():
                    tp = m.rollout(bb)["pred"].numpy()
                ref = R.rollout(p, dd["app"].astype(np.float64), bb.obs.numpy(), bb.vis.numpy(),
                                bb.u.numpy(), bb.dt.numpy(), "torus", frame)
                self.assertLess(np.abs(R.obs_diff(tp, ref, "torus")).max(), 1e-8, (name, frame))

    def test_ttt_equivalence_autograd(self):
        """G = C^T, beta = 1: the write equals -grad of 0.5 * sum ||y - C S k||^2."""
        m = M.AssocMemory(transition="static", learned_gain=False, beta_fixed=1.0).double()
        d, b = self._batch(n=2, T=4, dtype=torch.float64)
        b.kind = "plane"
        st = m.init_state(b)
        S = (0.1 * torch.randn_like(st["S"])).requires_grad_(True)
        st["S"] = S
        t = 2
        rd = m.read(st, b, t)
        y = torch.cat([torch.ones_like(b.obs[:, t, :, :1]), torch.zeros_like(b.obs[:, t, :, :1]), b.obs[:, t]], -1)
        r = torch.einsum("bvk,bnk->bnv", S, st["k"])[..., :4]
        loss = 0.5 * (((y - r) ** 2).sum(-1) * b.vis[:, t]).sum()
        g, = torch.autograd.grad(loss, S)
        st2, _ = m.write({k: (v.detach() if torch.is_tensor(v) else v) for k, v in st.items()}, b, t,
                         {k: v.detach() for k, v in rd.items()})
        self.assertLess(float((st2["S"] - S.detach() + g).abs().max()), 1e-10)

    def test_gdn_matches_fla_naive(self):
        torch.manual_seed(1)
        m = M.KeySideMemory(heads=1, d_k=32, kind_name="gdn", geom="torus").double()
        T = 16
        d = make_world("l0_sched:n_objects=1", 3, T, 5)
        d["vis"][:] = True
        d["obs"] = d["pos"].copy()
        b = to_tbatch(d, "cpu", torch.float64)
        st = m.init_state(b)
        for t in range(T):
            if t > 0:
                st = m.advance(st, b, t)
            rd = m.read(st, b, t)
            st, _ = m.write(st, b, t, rd)
        k = st["k"][:, 0]                                                  # [B,K]
        x = torch.cat([M.pos_embed(b.obs[:, :, 0], "torus"), b.app[:, None, 0].expand(-1, T, -1)], -1)
        v = m.val(x)                                                       # [B,T,V]
        beta = torch.sigmoid(m.beta(x))[..., 0]
        gstep = -torch.exp(m.A_log) * torch.nn.functional.softplus(m.a0 + m.dt_bias)
        g = (gstep * b.dt).clone()
        g[:, 0] = 0.0
        q = k[:, None, None].expand(-1, T, 1, -1)
        _, h = fla_naive_recurrent_gated_delta_rule(q, k[:, None, None].expand(-1, T, 1, -1), v[:, :, None],
                                                    beta[:, :, None], g[:, :, None], scale=1.0)
        diff = (st["S"] - h[:, 0].transpose(1, 2)).abs().max()
        self.assertLess(float(diff), 1e-5)

    def test_lact_update_matches_autograd(self):
        torch.manual_seed(2)
        m = M.LaCTMemory(geom="torus").double()
        d, b = self._batch(n=2, T=3, dtype=torch.float64)
        b.vis[:] = 1.0
        st = m.init_state(b)
        t = 1
        x = torch.cat([M.pos_embed(b.obs[:, t], "torus"), b.app], -1)
        B, N = 2, 5
        v = m.val(x).view(B, N, m.H, m.dv).transpose(1, 2)
        lr = torch.nn.functional.softplus(m.lr(x) + m.base_lr).view(B, N, m.H, 3).permute(0, 2, 1, 3)
        w = {n: st[n].clone().requires_grad_(True) for n in ("w0", "w1", "w2")}
        k = st["k"]
        new = {}
        for j, n in enumerate(("w0", "w1", "w2")):
            hid = torch.nn.functional.silu(torch.einsum("bhdk,bhnk->bhnd", w["w0"], k)) * \
                torch.einsum("bhdk,bhnk->bhnd", w["w2"], k)
            f = torch.einsum("bhvd,bhnd->bhnv", w["w1"], hid)
            obj = ((f * v).sum(-1) * lr[..., j]).sum()                     # ascent on v^T f(k), lr per weight
            gr, = torch.autograd.grad(obj, w[n])
            nw = w[n].detach() + gr
            new[n] = nw / (nw.norm(dim=-1, keepdim=True) + 1e-5) * st["n" + n[1]]
        st2, _ = m.write(st, b, t, m.read(st, b, t))
        for n in ("w0", "w1", "w2"):
            self.assertLess(float((st2[n] - new[n]).abs().max()), 1e-9, n)

    def test_hidden_truth_never_in_loss(self):
        d, b = self._batch()
        m = M.build("ours", n_objects=5, geom="torus")
        out = m.rollout(b)
        l1 = M.prequential_loss(out, b, M.seen_before_mask(b.vis))
        obs = b.obs.clone()
        obs[b.vis == 0] = 0.77
        b2 = M.TBatch(app=b.app, obs=obs, vis=b.vis, u=b.u, dt=b.dt, kind=b.kind)
        out2 = m.rollout(b2)
        l2 = M.prequential_loss(out2, b2, M.seen_before_mask(b2.vis))
        self.assertAlmostEqual(float(l1), float(l2), places=6)

    def test_train_and_eval_jobs_smoke(self):
        from pmr.predictive_memory_v6.train import train_job
        from pmr.predictive_memory_v6.evaluate import eval_job
        from pmr.predictive_memory_v6.experiments import C
        with tempfile.TemporaryDirectory() as root:
            for name in ("ours", "gru"):
                out = Path(root) / name
                st = train_job({"variant": name, "world": "l0_ctrl", "seed": 0, "steps": 3, "batch": 4,
                                "dev_episodes": 4, "pool_size": 16, "refresh": 100, "dev_every": 2, "device": "cpu",
                                "out": str(out), "T": 16})
                self.assertEqual(st["status"], "complete")
                conds = [C("std", "standard", "l0_ctrl", 16, 4), C("hist", "history_pair", "l0_sched", 16, 4),
                         C("chg", "change_pair", "l0_ctrl", 32, 8), C("blk", "blackout", "l0_blackout", 48, 4),
                         C("cnt", "count", "l0_tour_static:copies=2", 100, 4)]
                es = eval_job({"checkpoint": str(out / "best.pt"), "conditions": conds, "out": str(out / "eval"),
                               "device": "cpu", "training_seed": 0})
                self.assertEqual(es["status"], "complete", es)
            es = eval_job({"reference": "cv_kf", "conditions": conds[:3], "out": str(Path(root) / "ref")})
            self.assertEqual(es["status"], "complete", es)
            from pmr.predictive_memory_v6.report import write_report
            write_report(root)

    def test_numerics_job(self):
        from pmr.predictive_memory_v6.checks import numerics_job
        with tempfile.TemporaryDirectory() as root:
            res = numerics_job({"out": root})
            self.assertTrue(res["gates"]["spectral"])
            self.assertTrue(res["gates"]["fp64_replay"])


if __name__ == "__main__":
    unittest.main()
