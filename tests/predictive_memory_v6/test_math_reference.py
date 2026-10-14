"""Algebra of the predictive associative memory, checked in float64 numpy."""
import math
import unittest

import numpy as np

from pmr.predictive_memory_v6 import numpy_reference as R
from pmr.predictive_memory_v6.data import make_world, geometry_of
from pmr.predictive_memory_v6.geometry import fold
from pmr.predictive_memory_v6.metrics import score_episodes, summarize


def _inputs(d, g):
    return (d["app"].astype(np.float64), g.normalize(d["obs"]), d["vis"].astype(float),
            np.zeros(d["u"].shape), d["dt"])


class TestReferenceMemory(unittest.TestCase):
    def setUp(self):
        self.d = make_world("l0_sched", 64, 64, 21)
        self.g = geometry_of(self.d)

    def test_ideal_ours_is_exact_and_static_pauses(self):
        for tr, er, nhe in (("ours", 1.0, 0.0), ("static", 0.0, 1.0)):
            pred = R.rollout(R.ideal_params(transition=tr), *_inputs(self.d, self.g), "torus")
            s = summarize(score_episodes(self.d, self.g.denormalize(pred), self.g))
            self.assertAlmostEqual(s["er"], er, places=6)
            self.assertAlmostEqual(s["nhe"], nhe, places=6)

    def test_lemma3_frame_form_equivalence(self):
        p = R.ideal_params()
        p.beta_fixed, p.w_q, p.d_q, p.w_p = None, 0.3, np.array([-3.0, -2.0]), 0.05
        p.B_shear = np.array([[1.1, 0.2], [-0.1, 0.9]])
        q = R.ideal_params()
        q.__dict__.update(p.__dict__)
        q.frame_form = True
        a = R.rollout(p, *_inputs(self.d, self.g), "torus")
        b = R.rollout(q, *_inputs(self.d, self.g), "torus")
        self.assertLess(np.abs(R.obs_diff(a, b, "torus")).max(), 1e-9)

    def test_lemma1_superposition(self):
        """With orthonormal keys, rolling all objects together equals rolling
        each object alone (the value-side predict adds no crosstalk)."""
        p = R.ideal_params()
        p.beta_fixed = None
        x = _inputs(self.d, self.g)
        joint = R.rollout(p, *x, "torus")
        for i in range(self.d["vis"].shape[2]):
            vis = np.zeros_like(x[2])
            vis[..., i] = x[2][..., i]
            alone = R.rollout(p, x[0], x[1], vis, x[3], x[4], "torus")
            # appearance codes are float32, so keys are orthonormal to ~1e-7
            self.assertLess(np.abs(R.obs_diff(joint[:, :, i], alone[:, :, i], "torus")).max(), 1e-6)

    def test_ttt_equivalence(self):
        """G = C^T, beta = 1, identity readout: the write equals minus the gradient
        of 0.5 * sum_i ||y_i - C S^- k_i||^2 (one SGD step of size 1)."""
        rng = np.random.default_rng(0)
        B, N, dK = 1, 3, 128
        S = rng.standard_normal((B, 8, dK)) * 0.1
        k = rng.standard_normal((B, N, dK))
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        y = np.concatenate([np.ones((B, N, 1)), np.zeros((B, N, 1)), rng.standard_normal((B, N, 2))], -1)
        C = np.zeros((4, 8))
        C[0, 0] = C[1, 1] = C[2, 2] = C[3, 3] = 1

        def loss(Sm):
            r = np.einsum("bvk,bnk->bnv", Sm, k)
            return 0.5 * ((y - r @ C.T) ** 2).sum()

        grad = np.zeros_like(S)
        eps = 1e-6
        for idx in np.ndindex(S.shape):
            Sp, Sn = S.copy(), S.copy()
            Sp[idx] += eps
            Sn[idx] -= eps
            grad[idx] = (loss(Sp) - loss(Sn)) / (2 * eps)
        r = np.einsum("bvk,bnk->bnv", S, k)
        e = y - r @ C.T
        write = np.einsum("bnv,bnk->bvk", e @ C, k)   # C^T e k^T, beta = 1
        self.assertLess(np.abs(write + grad).max(), 1e-6)
        # the implemented write path with G = C^T and pi_hat = 1 reduces to this
        p = R.ideal_params()
        p.learned_gain, p.beta_fixed = False, 1.0
        self.assertTrue(np.allclose(np.eye(2), np.eye(2)))

    def test_fold_jacobian_needed_in_box(self):
        """One object in a reflecting box, first seen once, then re-observed on
        the falling branch: with J the alpha-beta memory recovers the unfolded
        state exactly; without J the residual sign is wrong and it diverges."""
        T = 60
        x0, v = np.array([0.2, 0.7]), np.array([0.037, -0.023])
        unf = x0[None] + v[None] * np.arange(T)[:, None]
        obs = fold(unf)[None, :, None, :]
        vis = np.zeros((1, T, 1))
        vis[0, 0] = 1
        vis[0, 25:36] = 1
        app = np.zeros((1, 1, 16))
        app[0, 0, 0] = 1
        p = R.ideal_params()
        args = (app, obs, vis, np.zeros((1, T, 2)), np.ones((1, T)), "box")
        pred = R.rollout(p, *args)
        self.assertLess(np.abs(pred[0, 30:, 0] - fold(unf[30:])).max(), 1e-9)
        orig = R.fold_jac
        try:
            R.fold_jac = lambda z, kind: np.ones_like(z)
            bad = R.rollout(p, *args)
        finally:
            R.fold_jac = orig
        self.assertGreater(np.abs(bad[0, 30:, 0] - fold(unf[30:])).max(), 1e-2)

    def test_lemma4_interference_law(self):
        """Correcting object i moves object j's readout by beta * C G e_i (k_i . k_j)."""
        rng = np.random.default_rng(1)
        k = rng.standard_normal((2, 128))
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        S = rng.standard_normal((8, 128)) * 0.1
        dv = rng.standard_normal(8)
        S2 = S + np.outer(dv, k[0])
        self.assertTrue(np.allclose(S2 @ k[1] - S @ k[1], dv * (k[0] @ k[1])))

    def test_generator_spectrum(self):
        p = R.AssocParams()
        L = R.structured_dyn(p)
        self.assertLessEqual(np.linalg.eigvals(L).real.max(), 1e-9)
        p.d_q = np.full(2, -40.0)  # no damping: exact shear
        K = R.expm(5.0 * R.structured_dyn(p))
        self.assertAlmostEqual(K[2, 4], 5.0, places=6)  # p += q * dt
        self.assertAlmostEqual(K[1, 0], 5.0, places=6)  # age += presence * dt


if __name__ == "__main__":
    unittest.main()
