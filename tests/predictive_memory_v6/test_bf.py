"""U1-U9: BF-Delta algebra in float64 numpy (the same code runs in torch)."""
import math
import unittest

import numpy as np

from pmr.predictive_memory_v6 import bf


def onehot(idx, dk):
    k = np.zeros((len(idx), dk))
    k[np.arange(len(idx)), idx] = 1.0
    return k


def rand_unit(rng, shape):
    k = rng.standard_normal(shape)
    return k / np.linalg.norm(k, axis=-1, keepdims=True)


class TestBF(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_u1_gamma0_onehot_is_posterior_mean(self):
        sbar, dk, D, T = 0.3, 5, 3, 60
        n0 = sbar / (1 - sbar)
        idx = self.rng.integers(0, dk, T)
        K = onehot(idx, dk)[None]
        R = self.rng.standard_normal((1, T, D))
        res = bf.scan("A6", K, R, np.ones((1, T, D)), np.ones((1, T)), sbar=sbar)
        S = res["S_final"][0]
        for j in range(dk):
            sel = idx == j
            want = R[0, sel].sum(0) / (n0 + sel.sum())
            np.testing.assert_allclose(S[:, j], want, atol=1e-12)

    def test_u2_gamma1_resets_to_shrunk_observation(self):
        sbar, dk, D = 0.5, 4, 10
        rho, n0 = 1 - sbar, sbar / (1 - sbar)
        k = onehot([1], dk)
        S, n = np.zeros((1, D, dk)), np.full((1, dk), n0)
        r1 = np.full((1, D), 1.0)
        for _ in range(20):
            S, n, _ = bf.bf_step(S, n, k, r1, np.ones((1, D)), np.ones(1), sbar=sbar, logit_pi=bf.logit(1e-3),
                                 arm="A8")
        r2 = np.full((1, D), -20.0)
        S, n, info = bf.bf_step(S, n, k, r2, np.ones((1, D)), np.ones(1), sbar=sbar, logit_pi=bf.logit(1e-3),
                                arm="A8")
        self.assertGreater(info["g"][0], 1 - 1e-12)
        np.testing.assert_allclose(S[0] @ k[0], rho * r2[0], atol=1e-6)
        self.assertAlmostEqual(float(n[0, 1]), n0 + 1.0, places=6)

    def test_u3_cold_start_exact(self):
        sbar, dk, D = 0.4, 6, 7
        rho, n0 = 1 - sbar, sbar / (1 - sbar)
        k = rand_unit(self.rng, (1, dk))
        S, n = np.zeros((1, D, dk)), np.full((1, dk), n0)
        r = self.rng.standard_normal((1, D))
        S, n, info = bf.bf_step(S, n, k, r, np.ones((1, D)), np.ones(1), sbar=sbar, logit_pi=bf.logit(1e-3),
                                arm="A8")
        self.assertAlmostEqual(float(info["ell"][0]), 0.0, places=12)
        np.testing.assert_allclose(S[0] @ k[0], rho * r[0], atol=1e-12)

    def test_u4_mean_equals_two_component_posterior(self):
        sbar, dk, D, pi = 0.35, 3, 4, 0.05
        rho, n0 = 1 - sbar, sbar / (1 - sbar)
        for trial in range(30):
            S = self.rng.standard_normal((1, D, dk)) * 0.5
            n = n0 + self.rng.integers(0, 6, (1, dk)).astype(float)
            j = trial % dk
            k = onehot([j], dk)
            r = self.rng.standard_normal((1, D)) * (0.5 + trial / 10)
            m = S[0, :, j]
            c = n[0, j]
            v_nc = sbar * (1 + 1 / c)
            lp_nc = -0.5 * (((r[0] - m) ** 2).sum() / v_nc + D * math.log(2 * math.pi * v_nc))
            lp_c = -0.5 * ((r[0] ** 2).sum() + D * math.log(2 * math.pi))
            a = math.log(pi) + lp_c
            b = math.log(1 - pi) + lp_nc
            post_c = 1 / (1 + math.exp(b - a))
            tau = 1 / (1 + c)
            want = (1 - post_c) * (m + tau * (r[0] - m)) + post_c * rho * r[0]
            S2, _, info = bf.bf_step(S.copy(), n.copy(), k, r, np.ones((1, D)), np.ones(1), sbar=sbar,
                                     logit_pi=bf.logit(pi), arm="A8", ell_clip=1e9)
            self.assertAlmostEqual(float(info["g"][0]), post_c, places=10)
            np.testing.assert_allclose(S2[0, :, j], want, atol=1e-10)

    def test_u5_matches_independent_nassar_observer(self):
        sbar, pi = 0.4, 0.02
        x = np.concatenate([self.rng.normal(1.0, math.sqrt(sbar), 30), self.rng.normal(-2.0, math.sqrt(sbar), 30)])
        ref = bf.nassar_reference(x, sbar, pi)
        K = np.ones((1, len(x), 1))
        res = bf.scan("A8", K, x[None, :, None], np.ones((1, len(x), 1)), np.ones((1, len(x))), sbar=sbar, pi=pi)
        g = res["g"][0]
        tau = res["tau"][0]
        m = res["m"][0, :, 0]
        for t, (beta, gamma, mu) in enumerate(ref):
            self.assertAlmostEqual(float(g[t]), gamma, places=10)
            self.assertAlmostEqual(float(g[t] + (1 - g[t]) * tau[t]), beta, places=10)
            self.assertAlmostEqual(float(m[t]), mu, places=10)

    def test_u6_global_is_gated_deltanet(self):
        sbar, dk, D, T = 0.5, 8, 5, 80
        rho = 1 - sbar
        K = rand_unit(self.rng, (1, T, dk))
        R = self.rng.standard_normal((1, T, D))
        R[0, 40:] += 3.0
        res = bf.scan("A8g", K, R, np.ones((1, T, D)), np.ones((1, T)), sbar=sbar, pi=1e-2)
        g, tau = res["g"][0], res["tau"][0]
        S = np.zeros((D, dk))
        ms = []
        for t in range(T):
            k = K[0, t]
            ms.append(S @ k)
            alpha, beta = 1 - g[t], tau[t]
            v = ((1 - g[t]) * tau[t] + g[t] * rho) * R[0, t] / tau[t]
            S = alpha * S @ (np.eye(dk) - beta * np.outer(k, k)) + beta * np.outer(v, k)
        np.testing.assert_allclose(np.array(ms), res["m"][0], atol=1e-10)

    def test_u7_chunk_equals_scan(self):
        sbar, dk, D, T, B = 0.3, 16, 6, 150, 3
        K = rand_unit(self.rng, (B, T, dk))
        R = self.rng.standard_normal((B, T, D))
        O = (self.rng.random((B, T, D)) > 0.3).astype(float)
        R = R * O
        W = np.ones((B, T))
        W[:, 77] = 0.0
        ref = bf.scan("A6", K, R, O, W, sbar=sbar)["m"]
        for chunk in (1, 16, 64, 200):
            M = bf.a6_chunk(K, R, O, W, sbar, chunk=chunk)
            np.testing.assert_allclose(M, ref, atol=1e-9)

    def test_u8_ml2_recovers_sbar(self):
        dk, D, T, B = 20, 8, 400, 8
        for strue in (0.25, 0.5, 0.75):
            idx = self.rng.integers(0, dk, (B, T))
            K = np.stack([onehot(i, dk) for i in idx])
            theta = self.rng.normal(0, math.sqrt(1 - strue), (B, dk, D))
            R = np.take_along_axis(theta, idx[..., None], 1) + self.rng.normal(0, math.sqrt(strue), (B, T, D))
            best, _ = bf.ml2_select(K, R, np.ones((B, T, D)), np.ones((B, T)), [0.05 * i for i in range(1, 20)])
            self.assertLessEqual(abs(best - strue), 0.05 + 1e-9)

    def test_u9_local_write_leaves_unobserved_rows(self):
        sbar, dk, D, T = 0.5, 6, 5, 40
        K = rand_unit(self.rng, (1, T, dk))
        R = self.rng.standard_normal((1, T, D))
        O = np.ones((1, T, D))
        O[0, :, 2] = 0.0
        R = R * O
        for arm, kw in (("A6", {}), ("A8", {}), ("A7", {}), ("A1", {"beta0": 0.4}), ("A1BF", {"beta0": 0.4})):
            res = bf.scan(arm, K, R, O, np.ones((1, T)), sbar=sbar, **kw)
            self.assertTrue(np.all(res["S_final"][0, 2] == 0.0), arm)

    def test_write_flag_and_oracle_reset(self):
        sbar, dk, D, T = 0.5, 4, 3, 30
        K = rand_unit(self.rng, (1, T, dk))
        R = self.rng.standard_normal((1, T, D))
        O, W = np.ones((1, T, D)), np.ones((1, T))
        W[0, 10] = 0.0
        a = bf.scan("A8", K, R, O, W, sbar=sbar)
        R2 = R.copy()
        R2[0, 10] += 100.0          # unwritten step cannot change the state
        b = bf.scan("A8", K, R2, O, W, sbar=sbar)
        np.testing.assert_allclose(a["S_final"], b["S_final"])
        reset = np.zeros((1, T))
        reset[0, 15] = 1.0
        oc = bf.scan("OC", K, R, O, W, sbar=sbar, reset=reset)
        fresh = bf.scan("A6", K[:, 15:], R[:, 15:], O[:, 15:], W[:, 15:], sbar=sbar)
        np.testing.assert_allclose(oc["m"][:, 15:], fresh["m"], atol=1e-12)

    def test_learned_arm_forms(self):
        sbar, dk, D, T = 0.5, 4, 3, 25
        K = rand_unit(self.rng, (1, T, dk))
        R = self.rng.standard_normal((1, T, D))
        O, W = np.ones((1, T, D)), np.ones((1, T))
        g = np.stack([np.full((1, T), 1.0), np.full((1, T), 0.3)], -1)
        a2 = bf.scan("A2", K, R, O, W, sbar=sbar, gate=g)
        a1 = bf.scan("A1", K, R, O, W, sbar=sbar, beta0=0.3)
        np.testing.assert_allclose(a2["m"], a1["m"], atol=1e-12)     # alpha = 1 GDN is DeltaNet
        a4 = bf.scan("A4", K, R, O, W, sbar=sbar, gate=(0.0, bf.logit(0.3)))
        np.testing.assert_allclose(a4["m"], a1["m"], atol=1e-12)
        a3 = bf.scan("A3", K, R, O, W, sbar=sbar, gate=np.full((1, T), 0.3))
        np.testing.assert_allclose(a3["m"], a1["m"], atol=1e-12)

        def g2p(f):
            B = f.shape[0]
            return np.ones(B), np.full(B, 0.3), np.ones(B)
        a2p = bf.scan("A2P", K, R, O, W, sbar=sbar, gate=g2p)
        np.testing.assert_allclose(a2p["m"], a1["m"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
