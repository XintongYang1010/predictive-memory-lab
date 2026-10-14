"""Torch parity: the GPU path must equal the float64 numpy reference.

Skipped where torch is not installed (CI installs the CPU wheel)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from pmr.predictive_memory_v6 import bf


def rand_unit(rng, shape):
    k = rng.standard_normal(shape)
    return k / np.linalg.norm(k, axis=-1, keepdims=True)


@unittest.skipIf(torch is None, "torch not installed")
class TestTorchParity(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        B, T, D, dk = 3, 90, 5, 12
        self.K = rand_unit(rng, (B, T, dk))
        self.O = (rng.random((B, T, D)) > 0.2).astype(float)
        self.R = rng.standard_normal((B, T, D)) * self.O
        self.R[:, 45:] += 2.0 * self.O[:, 45:]
        self.W = np.ones((B, T))
        self.W[:, 30] = 0.0
        self.t = lambda x: torch.as_tensor(x, dtype=torch.float64)

    def test_closed_form_arms(self):
        for arm, kw in (("A6", {}), ("A7", {}), ("A8", {}), ("A8g", {}), ("A1", {"beta0": 0.3}),
                        ("A1BF", {"beta0": 0.3}), ("A6L", {"lam": 0.99})):
            a = bf.scan(arm, self.K, self.R, self.O, self.W, sbar=0.5, pi=1e-2, **kw)
            b = bf.scan(arm, self.t(self.K), self.t(self.R), self.t(self.O), self.t(self.W), sbar=0.5, pi=1e-2, **kw)
            np.testing.assert_allclose(b["m"].numpy(), a["m"], atol=1e-10, err_msg=arm)

    def test_chunk_parity_and_grad(self):
        a = bf.a6_chunk(self.K, self.R, self.O, self.W, 0.3, chunk=16)
        K = self.t(self.K).requires_grad_(True)
        b = bf.a6_chunk(K, self.t(self.R), self.t(self.O), self.t(self.W), 0.3, chunk=16)
        np.testing.assert_allclose(b.detach().numpy(), a, atol=1e-9)
        b.square().sum().backward()
        self.assertTrue(torch.isfinite(K.grad).all())
        s = bf.scan("A6", self.t(self.K), self.t(self.R), self.t(self.O), self.t(self.W), sbar=0.3)["m"]
        np.testing.assert_allclose(s.numpy(), a, atol=1e-9)

    def test_learned_gates_backprop(self):
        from pmr.predictive_memory_v6 import models as Mo
        K, R, O, W = (self.t(x).float() for x in (self.K, self.R, self.O, self.W))
        g2 = Mo.GateA2(7)
        z = torch.randn(K.shape[0], K.shape[1], 7)
        res = bf.scan("A2", K, R, O, W, sbar=0.5, gate=g2(z))
        res["m"].square().sum().backward()
        self.assertTrue(all(p.grad is not None for p in g2.parameters() if p.requires_grad))
        g2p = Mo.GateA2P()
        res = bf.scan("A2P", K, R, O, W, sbar=0.5, gate=g2p)
        res["m"].square().sum().backward()
        sg = Mo.ScalarGate()
        res = bf.scan("A4", K, R, O, W, sbar=0.5, gate=(sg.a, sg.b))
        res["m"].square().sum().backward()
        self.assertIsNotNone(sg.a.grad)

    def test_save_load_roundtrip(self):
        from pmr.predictive_memory_v6 import models as Mo
        m = Mo.SlowModel(10, 4)
        m.sigma.fill_(0.5)
        with tempfile.TemporaryDirectory() as d:
            Mo.save(Path(d) / "s.pt", m, {"x": 1})
            m2, extra = Mo.load(Path(d) / "s.pt")
        z = torch.randn(3, 10)
        torch.testing.assert_close(m(z), m2(z))
        self.assertEqual(extra["x"], 1)
        self.assertEqual(float(m2.sigma[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
