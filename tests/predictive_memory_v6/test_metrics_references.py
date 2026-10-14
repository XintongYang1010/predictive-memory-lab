import unittest

import numpy as np

from pmr.predictive_memory_v6.data import make_world, geometry_of
from pmr.predictive_memory_v6.geometry import Geometry, fold, fold_sign, torus_diff
from pmr.predictive_memory_v6.metrics import (score_episodes, summarize, mra, auroc, best_f1_threshold,
                                              bootstrap, stat_er, tracking_masks, history_pair_scores,
                                              interference_scores, interference_summary, gap_closure)
from pmr.predictive_memory_v6.references import run_reference
from pmr.predictive_memory_v6.worlds_l0 import L0Config, history_pair, change_pair


class TestGeometry(unittest.TestCase):
    def test_torus_and_fold(self):
        self.assertAlmostEqual(float(torus_diff(0.1, 0.9)), 0.2)
        x = np.linspace(-3, 3, 601)
        f = fold(x, 1.0, 15.0)
        self.assertTrue(np.all((f >= 1) & (f <= 15)))
        self.assertEqual(float(fold(16.0, 1, 15)), 14.0)
        self.assertEqual(float(fold(0.0, 1, 15)), 2.0)
        self.assertEqual(float(fold_sign(16.0, 1, 15)), -1.0)
        g = Geometry("box", lo=1, hi=15)
        self.assertTrue(np.allclose(g.denormalize(g.normalize(np.array([3.0, 7.0]))), [3, 7]))


class TestReferencesMetrics(unittest.TestCase):
    def setUp(self):
        self.d = make_world("l0_sched", 256, 64, 11)
        self.g = geometry_of(self.d)

    def test_kf_gate_and_pause(self):
        m = tracking_masks(self.d)
        p = run_reference("analytic_kf", self.d, self.g)
        sel = m["hidden"] & m["identifiable"] & ~m["changed"]
        err = np.sqrt((self.g.diff(p, self.d["pos"])[sel] ** 2).sum(-1).mean())
        self.assertLess(err, 1e-3)
        s = summarize(score_episodes(self.d, run_reference("pause", self.d, self.g), self.g))
        self.assertAlmostEqual(s["er"], 0.0, places=9)
        self.assertAlmostEqual(s["nhe"], 1.0, places=9)
        s = summarize(score_episodes(self.d, p, self.g))
        self.assertAlmostEqual(s["er"], 1.0, places=6)

    def test_truth_scores_perfect(self):
        s = summarize(score_episodes(self.d, self.d["pos"].copy(), self.g))
        self.assertAlmostEqual(s["er"], 1.0, places=9)
        self.assertAlmostEqual(s["nhe"], 0.0, places=9)
        self.assertEqual(s["never_recovered_pred"], 0.0)

    def test_interp_oracle_exact_on_constant_velocity(self):
        p = run_reference("interp_oracle", self.d, self.g)
        ok = ~np.isnan(p[..., 0])
        self.assertTrue(ok.any())
        self.assertLess(np.abs(self.g.diff(p[ok], self.d["pos"][ok])).max(), 1e-9)

    def test_mra_cases(self):
        for (p, g), v in {(10, 10): 1.0, (15, 10): 0.1, (10.5, 10): 1.0, (0, 10): 0.0}.items():
            self.assertAlmostEqual(mra(p, g), v)

    def test_auroc_f1(self):
        self.assertEqual(auroc([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auroc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)
        thr, f = best_f1_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
        self.assertEqual(f, 1.0)
        self.assertTrue(0.2 < thr < 0.8)

    def test_bootstrap_pairing(self):
        rows = score_episodes(self.d, run_reference("cv_kf", self.d, self.g), self.g, training_seed=0)
        res = bootstrap(rows, stat_er, rows_b=rows, draws=50)
        self.assertEqual(res["point"], 0.0)

    def test_pairs(self):
        a, b = history_pair(L0Config(), 32, 64, 3)
        g = geometry_of(a)
        rows = history_pair_scores(a, b, a["pos"], b["pos"], g)
        self.assertLess(max(r["history_pair_error"] for r in rows), 1e-9)
        rows = history_pair_scores(a, b, run_reference("pause", a, g), run_reference("pause", b, g), g)
        self.assertGreater(np.mean([r["history_pair_error"] for r in rows]), 0)
        a, b = change_pair(L0Config(), 32, 64, 4)
        rows = interference_scores(a, b, a["pos"], b["pos"], g)
        self.assertEqual(interference_summary(rows)["mean_displacement"], 0.0)
        self.assertAlmostEqual(gap_closure(10.0, 4.0, 2.0), 0.75)


if __name__ == "__main__":
    unittest.main()
