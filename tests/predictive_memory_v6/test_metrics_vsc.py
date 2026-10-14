"""Metrics, pre-registered decision rules and the offline VSC aggregation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pmr.predictive_memory_v6 import jobs, metrics as Me, prereg, report, vsc


class TestMetrics(unittest.TestCase):
    def test_u10_mra_cases(self):
        for (pred, gt), want in prereg.GATES["mra_cases"].items():
            self.assertAlmostEqual(Me.mra(pred, gt), want)

    def test_auroc(self):
        self.assertEqual(Me.auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]), 1.0)
        self.assertEqual(Me.auroc([1, 1, 1, 1], [0, 1, 0, 1]), 0.5)

    def _rows(self, cond, vals):
        rows = []
        for i in range(20):
            r = {"uid": f"{cond}{i}", "cond": cond}
            for arm, v in vals.items():
                r[f"{arm}|POST|num"] = v * 10
                r[f"{arm}|POST|den"] = 10.0
            rows.append(r)
        return rows

    def test_nu_and_frontier(self):
        chg = self._rows("CHANGE", {"A6": 2.0, "OC": 1.0, "X": 1.2})
        prot = self._rows("EXACT", {"A6": 1.0, "OC": 1.5, "X": 1.1})
        self.assertAlmostEqual(Me.nu_c(chg, "X"), 0.8)
        self.assertAlmostEqual(Me.nu_r(prot, "X"), 0.8)
        F = Me.frontier([(0.0, 1.0), (1.0, 0.0)])
        self.assertAlmostEqual(F(0.25), 0.75)
        self.assertTrue(Me.dominated((0.2, 0.2), [(0.5, 0.5)]))
        self.assertFalse(Me.dominated((0.9, 0.9), [(0.5, 0.95)]))

    def test_passes_rule(self):
        self.assertTrue(Me.passes({"point": 0.4, "ci95": [0.1, 0.7]}, 0.3))
        self.assertFalse(Me.passes({"point": 0.4, "ci95": [-0.1, 0.7]}, 0.3))
        self.assertTrue(Me.passes({"point": -0.05, "ci95": [-0.08, 0.02]}, -0.1))
        self.assertFalse(Me.passes({"point": -0.05, "ci95": [-0.12, 0.02]}, -0.1))

    def test_h1_on_synthetic_rows(self):
        chg, prot = [], []
        vals_c = {"A6": 2.0, "OC": 1.0, "A8g": 1.1, "A8": 1.6}
        vals_p = {"A6": 1.0, "OC": 2.0, "A8g": 1.05, "A8": 1.0}
        for b in prereg.CALIB["a1_beta0"]:
            vals_c[f"A1@{b:g}"] = 2.0 - 0.8 * b
            vals_p[f"A1@{b:g}"] = 1.0 + 0.9 * b
            vals_c[f"A1BF@{b:g}"] = 2.0 - 0.85 * b
            vals_p[f"A1BF@{b:g}"] = 1.0 + 0.85 * b
        rng = np.random.default_rng(0)
        for i in range(60):
            rc = {"cond": "CHANGE", "uid": f"c{i}"}
            rp = {"cond": "EXACT", "uid": f"p{i}"}
            for a, v in vals_c.items():
                rc[f"{a}|POST|num"], rc[f"{a}|POST|den"] = v * 10 * (1 + 0.01 * rng.standard_normal()), 10.0
            for a, v in vals_p.items():
                rp[f"{a}|POST|num"], rp[f"{a}|POST|den"] = v * 10 * (1 + 0.01 * rng.standard_normal()), 10.0
            chg.append(rc)
            prot.append(rp)
        res = report.h1_seed(chg + prot, t0=True)
        self.assertTrue(res["i_not_dominated"])
        self.assertTrue(res["ii_pass"])
        self.assertTrue(res["iii_pass"])
        self.assertTrue(res["pass"])


class TestVSC(unittest.TestCase):
    def test_aggregate_merges_familiar(self):
        mus = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 0.01, 0])]
        pred, vet = vsc.aggregate(mus, [2.0, 3.0, 2.0], [0.0, 0.0, -5.0], thr=-1.0)
        self.assertEqual(pred, 5.0)
        self.assertEqual(vet, 1)
        pred, _ = vsc.aggregate(mus, [2.0, 3.0, 2.0], [0.0, 0.0, -5.0], thr=-np.inf)
        self.assertEqual(pred, 7.0)

    def test_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            res = jobs.vsc_synthetic({"out": str(Path(d) / "v")})
            self.assertEqual(res["status"], "complete")
            row = res["row"]
            self.assertIn("A8|pred", row)
            self.assertGreaterEqual(row["M0|pred"], row["M3|pred"])


if __name__ == "__main__":
    unittest.main()
