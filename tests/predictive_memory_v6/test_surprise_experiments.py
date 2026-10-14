import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pmr.predictive_memory_v6 import experiments, prereg
from pmr.predictive_memory_v6.data import make_world, geometry_of
from pmr.predictive_memory_v6.surprise import (rasterize, frame_surprise, segment_count, novelty_count,
                                               true_count, nn_novelty, reveal_events)


class TestSurprise(unittest.TestCase):
    def test_segment_sum_double_counts_at_cuts(self):
        vis = np.zeros((1, 10, 3), bool)
        vis[0, 1, 0] = vis[0, 2, 1] = vis[0, 6, 0] = vis[0, 7, 1] = True
        s = np.zeros((1, 10))
        self.assertEqual(segment_count(s, vis, 0.5)[0], 2)
        s[0, 5] = 1.0                      # a cut between the two passes
        self.assertEqual(segment_count(s, vis, 0.5)[0], 4)
        self.assertEqual(true_count(vis)[0], 2)

    def test_novelty_count_and_nn(self):
        d = make_world("l0_tour_static:copies=2", 4, 100, 0)
        g = geometry_of(d)
        nn = nn_novelty(d, g, admit_thr=0.5)
        cnt = novelty_count(nn, d["vis"], thr=0.5)
        self.assertTrue(np.array_equal(cnt, true_count(d["vis"])))   # repeats are recognised

    def test_raster_and_frame_surprise(self):
        d = make_world("l0_main", 2, 16, 1)
        F = rasterize(d, geometry_of(d))
        self.assertEqual(F.shape, (2, 16, 64, 18))
        s = frame_surprise(F, F)
        self.assertTrue(np.all(s[:, 0] == 1) and np.allclose(s[:, 1:], 0, atol=1e-6))

    def test_reveal_events(self):
        d = make_world("l0_ctrl", 32, 64, 2)
        ev = reveal_events(d, {"x": np.zeros(d["vis"].shape)})
        self.assertEqual(len(ev["label"]), len(ev["x"]))
        self.assertGreater(ev["label"].sum(), 0)


class TestStages(unittest.TestCase):
    def test_stage_jobs_well_formed(self):
        with tempfile.TemporaryDirectory() as root:
            for st in ("S1", "S2", "S3", "S4", "S6"):
                jobs = experiments.stage_jobs(st, root)
                ids = [j["id"] for j in jobs]
                self.assertEqual(len(ids), len(set(ids)), st)
                known = set(ids)
                for j in jobs:
                    for dep in j.get("deps", []):
                        self.assertIn(dep, known)
            for st in ("S5", "S7"):
                self.assertTrue(experiments.stage_jobs(st, root, cache="/tmp/x"))

    def test_s2_matrix_size(self):
        with tempfile.TemporaryDirectory() as root:
            jobs = experiments.stage_jobs("S2", root)
            trains = [j for j in jobs if j["type"] == "train"]
            self.assertEqual(len(trains), len(experiments.PRIMARY) * 3 * 2)

    def test_prereg_gate(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit):
                experiments.run_stage("S2", root, [], dry_run=True)
            self.assertEqual(experiments.run_stage("S2", root, [], prereg_commit="abc", dry_run=True), 0)

    def test_budget_table(self):
        b = prereg.BUDGET_GPUH
        self.assertLessEqual(sum(b[k] for k in ("S1", "S2", "S3", "S4", "S5", "S6", "S7")) + b["reserve"],
                             b["hard_cap"])
        self.assertLessEqual(b["S1"] + b["S2"] + b["S3"], b["first_decision_point"])

    def test_test_split_mapping(self):
        c = experiments.to_test({"name": "l1_dev", "world": "l1_dev", "T": 500, "n": 10, "kind": "standard"})
        self.assertEqual(c["world"], "l1_test")
        c = experiments.to_test({"name": "mot3_dev", "world": "real_mot_3_dev", "T": 64, "n": -1, "kind": "standard"})
        self.assertEqual(c["world"], "real_mot_3_test")


if __name__ == "__main__":
    unittest.main()
