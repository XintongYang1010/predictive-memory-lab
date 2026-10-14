import unittest

import numpy as np

from pmr.predictive_memory_v6 import worlds_l0
from pmr.predictive_memory_v6.data import make_world, parse_world, l0_config
from pmr.predictive_memory_v6.worlds_l0 import L0Config, generate, history_pair, change_pair


class TestL0(unittest.TestCase):
    def test_flowm_spec_and_truth(self):
        c = L0Config(visibility="window")
        d = generate(c, 32, 64, 0)
        self.assertEqual(d["pos"].shape, (32, 64, 5, 2))
        v = d["vel"]
        self.assertTrue(np.all(np.abs(v) <= 2) and np.allclose(v, np.round(v)))
        self.assertTrue(np.all(np.abs(d["u"]) <= 10))
        step = d["pos_unwrapped"][:, 1:] - d["pos_unwrapped"][:, :-1]
        self.assertTrue(np.allclose(step, v[:, :-1]))
        rel = np.mod(d["pos"] - d["ego"][:, :, None], 50)
        self.assertTrue(np.array_equal(d["vis"], np.all(rel < 32, -1)))
        self.assertTrue(np.all(d["obs"][~d["vis"]] == 0))

    def test_no_velocity_in_observations(self):
        d = generate(L0Config(visibility="schedule"), 8, 32, 1)
        self.assertNotIn("velocity", d)
        self.assertEqual(d["obs"].shape[-1], 2)

    def test_schedule_first_run_identifiable(self):
        d = generate(L0Config(visibility="schedule"), 64, 64, 2)
        self.assertTrue(d["vis"][:, :2].all())

    def test_changes_are_labelled_at_reveal(self):
        d = generate(L0Config(visibility="schedule", p_change=1.0), 64, 64, 3)
        n_changes = (d["change_step"] > 0).sum()
        self.assertGreater(n_changes, 0)
        self.assertTrue(d["vis"][d["change_step"] > 0].sum() == 0)  # changes happen while hidden
        self.assertLessEqual(d["reveal_changed"].sum(), n_changes)
        self.assertGreater(d["reveal_changed"].sum(), 0)

    def test_appearance_cosine(self):
        for rho in (0.0, 0.5, 0.9):
            a = worlds_l0.appearance_codes(np.random.default_rng(0), 4, 5, 16, rho)
            g = np.einsum("bik,bjk->bij", a, a)
            off = g[:, ~np.eye(5, dtype=bool)]
            self.assertTrue(np.allclose(off, rho, atol=1e-5))
            self.assertTrue(np.allclose(np.diagonal(g, axis1=1, axis2=2), 1, atol=1e-5))

    def test_history_pair_identical_inputs(self):
        a, b = history_pair(L0Config(), 16, 64, 4)
        self.assertTrue(np.array_equal(a["vis"], b["vis"]))
        self.assertTrue(np.array_equal(a["obs"][:, 4:], b["obs"][:, 4:]))
        self.assertTrue(np.array_equal(a["app"], b["app"]))
        self.assertGreater(np.abs(a["pos"][:, -1, 0] - b["pos"][:, -1, 0]).max(), 0)
        self.assertTrue(np.array_equal(a["pos"][:, :, 1:], b["pos"][:, :, 1:]))

    def test_change_pair_differs_only_in_probe(self):
        a, b = change_pair(L0Config(), 32, 64, 5)
        self.assertTrue(a["pair_valid"].mean() > 0.5)
        self.assertTrue(np.array_equal(a["pos"][:, :, 1:], b["pos"][:, :, 1:]))
        self.assertTrue(np.array_equal(a["vis"], b["vis"]))
        for i in np.nonzero(a["pair_valid"])[0]:
            r = a["probe_reveal"][i]
            self.assertTrue(np.array_equal(a["obs"][i, :r], b["obs"][i, :r]))  # identical inputs before reveal

    def test_tour_copies_identical(self):
        d = make_world("l0_tour_static:copies=3", 4, 150, 6)
        self.assertTrue(np.array_equal(d["vis"][:, :50], d["vis"][:, 50:100]))
        self.assertTrue(np.array_equal(d["obs"][:, :50], d["obs"][:, 100:]))

    def test_blind_protocol_and_overrides(self):
        d = make_world("l0_main:blind_after=50", 4, 80, 7)
        self.assertFalse(d["vis"][:, 50:].any())
        name, over = parse_world("l0_ctrl:rho=0.5,n_objects=8")
        self.assertEqual((name, over["rho"], over["n_objects"]), ("l0_ctrl", 0.5, 8))
        cfg, _ = l0_config("l0_sched", 576)
        self.assertEqual(cfg.hidden_scale, 9.0)

    def test_ego_frame(self):
        d = make_world("l0_main:frame=ego", 4, 32, 8)
        self.assertEqual(d["meta"]["frame"], "ego")
        vis = d["vis"]
        self.assertTrue(np.all(d["obs"][vis] < 32))

    def test_drift_world(self):
        d = make_world("l0_drift", 8, 64, 9)
        sp = np.linalg.norm(d["vel"], axis=-1)
        self.assertTrue(np.all(sp[:, -1] <= sp[:, 0] + 1e-9))  # damping never increases speed


if __name__ == "__main__":
    unittest.main()
