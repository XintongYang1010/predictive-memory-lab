"""Software-only counterexamples: never instantiate fit/confirm or reserved Test."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from v5_temporal_confirmation import (ORDERS, choose_reader, contrast_skills,
                                      kernel_predict, make_quartet, score_quartets, seed_for)


class TemporalConfirmation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = [make_quartet("software_test", i) for i in range(8)]
        cls.targets = np.stack([b.targets for b in cls.blocks])

    def test_end_and_start_anchor_pixels_identical(self):
        for b in self.blocks:
            x = b.actor_inputs()["rgb"]
            np.testing.assert_array_equal(x[0, 1], x[1, 1])
            np.testing.assert_array_equal(x[2, 0], x[3, 0])

    def test_exact_reversals_pixels_and_labels(self):
        for b in self.blocks:
            x = b.actor_inputs()["rgb"]
            np.testing.assert_array_equal(x[0], x[3, ::-1])
            np.testing.assert_array_equal(x[1], x[2, ::-1])
            np.testing.assert_array_equal(b.targets[0, 4:], -b.targets[3, 4:])
            np.testing.assert_array_equal(b.targets[1, 4:], -b.targets[2, 4:])

    def test_no_reflection_and_displacement_matches_positions(self):
        for b in self.blocks:
            self.assertTrue(np.all(b.positions > [.1, .15, .1, .6]))
            self.assertTrue(np.all(b.positions < [.9, .4, .9, .85]))
            np.testing.assert_array_equal(b.targets[:, 4:], b.positions[ORDERS[:, 1]]-b.positions[ORDERS[:, 0]])
            self.assertTrue(np.all((np.abs(b.targets[:, 4:]) > .00799) & (np.abs(b.targets[:, 4:]) < .01801)))

    def test_actor_has_no_group_or_variant_or_state(self):
        for b in self.blocks:
            a = b.actor_inputs()
            self.assertEqual(set(a), {"rgb", "times", "shutters"})
            np.testing.assert_array_equal(a["times"], np.tile([6., 7.], (4, 1)))
            self.assertTrue(np.all(a["shutters"] == 1))

    def test_any_last_frame_function_fails_end_contrast(self):
        p7 = self.targets[:, :, :4].astype(float)
        arbitrary = np.sin(11*p7)+7*p7**2
        scores = contrast_skills(arbitrary, self.targets[:, :, 4:])
        np.testing.assert_array_equal(scores["end_anchor"], np.zeros(4))

    def test_any_first_frame_function_fails_start_contrast(self):
        p6 = (self.targets[:, :, :4]-self.targets[:, :, 4:]).astype(float)
        arbitrary = np.cos(9*p6)+3*p6**3
        scores = contrast_skills(arbitrary, self.targets[:, :, 4:])
        np.testing.assert_array_equal(scores["start_anchor"], np.zeros(4))

    def test_perfect_two_frame_coordinate_oracle_passes_metrics_only(self):
        result = score_quartets(self.targets, self.targets, np.zeros((8, 4)), self.targets[:, :, :4].mean((0, 1)))
        self.assertTrue(result["metric_conditions_met"])
        self.assertFalse(result["p0_authorized"])
        self.assertEqual(result["primary_lower_bounds"], [1.]*9)

    def test_zero_motion_prediction_cannot_pass(self):
        p = self.targets.copy(); p[:, :, 4:] = 0
        result = score_quartets(p, self.targets, np.zeros((8, 4)), self.targets[:, :, :4].mean((0, 1)))
        self.assertFalse(result["metric_conditions_met"])
        self.assertEqual(result["primary_lower_bounds"], [0.]*9)

    def test_reject_test_namespace_and_bad_order(self):
        for role in ("test", "Test", "dev", ""):
            with self.assertRaises(ValueError):
                seed_for(role, 0)
        bad = self.targets.copy(); bad[0, 0, 4] += .01
        with self.assertRaises(ValueError):
            contrast_skills(bad[:, :, 4:], bad[:, :, 4:])

    def test_quadratic_matches_explicit_feature_map(self):
        rng = np.random.default_rng(4)
        x, q, y = rng.normal(size=(12, 3)), rng.normal(size=(4, 3)), rng.normal(size=(12, 8))
        xc, qc = x-x.mean(0), q-x.mean(0)
        scale = np.sqrt((xc**2).sum(1).mean()); xc, qc = xc/scale, qc/scale
        def phi(a):
            return np.c_[np.ones(len(a)), np.sqrt(2)*a, np.einsum("bi,bj->bij", a, a).reshape(len(a), -1)]
        a, b = phi(xc), phi(qc)
        mean = a.mean(0); a, b = a-mean, b-mean
        s = (a*a).sum(1).mean()
        expected = (b@a.T/s)@np.linalg.solve(a@a.T/s+12*.001*np.eye(12), y-y.mean(0))+y.mean(0)
        actual = kernel_predict(x@x.T, q@x.T, y, "quadratic", .001)
        np.testing.assert_allclose(actual, expected, atol=1e-11)

    def test_fixed_train_selection_keeps_all_four_views_together(self):
        x = np.stack([b.positions[ORDERS].reshape(4, -1) for b in self.blocks]).reshape(32, -1)
        y = self.targets.reshape(32, 8)
        groups = [f"software-{i}" for i in range(8) for _ in range(4)]
        result = choose_reader(x@x.T, y, groups)
        self.assertEqual(len(result["trials"]), 12)
        held = result["held_groups"]
        self.assertEqual(sorted(sum(held, [])), sorted(set(groups)))
        self.assertTrue(all(len(s) == 2 for s in held))


if __name__ == "__main__":
    unittest.main()
