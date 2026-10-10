"""Synthetic counterexamples for fixed-reader counterfactual scoring; CPU only."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import score_v5_2_counterfactual as scorer


class CounterfactualScore(unittest.TestCase):
    def test_removing_shared_dx_error_gives_known_effect(self):
        target = np.ones((6, 2, 8))
        prediction = np.repeat(target[:, :, None], 4, axis=2)
        # Original/swap errors are shared by both worlds' variants. The static
        # conditions recover the target exactly, with no reader refit.
        for pair in range(6):
            bias = .5 if pair % 2 else -.5
            prediction[pair, :, 0, [4, 6]] += bias
            prediction[pair, :, 3, [4, 6]] -= bias
        result = scorer.paired_effect(prediction, target, np.zeros(8), replicates=300)
        self.assertAlmostEqual(result["primary_dx_mse_reduction"], .25)
        np.testing.assert_allclose(result["primary_dx_paired_95ci"], [.25, .25])
        self.assertAlmostEqual(result["secondary_balanced_dx_mse_reduction"], .25)
        self.assertEqual(result["conditions"]["static_t6"]["dx_mse"], 0)
        self.assertEqual(result["conditions"]["static_t7"]["dx_mse"], 0)
        self.assertEqual(result["bootstrap"]["primary_endpoints"], 1)

    def test_static_comparison_averages_losses_not_predictions(self):
        target = np.ones((5, 2, 8))
        prediction = np.repeat(target[:, :, None], 4, axis=2)
        prediction[:, :, 0, 4:] += 1
        prediction[:, :, 1, 4:] += 1
        prediction[:, :, 2, 4:] -= 1
        prediction[:, :, 3, 4:] += 2
        result = scorer.paired_effect(prediction, target, np.zeros(8), replicates=300)
        # The average static prediction is perfect, but each static condition
        # has MSE1. Therefore the registered primary reduction is0, not1.
        np.testing.assert_array_equal((prediction[:, :, 1]+prediction[:, :, 2])/2, target)
        self.assertEqual(result["primary_dx_mse_reduction"], 0)
        np.testing.assert_array_equal(result["primary_dx_paired_95ci"], [0, 0])
        self.assertEqual(result["secondary_balanced_dx_mse_reduction"], 1.5)
        self.assertEqual(result["secondary_static_average_motion_nmse_reduction"], 0)

    def test_bootstrap_preserves_all_variants_and_conditions_within_pair(self):
        rng = np.random.default_rng(117)
        target = rng.uniform(.3, 2.4, (7, 2, 8))
        prediction = target[:, :, None]+rng.normal(size=(7, 2, 4, 8))*.3
        mean = np.linspace(-.1, .1, 8)
        replicates = 301  # Exercises the scorer's250-row bootstrap batch edge.
        result = scorer.paired_effect(prediction, target, mean, replicates)
        # Independent explicit loops expose pair/variant/condition/dimension
        # indexing, rather than deriving the effect from the returned scores.
        losses = np.empty((7, 4))
        motion_losses = np.empty((7, 4, 4))
        baseline = np.empty((7, 4))
        for pair in range(7):
            for condition in range(4):
                losses[pair, condition] = np.mean([
                    (prediction[pair, variant, condition, dim]-target[pair, variant, dim])**2
                    for variant in range(2) for dim in (4, 6)])
                for d, dim in enumerate(range(4, 8)):
                    motion_losses[pair, condition, d] = np.mean([
                        (prediction[pair, variant, condition, dim]-target[pair, variant, dim])**2
                        for variant in range(2)])
            for d, dim in enumerate(range(4, 8)):
                baseline[pair, d] = np.mean([(target[pair, variant, dim]-mean[dim])**2 for variant in range(2)])
        primary = losses[:, 0]-(losses[:, 1]+losses[:, 2])/2
        balanced = (losses[:, 0]+losses[:, 3]-losses[:, 1]-losses[:, 2])/2
        ids = np.random.default_rng(scorer.BOOT_SEED).integers(0, 7, (replicates, 7))
        np.testing.assert_allclose(result["primary_dx_mse_reduction"], primary.mean(), atol=1e-14)
        np.testing.assert_allclose(result["primary_dx_paired_95ci"], np.quantile(primary[ids].mean(1), [.025, .975]), atol=1e-14)
        np.testing.assert_allclose(result["secondary_balanced_dx_paired_95ci"], np.quantile(balanced[ids].mean(1), [.025, .975]), atol=1e-14)
        improvement = motion_losses[:, 0]-(motion_losses[:, 1]+motion_losses[:, 2])/2
        normalized = (improvement[ids].mean(1)/baseline[ids].mean(1)).mean(1)
        np.testing.assert_allclose(result["secondary_motion_paired_95ci"], np.quantile(normalized, [.025, .975]), atol=1e-14)
        self.assertEqual(result["bootstrap"]["unit"], "paired_world_with_all_variants_and_conditions")

    def test_four_unique_image_indices_preserve_frame_content(self):
        # Unique images encode [frame identity, yellow-position identity].
        unique = np.array([[6, 6], [6, 7], [7, 6], [7, 7]])
        expected = np.array([[[6, 6], [7, 7]], [[6, 6], [7, 6]],
                             [[6, 7], [7, 7]], [[6, 7], [7, 6]]])
        np.testing.assert_array_equal(unique[scorer.INDICES], expected)
        self.assertEqual(scorer.CONDITIONS, ("original", "static_t6", "static_t7", "swap_nuisance"))
        # The flattened scorer indexing must never cross a pair/variant boundary.
        sample_ids = np.repeat(np.arange(6), 4)
        indices = (np.arange(6)[:, None, None]*4+scorer.INDICES[None]).reshape(-1, 2)
        np.testing.assert_array_equal(sample_ids[indices], np.repeat(np.arange(6), 4)[:, None].repeat(2, axis=1))
        balanced = np.concatenate((unique[scorer.INDICES[0]], unique[scorer.INDICES[3]])).reshape(2, 2, 2)
        static = np.concatenate((unique[scorer.INDICES[1]], unique[scorer.INDICES[2]])).reshape(2, 2, 2)
        for slot in range(2):
            np.testing.assert_array_equal(np.sort(balanced[:, slot, 1]), np.sort(static[:, slot, 1]))

    def test_wrong_condition_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pair,variant,condition"):
            scorer.paired_effect(np.zeros((3, 2, 3, 8)), np.ones((3, 2, 8)), np.zeros(8), 20)

    def test_original_parity_one_percent_rms_boundary(self):
        target = np.ones((8, 2, 8))
        prior = target.copy()
        for change, accepted in ((.0099, True), (.0101, False)):
            current = prior.copy(); current[:, :, 4] += change
            result = scorer.original_parity(current, prior, target, np.zeros(8))
            self.assertEqual(result["checks"]["motion_rms_drift_fraction_at_most_001"], accepted)
            self.assertTrue(result["checks"]["motion_max_drift_fraction_at_most_01"])
            self.assertTrue(result["checks"]["each_motion_skill_drift_at_most_002"])
            self.assertEqual(result["passed"], accepted)

    def test_original_parity_ten_percent_max_boundary(self):
        target = np.ones((1000, 2, 8))
        prior = target.copy()
        for change, accepted in ((.099, True), (.101, False)):
            current = prior.copy(); current[0, 0, 4] += change
            result = scorer.original_parity(current, prior, target, np.zeros(8))
            self.assertTrue(result["checks"]["motion_rms_drift_fraction_at_most_001"])
            self.assertEqual(result["checks"]["motion_max_drift_fraction_at_most_01"], accepted)
            self.assertTrue(result["checks"]["each_motion_skill_drift_at_most_002"])
            self.assertEqual(result["passed"], accepted)

    def test_original_parity_point_zero_two_skill_boundary(self):
        target = np.ones((8, 2, 8))
        prior = target.copy(); prior[:, :, 4] += 3
        for change, accepted in ((.0033, True), (.0034, False)):
            current = prior.copy(); current[:, :, 4] += change
            result = scorer.original_parity(current, prior, target, np.zeros(8))
            self.assertTrue(result["checks"]["motion_rms_drift_fraction_at_most_001"])
            self.assertTrue(result["checks"]["motion_max_drift_fraction_at_most_01"])
            self.assertEqual(result["checks"]["each_motion_skill_drift_at_most_002"], accepted)
            self.assertEqual(result["passed"], accepted)

    def test_perfect_residuals_use_null_for_undefined_diagnostics(self):
        target = np.ones((3, 2, 8))
        prediction = np.repeat(target[:, :, None], 4, axis=2)
        summary, rows = scorer.residual_diagnostics(prediction, target, np.array([128, 129, 130]))
        for condition in scorer.CONDITIONS:
            for axis in ("A_dx", "A_dy", "B_dx", "B_dy"):
                result = summary[condition][axis]
                self.assertEqual(result["mse"], 0)
                self.assertIsNone(result["pair_common_mse_fraction"])
                self.assertIsNone(result["pair_common_residual_corr_original_nuisance_dx"])
                self.assertEqual(result["direction_error_rate"], 0)
        self.assertEqual(len(rows), 3*2*4)
        self.assertIn("null", json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
