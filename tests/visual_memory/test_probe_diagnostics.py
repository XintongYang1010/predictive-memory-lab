import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("probe_diagnostics", ROOT/"scripts/diagnose_visual_probe_v5.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class ProbeDiagnostics(unittest.TestCase):
    def test_gram_centering_matches_explicit_original_dual_fit(self):
        generator = np.random.default_rng(31)
        x = generator.normal(size=(12, 19)) + 7
        z = generator.normal(size=(7, 19)) - 3
        y = generator.normal(size=(12, 8))
        # Independent reference: original feature-space centering and torch solve.
        a, b = torch.from_numpy(x), torch.from_numpy(z)
        mean = a.mean(0); a = a-mean; b = b-mean
        scale = a.square().sum(1).mean().clamp_min(1e-12)
        target = torch.from_numpy(y); ym = target.mean(0)
        coefficients = torch.linalg.solve(a@a.T/scale + .01*torch.eye(len(a)), target-ym)
        expected = ((b@a.T/scale)@coefficients+ym).numpy()
        actual = diagnostics.dual_predict(x@x.T, z@x.T, y, .01)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_chunked_frame_kernels_match_explicit_concat_and_difference(self):
        generator = torch.Generator().manual_seed(44)
        def records(n):
            result = []
            for _ in range(n):
                raw = torch.randn(2, 3, 5, generator=generator).bfloat16()
                pooled = torch.randn(8, 2, 4, generator=generator).bfloat16()
                result.append(dict(raw_cue=raw, features=pooled))
            return result
        train, dev = records(6), records(3)
        for mode in ("raw", "pooled"):
            source = lambda row: row["raw_cue"] if mode == "raw" else row["features"][6:8]
            x = np.stack([source(row).double().numpy().reshape(2, -1) for row in train])
            z = np.stack([source(row).double().numpy().reshape(2, -1) for row in dev])
            gram, cross, _ = diagnostics.frame_grams(train, dev, mode, chunk=2)
            views = diagnostics.kernel_views(gram, cross)
            for name, a, b in (("concat", x.reshape(6, -1), z.reshape(3, -1)),
                               ("difference_hybrid", x[:, 1]-x[:, 0], z[:, 1]-z[:, 0])):
                np.testing.assert_allclose(views[name][0], a@a.T, atol=1e-12)
                np.testing.assert_allclose(views[name][1], b@a.T, atol=1e-12)

    def test_grouped_folds_keep_both_variants_and_frames_together(self):
        pairs = [f"train-{pair:06d}" for pair in range(12) for _ in (0, 1)]
        folds = diagnostics.grouped_folds(pairs, np.arange(24), folds=4)
        held_indices = []
        for fit, held in folds:
            fit_pairs, held_pairs = {pairs[i] for i in fit}, {pairs[i] for i in held}
            self.assertFalse(fit_pairs & held_pairs)
            for pair in held_pairs:
                self.assertEqual(sum(pairs[i] == pair for i in held), 2)
            self.assertFalse(set(diagnostics.frame_indices(fit)) & set(diagnostics.frame_indices(held)))
            held_indices.extend(held)
        self.assertEqual(sorted(held_indices), list(range(24)))

    def test_shared_position_decoder_uses_derived_labels_and_differences(self):
        generator = np.random.default_rng(67)
        # An exact shared affine frame->position rule supplies a semantic oracle.
        frames = generator.normal(size=(24, 7)); query = generator.normal(size=(8, 7))
        weight = generator.normal(size=(7, 4)); bias = generator.normal(size=4)
        positions = (frames@weight+bias).reshape(12, 2, 4)
        labels = np.concatenate([positions[:, 1], positions[:, 1]-positions[:, 0]], 1)
        np.testing.assert_allclose(diagnostics.position_targets(labels), frames@weight+bias, atol=1e-15)
        views = diagnostics.kernel_views(frames@frames.T, query@frames.T)
        prediction = diagnostics.method_predict("shared_position", views, np.arange(12), np.arange(4), labels, 1e-8)
        expected_positions = (query@weight+bias).reshape(4, 2, 4)
        expected = np.concatenate([expected_positions[:, 1], expected_positions[:, 1]-expected_positions[:, 0]], 1)
        np.testing.assert_allclose(prediction, expected, rtol=1e-6, atol=1e-6)

    def test_alpha_selection_cannot_see_dev_kernel_or_targets(self):
        generator = np.random.default_rng(18)
        frames = generator.normal(size=(32, 11)); y = generator.normal(size=(16, 8))
        gram = frames@frames.T
        pairs = [f"train-{pair:06d}" for pair in range(8) for _ in (0, 1)]
        plan = json.loads(diagnostics.PLAN_PATH.read_text())
        plan["alpha_grid"] = [.001, .01, .1]
        clean = diagnostics.kernel_views(gram, np.zeros((6, 32)))
        poisoned = diagnostics.kernel_views(gram, np.full((6, 32), np.nan))
        for method in plan["methods"]:
            first = diagnostics.select_alpha(method, clean, np.arange(16), y, pairs, plan)
            second = diagnostics.select_alpha(method, poisoned, np.arange(16), y, pairs, plan)
            self.assertEqual(first, second)

    def test_pair_bootstrap_identity_and_incomplete_pairs(self):
        generator = np.random.default_rng(5)
        target = generator.normal(size=(8, 8)); mean = np.zeros(8)
        pairs = [f"dev-{i}" for i in range(4) for _ in (0, 1)]
        plan = dict(seed=9, replicates=50, confidence=.95)
        result = diagnostics.paired_bootstrap(target, target, mean, pairs, plan, reference=target)
        self.assertEqual(result["displacement_r2_ci"], [1., 1.])
        self.assertEqual(result["difference_from_original_ci"], [0., 0.])
        with self.assertRaisesRegex(ValueError, "complete variants"):
            diagnostics.paired_bootstrap(target, target, mean, list(range(8)), plan)


if __name__ == "__main__":
    unittest.main()
