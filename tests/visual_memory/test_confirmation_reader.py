"""Software-only reader tests: never generate registered FIT/Confirm/bridge data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import v5_confirmation_reader as reader
from v5_temporal_confirmation import kernel_predict, make_quartet


class ConfirmationReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blocks = [make_quartet("software_test", i) for i in range(8)]
        cls.targets = np.stack([b.targets for b in cls.blocks])
        # Synthetic features derived only from the software-test namespace.
        rng = np.random.default_rng(481)
        cls.features = rng.normal(size=(8, 3, 16)).astype(np.float32)

    def test_serialized_moments_match_reference_both_kernels(self):
        rng = np.random.default_rng(29)
        x, q, y = rng.normal(size=(19, 7)), rng.normal(size=(5, 7)), rng.normal(size=(19, 8))
        for kernel in ("linear", "quadratic"):
            state = reader.fit_kernel(x@x.T, y, kernel, .0007)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td)/"state.npz"
                np.savez(path, **state)
                with np.load(path, allow_pickle=False) as z:
                    restored = {k: z[k] for k in z.files}
            actual = reader.predict_kernel(q@x.T, restored, kernel)
            expected = kernel_predict(x@x.T, q@x.T, y, kernel, .0007)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            # Query batch membership must not change prediction/normalization.
            single = np.concatenate([reader.predict_kernel(q[i:i+1]@x.T, restored, kernel) for i in range(len(q))])
            np.testing.assert_allclose(actual, single, rtol=1e-10, atol=1e-10)

    def test_indexed_grams_equal_explicit_cue_inputs(self):
        q = self.features[:3]
        a, b = q.reshape(-1, 16), self.features.reshape(-1, 16)
        qi, bi = reader.quartet_indices(3), reader.quartet_indices(8)
        raw = reader.dot_gram(a, b, chunk=5)
        for name in reader.INPUTS:
            cols = [0, 1] if name == "concat" else ([0] if name == "t6" else [1])
            left = a[qi[:, cols]].reshape(len(qi), -1).astype(float)
            right = b[bi[:, cols]].reshape(len(bi), -1).astype(float)
            np.testing.assert_allclose(reader.view_gram(raw, qi, bi, name), left@right.T, atol=1e-12)

    def test_all_three_readers_use_same_grouped_cv_and_survive_reload(self):
        arrays, selection = reader.train_models(self.features, self.targets, np.arange(8))
        groups = selection["concat"]["held_groups"]
        for name in reader.INPUTS:
            self.assertEqual(len(selection[name]["trials"]), 12)
            self.assertEqual(selection[name]["held_groups"], groups)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"model.npz"
            np.savez(path, **arrays)
            with np.load(path, allow_pickle=False) as z:
                restored = {k: z[k] for k in z.files}
        index = reader.quartet_indices(8)
        before = reader.predict_readers(arrays, selection, self.features.reshape(-1, 16), index)
        after = reader.predict_readers(restored, selection, self.features.reshape(-1, 16), index)
        for name in reader.INPUTS:
            np.testing.assert_array_equal(before[name], after[name])

    def test_production_rejects_small_cache_and_wrong_role(self):
        with tempfile.TemporaryDirectory() as td:
            path, receipt_path = Path(td)/"cache.npz", Path(td)/"receipt.json"
            np.savez(path, features=self.features, targets=self.targets, block_ids=np.arange(8), role=np.array("fit"))
            receipt = dict(schema=reader.SCHEMA, role="fit", cache_sha256=reader.sha256(path),
                plan_sha256=reader.sha256(reader.PLAN), source_commit="software-only-fixture",
                encoder_identity={}, source_identity={}, pooling="4x4_flattened", feature_dtype="float32",
                entry_hashes={"software_fixture": "unused"}, unique_image_rule="encode_once_reuse")
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "shape/count/dtype"):
                reader.load_cache(path, receipt_path, "fit")
            with self.assertRaisesRegex(ValueError, "role mismatch"):
                reader.load_cache(path, receipt_path, "confirm", {})
            with self.assertRaisesRegex(ValueError, "Only registered"):
                reader.load_cache(path, receipt_path, "test")
            receipt["cache_sha256"] = "0"*64
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                reader.load_cache(path, receipt_path, "fit")

    def test_bridge_pair_bootstrap_known_relative_error(self):
        # Synthetic paired data with exact proportional errors; every resample
        # has the same score, proving denominator and paired indexing behavior.
        rng = np.random.default_rng(718)
        target = rng.normal(size=(8, 2, 8))
        fit_mean = np.arange(8)/20
        fraction = np.arange(1, 9)/10
        prediction = target-(target-fit_mean)*fraction
        score = reader.score_bridge(prediction, target, fit_mean)
        expected = 1-fraction**2
        np.testing.assert_allclose(score["per_dimension_skill"], expected, atol=1e-14)
        self.assertAlmostEqual(score["motion_mean_lower_bound"], expected[4:].mean(), places=14)
        self.assertTrue(score["metric_conditions_met"])
        self.assertFalse(score["p0_authorized"])
        self.assertEqual(score["bootstrap"]["unit"], "paired_world")

    def test_frozen_files_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            np.savez(out/"frozen_model.npz", fake=np.array([1]))
            (out/"train_selection.json").write_text("{}")
            receipt = dict(schema=reader.SCHEMA, status="frozen_before_confirmation", fit_blocks=256,
                fit_views=1024, sources=reader.current_sources(), files={
                    name: reader.sha256(out/name) for name in ("frozen_model.npz", "train_selection.json")})
            (out/"freeze_receipt.json").write_text(json.dumps(receipt))
            (out/"train_selection.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reader.load_frozen(out)


if __name__ == "__main__":
    unittest.main()
