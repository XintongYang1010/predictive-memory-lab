"""CPU-only extraction tests; no scientific FIT/confirm/bridge inputs generated."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import extract_v5_confirmation as extract
from v5_confirmation_reader import validate_entry_hashes
from v5_temporal_confirmation import ORDERS, make_quartet


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, images):
        self.calls.append(images.copy())
        values = images.astype(np.float32).mean((1, 2, 3))
        return torch.from_numpy(np.broadcast_to(values[:, None, None], (len(images), 16, 2)).copy())


class ConfirmationExtraction(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_scalar_role_hash_preserves_npz_scalar_shape(self):
        value = np.array("software_test")
        receipt = extract.array_identity(value)
        self.assertEqual(receipt["shape"], [])
        self.assertEqual(receipt["dtype"], str(value.dtype))
        self.assertEqual(receipt["sha256"], extract.hashlib.sha256(value.tobytes()).hexdigest())

    def test_npz_entry_hashes_are_accepted_by_reader(self):
        arrays = dict(features=np.zeros((2, 3, 32), dtype=np.float32),
                      targets=np.zeros((2, 4, 8), dtype=np.float32),
                      block_ids=np.arange(2, dtype=np.int64), role=np.array("software_test"))
        path = self.root/"software_fixture.npz"
        extract.atomic_npz(path, **arrays)
        entries = {name: extract.array_identity(value) for name, value in arrays.items()}
        with np.load(path, allow_pickle=False) as contents:
            validate_entry_hashes({name: contents[name] for name in contents.files}, entries)

    def test_unique_images_encoded_once_and_ordered_views_reuse_exact_tensors(self):
        quartet = make_quartet("software_test", 2)
        encoder = FakeEncoder()
        features, targets, receipt = extract.encode_quartet(quartet, encoder)
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(len(encoder.calls[0]), 3)
        np.testing.assert_array_equal(encoder.calls[0], quartet.unique_rgb)
        np.testing.assert_array_equal(targets, quartet.targets)
        views = features[ORDERS]
        np.testing.assert_array_equal(views[0, 1], views[1, 1])
        np.testing.assert_array_equal(views[2, 0], views[3, 0])
        np.testing.assert_array_equal(views[0], views[3, ::-1])
        self.assertEqual(receipt["unique_images_encoded"], 3)
        self.assertEqual(receipt["metadata"]["role"], "software_test")

    def bridge_fixture(self):
        rows = []
        for variant in (0, 1):
            frames = np.stack([np.full((224, 224, 3), variant*10+i, dtype=np.uint8) for i in range(5)])
            rows.append(SimpleNamespace(rgb=frames, times=np.array([0., 6., 6.5, 7., 8.]),
                shutters=np.ones((5, 2)), probe_state=np.zeros(8, dtype=np.float32),
                metadata=dict(namespace="software_test", variant=variant)))
        return rows

    def test_bridge_uses_actual_cue_times_and_keeps_both_variants(self):
        encoder = FakeEncoder()
        features, targets, inputs = extract.encode_bridge(self.bridge_fixture(), encoder)
        self.assertEqual(features.shape, (2, 2, 32))
        self.assertEqual(targets.shape, (2, 8))
        self.assertEqual(len(encoder.calls), 1)
        np.testing.assert_array_equal(encoder.calls[0][:, 0, 0, 0], [1, 3, 11, 13])
        self.assertEqual(inputs["images_encoded"], 4)

    def test_missing_or_ambiguous_bridge_time_is_rejected_before_encoding(self):
        for times in ([0., 5., 6.5, 7., 8.], [0., 6., 6., 7., 8.]):
            rows = self.bridge_fixture(); rows[0].times = np.array(times)
            encoder = FakeEncoder()
            with self.assertRaises(ValueError):
                extract.encode_bridge(rows, encoder)
            self.assertEqual(encoder.calls, [])

    def test_shard_resume_validates_data_and_configuration(self):
        (self.root/"shards").mkdir()
        x = np.ones((3, 32), dtype=np.float32); y = np.ones((4, 8), dtype=np.float32)
        extract.save_unit(self.root, 2, "software_test", "config-sha", x, y, {"fake": True})
        arrays, _ = extract.read_unit(self.root, 2, "config-sha", "software_test")
        np.testing.assert_array_equal(arrays["features"], x)
        with self.assertRaises(ValueError):
            extract.read_unit(self.root, 2, "different-config", "software_test")
        path, _ = extract.unit_paths(self.root, 2)
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaises(ValueError):
            extract.read_unit(self.root, 2, "config-sha", "software_test")

    def test_orphan_shard_is_preserved_and_not_silently_overwritten(self):
        (self.root/"shards").mkdir()
        path, _ = extract.unit_paths(self.root, 2); path.write_bytes(b"preserved")
        with self.assertRaises(ValueError):
            extract.read_unit(self.root, 2, "config", "software_test")
        with self.assertRaises(FileExistsError):
            extract.save_unit(self.root, 2, "software_test", "config", np.zeros((3, 32)), np.zeros((4, 8)), {})
        self.assertEqual(path.read_bytes(), b"preserved")

    def freeze_fixture(self):
        identity = {key: key for key in extract.IDENTITY_KEYS}
        files = {}
        for name in ("frozen_model.npz", "train_selection.json"):
            (self.root/name).write_bytes(b"software fixture")
            files[name] = extract.digest(self.root/name)
        receipt = dict(schema=extract.SCHEMA, status="frozen_before_confirmation", fit_blocks=256,
            fit_views=1024, cache_identity=identity, files=files,
            sources={name: extract.digest(extract.ROOT/"scripts"/name) for name in
                     ("v5_confirmation_reader.py", "v5_temporal_confirmation.py", "v5_temporal_confirmation_plan.json")})
        path = self.root/"freeze_receipt.json"; path.write_text(json.dumps(receipt))
        return identity, path

    def test_evaluation_cannot_generate_before_frozen_reader_and_attestation(self):
        identity, path = self.freeze_fixture()
        for role in ("confirm", "bridge"):
            with self.assertRaises(ValueError):
                extract.check_freeze(role, None, identity, True)
            with self.assertRaises(ValueError):
                extract.check_freeze(role, path, identity, False)
            self.assertEqual(extract.check_freeze(role, path, identity, True), extract.digest(path))
        (self.root/"frozen_model.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            extract.check_freeze("confirm", path, identity, True)

    def test_budget_is_inherited_and_refunded_even_on_error(self):
        prior = dict(charged_gpuh=.3916116764, stages={}, leases=[])
        (self.root/"budget.json").write_text(json.dumps(prior))
        args = argparse.Namespace(role="fit", workers=8, budget_root=self.root,
                                  out=self.root/"role", lease_hours=.20)
        with patch.object(extract, "run_reserved", side_effect=RuntimeError("software-only failure")):
            with self.assertRaisesRegex(RuntimeError, "software-only failure"):
                extract.run(args)
        ledger = json.loads((self.root/"budget.json").read_text())
        self.assertGreaterEqual(ledger["charged_gpuh"], prior["charged_gpuh"])
        self.assertLess(ledger["charged_gpuh"], prior["charged_gpuh"]+.01)
        self.assertEqual(ledger["leases"][0]["gpu_count"], 8)
        self.assertEqual(ledger["leases"][0]["status"], "partial")

    def test_extractor_never_initializes_empty_budget(self):
        args = argparse.Namespace(role="fit", workers=8, budget_root=self.root,
                                  out=self.root/"role", lease_hours=.20)
        with patch.object(extract, "make_quartet") as generate:
            with self.assertRaisesRegex(ValueError, "Existing inherited"):
                extract.run(args)
            generate.assert_not_called()
        self.assertFalse((self.root/"budget.json").exists())


if __name__ == "__main__":
    unittest.main()
