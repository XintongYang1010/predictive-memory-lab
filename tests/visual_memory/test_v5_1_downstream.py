"""Evidence-boundary tests only: synthetic arrays, no CUDA or scientific data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_v5_1_downstream as downstream


class DownstreamPrerequisites(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ("frozen_model.npz", "train_selection.json"):
            (self.model / name).write_text("isolated verifier fixture\n")
        encoder = downstream.read_json(downstream.reader.PLAN)["encoder"]
        self.identity = dict(plan_sha256=downstream.digest(downstream.reader.PLAN),
            source_commit="a"*40,
            encoder_identity={k: encoder[k] for k in ("source_commit", "checkpoint_sha256", "model")},
            source_identity={"scripts/v5_temporal_confirmation.py":
                downstream.digest(downstream.ROOT / "scripts/v5_temporal_confirmation.py")},
            pooling="4x4_flattened", feature_dtype="float32")
        self.frozen = dict(cache_identity=self.identity, receipt_sha256="b"*64,
            predictor_input_audit=dict(features_only=True, labels_or_group_ids_used_at_prediction=False))
        self.arrays = {"concat__target_mean": np.array([.5, .275, .5, .725, 0, 0, 0, 0])}
        self.selections = {"concat": {"selected": {"kernel": "linear", "mean_loss_alpha": .001}}}
        # These are deterministic unit-test labels, not FIT/confirmation generator calls.
        center = np.tile(np.array([.5, .275, .5, .725], dtype=np.float32), (256, 1))
        center[:, 0] += np.linspace(-.1, .1, 256, dtype=np.float32)
        delta = np.tile(np.array([1, -1, -1, 1], dtype=np.float32)/64, (256, 1))
        delta[1::2] *= -1
        y = np.zeros((256, 4, 8), dtype=np.float32)
        y[:, 0, :4] = y[:, 1, :4] = center
        y[:, 2, :4], y[:, 3, :4] = center+delta, center-delta
        y[:, 0, 4:] = y[:, 2, 4:] = delta
        y[:, 1, 4:] = y[:, 3, 4:] = -delta
        self.make_report("confirm", y)
        self.make_report("bridge", y[:, [2, 3]])

    def make_report(self, role, target):
        folder = self.root / role
        folder.mkdir()
        arrays = dict(concat=target.copy(), targets=target)
        if role == "confirm":
            arrays["concat_repeat"] = np.zeros((256, 4), dtype=np.float64)
            score = downstream.score_quartets(target, target, arrays["concat_repeat"],
                                              self.arrays["concat__target_mean"][:4])
        else:
            score = downstream.reader.score_bridge(target, target, self.arrays["concat__target_mean"])
        np.savez(folder / "predictions.npz", **arrays)
        ids_key = "block_ids" if role == "confirm" else "pair_ids"
        ids = np.arange(256, dtype=np.int64) if role == "confirm" else np.arange(128, 384, dtype=np.int64)
        cache = dict(schema=downstream.CONFIRM_SCHEMA, role=role, **self.identity,
            cache_sha256="c"*64, frozen_reader_sha256=self.frozen["receipt_sha256"],
            frozen_before_generation=True, unseen_before_release=True,
            unique_image_rule="encode_once_reuse" if role == "confirm" else "original_variant_cues",
            groups=256, examples=1024 if role == "confirm" else 512, test_opened=False,
            entry_hashes={ids_key: downstream.array_identity(ids),
                          "targets": downstream.array_identity(target)})
        downstream.atomic_json(folder / "cache_receipt.json", cache)
        report = dict(schema=downstream.CONFIRM_SCHEMA, role=role,
            counts=downstream.EXPECTED_COUNTS[role], test_access=False, primary_reader="concat",
            primary_metric_conditions_met=True, original_v5_gate="failed_unchanged",
            sources=downstream.reader.current_sources(), frozen_reader_sha256=self.frozen["receipt_sha256"],
            cache_identity=self.identity, cache_sha256=cache["cache_sha256"],
            cache_receipt_sha256=downstream.digest(folder / "cache_receipt.json"),
            predictions_sha256=downstream.digest(folder / "predictions.npz"), scores={"concat": score})
        downstream.atomic_json(folder / "report.json", report)

    def validate(self):
        # Frozen-model serialization/source checks are separately covered by the reader tests.
        # This isolates downstream report binding; it cannot invoke any GPU stage.
        with mock.patch.object(downstream.reader, "load_frozen",
                               return_value=(self.arrays, self.selections, self.frozen)):
            return downstream.validate_prerequisites(self.model, self.root / "confirm", self.root / "bridge")

    def change_report(self, role, update):
        path = self.root / role / "report.json"
        report = downstream.read_json(path)
        update(report)
        downstream.atomic_json(path, report)

    def test_valid_fixture_passes_recomputed_boundary_without_gpu(self):
        with mock.patch.object(downstream.torch.cuda, "is_available", side_effect=AssertionError("No CUDA query")):
            result = self.validate()
        self.assertTrue(result["p0_authorized"])
        self.assertEqual(result["original_v5_gate"], "failed_unchanged")
        self.assertEqual(set(result["evidence"]), {"confirm", "bridge"})

    def test_failed_gate_refuses(self):
        self.change_report("bridge", lambda r: r.update(primary_metric_conditions_met=False))
        with self.assertRaisesRegex(ValueError, "prerequisite failed"):
            self.validate()

    def test_prediction_hash_mismatch_refuses(self):
        path = self.root / "confirm/predictions.npz"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.validate()

    def test_forged_metric_with_same_predictions_refuses(self):
        self.change_report("confirm", lambda r: r["scores"]["concat"].update(position_skill=.9))
        with self.assertRaisesRegex(ValueError, "independently recomputed"):
            self.validate()

    def test_wrong_count_or_test_access_refuses(self):
        self.change_report("confirm", lambda r: r.update(counts={"blocks": 8, "views": 32}))
        with self.assertRaisesRegex(ValueError, "role/count/Test"):
            self.validate()

    def test_changed_source_refuses(self):
        self.frozen = copy.deepcopy(self.frozen)
        self.frozen["cache_identity"]["source_identity"]["scripts/v5_temporal_confirmation.py"] = "0"*64
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.validate()

    def test_duplicate_group_ids_receipt_refuses_even_if_rehashed(self):
        path = self.root / "confirm/cache_receipt.json"
        receipt = downstream.read_json(path)
        receipt["entry_hashes"]["block_ids"] = downstream.array_identity(np.zeros(256, dtype=np.int64))
        downstream.atomic_json(path, receipt)
        self.change_report("confirm", lambda r: r.update(cache_receipt_sha256=downstream.digest(path)))
        with self.assertRaisesRegex(ValueError, "unique group IDs"):
            self.validate()

    def test_budget_cannot_drop_original_charge(self):
        old, new = self.root / "old", self.root / "ledger"
        old.mkdir(); new.mkdir()
        ledger = dict(charged_gpuh=.4, leases=[dict(lease_id="original", charged_gpuh=.4, status="partial")])
        downstream.atomic_json(old / "budget.json", ledger)
        downstream.atomic_json(new / "budget.json", ledger)
        downstream.validate_cumulative_budget(old / "profile.json", new)
        downstream.atomic_json(new / "budget.json", dict(charged_gpuh=0., leases=[]))
        with self.assertRaisesRegex(ValueError, "original charged leases"):
            downstream.validate_cumulative_budget(old / "profile.json", new)


if __name__ == "__main__":
    unittest.main()
