"""Causal information and matched-parent scientific-contract regressions."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
import run as runner

API = runner.imports()
torch, MemoryEntry, StreamingMemory, lib = API


def entry(t, key, value=1., mask=None):
    r = np.full(core.D, value, np.float32)
    o = np.ones(core.D, np.float32) if mask is None else np.asarray(mask, np.float32)
    return MemoryEntry("e000", t, core.representation(r, o, key))


class Contracts(unittest.TestCase):
    def test_all_keys_roundtrip(self):
        for key in range(core.DK):
            self.assertEqual(core.item_key(entry(0, key)), key)

    def test_masked_attention_reconstructs_prediction(self):
        mask = np.zeros(core.D, np.float32); mask[:17] = 1
        entries = (entry(0, 4, 2.), entry(1, 4, 6., mask), entry(2, 8, 100.))
        prediction, weights = core.read(entries, 4, 3)
        np.testing.assert_allclose(prediction[:17], 4)
        np.testing.assert_allclose(prediction[17:], 2)
        np.testing.assert_allclose(weights.sum(0), 1)
        np.testing.assert_array_equal(weights[2], 0)

    def test_current_and_future_entries_rejected(self):
        with self.assertRaisesRegex(ValueError, "current/future"):
            core.read((entry(3, 4),), 4, 3)

    def test_teacher_requires_future_after_decision(self):
        with self.assertRaisesRegex(ValueError, "genuinely future"):
            runner.future_attention_teacher(lib, (entry(0, 4),), (entry(1, 4),), [4], [1], 1)

    def test_teacher_is_real_conditional_attention(self):
        candidates = (entry(0, 4), entry(1, 9))
        target, trace = runner.future_attention_teacher(lib, candidates, (entry(2, 4),),
                                                        np.array([4, 4]), np.array([3, 4]), 2)
        np.testing.assert_array_equal(target, [1., 0.])
        self.assertEqual(trace["history_attention_mass"], [.5, .5])
        self.assertTrue(trace["future_group_only"])

    def test_no_attention_does_not_invent_uniform_labels(self):
        target, trace = runner.future_attention_teacher(lib, (entry(0, 4),), (entry(1, 8),), [9], [2], 1)
        self.assertIsNone(target)
        self.assertEqual(trace["valid_queries"], 0)

    def test_local_oracle_same_candidates_and_recent(self):
        candidates = (entry(0, 4, 1), entry(1, 4, 10))
        data = (np.ones((8, core.D), np.float32), np.ones((8, core.D), np.float32),
                np.ones(8, np.float32), np.full(8, 4, np.int64))
        result = runner.frozen_parent_oracle(candidates, (), data, 2, candidates[0].entry_id)
        self.assertEqual(result["oracle_drop"], 1)
        self.assertEqual(result["chosen_drop"], 0)
        self.assertAlmostEqual(result["regret"], 81.)
        self.assertEqual(result["candidate_times"], [0, 1])

    def test_source_split_purges_shared_boundary(self):
        ids = [f"source{i}" for i in range(33)]
        split = core.source_split({"ordered_a_ids": ids[:32], "original_b_pairs": [[ids[i], ids[i+1]] for i in range(32)]})
        self.assertFalse(set(split["train_sources"]) & set(split["dev_sources"]))
        self.assertEqual(split["excluded_indices"], [23])

    def test_future_values_cannot_change_online_prefix(self):
        rng = np.random.default_rng(2)
        r = rng.normal(size=(35, core.D)).astype(np.float32)
        o = np.ones_like(r); w = np.ones(35, np.float32)
        keys = rng.integers(0, 5, 35)
        modified = r.copy(); modified[20:] += 100
        for method in ("FIFO", "WR"):
            p1, _, a = runner.run_stream((r, o, w, keys), 0, 4, method, None, API, audit=False)
            p2, _, _ = runner.run_stream((modified, o, w, keys), 0, 4, method, None, API, audit=False)
            np.testing.assert_array_equal(p1[:21], p2[:21])
            self.assertLessEqual(a["peak_actual_controller_bytes"], core.CONTROLLER_RESERVATION)

    def test_current_target_not_in_prediction_and_invalid_write_skipped(self):
        r = np.ones((8, core.D), np.float32); r[0] = 9; r[3] = 1000
        o = np.ones_like(r); w = np.ones(8, np.float32); w[3] = 0
        p, _, _ = runner.run_stream((r, o, w, np.zeros(8, np.int64)), 0, 4, "FIFO", None, API, audit=False)
        np.testing.assert_array_equal(p[0], 0)
        np.testing.assert_array_equal(p[1], 9)
        np.testing.assert_allclose(p[4], 11 / 3)

    def test_pafu_future_values_do_not_change_prefix(self):
        torch.manual_seed(17)
        model = lib.SalienceStudent(5 * core.FEATURE_DIM + 1, 32).eval()
        rng = np.random.default_rng(2)
        r = rng.normal(size=(25, core.D)).astype(np.float32)
        modified = r.copy(); modified[16:] -= 100
        o = np.ones_like(r); w = np.ones(25, np.float32); keys = rng.integers(0, 6, 25)
        p1, _, a = runner.run_stream((r, o, w, keys), 0, 4, "PaFu", model, API, audit=False)
        p2, _, _ = runner.run_stream((modified, o, w, keys), 0, 4, "PaFu", model, API, audit=False)
        np.testing.assert_array_equal(p1[:17], p2[:17])
        self.assertGreater(a["student_parameter_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
