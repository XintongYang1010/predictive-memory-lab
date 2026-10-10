import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import torch

from pmr.visual_memory.common import BudgetExceeded
from pmr.visual_memory.common import digest
from pmr.visual_memory.data import specs
from pmr.visual_memory.feature_workers import WorkerBudget, paired_shards
from pmr.visual_memory.features import begin_cache, cache_records, cache_split, finish_cache, load_records


class Unlimited:
    def check(self):
        pass


class TinyEncoder:
    # CPU software test only; not reachable from the scientific CLI.
    grid = 2
    identity = {"test_encoder": True, "grid": 2}

    def __call__(self, rgb, raw=False):
        x = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float() / 255
        z = torch.nn.functional.adaptive_avg_pool2d(x, 2).flatten(2).transpose(1, 2).bfloat16()
        return (z, z) if raw else z


class FeatureWorkers(unittest.TestCase):
    def test_shards_are_disjoint_complete_and_keep_pairs_together(self):
        shards = paired_shards("train", 2048, 8)
        flattened = [spec for shard in shards for spec in shard]
        self.assertEqual(set(flattened), set(specs("train", 2048)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual([len(shard) for shard in shards], [256] * 8)
        for index, shard in enumerate(shards):
            self.assertTrue(all(spec[1] % 8 == index for spec in shard))
            self.assertEqual({spec[1] for spec in shard if spec[2] == 0},
                             {spec[1] for spec in shard if spec[2] == 1})

    def test_parallel_record_writes_match_serial_and_parent_owns_completion(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); encoder = TinyEncoder(); budget = Unlimited()
            folder = begin_cache(root / "parallel", "dev", 4, encoder.identity)
            shards = paired_shards("dev", 4, 2)
            cache_records(encoder, folder, shards[0], budget, 5)
            self.assertEqual(json.loads((folder / "manifest.json").read_text())["status"], "partial")
            with self.assertRaisesRegex(ValueError, "inventory"):
                finish_cache(root / "parallel", "dev", 4, encoder.identity, budget)
            cache_records(encoder, folder, shards[1], budget, 5)
            finish_cache(root / "parallel", "dev", 4, encoder.identity, budget)
            inventory=json.loads((folder / "manifest.json").read_text())["files"]
            self.assertEqual(len(inventory),4)
            for name,entry in inventory.items():
                self.assertEqual(entry,{"bytes":(folder/name).stat().st_size,"sha256":digest(folder/name)})
            cache_split(encoder, root / "serial", "dev", 4, budget, 5)
            parallel = load_records(root / "parallel", "dev")
            serial = load_records(root / "serial", "dev")
            for left, right in zip(parallel, serial):
                self.assertEqual(left["metadata"], right["metadata"])
                for key in ("features", "targets", "raw_cue", "times", "shutters", "probe_state", "target_masks"):
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            with (folder/"000000-0.pt").open("ab") as stream:
                stream.write(b"byte change")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                load_records(root/"parallel","dev")

    def test_resume_rejects_wrong_episode_and_test_is_never_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); encoder = TinyEncoder(); budget = Unlimited()
            with self.assertRaisesRegex(ValueError, "Train/Dev"):
                begin_cache(root, "test", 2, encoder.identity)
            self.assertFalse((root / "test").exists())
            folder = begin_cache(root, "train", 2, encoder.identity)
            cache_records(encoder, folder, specs("train", 2), budget, 32)
            path = folder / "000000-0.pt"
            row = torch.load(path, weights_only=True)
            row["metadata"]["variant"] = 1
            torch.save(row, path)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                cache_records(encoder, folder, specs("train", 2), budget, 32)

    def test_worker_stops_for_shared_cancellation_or_deadline(self):
        cancelled = threading.Event()
        budget = WorkerBudget(time.monotonic() + 60, cancelled)
        budget.check()
        cancelled.set()
        with self.assertRaises(BudgetExceeded):
            budget.check()
        with self.assertRaises(BudgetExceeded):
            WorkerBudget(time.monotonic() - 1, threading.Event()).check()


if __name__ == "__main__":
    unittest.main()
