import json
from pathlib import Path
import tempfile
import unittest

from pmr.visual_memory.memory import VARIANTS
from pmr.visual_memory.report import report


class ReportCompleteness(unittest.TestCase):
    def test_requires_reader_results_and_successful_replay_status(self):
        variants=["full","recent","fifo",*VARIANTS,"reader_only_C0"]
        row=dict(family="hold",gap=16,interference_writes=8,
                 episode_id="dev-000000-0",pair_id="dev-000000",
                 mse=1.,roi_mse=1.,object_a_mse=1.,object_b_mse=1.,
                 paired_difference_mse=1.)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"run_manifest.json").write_text(json.dumps(dict(status="partial",configuration={"seed":42})))
            for variant in variants[:-1]:
                folder=root/variant;folder.mkdir()
                (folder/"per_episode.jsonl").write_text(json.dumps(row)+"\n")
            stages=["profile","features","p0",*("adapt_"+v for v in variants[:-1])]
            for stage in stages:
                (root/f"{stage}_status.json").write_text(json.dumps(dict(status="complete")))
            report(root)
            status=json.loads((root/"report_status.json").read_text())
            self.assertEqual(status["status"],"partial")
            self.assertEqual(status["missing_variants"],["reader_only_C0"])
            self.assertEqual(status["incomplete_stages"],["replay"])
            self.assertEqual(json.loads((root/"run_manifest.json").read_text())["status"],"partial")

            folder=root/"reader_only_C0";folder.mkdir()
            (folder/"per_episode.jsonl").write_text(json.dumps(row)+"\n")
            (root/"replay_status.json").write_text(json.dumps(dict(status="partial")))
            report(root)
            status=json.loads((root/"report_status.json").read_text())
            self.assertEqual(status["status"],"partial")
            self.assertEqual(status["missing_variants"],[])
            self.assertEqual(status["incomplete_stages"],["replay"])

            (root/"replay_status.json").write_text(json.dumps(dict(status="complete")))
            report(root)
            status=json.loads((root/"report_status.json").read_text())
            self.assertEqual(status["status"],"pilot_complete")
            self.assertEqual(status["incomplete_stages"],[])
            manifest=json.loads((root/"run_manifest.json").read_text())
            self.assertEqual(manifest["status"],"complete")
            self.assertEqual(manifest["configuration"],{"seed":42})


if __name__=="__main__":
    unittest.main()
