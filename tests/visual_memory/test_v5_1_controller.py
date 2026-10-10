"""Mocked controller boundaries; no Slurm, scientific samples, or GPU access."""
from argparse import Namespace
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import launch_v5_1 as launch


class V51Controller(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root/"logs").mkdir()
        (self.root/"ledger").mkdir()
        (self.root/"reader").mkdir()
        stat = self.root.stat()
        launch.save(self.root/"creation.json", dict(root=str(self.root), inode=stat.st_ino, device=stat.st_dev))
        self.commit = "a"*40
        launch.save(self.root/"source_manifest.json", dict(commit=self.commit,
            files={"scripts/launch_v5_1.py": launch.sha(Path(launch.__file__))}))
        launch.save(self.root/"ledger/budget.json", dict(charged_gpuh=.39161167637833216,
            stages={"profile": .007657659230495106, "features": .38395401714783617}, leases=[]))
        launch.save(self.root/"reader/freeze_receipt.json", {"software_test_placeholder": True})

    def controller(self, action):
        return launch.Controller(Namespace(action=action, root=self.root, source_commit=self.commit,
            python=Path("/software-test/python"), assets=self.root/"unused-assets",
            prior_profile=self.root/"unused-prior-profile.json", allocation_job="415453",
            account="hansirui", partition="llm", qos="llm"))

    def publication(self, verified=True):
        launch.save(self.root/"freeze_publication.json", dict(verified_readback=verified,
            freeze_sha256=launch.sha(self.root/"reader/freeze_receipt.json"), source_commit=self.commit))

    def test_extraction_uses_all_eight_allocated_gpus(self):
        c = self.controller("fit")
        with patch.object(launch.subprocess, "run", return_value=Namespace(returncode=0)) as run, contextlib.redirect_stdout(io.StringIO()):
            c.extraction("fit")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "srun")
        self.assertIn("--jobid=415453", command)
        self.assertIn("--gres=gpu:8", command)
        self.assertEqual(command[command.index("--worker-gpus")+1], "8")
        self.assertEqual(command[command.index("--workers")+1], "8")
        self.assertIn("pmr.visual_memory.bind_gpu", command)
        self.assertNotIn("--gpu-index", command)
        self.assertNotIn("--freeze-receipt", command)
        self.assertEqual(c.state["jobs"][0]["selected_gpus"], 8)
        ledger = json.loads((self.root/"ledger/budget.json").read_text())
        self.assertEqual(ledger["controller_envelopes"][0]["status"], "closed")
        self.assertEqual(ledger["overhead_receipts"][0]["stage"], "temporal_fit")

    def test_cpu_reader_explicitly_releases_gpu_step(self):
        c = self.controller("fit")
        before = (self.root/"ledger/budget.json").read_bytes()
        with patch.object(launch.subprocess, "run", return_value=Namespace(returncode=0)) as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(c.reader("fit"))
        command = run.call_args.args[0]
        self.assertIn("--gres=none", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertNotIn("pmr.visual_memory.bind_gpu", command)
        self.assertNotIn("--worker-gpus", command)
        self.assertEqual(c.state["jobs"][0]["selected_gpus"], 0)
        self.assertEqual(before, (self.root/"ledger/budget.json").read_bytes())

    def test_failed_confirmation_prevents_bridge_release(self):
        self.publication()
        c = self.controller("confirm")
        with patch.object(c, "extraction") as extract, patch.object(c, "reader", return_value=False) as read:
            c.run()
        extract.assert_called_once_with("confirm")
        read.assert_called_once_with("confirm")
        status = json.loads(c.status_path.read_text())
        self.assertEqual(status["status"], "scientific_prerequisite_failed")
        self.assertEqual(status["failed_role"], "confirm")

    def test_confirmation_requires_verified_matching_publication(self):
        self.publication(verified=False)
        c = self.controller("confirm")
        with patch.object(c, "extraction") as extract, patch.object(c, "reader") as read:
            with self.assertRaisesRegex(ValueError, "Verified GitHub freeze publication required"):
                c.run()
        extract.assert_not_called()
        read.assert_not_called()
        self.assertEqual(json.loads(c.status_path.read_text())["status"], "execution_failed")

    def test_changed_freeze_blocks_publication_release(self):
        self.publication()
        launch.save(self.root/"reader/freeze_receipt.json", {"changed_after_publication": True})
        c = self.controller("confirm")
        with patch.object(c, "extraction") as extract:
            with self.assertRaisesRegex(ValueError, "publication required"):
                c.run()
        extract.assert_not_called()

    def test_outer_budget_rejects_before_launch(self):
        c = self.controller("fit")
        launch.save(self.root/"ledger/budget.json", dict(charged_gpuh=59., stages={}, leases=[]))
        with patch.object(launch.subprocess, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "Cumulative budget"):
                c.extraction("fit")
        run.assert_not_called()

    def test_partial_final_report_cannot_be_labeled_complete(self):
        (self.root/"p0_run").mkdir()
        launch.save(self.root/"p0_run/report_status.json", dict(status="partial", missing_variants=["fifo"], incomplete_stages=[]))
        launch.save(self.root/"p0_run/run_manifest.json", dict(status="partial"))
        c = self.controller("downstream")
        with patch.object(c, "downstream"):
            with self.assertRaisesRegex(RuntimeError, "Final report is incomplete"):
                c.run()
        self.assertEqual(json.loads(c.status_path.read_text())["status"], "execution_failed")


if __name__ == "__main__":
    unittest.main()
