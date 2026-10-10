import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("archive_visual_v5", Path(__file__).resolve().parents[2] / "scripts/archive_visual_v5.py")
archive_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive_module)


class Archive(unittest.TestCase):
    def workspace(self, root):
        work = root / "downloaded"
        for name in ("START_HERE.md", "docs/CODEX_DGX_HANDOFF.md", "docs/VISUAL_MEMORY_V5.md",
                     "pmr/visual_memory/__main__.py", "scripts/launch_visual_v5.py"):
            path = work / "code" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source fixture\n")
        (work / "creation_record.json").write_text('{"created": true}\n')
        run = work / "run1"; run.mkdir()
        (run / "features_status.json").write_text('{"status":"partial","reason":"feature gate failed"}')
        (run / "feature_gate.json").write_text('{"passed":false}')
        (run / "run_manifest.json").write_text('{"configuration":{"source":"fixture"},"status":"partial"}')
        (run / "profile.json").write_text('{"runtime":{"gpu":"NVIDIA H800","job_id":"123"},"encoder":{"checkpoint_sha256":"fixture"}}')
        cache = run / "probe_features/train"; cache.mkdir(parents=True)
        (cache / "000000-0.pt").write_bytes(b"regenerate this cache")
        (cache / "manifest.json").write_text('{"status":"complete"}')
        (work / "assets").mkdir(); (work / "assets/official.pth").write_bytes(b"official weights")
        (work / ".venv").mkdir(); (work / ".venv/private.txt").write_text("environment")
        return work

    def test_failed_gate_has_no_checkpoints_and_retains_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = self.workspace(root); out = root / "results"
            manifest = archive_module.archive(work, out)
            self.assertEqual(manifest["experiment_status"], "partial")
            self.assertIsNone(manifest["checkpoint_archive"])
            self.assertTrue((out / "run/probe_features/train/manifest.json").exists())
            self.assertFalse((out / "run/probe_features/train/000000-0.pt").exists())
            self.assertFalse((out / "assets").exists())
            self.assertIn("H800", (out / "README.md").read_text())
            self.assertTrue(archive_module.verify(out)["verified"])
            self.assertTrue((work / "assets/official.pth").exists())

    def test_training_checkpoint_archive_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = self.workspace(root); out = root / "results"
            (work / "run1/p0.pt").write_bytes(b"trained weights")
            manifest = archive_module.archive(work, out)
            self.assertEqual(archive_module.verify(out)["checkpoints"], 1)
            self.assertNotIn("run/p0.pt", manifest["files"])
            asset = out.parent / manifest["checkpoint_archive"]["filename"]
            with asset.open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                archive_module.verify(out)

    def test_missing_source_refused_before_output_and_inventory_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); work = self.workspace(root); out = root / "results"
            (work / "code/START_HERE.md").unlink()
            with self.assertRaises(FileNotFoundError):
                archive_module.archive(work, out)
            self.assertFalse(out.exists())
            (work / "code/START_HERE.md").write_text("restored fixture")
            archive_module.archive(work, out)
            manifest = json.loads((out / "archive_manifest.json").read_text())
            manifest["experiment_status"] = "complete"
            (out / "archive_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA256SUMS"):
                archive_module.verify(out)


if __name__ == "__main__":
    unittest.main()
