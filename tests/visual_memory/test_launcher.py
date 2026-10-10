import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("v5_launcher",Path(__file__).resolve().parents[2]/"scripts/launch_visual_v5.py")
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)


class Launcher(unittest.TestCase):
    def test_dry_run_uses_live_gpu_type_and_never_submits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/"assets";assets.mkdir()
            (assets/"ijepa").mkdir();(assets/"IN1K-vit.h.14-300e.pth.tar").touch()
            def fake_run(args):
                if args[0]=="hostname":return "b-3-17"
                if args[:3]==["scontrol","show","config"]:return "ClusterName = burst"
                if args[0]=="sinfo":return "a100-live|gpu:a100_40gb:1(S:0)|87040|12"
                if args[0]=="sacct":return "123|COMPLETED|3600|cpu=4,gres/gpu=1,gres/gpu:a100_40gb=1|"
                if args[0]=="squeue":return "10"
                if args[:3]==["scontrol","show","job"]:return "JobId=10 ReqTRES=cpu=4,mem=4G"
                if args[0]==sys.executable:return ""
                raise AssertionError(args)
            argv=["launch","--out",str(root/"out"),"--assets",str(assets),"--partition","a100-live","--all"]
            with patch.object(sys,"argv",argv),patch.object(launcher,"run",side_effect=fake_run),patch.object(launcher.subprocess,"run") as submit,contextlib.redirect_stdout(io.StringIO()):
                launcher.main();submit.assert_not_called()
            plan=json.loads((root/"out"/"submission_plan.json").read_text())
            self.assertEqual(plan["requested_max_gpuh"],54.5)
            self.assertEqual(plan["observed_allocated_gpuh"],1)
            self.assertEqual(plan["resources"]["gres"],"gpu:a100_40gb:1")
            self.assertFalse(plan["submitted"])

    def test_refuse_torch_and_unknown_partition(self):
        with patch.object(launcher,"run",side_effect=["login.torch","ClusterName = torch"]):
            with self.assertRaises(RuntimeError):launcher.query_resources("a100-live")
        with patch.object(launcher,"run",side_effect=["b-3-17","ClusterName = burst","l4-live|gpu:l4:1|87040|12"]):
            with self.assertRaises(RuntimeError):launcher.query_resources("a100-live")


if __name__=="__main__":unittest.main()
