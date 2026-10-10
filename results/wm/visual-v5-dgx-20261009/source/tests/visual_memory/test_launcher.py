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

    def dgx_run(self, args):
        if args[0]=="hostname":return "dgx-127"
        if args[:3]==["scontrol","show","config"]:return "ClusterName = slurm"
        if args[0]=="sinfo":return "llm|gpu:8(S:0-1)|1048576|128"
        if args[0]=="sacctmgr":return "slurm|hansirui|guhao||llm|"
        if args[:3]==["scontrol","show","job"]:
            return "JobId=415453 Account=hansirui UserId=guhao(1000) JobState=RUNNING Partition=llm QOS=llm AllocTRES=cpu=64,gres/gpu=8"
        if args[0]==sys.executable:return ""
        if args[0]=="squeue":return ""
        raise AssertionError(args)

    def test_dgx_shared_dry_run_records_reserved_eight_selected_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/"assets";assets.mkdir()
            (assets/"ijepa").mkdir();(assets/"IN1K-vit.h.14-300e.pth.tar").touch()
            argv=["launch","--out",str(root/"out"),"--assets",str(assets),"--partition","llm",
                  "--backend","dgx","--account","hansirui","--qos","llm","--allocation-job","415453",
                  "--gpu-model","H800","--shared-allocation-gpus","8","--gpu-index","3"]
            output=io.StringIO()
            with patch.object(sys,"argv",argv),patch.dict("os.environ",{"USER":"guhao"}),patch.object(launcher,"run",side_effect=self.dgx_run),patch.object(launcher.subprocess,"run") as submit,contextlib.redirect_stdout(output):
                launcher.main();submit.assert_not_called()
            plan=json.loads((root/"out"/"submission_plan.json").read_text())
            self.assertEqual(plan["parent_allocated_gpu_count"],8)
            self.assertEqual(plan["step_allocated_gpu_count"],8)
            self.assertEqual(plan["task_selected_gpu_count"],1)
            self.assertIsNone(plan["observed_allocated_gpuh"])
            self.assertIn("pmr.visual_memory.bind_gpu",output.getvalue())
            self.assertIn("--gres=gpu:8",output.getvalue())

    def test_dgx_requires_explicit_account(self):
        argv=["launch","--out","/unused","--assets","/unused","--partition","llm","--backend","dgx"]
        with patch.object(sys,"argv",argv),self.assertRaisesRegex(RuntimeError,"--account"):
            launcher.main()

    def test_dgx_60h_round_cap_prevents_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/"assets";assets.mkdir();out=root/"out";out.mkdir()
            (assets/"ijepa").mkdir();(assets/"IN1K-vit.h.14-300e.pth.tar").touch()
            (out/"budget.json").write_text(json.dumps({"charged_gpuh":6,"stages":{}}))
            argv=["launch","--out",str(out),"--assets",str(assets),"--partition","llm",
                  "--backend","dgx","--account","hansirui","--allocation-job","415453","--all","--submit"]
            with patch.object(sys,"argv",argv),patch.dict("os.environ",{"USER":"guhao"}),patch.object(launcher,"run",side_effect=self.dgx_run),patch.object(launcher.subprocess,"run") as submit,self.assertRaisesRegex(RuntimeError,"60 GPUh"):
                launcher.main()
            submit.assert_not_called()

    def test_dgx_rejects_unverified_account_and_untyped_new_allocation(self):
        with patch.dict("os.environ",{"USER":"guhao"}),patch.object(launcher,"run",return_value="slurm|other|guhao||llm|"):
            with self.assertRaisesRegex(RuntimeError,"association"):
                launcher.validate_association("hansirui","slurm","llm")
        with patch.object(launcher,"run",side_effect=self.dgx_run):
            with self.assertRaisesRegex(RuntimeError,"untyped"):
                launcher.query_resources("llm","dgx","A100")

    def test_dgx_failed_step_persists_receipt_and_stops_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/"assets";assets.mkdir()
            (assets/"ijepa").mkdir();(assets/"IN1K-vit.h.14-300e.pth.tar").touch()
            argv=["launch","--out",str(root/"out"),"--assets",str(assets),"--partition","llm",
                  "--backend","dgx","--account","hansirui","--allocation-job","415453","--all","--submit"]
            from types import SimpleNamespace
            with patch.object(sys,"argv",argv),patch.dict("os.environ",{"USER":"guhao"}),patch.object(launcher,"run",side_effect=self.dgx_run),patch.object(launcher.subprocess,"run",return_value=SimpleNamespace(returncode=17)) as submit,contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,"subsequent steps not started"):
                launcher.main()
            self.assertEqual(submit.call_count,1)
            receipt=json.loads((root/"out"/"submission.json").read_text())
            self.assertEqual(receipt["jobs"][0]["returncode"],17)
            self.assertEqual(receipt["jobs"][0]["status"],"failed")
            self.assertEqual(receipt["jobs"][0]["job_id"],"415453")

    def test_unreconciled_interrupted_budget_never_submits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/"assets";assets.mkdir();out=root/"out";out.mkdir()
            (assets/"ijepa").mkdir();(assets/"IN1K-vit.h.14-300e.pth.tar").touch()
            (out/"budget.json").write_text(json.dumps({"charged_gpuh":.5,"stages":{"profile":.5},
                "leases":[{"status":"reserved_or_interrupted"}]}))
            argv=["launch","--out",str(out),"--assets",str(assets),"--partition","llm",
                  "--backend","dgx","--account","hansirui","--allocation-job","415453","--resume","--submit"]
            with patch.object(sys,"argv",argv),patch.dict("os.environ",{"USER":"guhao"}),patch.object(launcher,"run",side_effect=self.dgx_run),patch.object(launcher.subprocess,"run") as submit,self.assertRaisesRegex(RuntimeError,"reconciliation"):
                launcher.main()
            submit.assert_not_called()

    def test_parallel_dry_run_keeps_aggregate_budget_and_uses_eight_feature_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=root/"assets";assets.mkdir()
            (assets/"ijepa").mkdir();(assets/"IN1K-vit.h.14-300e.pth.tar").touch()
            argv=["launch","--out",str(root/"out"),"--assets",str(assets),"--partition","llm",
                  "--backend","dgx","--account","hansirui","--allocation-job","415453",
                  "--gpu-model","H800","--shared-allocation-gpus","8","--gpu-index","3","--parallel-gpus","8","--all"]
            output=io.StringIO()
            with patch.object(sys,"argv",argv),patch.dict("os.environ",{"USER":"guhao"}),patch.object(launcher,"run",side_effect=self.dgx_run),patch.object(launcher.subprocess,"run") as submit,patch.object(launcher.subprocess,"Popen") as spawn,contextlib.redirect_stdout(output):
                launcher.main();submit.assert_not_called();spawn.assert_not_called()
            plan=json.loads((root/"out"/"submission_plan.json").read_text())
            self.assertEqual(plan["requested_max_gpuh"],54.5)
            features=next(x for x in plan["planned_stages"] if x["stage"]=="features")
            self.assertEqual((features["minutes"],features["selected_gpus"]),(60,8))
            self.assertEqual(len(plan["parallel_plan"]),9)
            self.assertEqual([x["gpu_index"] for x in plan["parallel_plan"][:8]],list(range(8)))
            self.assertIn("--cpus-per-task=32",output.getvalue())
            self.assertIn("--mem=384G",output.getvalue())
            self.assertIn("--feature-workers 8",output.getvalue())
            self.assertNotIn("--allow-gpu-sharing",output.getvalue())

    def test_parallel_scheduler_never_exceeds_eight_and_waits_before_ninth(self):
        self.run_parallel_scheduler_case(fail_first=False)

    def test_parallel_failure_drains_started_siblings_but_never_launches_ninth(self):
        self.run_parallel_scheduler_case(fail_first=True)

    def run_parallel_scheduler_case(self,fail_first):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=SimpleNamespace(out=root,parallel_gpus=8,submit=True,allocation_job="415453")
            items=[dict(stage=f"adapt_{i}",phase="adapt",variant=str(i),minutes=240,selected_gpus=1) for i in range(9)]
            data={"jobs":[],"submitted":False};receipt=root/"submission.json"
            live=set();started=[];completed=[];peak=[0]
            class Process:
                def __init__(self,invocation):
                    self.stage,self.gpu=invocation;self.pid=1000+len(started);self.polls=0
                    if self.gpu in live:raise AssertionError("GPU assigned twice")
                    live.add(self.gpu);started.append(self.stage);peak[0]=max(peak[0],len(live))
                    if len(started)==9 and not completed:raise AssertionError("Ninth started before slot free")
                def poll(self):
                    self.polls+=1
                    if self.polls==1 and not (fail_first and self.stage=="adapt_0"):return None
                    live.remove(self.gpu);completed.append(self.stage)
                    return 7 if fail_first and self.stage=="adapt_0" else 0
            def command(args,item,project,resources,upstream,checkpoint,gpu):return [item["stage"],gpu]
            with patch.object(launcher,"step_command",side_effect=command),patch.object(launcher.shlex,"join",return_value="fake srun"),patch.object(launcher.subprocess,"Popen",side_effect=Process),patch.object(launcher.time,"sleep"),contextlib.redirect_stdout(io.StringIO()):
                if fail_first:
                    with self.assertRaisesRegex(RuntimeError,"failed"):
                        launcher.run_parallel_adaptations(a,items,data,receipt,None,None,None,None,poll_seconds=0)
                else:
                    launcher.run_parallel_adaptations(a,items,data,receipt,None,None,None,None,poll_seconds=0)
            self.assertEqual(peak[0],8)
            self.assertFalse(live)
            self.assertEqual(len(started),8 if fail_first else 9)
            self.assertEqual(len(completed),len(started))
            durable=json.loads(receipt.read_text())
            if fail_first:self.assertEqual(durable["not_started_after_failure"],["adapt_8"])
            else:self.assertTrue(all(entry["status"]=="complete" for entry in durable["jobs"]))


if __name__=="__main__":unittest.main()
