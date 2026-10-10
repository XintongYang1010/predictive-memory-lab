import os
import unittest
from unittest.mock import Mock
from pmr.visual_memory.slurm_backend import fields, gpu_count, validate_runtime, shared_gpu_identity


JOB = "JobId=415453 Account=hansirui UserId=guhao(1000) JobState=RUNNING Partition=llm QOS=llm AllocTRES=cpu=64,gres/gpu=8"
STEP = "StepId=415453.12 State=RUNNING TRES=cpu=4,gres/gpu=1,mem=48G,node=1"
ENV = {"SLURM_JOB_ID": "415453", "SLURM_STEP_ID": "12", "SLURM_STEP_GPUS": "0"}


class SlurmBackend(unittest.TestCase):
    def test_dgx_account_derived_from_live_job_not_missing_env(self):
        runner = Mock(side_effect=[JOB, "ClusterName = slurm", STEP])
        result = validate_runtime("dgx", "hansirui", "llm", "H800", 1, "NVIDIA H800", "415453", "llm", ENV, runner)
        self.assertEqual(result["account"], "hansirui")
        self.assertEqual(result["parent_allocated_gpu_count"], 8)
        self.assertEqual(result["step_allocated_gpu_count"], 1)

    def test_real_account_cannot_be_spoofed_by_environment(self):
        env = dict(ENV, SLURM_JOB_ACCOUNT="csci_ga_2271-2026fa")
        with self.assertRaisesRegex(RuntimeError, "account"):
            validate_runtime("nyu", "csci_ga_2271-2026fa", "llm", "A100", 1, "A100", env=env, runner=Mock(return_value=JOB))

    def test_wrong_gpu_and_multiple_visible_devices_are_refused(self):
        for count, name in [(1, "NVIDIA H800"), (8, "NVIDIA A100")]:
            with self.subTest(count=count), self.assertRaises(RuntimeError):
                validate_runtime("dgx", "hansirui", "llm", "A100", count, name,
                                 env=ENV, runner=Mock(side_effect=[JOB, "ClusterName = slurm"]))

    def test_unapproved_multi_gpu_step_is_refused(self):
        runner = Mock(side_effect=[JOB, "ClusterName = slurm", STEP.replace("gres/gpu=1", "gres/gpu=8")])
        with self.assertRaisesRegex(RuntimeError, "GPU count"):
            validate_runtime("dgx", "hansirui", "llm", "H800", 1, "NVIDIA H800", env=ENV, runner=runner)

    def test_shared_selection_requires_exact_uuid_and_records_allocation_eight(self):
        env = dict(ENV, SLURM_STEP_GPUS="0,1,2,3,4,5,6,7", CUDA_VISIBLE_DEVICES="GPU-selected")
        runner = Mock(side_effect=[JOB, "ClusterName = slurm", STEP.replace("gres/gpu=1", "gres/gpu=8"),
                                   "3, GPU-selected, NVIDIA H800, 81920, 81920", str(os.getpid())])
        result = validate_runtime("dgx", "hansirui", "llm", "H800", 1, "NVIDIA H800", "415453",
                                  env=env, runner=runner, shared_allocation_gpus=8, gpu_index=3)
        self.assertTrue(result["shared_parent_allocation"])
        self.assertEqual(result["step_allocated_gpu_count"], 8)
        self.assertEqual(result["selected_gpu"]["selected_gpus"], 1)
        self.assertEqual(result["selected_gpu"]["uuid"], "GPU-selected")

    def test_shared_selection_never_takes_gpu_outside_step_or_busy_gpu(self):
        step = fields(STEP.replace("gres/gpu=1", "gres/gpu=8"))
        with self.assertRaisesRegex(RuntimeError, "outside"):
            shared_gpu_identity(fields(JOB), step, 8, 3, ENV, Mock())
        env = dict(ENV, SLURM_STEP_GPUS="0,1,2,3,4,5,6,7")
        with self.assertRaisesRegex(RuntimeError, "another compute"):
            shared_gpu_identity(fields(JOB), step, 8, 3, env, Mock(side_effect=["3, GPU-selected, NVIDIA H800, 81920, 81920", "999999"]))

    def test_gpu_totals_do_not_double_count_types(self):
        self.assertEqual(gpu_count("cpu=4,gres/gpu=1,gres/gpu:a100=1"), 1)
        self.assertEqual(gpu_count("cpu=4,gres/gpu:a100=1,gres/gpu:h800=2"), 3)

    def test_real_pending_job_and_wrong_step_identity_refused(self):
        with self.assertRaisesRegex(RuntimeError, "RUNNING"):
            validate_runtime("dgx", "hansirui", "llm", "H800", 1, "H800", env=ENV,
                             runner=Mock(return_value=JOB.replace("RUNNING", "PENDING")))
        with self.assertRaisesRegex(RuntimeError, "matching live Slurm step"):
            validate_runtime("dgx", "hansirui", "llm", "H800", 1, "H800", env=ENV,
                             runner=Mock(side_effect=[JOB, "ClusterName = slurm", STEP.replace("415453.12", "415453.11")]))

    def test_selector_changes_only_cuda_visibility_and_preserves_receipt(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import patch
        from pmr.visual_memory import bind_gpu
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(ENV,SLURM_STEP_GPUS="0,1,2,3,4,5,6,7",CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
            argv=["bind", "--allocation-job","415453","--account","hansirui","--partition","llm",
                  "--shared-allocation-gpus","8","--gpu-index","3","--out",tmp,"--stage","profile",
                  "--",sys.executable,"-m","pmr.visual_memory","profile"]
            captured={}
            def capture(executable,command,actual_env):
                captured.update(actual_env)
            with patch.object(sys,"argv",argv),patch.dict(os.environ,env,clear=True),patch.object(bind_gpu,"validate_job",return_value=fields(JOB)),patch.object(bind_gpu,"run",return_value=STEP.replace("gres/gpu=1","gres/gpu=8")),patch.object(bind_gpu,"shared_gpu_selection",return_value=[{"uuid":"GPU-selected"}]),patch.object(bind_gpu.os,"execvpe",side_effect=capture),contextlib.redirect_stdout(io.StringIO()):
                bind_gpu.main()
            self.assertEqual(captured["CUDA_VISIBLE_DEVICES"],"GPU-selected")
            self.assertEqual({k:v for k,v in captured.items() if k.startswith("SLURM_")},
                             {k:v for k,v in env.items() if k.startswith("SLURM_")})
            receipt=json.loads((Path(tmp)/"gpu_selection/profile_415453_12.json").read_text())
            self.assertEqual(receipt["status"],"validated")
            self.assertEqual(receipt["step_id"],"12")
            self.assertEqual(receipt["original_cuda_visible_devices"],env["CUDA_VISIBLE_DEVICES"])

    def test_explicit_eight_worker_selection_validates_all_uuids(self):
        env=dict(ENV,SLURM_STEP_GPUS="0-7",CUDA_VISIBLE_DEVICES=",".join(f"GPU-{i}" for i in range(8)))
        outputs=[JOB,"ClusterName = slurm",STEP.replace("gres/gpu=1","gres/gpu=8")]
        for i in range(8):
            outputs.extend([f"{i}, GPU-{i}, NVIDIA H800, 81920, 81920", ""])
        result=validate_runtime("dgx","hansirui","llm","H800",8,"NVIDIA H800","415453",env=env,
                                runner=Mock(side_effect=outputs),shared_allocation_gpus=8,expected_worker_gpus=8)
        self.assertEqual(result["task_selected_gpu_count"],8)
        self.assertEqual(len(result["selected_gpus"]),8)
        self.assertEqual([g["index"] for g in result["selected_gpus"]],list(range(8)))

    def test_explicit_sharing_records_occupancy_but_requires_free_memory(self):
        job=fields(JOB);step=fields(STEP.replace("gres/gpu=1","gres/gpu=8"))
        env=dict(ENV,SLURM_STEP_GPUS="0-7")
        selected=shared_gpu_identity(job,step,8,3,env,
                    Mock(side_effect=["3, GPU-selected, NVIDIA H800, 61440, 81920","999999"]),
                    allow_gpu_sharing=True,min_free_gpu_gib=40)
        self.assertEqual(selected["occupied_by_other_pids"],["999999"])
        self.assertEqual(selected["free_memory_gib"],60)
        with self.assertRaisesRegex(RuntimeError,"requires 40"):
            shared_gpu_identity(job,step,8,3,env,
                    Mock(side_effect=["3, GPU-selected, NVIDIA H800, 20480, 81920","999999"]),
                    allow_gpu_sharing=True,min_free_gpu_gib=40)


if __name__ == "__main__":
    unittest.main()
