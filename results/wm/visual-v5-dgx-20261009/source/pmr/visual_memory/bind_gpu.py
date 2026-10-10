"""Select allocated GPU UUIDs with occupancy checks before CUDA imports."""
import argparse
import json
import os
from pathlib import Path
import time
from .slurm_backend import fields, run, shared_gpu_selection, validate_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allocation-job", required=True)
    p.add_argument("--account", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--qos")
    p.add_argument("--shared-allocation-gpus", required=True, type=int)
    p.add_argument("--gpu-index", type=int)
    p.add_argument("--worker-gpus", type=int, default=1)
    p.add_argument("--allow-gpu-sharing", action="store_true")
    p.add_argument("--min-free-gpu-gib", type=float, default=40)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stage", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    step_id = os.environ.get("SLURM_STEP_ID", "")
    if not job_id.isdigit() or not step_id.isdigit() or not a.stage.replace("_", "").isalnum():
        raise RuntimeError("GPU selection requires real numeric Slurm job/step IDs and a valid stage")
    receipt = a.out / "gpu_selection" / f"{a.stage}_{job_id}_{step_id}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(job_id=job_id, step_id=step_id, stage=a.stage, status="validating", start_unix=time.time(),
                    original_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    slurm_step_gpus=os.environ.get("SLURM_STEP_GPUS"), selected_index=a.gpu_index)

    def save():
        temp = receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(evidence, indent=2) + "\n")
        os.replace(temp, receipt)

    save()
    try:
        if job_id != a.allocation_job:
            raise RuntimeError("GPU selection must execute inside the requested real allocation")
        job = validate_job(a.allocation_job, a.account, a.partition, a.qos)
        identity = job_id + "." + step_id
        step = fields(run(["scontrol", "show", "step", identity, "-o"]))
        if step.get("StepId") != identity or step.get("State") != "RUNNING":
            raise RuntimeError("GPU selection step is not live")
        selected = shared_gpu_selection(job, step, a.shared_allocation_gpus, a.gpu_index, os.environ,
                                        expected_worker_gpus=a.worker_gpus, allow_gpu_sharing=a.allow_gpu_sharing,
                                        min_free_gpu_gib=a.min_free_gpu_gib)
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not command:
            raise RuntimeError("A scientific command is required")
        evidence.update(status="validated", selected_gpus=selected, selected_gpu_count=a.worker_gpus,
                        allow_gpu_sharing=a.allow_gpu_sharing, slurm_job=job, slurm_step=step)
        save()
        print(f"V5 shared allocation: parent={job_id} step={identity} "
              f"allocated_gpus={a.shared_allocation_gpus} selected_gpu={selected}", flush=True)
        # Preserve all actual SLURM_* identity variables unchanged. This selects
        # a subset of task CUDA devices without changing the Slurm reservation.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in selected)
        os.execvpe(command[0], command, os.environ)
    except BaseException as exc:
        evidence.update(status="failed", reason=str(exc), end_unix=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
