"""Live Slurm validation, independent of CUDA imports and platform conventions."""
import os
import re
import subprocess

NYU_ACCOUNT = "csci_ga_2271-2026fa"


def run(command):
    return subprocess.check_output(command, text=True).strip()


def fields(output):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(\S+)", output))


def gpu_count(tres):
    """Untyped totals take priority: typed subtotals must not be double counted."""
    match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
    if match:
        return int(match.group(1))
    return sum(int(n) for n in re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)", tres))


def validate_job(job_id, account, partition=None, qos=None, runner=run):
    if not str(job_id).isdigit():
        raise RuntimeError("An explicit numeric Slurm allocation job ID is required")
    details = fields(runner(["scontrol", "show", "job", str(job_id), "-o"]))
    if details.get("JobId") != str(job_id) or details.get("JobState") != "RUNNING":
        raise RuntimeError("Slurm allocation is not the requested RUNNING job")
    if details.get("Account") != account:
        raise RuntimeError("Real Slurm job account does not match the requested account")
    if partition and details.get("Partition") != partition:
        raise RuntimeError("Real Slurm job partition does not match the requested partition")
    if qos and details.get("QOS") != qos:
        raise RuntimeError("Real Slurm job QOS does not match the requested QOS")
    if gpu_count(details.get("AllocTRES", "")) < 1:
        raise RuntimeError("Slurm allocation does not report allocated GPUs")
    return details


def allocated_gpu_indices(env):
    indices = []
    for value in env.get("SLURM_STEP_GPUS", "").split(","):
        if re.fullmatch(r"\d+", value):
            indices.append(int(value))
        elif re.fullmatch(r"\d+-\d+", value):
            first, last = map(int, value.split("-"))
            indices.extend(range(first, last + 1))
        else:
            raise RuntimeError("Slurm step must report actual numeric GPU indices")
    if len(indices) != len(set(indices)):
        raise RuntimeError("Duplicate physical GPU indices in Slurm step")
    return sorted(indices)


def shared_gpu_selection(job, step, gpu_count_requested, gpu_index, env, runner=run,
                         expected_worker_gpus=1, allow_gpu_sharing=False, min_free_gpu_gib=40):
    if expected_worker_gpus not in (1, gpu_count_requested):
        raise RuntimeError("Shared execution selects one GPU or every allocated GPU")
    indices = [gpu_index] if expected_worker_gpus == 1 else allocated_gpu_indices(env)
    if len(indices) != expected_worker_gpus:
        raise RuntimeError("Actual Slurm GPU index count differs from feature worker count")
    return [shared_gpu_identity(job, step, gpu_count_requested, index, env, runner,
                                allow_gpu_sharing, min_free_gpu_gib) for index in indices]


def shared_gpu_identity(job, step, gpu_count_requested, gpu_index, env, runner=run,
                        allow_gpu_sharing=False, min_free_gpu_gib=40):
    """Verify a selected physical GPU belongs to a real reused multi-GPU step.

    Some DGX installations do not narrow CUDA visibility with --gpu-bind. A
    selected UUID is a subset of already allocated devices, never a fake Slurm
    allocation. Parent and step reservations remain explicitly multi-GPU.
    """
    if gpu_count_requested < 2 or gpu_index is None or gpu_index < 0:
        raise RuntimeError("Explicit shared GPU count and physical index are required")
    if gpu_count(job.get("AllocTRES", "")) != gpu_count_requested:
        raise RuntimeError("Shared parent GPU count differs from the explicit request")
    if gpu_count(step.get("TRES", step.get("AllocTRES", ""))) != gpu_count_requested:
        raise RuntimeError("Shared step GPU count differs from the explicit request")
    if gpu_index not in allocated_gpu_indices(env):
        raise RuntimeError("Selected GPU is outside the actual Slurm step GPU set")
    if min_free_gpu_gib < 0:
        raise RuntimeError("Minimum free GPU memory cannot be negative")
    identity = runner(["nvidia-smi", "-i", str(gpu_index),
                       "--query-gpu=index,uuid,name,memory.free,memory.total", "--format=csv,noheader,nounits"]).split(",")
    if len(identity) != 5 or identity[0].strip() != str(gpu_index) or not identity[1].strip().startswith("GPU-"):
        raise RuntimeError("Could not resolve selected physical GPU to an NVIDIA UUID")
    processes = runner(["nvidia-smi", "-i", str(gpu_index),
                        "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    others = [p.strip() for p in processes.splitlines() if p.strip() != str(os.getpid())]
    if others and not allow_gpu_sharing:
        raise RuntimeError("Selected GPU already has another compute process: " + ",".join(others))
    free_gib, total_gib = float(identity[3]) / 1024, float(identity[4]) / 1024
    if free_gib < min_free_gpu_gib:
        raise RuntimeError(f"Selected GPU {gpu_index} has {free_gib:.2f} GiB free; requires {min_free_gpu_gib:g}")
    return dict(index=gpu_index, uuid=identity[1].strip(), name=identity[2].strip(),
                occupied_by_other_pids=others, allow_gpu_sharing=bool(allow_gpu_sharing),
                free_memory_gib=free_gib, total_memory_gib=total_gib,
                minimum_free_memory_gib=min_free_gpu_gib,
                selection="explicit CUDA UUID subset of verified Slurm step allocation",
                scheduler_gpu_binding=step.get("TresBind"), parent_gpus=gpu_count_requested,
                step_gpus=gpu_count_requested, selected_gpus=1)


def validate_runtime(backend, account, partition, gpu_model, cuda_count, cuda_name,
                     allocation_job=None, qos=None, env=None, runner=run,
                     shared_allocation_gpus=None, gpu_index=None, expected_worker_gpus=1,
                     allow_gpu_sharing=False, min_free_gpu_gib=40):
    env = os.environ if env is None else env
    if backend == "nyu" and account != NYU_ACCOUNT:
        raise RuntimeError("NYU backend requires the CV course account")
    if not account:
        raise RuntimeError("DGX backend requires an explicit account from live Slurm inspection")
    job_id = env.get("SLURM_JOB_ID", "")
    if allocation_job and job_id != str(allocation_job):
        raise RuntimeError("Scientific process is not inside the requested existing allocation")
    job = validate_job(job_id, account, partition, qos, runner)
    config = runner(["scontrol", "show", "config"])
    cluster = re.search(r"^ClusterName\s*=\s*(\S+)", config, re.M)
    cluster = cluster.group(1) if cluster else "unknown"
    if backend == "nyu" and "torch" in cluster.lower():
        raise RuntimeError("Course policy requires Cloud Bursting, not Torch")
    if int(env.get("SLURM_RESTART_COUNT", "0")) > 2:
        raise RuntimeError("Repeated preemption; inspect state before another allocation")
    if expected_worker_gpus < 1 or cuda_count != expected_worker_gpus:
        raise RuntimeError("Visible CUDA GPU count differs from explicit worker GPU count")
    if not re.search(r"(?<![A-Za-z0-9])" + re.escape(gpu_model) + r"(?![A-Za-z0-9])", cuda_name, re.I):
        raise RuntimeError(f"Requested {gpu_model}, but actual CUDA device is {cuda_name}")
    if shared_allocation_gpus is not None:
        if backend != "dgx" or not allocation_job or shared_allocation_gpus < 2 or (expected_worker_gpus == 1 and gpu_index is None):
            raise RuntimeError("Shared binding needs explicit DGX allocation, GPU count and selected GPU(s)")
        if gpu_count(job.get("AllocTRES", "")) != shared_allocation_gpus:
            raise RuntimeError("Shared allocation GPU count differs from the explicit request")
    elif gpu_index is not None or expected_worker_gpus != 1 or allow_gpu_sharing:
        raise RuntimeError("GPU selection/sharing requires explicit shared allocation mode")
    step_id = env.get("SLURM_STEP_ID")
    step = None
    selected_gpu = None
    selected_gpus = []
    if step_id and step_id.isdigit():
        step_identity = f"{job_id}.{step_id}"
        step = fields(runner(["scontrol", "show", "step", step_identity, "-o"]))
        if step.get("StepId") != step_identity or step.get("State") != "RUNNING":
            raise RuntimeError("Scientific process has no matching live Slurm step")
        expected_step_gpus = shared_allocation_gpus or 1
        if gpu_count(step.get("TRES", step.get("AllocTRES", ""))) != expected_step_gpus:
            raise RuntimeError("Live Slurm step GPU count differs from requested binding mode")
        if shared_allocation_gpus:
            selected_gpus = shared_gpu_selection(job, step, shared_allocation_gpus, gpu_index, env, runner,
                                                expected_worker_gpus, allow_gpu_sharing, min_free_gpu_gib)
            selected_gpu = selected_gpus[0] if expected_worker_gpus == 1 else None
            if env.get("CUDA_VISIBLE_DEVICES") != ",".join(gpu["uuid"] for gpu in selected_gpus):
                raise RuntimeError("CUDA visibility must equal UUIDs selected from the real shared allocation")
            if any(not re.search(r"(?<![A-Za-z0-9])" + re.escape(gpu_model) + r"(?![A-Za-z0-9])", gpu["name"], re.I)
                   for gpu in selected_gpus):
                raise RuntimeError("A selected physical GPU does not match the requested GPU model")
    elif allocation_job or gpu_count(job.get("AllocTRES", "")) != 1:
        raise RuntimeError("A multi-GPU/reused allocation requires a verified one-GPU Slurm step")
    return dict(backend=backend, account=account, partition=job.get("Partition"),
                qos=job.get("QOS"), cluster=cluster, job_id=job_id, step_id=step_id,
                requested_gpu_model=gpu_model, actual_gpu=cuda_name,
                visible_gpu_count=cuda_count, parent_allocated_gpu_count=gpu_count(job.get("AllocTRES", "")),
                step_allocated_gpu_count=shared_allocation_gpus or 1, task_selected_gpu_index=gpu_index,
                task_selected_gpu_count=expected_worker_gpus,
                pilot_accounting=f"{expected_worker_gpus} explicitly selected GPU(s); existing parent allocation accounted separately" if shared_allocation_gpus else "one allocated GPU",
                allow_gpu_sharing=allow_gpu_sharing, minimum_free_gpu_gib=min_free_gpu_gib,
                shared_parent_allocation=bool(shared_allocation_gpus), selected_gpu=selected_gpu, selected_gpus=selected_gpus,
                slurm_job=job, slurm_step=step)
