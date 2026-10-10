#!/usr/bin/env python3
"""Inspect live Slurm and run a bounded V5 chain (NYU or explicit DGX backend).

Dry run is the default. DGX can reuse an existing allocation for eight feature
workers and concurrent independent adaptations. Keep the controller in a durable
CPU session (e.g. nohup). No unrelated
job is stopped or modified. Actual GPU model is validated inside every stage.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmr.visual_memory.slurm_backend import NYU_ACCOUNT, fields, gpu_count, validate_job

ACCOUNT = NYU_ACCOUNT
PLAN = [("profile", None, 30), ("features", None, 480), ("p0", None, 720),
        *[("adapt", x, 240) for x in ("A0", "A1", "B0", "B1", "C0", "C1")],
        *[("adapt", x, 120) for x in ("full", "recent", "fifo")], ("replay", None, 240)]


def run(args):
    return subprocess.check_output(args, text=True).strip()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temp, path)


def query_resources(partition, backend="nyu", gpu_model="A100", allocation_job=None):
    host = run(["hostname"])
    config = run(["scontrol", "show", "config"])
    cluster = re.search(r"^ClusterName\s*=\s*(\S+)", config, re.M)
    cluster = cluster.group(1) if cluster else "unknown"
    if backend == "nyu":
        if "torch" in cluster.lower() or "torch" in host.lower():
            raise RuntimeError("Course computing must be on Cloud Bursting; refusing Torch")
        if host.startswith(("log-", "login")) or "ood" in host.lower():
            raise RuntimeError("Prepare/submit from a Burst CPU compute session: login and compute filesystems differ")
    rows = run(["sinfo", "-N", "-h", "-o", "%P|%G|%m|%c"])
    candidates = []
    for line in rows.splitlines():
        part, gres, mem, cpus = line.split("|")
        if part.removesuffix("*") != partition:
            continue
        typed = re.search(r"(?:^|,)gpu:([^:,()]+):([0-9]+)", gres)
        untyped = re.search(r"(?:^|,)gpu:([0-9]+)(?:\(|,|$)", gres)
        if typed and gpu_model.lower() in (typed.group(1) + part).lower():
            request = f"gpu:{typed.group(1)}:1"
            model_evidence = "sinfo typed GRES; actual CUDA model checked in stage"
        elif backend == "dgx" and allocation_job and untyped:
            request = "gpu:1"
            model_evidence = "untyped GRES; model unverified until allocated profile"
        else:
            continue
        candidates.append(dict(partition=part.removesuffix("*"), gres=request,
                               memory_mb=int(mem.rstrip("+")), cpus=int(cpus),
                               model_evidence=model_evidence))
    if not candidates:
        raise RuntimeError(f"No live {gpu_model} GRES for selected partition; untyped DGX GRES needs an existing allocation")
    resource = min(candidates, key=lambda x: x["memory_mb"])
    if resource["memory_mb"] < 49152:
        raise RuntimeError("Pilot requests 48 GiB host RAM; inspect partition capacity")
    return dict(host=host, cluster=cluster, **resource)


def validate_association(account, cluster, partition):
    user = os.environ.get("USER", "")
    if not user:
        raise RuntimeError("USER is required to inspect live Slurm associations")
    rows = run(["sacctmgr", "-n", "-P", "show", "assoc", f"where", f"user={user}",
                "format=Cluster,Account,User,Partition,QOS"])
    for row in rows.splitlines():
        values = row.split("|")
        if len(values) >= 4 and values[:3] == [cluster, account, user] and values[3] in ("", partition):
            return rows
    raise RuntimeError("Requested account/partition has no live user Slurm association")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--assets", required=True, type=Path)
    p.add_argument("--partition", required=True, help="Exact live Slurm partition")
    p.add_argument("--backend", choices=("nyu", "dgx"), default="nyu")
    p.add_argument("--account", help="Required for DGX; must match live Slurm")
    p.add_argument("--qos", help="Optional real Slurm QOS")
    p.add_argument("--gpu-model", choices=("A100", "H800"), default="A100",
                   help="A100 protocol default; H800 requires an explicitly approved protocol change")
    p.add_argument("--allocation-job", help="Use sequential one-GPU steps in this existing DGX job")
    p.add_argument("--shared-allocation-gpus", type=int,
                   help="Explicit reuse of a preexisting multi-GPU allocation with one explicitly selected GPU")
    p.add_argument("--gpu-index", type=int, help="Physical GPU index selected by validated UUID in shared mode")
    p.add_argument("--parallel-gpus", type=int, choices=(1, 8), default=1,
                   help="Eight feature workers and up to eight independent adaptation stages")
    p.add_argument("--allow-gpu-sharing", action="store_true",
                   help="Explicitly permit coexisting compute processes; never stop them")
    p.add_argument("--min-free-gpu-gib", type=float, default=40,
                   help="Minimum free memory on each selected GPU, checked before starting")
    p.add_argument("--python", default=sys.executable, type=Path)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--all", action="store_true", help="Run full gated chain; default is profile only")
    p.add_argument("--resume", action="store_true", help="Continue inactive interrupted chain within original budgets")
    a = p.parse_args()
    if a.backend == "nyu":
        a.account = a.account or ACCOUNT
        if a.account != ACCOUNT or a.allocation_job or a.gpu_model != "A100":
            raise RuntimeError("NYU backend requires course account, A100 and dedicated batch allocations")
    elif not a.account:
        raise RuntimeError("DGX requires --account from live Slurm associations")
    if a.shared_allocation_gpus is not None:
        if a.backend != "dgx" or not a.allocation_job or a.shared_allocation_gpus < 2 or a.gpu_index is None or a.gpu_index < 0:
            raise RuntimeError("Shared mode requires explicit DGX job, GPU count >=2 and nonnegative GPU index")
    elif a.gpu_index is not None:
        raise RuntimeError("--gpu-index requires --shared-allocation-gpus")
    if a.parallel_gpus > 1 and (a.backend != "dgx" or a.shared_allocation_gpus != a.parallel_gpus):
        raise RuntimeError("Parallel mode requires explicit reuse of an eight-GPU DGX allocation")
    if a.allow_gpu_sharing and not a.shared_allocation_gpus:
        raise RuntimeError("GPU coexistence requires explicit shared allocation mode")
    if a.min_free_gpu_gib < 0:
        raise RuntimeError("Minimum free GPU memory cannot be negative")
    project = Path(__file__).resolve().parents[1]
    a.out = a.out.resolve(); a.assets = a.assets.resolve(); a.python = a.python.absolute()
    checkpoint = a.assets / "IN1K-vit.h.14-300e.pth.tar"; upstream = a.assets / "ijepa"
    for path in (a.python, checkpoint, upstream):
        if not path.exists():
            raise FileNotFoundError(path)
    resources = query_resources(a.partition, a.backend, a.gpu_model, a.allocation_job)
    association = validate_association(a.account, resources["cluster"], a.partition) if a.backend == "dgx" else None
    allocation = validate_job(a.allocation_job, a.account, a.partition, a.qos, run) if a.allocation_job else None
    if allocation and allocation.get("UserId", "").split("(")[0] != os.environ.get("USER"):
        raise RuntimeError("Existing allocation is not owned by the invoking Slurm user")
    if a.shared_allocation_gpus and gpu_count(allocation.get("AllocTRES", "")) != a.shared_allocation_gpus:
        raise RuntimeError("Shared GPU count does not match the real parent allocation")
    run([str(a.python), "-c", "import torch,numpy,PIL,matplotlib; assert torch.version.cuda, 'CUDA PyTorch is required'"])
    used = None
    if a.backend == "nyu":
        usage = run(["sacct", "-X", "-n", "-P", "-S", "2026-09-01", "-A", a.account,
                     "-u", os.environ.get("USER", "xy3257"), "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"])
        used = sum(int(row.split("|")[2]) * gpu_count(row.split("|")[3]) / 3600
                   for row in usage.splitlines() if len(row.split("|")) >= 4)
        active = run(["squeue", "-h", "-u", os.environ.get("USER", "xy3257"), "-A", a.account, "-o", "%i"])
        for job in active.splitlines():
            details = run(["scontrol", "show", "job", job, "-o"])
            if re.search(r"(?:TRES|AllocTRES|ReqTRES|TresPerNode)=\S*gres/gpu", details):
                raise RuntimeError("An existing course GPU job needs accounting/reconciliation first: " + job)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "submission.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch(a, project, resources, association, allocation, used, upstream, checkpoint)


def stage_plan(a, ledger):
    plan = []
    for phase, variant, nominal_minutes in (PLAN if a.all else PLAN[:1]):
        stage = phase + ("_" + variant if variant else "")
        if a.resume:
            status_path = a.out / f"{stage}_status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                continue
        gpus = a.parallel_gpus if phase == "features" else 1
        charged_minutes = ledger["stages"].get(stage, 0.) * 60 if a.resume else 0
        minutes = int((nominal_minutes - charged_minutes) / gpus)
        if minutes < 2:
            raise RuntimeError("No stage budget remains for " + stage)
        plan.append(dict(stage=stage, phase=phase, variant=variant, minutes=minutes, selected_gpus=gpus))
    return plan


def scientific_command(a, item, upstream, checkpoint, gpu_index):
    phase, variant, minutes = item["phase"], item["variant"], item["minutes"]
    command = [str(a.python), "-m", "pmr.visual_memory", phase, "--out", str(a.out),
               "--lease-hours", str(minutes / 60), "--backend", a.backend,
               "--account", a.account, "--partition", a.partition, "--gpu-model", a.gpu_model,
               "--feature-workers", str(a.parallel_gpus), "--min-free-gpu-gib", str(a.min_free_gpu_gib)]
    if a.qos:
        command += ["--qos", a.qos]
    if phase in ("profile", "features"):
        command += ["--ijepa-code", str(upstream), "--ijepa-checkpoint", str(checkpoint)]
    if variant:
        command += ["--variant", variant]
    if a.allocation_job:
        command += ["--allocation-job", a.allocation_job]
    if a.shared_allocation_gpus:
        command += ["--shared-allocation-gpus", str(a.shared_allocation_gpus)]
        if item["selected_gpus"] == 1:
            command += ["--gpu-index", str(gpu_index)]
        if a.allow_gpu_sharing:
            command += ["--allow-gpu-sharing"]
        selector = [str(a.python), "-m", "pmr.visual_memory.bind_gpu", "--allocation-job", a.allocation_job,
                    "--account", a.account, "--partition", a.partition,
                    "--shared-allocation-gpus", str(a.shared_allocation_gpus),
                    "--worker-gpus", str(item["selected_gpus"]), "--min-free-gpu-gib", str(a.min_free_gpu_gib),
                    "--out", str(a.out), "--stage", item["stage"]]
        if item["selected_gpus"] == 1:
            selector += ["--gpu-index", str(gpu_index)]
        if a.qos:
            selector += ["--qos", a.qos]
        if a.allow_gpu_sharing:
            selector += ["--allow-gpu-sharing"]
        command = selector + ["--"] + command
    return ["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", *command]


def step_command(a, item, project, resources, upstream, checkpoint, gpu_index):
    command = scientific_command(a, item, upstream, checkpoint, gpu_index)
    step_gres = f"gpu:{a.shared_allocation_gpus}" if a.shared_allocation_gpus else resources["gres"]
    binding = [f"--gpu-bind=map_gpu:{gpu_index}"] if a.shared_allocation_gpus and item["selected_gpus"] == 1 else []
    cpus = 4 * item["selected_gpus"]
    memory = 48 * item["selected_gpus"]
    return ["srun", "--jobid=" + a.allocation_job, "--overlap", "--exact", *binding,
            "--gres=" + step_gres, "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus}",
            f"--mem={memory}G", f"--time={item['minutes']}", "--signal=USR1@120", "--kill-on-bad-exit=1",
            "--job-name=v5_" + item["stage"], "--chdir=" + str(project),
            "--output=" + str(a.out / f"{item['stage']}_%J.log"), *command]


def attach_step_receipts(a, entry):
    stage = entry["stage"]
    entry["steps"] = [json.loads(path.read_text()) for path in (a.out / "steps").glob(f"{stage}_*.json")]
    entry["gpu_selection"] = [json.loads(path.read_text()) for path in (a.out / "gpu_selection").glob(f"{stage}_*.json")]


def start_receipt(a, item, data, receipt, gpu_index):
    entry = dict(stage=item["stage"], kind="step", job_id=a.allocation_job, minutes=item["minutes"],
                 selected_gpus=item["selected_gpus"], selected_gpu_index=gpu_index,
                 max_attributable_gpuh=item["minutes"] * item["selected_gpus"] / 60,
                 status="starting", start_unix=time.time(), step_receipts=str(a.out / "steps"),
                 selection_receipts=str(a.out / "gpu_selection"))
    data["jobs"].append(entry); data["submitted"] = True; atomic_json(receipt, data)
    return entry


def finish_receipt(a, entry, returncode, data, receipt):
    entry.update(status="complete" if returncode == 0 else "failed", returncode=returncode, end_unix=time.time())
    attach_step_receipts(a, entry)
    atomic_json(receipt, data)


def run_parallel_adaptations(a, items, data, receipt, project, resources, upstream, checkpoint,
                             poll_seconds=5):
    """Distinct physical GPUs per active stage; never cancel sibling or other jobs.

    A failure stops pending launches. Already started stages finish under their
    original scheduler/budget ceilings, then the controller reports failure.
    """
    pending = list(items)
    free_gpus = list(range(a.parallel_gpus))
    active = []
    failures = []
    while pending or active:
        while pending and free_gpus and not failures:
            item = pending.pop(0); gpu = free_gpus.pop(0)
            invocation = step_command(a, item, project, resources, upstream, checkpoint, gpu)
            print(shlex.join(invocation), flush=True)
            if not a.submit:
                # Dry run expresses the first eight placements and the ninth's
                # dependency on an available slot; no process is started.
                data.setdefault("parallel_plan", []).append(dict(stage=item["stage"], gpu_index=gpu,
                                                                 placement="first available GPU slot"))
                if not free_gpus:
                    free_gpus = list(range(a.parallel_gpus))
                continue
            entry = start_receipt(a, item, data, receipt, gpu)
            try:
                process = subprocess.Popen(invocation)
            except BaseException as exc:
                entry["reason"] = str(exc)
                finish_receipt(a, entry, -1, data, receipt)
                failures.append(item["stage"])
                break
            entry.update(status="running", controller_child_pid=process.pid)
            atomic_json(receipt, data)
            active.append((process, entry, gpu))
        if not a.submit:
            break
        remaining = []
        for process, entry, gpu in active:
            code = process.poll()
            if code is None:
                remaining.append((process, entry, gpu))
                attach_step_receipts(a, entry)
            else:
                finish_receipt(a, entry, code, data, receipt)
                free_gpus.append(gpu)
                if code:
                    failures.append(entry["stage"])
        active = remaining
        atomic_json(receipt, data)
        if failures and not active:
            data["not_started_after_failure"] = [item["stage"] for item in pending]
            atomic_json(receipt, data)
            raise RuntimeError("Adaptation stages failed; subsequent steps not started: " + ", ".join(failures))
        if active:
            time.sleep(poll_seconds)


def launch(a, project, resources, association, allocation, used, upstream, checkpoint):
    receipt = a.out / "submission.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else None
    if previous and previous.get("jobs") and not a.resume:
        raise RuntimeError("Output has submission receipts; reconcile them and use --resume")
    if previous and a.resume:
        for entry in previous.get("jobs", []):
            if entry.get("kind") != "step":
                active = run(["squeue", "-h", "-j", entry["job_id"], "-o", "%i"])
                if active:
                    raise RuntimeError("Previous V5 job is still active: " + entry["job_id"])
        identities = set()
        for folder in ("steps", "gpu_selection"):
            for path in (a.out / folder).glob("*.json"):
                entry = json.loads(path.read_text())
                if entry.get("status") in ("running", "validated", "validating"):
                    identities.add(entry["job_id"] + "." + entry["step_id"])
        for identity in sorted(identities):
            try:
                active = fields(run(["scontrol", "show", "step", identity, "-o"]))
            except subprocess.CalledProcessError:
                active = {}
            if active.get("State") == "RUNNING":
                raise RuntimeError("Previous V5 Slurm step remains active: " + identity)
    ledger_path = a.out / "budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"charged_gpuh": 0., "stages": {}}
    if ledger.get("leases") and not a.resume:
        raise RuntimeError("Existing budget requires --resume and accounting reconciliation")
    if any(x.get("status") == "reserved_or_interrupted" for x in ledger.get("leases", [])):
        raise RuntimeError("Interrupted budget lease needs Slurm accounting reconciliation before resume")
    plan = stage_plan(a, ledger)
    requested = sum(item["minutes"] * item["selected_gpus"] for item in plan) / 60
    if ledger.get("charged_gpuh", 0.) + requested > 60 + 1e-8:
        raise RuntimeError("Charged time plus requested stage ceilings exceeds persistent 60 GPUh cap")
    if used is not None and used + requested > 300:
        raise RuntimeError(f"Observed {used:.2f} GPUh + requested {requested:.2f} exceeds initial NYU 300h")
    data = dict(backend=a.backend, account=a.account, qos=a.qos, gpu_model=a.gpu_model,
                resources=resources, association_evidence=association, allocation_evidence=allocation,
                observed_allocated_gpuh=used, requested_max_gpuh=requested,
                charged_round_gpuh=ledger.get("charged_gpuh", 0.), round_limit_gpuh=60,
                parent_allocated_gpu_count=gpu_count(allocation.get("AllocTRES", "")) if allocation else 1,
                step_allocated_gpu_count=a.shared_allocation_gpus or 1,
                task_selected_gpu_count=1 if a.parallel_gpus == 1 else None,
                selected_gpu_count_by_phase=dict(profile=1, features=a.parallel_gpus, p0=1, adapt=1, replay=1),
                task_selected_gpu_index=a.gpu_index, feature_workers=a.parallel_gpus,
                max_concurrent_adaptation_stages=a.parallel_gpus, allow_gpu_sharing=a.allow_gpu_sharing,
                minimum_free_gpu_gib=a.min_free_gpu_gib, planned_stages=plan,
                accounting_note="60 GPUh caps the sum of selected GPU walltime across all workers and stages; the preexisting shared parent reservation is separately reported" if a.shared_allocation_gpus else "60 GPUh caps this run's allocated GPU time",
                jobs=[], submitted=False)
    if previous:
        data["previous_submissions"] = previous
    print(json.dumps(data, indent=2), flush=True)
    dependency = None
    adaptations = [item for item in plan if item["phase"] == "adapt"]
    adaptations_done = False
    for item in plan:
        phase, stage = item["phase"], item["stage"]
        if phase == "adapt" and a.parallel_gpus > 1:
            if not adaptations_done:
                run_parallel_adaptations(a, adaptations, data, receipt, project, resources, upstream, checkpoint)
                adaptations_done = True
            continue
        if a.allocation_job:
            gpu_index = a.gpu_index if item["selected_gpus"] == 1 else None
            invocation = step_command(a, item, project, resources, upstream, checkpoint, gpu_index)
            print(shlex.join(invocation), flush=True)
            if a.submit:
                entry = start_receipt(a, item, data, receipt, gpu_index)
                result = subprocess.run(invocation, check=False)
                finish_receipt(a, entry, result.returncode, data, receipt)
                if result.returncode:
                    raise RuntimeError(f"Stage {stage} failed; subsequent steps not started, inspect retained logs")
        else:
            command = scientific_command(a, item, upstream, checkpoint, None)
            script = "#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(project)) + "\nexec " + shlex.join(command) + "\n"
            invocation = ["sbatch", "--parsable", "--account=" + a.account, "--partition=" + a.partition,
                          "--gres=" + resources["gres"], "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=48G",
                          f"--time={item['minutes']}", "--signal=B:USR1@120", "--no-requeue", "--job-name=v5_" + stage,
                          "--chdir=" + str(project), "--output=" + str(a.out / f"{stage}_%j.log")]
            if a.qos:
                invocation += ["--qos=" + a.qos]
            if dependency:
                invocation += ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
            print(shlex.join(invocation) + " <<'SCRIPT'\n" + script + "SCRIPT", flush=True)
            if a.submit:
                result = subprocess.run(invocation, input=script, text=True, capture_output=True, check=True)
                job = result.stdout.strip().split(";")[0]
                if not job.isdigit():
                    raise RuntimeError("Unknown sbatch response; inspect queue before retry: " + result.stdout)
                data["jobs"].append(dict(stage=stage, kind="batch", job_id=job, minutes=item["minutes"], dependency=dependency))
                data["submitted"] = True; atomic_json(receipt, data); dependency = job
    if not a.submit:
        atomic_json(a.out / "submission_plan.json", data)
    elif data["jobs"]:
        ids = sorted(set(j["job_id"] for j in data["jobs"]))
        print(run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i %T %R"]), flush=True)


if __name__ == "__main__":
    main()
