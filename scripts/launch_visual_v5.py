#!/usr/bin/env python3
"""Inspect live Burst partitions and prepare/submit a bounded dependency chain.

No guessed partition, hidden authentication, shell-based credential handling,
Torch fallback, GPU keepalive, or automatic quota increase. Defaults to dry run.
Run from a CPU compute session on Cloud Bursting after copying this repo/assets
to its persistent filesystem; Burst login-node mounts can differ.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ACCOUNT="csci_ga_2271-2026fa"
PLAN=[("profile",None,30),("features",None,480),("p0",None,720),
      *[("adapt",x,240) for x in ("A0","A1","B0","B1","C0","C1")],
      *[("adapt",x,120) for x in ("full","recent","fifo")],("replay",None,240)]


def run(args):
    return subprocess.check_output(args,text=True).strip()


def query_resources(partition):
    host=run(["hostname"])
    config=run(["scontrol","show","config"])
    cluster=re.search(r"^ClusterName\s*=\s*(\S+)",config,re.M)
    cluster=cluster.group(1) if cluster else "unknown"
    if "torch" in cluster.lower() or "torch" in host.lower():
        raise RuntimeError("Course computing must be on Cloud Bursting; refusing Torch")
    if host.startswith(("log-","login")) or "ood" in host.lower():
        raise RuntimeError("Prepare/submit from a Burst CPU compute session: login and compute filesystems differ")
    rows=run(["sinfo","-N","-h","-o","%P|%G|%m|%c"])
    candidates=[]
    for line in rows.splitlines():
        part,gres,mem,cpus=line.split("|")
        if part.removesuffix("*")!=partition:continue
        match=re.search(r"gpu:([^:,()]+):([0-9]+)",gres)
        if match and "a100" in (match.group(1)+part).lower():
            candidates.append(dict(partition=part.removesuffix("*"),gres=f"gpu:{match.group(1)}:1",memory_mb=int(mem.rstrip('+')),cpus=int(cpus)))
    if not candidates:
        raise RuntimeError("Selected partition has no live A100 entry in sinfo; no resource name is guessed")
    resource=min(candidates,key=lambda x:x["memory_mb"])
    if resource["memory_mb"]<49152:raise RuntimeError("Pilot requests 48 GiB host RAM; inspect partition capacity")
    return dict(host=host,cluster=cluster,**resource)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--assets",required=True,type=Path)
    p.add_argument("--partition",required=True,help="Exact live A100 partition from sinfo")
    p.add_argument("--python",default=sys.executable,type=Path)
    p.add_argument("--submit",action="store_true",help="Actually submit; otherwise inspect and print only")
    p.add_argument("--all",action="store_true",help="Queue all gated stages; default submits profile only")
    p.add_argument("--resume",action="store_true",help="Resume incomplete stages after reconciling inactive previous jobs")
    a=p.parse_args()
    project=Path(__file__).resolve().parents[1];a.out=a.out.resolve();a.assets=a.assets.resolve()
    # Preserve a venv interpreter symlink: resolving it would silently leave the
    # environment and use the system interpreter's packages instead.
    a.python=a.python.absolute()
    checkpoint=a.assets/"IN1K-vit.h.14-300e.pth.tar";upstream=a.assets/"ijepa"
    for path in (a.python,checkpoint,upstream):
        if not path.exists():raise FileNotFoundError(path)
    resources=query_resources(a.partition)
    run([str(a.python),"-c","import torch,numpy,PIL,matplotlib; assert torch.version.cuda, 'CPU-only PyTorch cannot run A100 jobs'"])
    # Report authoritative accounting; 300h is the initial course allocation,
    # not an assertion that the user still has 300h remaining.
    usage=run(["sacct","-X","-n","-P","-S","2026-09-01","-A",ACCOUNT,"-u",os.environ.get("USER","xy3257"),
               "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"])
    used=0.
    for line in usage.splitlines():
        fields=line.split("|")
        if len(fields)<4:continue
        match=re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)",fields[3])
        gpu_count=int(match.group(1)) if match else sum(int(x) for x in re.findall(r"(?:^|,)gres/gpu:[^=,]+=(\d+)",fields[3]))
        used+=int(fields[2])*gpu_count/3600
    # Refuse concurrent user allocations so reservations cannot invalidate this
    # simple cap. Report accounting and never cancel their other work.
    active=run(["squeue","-h","-u",os.environ.get("USER","xy3257"),"-A",ACCOUNT,"-o","%i"])
    for job in active.splitlines():
        details=run(["scontrol","show","job",job,"-o"])
        if re.search(r"(?:TRES|AllocTRES|ReqTRES|TresPerNode)=\S*gres/gpu",details):
            raise RuntimeError("An existing course GPU job needs accounting/reconciliation first: "+job)
    plan=PLAN if a.all else PLAN[:1]
    if a.resume:
        ledger_path=a.out/"budget.json"
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {"stages":{}}
        remaining=[]
        for phase,variant,minutes in plan:
            stage=phase+("_"+variant if variant else "")
            status_path=a.out/f"{stage}_status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status")=="complete":continue
            charged=ledger["stages"].get(stage,0.)
            minutes=int(minutes-charged*60)
            if minutes<2:raise RuntimeError("No stage budget remains for "+stage+"; inspect interrupted leases")
            remaining.append((phase,variant,minutes))
        plan=remaining
    requested=sum(minutes for _,_,minutes in plan)/60
    if used+requested>300:raise RuntimeError(f"Observed {used:.2f} GPUh + requested {requested:.2f} exceeds initial 300h")
    a.out.mkdir(parents=True,exist_ok=True)
    receipt=a.out/"submission.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("jobs") and not a.resume:
        raise RuntimeError("This output already has submission receipts; reconcile them before a new chain")
    data=dict(account=ACCOUNT,resources=resources,observed_allocated_gpuh=used,requested_max_gpuh=requested,
              quota_note="sacct sum, not a live balance; sbatch account/QOS validation is still required",jobs=[],submitted=False)
    if a.resume and receipt.exists():data["previous_submissions"]=json.loads(receipt.read_text())
    print(json.dumps(data,indent=2))
    dependency=None
    for phase,variant,minutes in plan:
        stage=phase+("_"+variant if variant else "")
        command=[str(a.python),"-m","pmr.visual_memory",phase,"--out",str(a.out),"--lease-hours",str(minutes/60)]
        if phase in ("profile","features"):
            command += ["--ijepa-code",str(upstream),"--ijepa-checkpoint",str(checkpoint)]
        if variant:command += ["--variant",variant]
        script="#!/bin/bash\nset -euo pipefail\ncd "+shlex.quote(str(project))+"\nexec "+shlex.join(command)+"\n"
        sbatch=["sbatch","--parsable","--account="+ACCOUNT,"--partition="+a.partition,
                "--gres="+resources["gres"],"--nodes=1","--ntasks=1","--cpus-per-task=4","--mem=48G",
                f"--time={minutes}","--signal=B:USR1@120","--no-requeue","--job-name=v5_"+stage,
                "--chdir="+str(project),"--output="+str(a.out/f"{stage}_%j.log")]
        if dependency:sbatch += ["--dependency=afterok:"+dependency,"--kill-on-invalid-dep=yes"]
        print(shlex.join(sbatch)+" <<'SCRIPT'\n"+script+"SCRIPT")
        if a.submit:
            result=subprocess.run(sbatch,input=script,text=True,capture_output=True,check=True)
            job=result.stdout.strip().split(";")[0]
            if not job.isdigit():raise RuntimeError("Unknown sbatch response; inspect queue before retrying: "+result.stdout)
            data["jobs"].append(dict(stage=stage,job_id=job,minutes=minutes,dependency=dependency))
            data["submitted"]=True
            temp=receipt.with_suffix(".tmp");temp.write_text(json.dumps(data,indent=2));os.replace(temp,receipt)
            dependency=job
    if a.submit and data["jobs"]:
        print(run(["squeue","-h","-j",",".join(j["job_id"] for j in data["jobs"]),"-o","%i %T %R"]))
    else:
        (a.out/"submission_plan.json").write_text(json.dumps(data,indent=2))


if __name__=="__main__":main()
