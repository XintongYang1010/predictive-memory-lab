import hashlib
import json
import os
from pathlib import Path
import signal
import time
import torch


def digest(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    os.replace(tmp,path)


def atomic_save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    torch.save(obj,tmp); os.replace(tmp,path)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Persistent per-stage budget. Charge reserved GPU wall time, including idle.

    Record the full requested lease BEFORE starting a GPU job. This conservative
    accounting survives SIGKILL/requeues; clean exit refunds unused reservation.
    Scheduler wall limits are the hard stop. One process/run, enforced by flock.
    """
    def __init__(self, root, stage, limit_hours, lease_hours, gpu_count=1):
        import fcntl
        root = Path(root); root.mkdir(parents=True,exist_ok=True)
        self.lock = open(root/"active.lock","a+")
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.path = root/"budget.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"charged_gpuh":0.,"stages":{},"leases":[]}
        self.stage, self.gpus = stage,gpu_count
        stage_used = self.data["stages"].get(stage,0.)
        allowed = min(limit_hours-stage_used,60-self.data["charged_gpuh"])
        charge = lease_hours*gpu_count
        if charge <= 0 or charge > allowed + 1e-8:
            self.lock.close()
            raise BudgetExceeded(f"Requested lease {charge:g} GPUh exceeds remaining stage/round budget {allowed:g}")
        self.charge,self.start,self.stop = charge,time.monotonic(),False
        self.data["charged_gpuh"] += charge
        self.data["stages"][stage] = stage_used+charge
        self.lease = dict(stage=stage,job_id=os.environ.get("SLURM_JOB_ID"),
                          charged_gpuh=charge,status="reserved_or_interrupted",start_unix=time.time())
        self.data["leases"].append(self.lease)
        atomic_json(self.path,self.data)
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
            signal.signal(sig,lambda *_:setattr(self,"stop",True))

    def check(self):
        if self.stop or self.elapsed() >= max(0,self.charge-60/3600*self.gpus):
            raise BudgetExceeded("Lease nearly exhausted or checkpoint signal received")

    def elapsed(self):
        return (time.monotonic()-self.start)/3600*self.gpus

    def close(self, status):
        used = min(self.elapsed(), self.charge)
        self.data["charged_gpuh"] -= self.charge-used
        self.data["stages"][self.stage] -= self.charge-used
        self.lease.update(charged_gpuh=used,status=status,end_unix=time.time())
        atomic_json(self.path,self.data)
        self.lock.close()


def runtime():
    import platform
    return dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                job_id=os.environ.get("SLURM_JOB_ID"))


def source_identity():
    """Hash actual code, including uncommitted/ZIP changes, for safe resumes."""
    import subprocess
    root=Path(__file__).resolve().parents[2]
    files=sorted((root/"pmr/visual_memory").glob("*.py"))+sorted((root/"scripts").glob("*visual_v5*"))
    hashes={str(p.relative_to(root)):digest(p) for p in files}
    encoded=json.dumps(hashes,sort_keys=True).encode()
    try:
        commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        commit=None
    return dict(git_commit=commit,source_sha256=hashlib.sha256(encoded).hexdigest(),files=hashes)
