import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import signal
import time
import uuid
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
    Scheduler wall limits are the hard stop. Different stages may run in
    parallel; each stage has one active owner, and ledger updates are serialized.
    """
    def __init__(self, root, stage, limit_hours, lease_hours, gpu_count=1):
        import fcntl
        if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1:
            raise ValueError("gpu_count must be a positive integer")
        if not all(math.isfinite(x) and x > 0 for x in (limit_hours, lease_hours)):
            raise ValueError("Stage and lease hours must be positive and finite")
        root = Path(root); root.mkdir(parents=True,exist_ok=True)
        locks = root/"budget_locks"; locks.mkdir(exist_ok=True)
        stage_key = hashlib.sha256(stage.encode()).hexdigest()
        self.lock = open(locks/(stage_key+".lock"),"a+")
        self.path = root/"budget.json"
        # Keep the former run-wide lock name for compatibility with an older
        # process, but hold it only while reading and updating the ledger.
        self.ledger_lock_path = root/"active.lock"
        self.stage, self.gpus = stage,gpu_count
        charge = lease_hours*gpu_count
        self.charge,self.start,self.stop = charge,time.monotonic(),False
        self._closed = False
        self._owner_pid = os.getpid()
        self._signal_handlers = {}
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self._locked_ledger() as data:
                stage_used = data["stages"].get(stage,0.)
                allowed = min(limit_hours-stage_used,60-data["charged_gpuh"])
                if charge > allowed + 1e-8:
                    raise BudgetExceeded(f"Requested lease {charge:g} GPUh exceeds remaining stage/round budget {allowed:g}")
                self.lease = dict(lease_id=uuid.uuid4().hex,stage=stage,
                                  job_id=os.environ.get("SLURM_JOB_ID"),pid=self._owner_pid,
                                  gpu_count=gpu_count,lease_hours=lease_hours,
                                  charged_gpuh=charge,status="reserved_or_interrupted",start_unix=time.time())
                data["charged_gpuh"] += charge
                data["stages"][stage] = stage_used+charge
                data["leases"].append(self.lease)
                atomic_json(self.path,data)
                self.data = data
            self._stop_handler = lambda *_:setattr(self,"stop",True)
            for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
                self._signal_handlers[sig] = signal.signal(sig,self._stop_handler)
        except BaseException:
            self.lock.close()
            raise

    @contextmanager
    def _locked_ledger(self):
        import fcntl
        with open(self.ledger_lock_path,"a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield (json.loads(self.path.read_text()) if self.path.exists()
                   else {"charged_gpuh":0.,"stages":{},"leases":[]})

    def check(self):
        if self.stop or self.elapsed() >= max(0,self.charge-60/3600*self.gpus):
            raise BudgetExceeded("Lease nearly exhausted or checkpoint signal received")

    def elapsed(self):
        return (time.monotonic()-self.start)/3600*self.gpus

    def close(self, status):
        if self._closed:
            return
        if os.getpid() != self._owner_pid:
            raise RuntimeError("Only the process that reserved a lease may close it")
        used = min(max(0.,self.elapsed()), self.charge)
        with self._locked_ledger() as data:
            lease = next(x for x in data["leases"] if x.get("lease_id") == self.lease["lease_id"])
            if lease["status"] != "reserved_or_interrupted":
                raise RuntimeError("Budget lease was already reconciled")
            refund = max(0.,lease["charged_gpuh"]-used)
            data["charged_gpuh"] -= refund
            data["stages"][self.stage] -= refund
            lease.update(charged_gpuh=used,status=status,end_unix=time.time())
            atomic_json(self.path,data)
            self.data,self.lease = data,lease
        self._closed = True
        self.lock.close()
        for sig,previous in self._signal_handlers.items():
            if signal.getsignal(sig) is self._stop_handler:
                signal.signal(sig,previous)


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
