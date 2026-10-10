from pathlib import Path
import hashlib
import json
import random
import platform
import subprocess
import time
import importlib.metadata
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone(v) for v in value)
    return value


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k:move(v,device) for k,v in value.items()}
    if isinstance(value, list):
        return [move(v,device) for v in value]
    return value


def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def clock(device):
    if torch.device(device).type=="cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def environment(device):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu only for smoke/debug")
    try:
        commit=subprocess.check_output(["git","rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
    except subprocess.CalledProcessError:
        commit=None
    versions={}
    for name in ("numpy","av2","pandas","pyarrow","matplotlib","pytest"):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {"python":platform.python_version(),"torch":str(torch.__version__),"dependencies":versions,
            "cuda_runtime":torch.version.cuda,"device":str(device),"code_commit":commit,
            "gpu":torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None}


def code_fingerprint():
    root=Path(__file__).resolve().parents[1]
    h=hashlib.sha256()
    for path in sorted((root/"pmr").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()
