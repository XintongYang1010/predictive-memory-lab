import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from .common import Budget,BudgetExceeded,atomic_json,digest,runtime,source_identity
from .data import episode,SCHEMA
from .features import Encoder,cache_split,load_records,probe
from .train import train_p0,train_candidate,reader_only
from .report import report
from .memory import VARIANTS


def main():
    p=argparse.ArgumentParser(description="V5 controlled RGB memory pilot; official I-JEPA and NYU Burst A100")
    p.add_argument("phase",choices=["preview","profile","features","p0","adapt","replay","report"])
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--ijepa-code",type=Path)
    p.add_argument("--ijepa-checkpoint",type=Path)
    p.add_argument("--lease-hours",type=float,default=.5)
    p.add_argument("--variant",choices=[*VARIANTS,"full","recent","fifo"])
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--frame-batch",type=int,default=32)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.phase=="preview":
        from PIL import Image
        for family in ("hold","interference","update"):
            variants=[episode("train",0,v,family=family,writes=8) for v in (0,1)]
            canvas=Image.new("RGB",(224*7,224*2))
            for row,ep in enumerate(variants):
                indices=[0,7,len(ep.rgb)//2,len(ep.rgb)-4]
                frames=[ep.rgb[i] for i in indices]+list(ep.target_rgb)
                for col,frame in enumerate(frames):canvas.paste(Image.fromarray(frame),(224*col,224*row))
            canvas.save(a.out/f"pair_{family}.png")
        return
    if a.phase=="report":report(a.out);return
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("Scientific runner requires the requested A100; CPU smoke is a separate test")
    if os.environ.get("SLURM_JOB_ACCOUNT")!="csci_ga_2271-2026fa":
        raise RuntimeError("Run inside the CV course Cloud Bursting allocation csci_ga_2271-2026fa")
    cluster=os.environ.get("SLURM_CLUSTER_NAME","").lower()
    if "torch" in cluster:
        raise RuntimeError("Course policy requires Cloud Bursting, not Torch")
    if int(os.environ.get("SLURM_RESTART_COUNT","0"))>2:
        raise RuntimeError("Repeated preemption; inspect state before requesting another allocation")
    # A visible allocation may contain extra devices; fail rather than silently
    # charge a multi-GPU allocation as one GPU-hour per hour.
    if torch.cuda.device_count()!=1:
        raise RuntimeError("This pilot expects exactly one allocated/visible GPU")
    stage="adapt_"+a.variant if a.phase=="adapt" and a.variant else a.phase
    limits={"profile":.5,"features":8.,"p0":12.,"replay":4.}
    limit=4. if a.variant in VARIANTS else 2.
    budget=Budget(a.out,stage,limits.get(a.phase,limit),a.lease_hours)
    status="partial"
    try:
        info=runtime()
        manifest=a.out/"run_manifest.json"
        config=dict(schema=SCHEMA,train=2048,dev=256,test=512,seed=42,round_limit_gpuh=60,
                    prototype="two moving targets; independent gap/write axes; terminal multi-horizon prediction",
                    source=source_identity())
        if manifest.exists() and json.loads(manifest.read_text())["configuration"]!=config:
            raise RuntimeError("Run identity changed; use a fresh output directory")
        atomic_json(manifest,dict(configuration=config,latest_runtime=info,status="partial"))
        cache=a.out/"features"
        if a.phase in ("profile","features"):
            if not a.ijepa_code or not a.ijepa_checkpoint:
                raise ValueError("Supply pinned official I-JEPA code and checkpoint paths")
            encoder=Encoder(a.ijepa_code,a.ijepa_checkpoint)
            if a.phase=="profile":
                ep=episode("profile",0,0)
                start=time.monotonic();frames=0
                for i in range(0,len(ep.rgb),a.frame_batch):
                    budget.check();encoder(ep.rgb[i:i+a.frame_batch]);frames+=len(ep.rgb[i:i+a.frame_batch])
                torch.cuda.synchronize()
                seconds=time.monotonic()-start
                estimate=2816*64*seconds/frames/3600
                atomic_json(a.out/"profile.json",dict(status="complete",seconds=seconds,frames=frames,
                    seconds_per_frame=seconds/frames,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                    encoder=encoder.identity,runtime=info,estimated_feature_gpuh_upper=estimate))
                if estimate>7.5:
                    raise RuntimeError("Feature throughput projects beyond the 8 GPUh budget; inspect batching before continuation")
            else:
                if not (a.out/"profile.json").exists():raise RuntimeError("Profile must run first")
                # Probe before spending on the full feature cache.
                pilot=a.out/"probe_features"
                cache_split(encoder,pilot,"train",128,budget,a.frame_batch)
                cache_split(encoder,pilot,"dev",128,budget,a.frame_batch)
                check=probe(load_records(pilot,"train"),load_records(pilot,"dev"))
                atomic_json(a.out/"feature_gate.json",check)
                if not check["passed"]:
                    raise RuntimeError("Pooled visual cue probe failed; inspect raw vs pooled before training P0")
                for split,count in [("train",2048),("dev",256)]:
                    cache_split(encoder,cache,split,count,budget,a.frame_batch)
                # Test identities reserved now; Test features/evaluation remain unopened.
                atomic_json(a.out/"data_manifest.json",dict(schema=SCHEMA,train=2048,dev=256,test=512,
                    test_status="reserved_unopened",paired_variants_stay_together=True,encoder=encoder.identity))
        else:
            gate=json.loads((a.out/"feature_gate.json").read_text())
            if not gate["passed"]:raise RuntimeError("Feature gate has not passed")
            train,dev=load_records(cache,"train"),load_records(cache,"dev")
            if a.phase=="p0":train_p0(train,dev,a.out,"cuda",budget,batch_size=a.batch_size)
            elif a.phase=="adapt":
                if not a.variant:raise ValueError("adapt needs --variant")
                train_candidate(a.variant,train,dev,a.out,"cuda",budget,batch_size=a.batch_size)
            else:
                reader_only(train,dev,a.out,"cuda",budget);report(a.out)
        status="complete"
    except BaseException as exc:
        atomic_json(a.out/f"{stage}_status.json",dict(status="partial",reason=str(exc),runtime=runtime()))
        raise
    finally:
        budget.close(status)
    atomic_json(a.out/f"{stage}_status.json",dict(status=status,runtime=runtime()))


if __name__=="__main__":main()
