import argparse
from contextlib import nullcontext
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
from .slurm_backend import NYU_ACCOUNT,validate_runtime
from .profiling import measure_encoder
from .feature_workers import FeatureWorkers


def main():
    p=argparse.ArgumentParser(description="V5 controlled RGB memory pilot; official I-JEPA in a validated Slurm allocation")
    p.add_argument("phase",choices=["preview","profile","features","p0","adapt","replay","report"])
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--ijepa-code",type=Path)
    p.add_argument("--ijepa-checkpoint",type=Path)
    p.add_argument("--backend",choices=("nyu","dgx"),default="nyu")
    p.add_argument("--account")
    p.add_argument("--partition")
    p.add_argument("--qos")
    p.add_argument("--gpu-model",choices=("A100","H800"),default="A100")
    p.add_argument("--allocation-job")
    p.add_argument("--shared-allocation-gpus",type=int)
    p.add_argument("--gpu-index",type=int)
    p.add_argument("--feature-workers",type=int,choices=(1,8),default=1,
                   help="Persistent GPU encoders for the features stage; other stages retain one GPU")
    p.add_argument("--allow-gpu-sharing",action="store_true")
    p.add_argument("--min-free-gpu-gib",type=float,default=40.)
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
    if not torch.cuda.is_available():
        raise RuntimeError("Scientific runner requires CUDA; CPU smoke is a separate test")
    if a.frame_batch<1 or a.batch_size<1:raise ValueError("Batch sizes must be positive")
    worker_gpus=a.feature_workers if a.phase=="features" else 1
    account=a.account or (NYU_ACCOUNT if a.backend=="nyu" else None)
    if a.backend=="nyu" and (a.gpu_model!="A100" or a.allocation_job):
        raise RuntimeError("NYU backend requires A100 and a dedicated course allocation")
    slurm=validate_runtime(a.backend,account,a.partition,a.gpu_model,
                           torch.cuda.device_count(),torch.cuda.get_device_name(),
                           allocation_job=a.allocation_job,qos=a.qos,
                           shared_allocation_gpus=a.shared_allocation_gpus,gpu_index=a.gpu_index,
                           expected_worker_gpus=worker_gpus,allow_gpu_sharing=a.allow_gpu_sharing,
                           min_free_gpu_gib=a.min_free_gpu_gib)
    stage="adapt_"+a.variant if a.phase=="adapt" and a.variant else a.phase
    limits={"profile":.5,"features":8.,"p0":12.,"replay":4.}
    limit=4. if a.variant in VARIANTS else 2.
    budget=Budget(a.out,stage,limits.get(a.phase,limit),a.lease_hours,gpu_count=worker_gpus)
    status="partial"
    step_receipt=a.out/"steps"/f"{stage}_{slurm['job_id']}_{slurm['step_id'] or 'batch'}.json"
    receipt=dict(stage=stage,job_id=slurm["job_id"],step_id=slurm["step_id"] or "batch",
                 status="running",start_unix=time.time(),allocation=slurm,lease_hours=a.lease_hours,worker_gpus=worker_gpus)
    atomic_json(step_receipt,receipt)
    try:
        info=dict(runtime(),slurm=slurm)
        manifest=a.out/"run_manifest.json"
        config=dict(schema=SCHEMA,train=2048,dev=256,test=512,seed=42,round_limit_gpuh=60,
                    prototype="two moving targets; independent gap/write axes; terminal multi-horizon prediction",
                    source=source_identity(),
                    execution=dict(backend=a.backend,account=account,partition=slurm["partition"],
                                   requested_gpu_model=a.gpu_model,
                                   shared_parent_allocation=slurm["shared_parent_allocation"],
                                   parent_allocated_gpu_count=slurm["parent_allocated_gpu_count"],
                                   feature_workers=a.feature_workers,batch_size=a.batch_size,frame_batch=a.frame_batch))
        if manifest.exists() and json.loads(manifest.read_text())["configuration"]!=config:
            raise RuntimeError("Run identity changed; use a fresh output directory")
        # Profile creates this immutable run identity. Concurrent adaptation
        # stages have their own receipts; only the final report updates status.
        if not manifest.exists():
            atomic_json(manifest,dict(configuration=config,latest_runtime=info,status="partial"))
        cache=a.out/"features"
        if a.phase in ("profile","features"):
            if not a.ijepa_code or not a.ijepa_checkpoint:
                raise ValueError("Supply pinned official I-JEPA code and checkpoint paths")
            if a.phase=="profile":
                load_start=time.monotonic()
                encoder=Encoder(a.ijepa_code,a.ijepa_checkpoint)
                encoder_load_seconds=time.monotonic()-load_start
                import numpy as np
                if a.frame_batch<1:raise ValueError("frame-batch must be positive")
                warmup_source=episode("profile",0,0,writes=32).rgb
                copies=(a.frame_batch+len(warmup_source)-1)//len(warmup_source)
                warmup_rgb=np.concatenate([warmup_source]*copies)[:a.frame_batch]
                measurements=[(writes,repeat,episode("profile",repeat,0,writes=writes).rgb)
                              for writes in (0,8,32) for repeat in range(3)]
                timing=measure_encoder(encoder,budget,warmup_rgb,measurements,a.frame_batch,
                                       torch.cuda.synchronize,torch.cuda.reset_peak_memory_stats,
                                       torch.cuda.max_memory_allocated)
                estimate=2816*64*timing["seconds_per_frame"]/3600
                profile=dict(status="complete" if estimate<=7.5 else "partial",**timing,
                    encoder_load_seconds=encoder_load_seconds,
                    encoder=encoder.identity,runtime=info,estimated_feature_gpuh_upper=estimate,
                    estimate_frame_count=2816*64,throughput_gate_limit_gpuh=7.5,
                    throughput_gate_passed=estimate<=7.5)
                atomic_json(a.out/"profile.json",profile)
                if not profile["throughput_gate_passed"]:
                    raise RuntimeError("Feature throughput projects beyond the 8 GPUh budget; inspect batching before continuation")
            else:
                profile=json.loads((a.out/"profile.json").read_text())
                profile_status=json.loads((a.out/"profile_status.json").read_text())
                if (profile.get("status")!="complete" or not profile.get("throughput_gate_passed")
                        or profile_status.get("status")!="complete" or not profile.get("encoder")):
                    raise RuntimeError("A successful profile must precede features")
                if a.feature_workers>1:
                    # No encoder is allocated in the parent: each persistent
                    # child owns exactly one of the eight validated GPUs.
                    context=FeatureWorkers(a.ijepa_code,a.ijepa_checkpoint,a.feature_workers,
                                           budget,a.out/"workers",profile["encoder"])
                else:
                    context=nullcontext(Encoder(a.ijepa_code,a.ijepa_checkpoint))
                with context as extractor:
                    if extractor.identity!=profile["encoder"]:
                        raise RuntimeError("A successful profile of this exact encoder must precede features")
                    def extract(root,split,count):
                        if a.feature_workers>1:
                            extractor.cache_split(root,split,count,a.frame_batch)
                        else:
                            cache_split(extractor,root,split,count,budget,a.frame_batch)
                    # Probe before spending on the full feature cache. All
                    # pair variants remain in the same worker and data split.
                    pilot=a.out/"probe_features"
                    extract(pilot,"train",128)
                    extract(pilot,"dev",128)
                    budget.check()
                    check=probe(load_records(pilot,"train"),load_records(pilot,"dev"))
                    atomic_json(a.out/"feature_gate.json",check)
                    budget.check()
                    if not check["passed"]:
                        raise RuntimeError("Pooled visual cue probe failed; inspect raw vs pooled before training P0")
                    for split,count in [("train",2048),("dev",256)]:
                        extract(cache,split,count)
                    atomic_json(a.out/"data_manifest.json",dict(schema=SCHEMA,train=2048,dev=256,test=512,
                        test_status="reserved_unopened",paired_variants_stay_together=True,
                        feature_workers=a.feature_workers,encoder=extractor.identity))
        else:
            gate=json.loads((a.out/"feature_gate.json").read_text())
            if not gate["passed"]:raise RuntimeError("Feature gate has not passed")
            train,dev=load_records(cache,"train"),load_records(cache,"dev")
            if a.phase=="p0":train_p0(train,dev,a.out,"cuda",budget,batch_size=a.batch_size)
            elif a.phase=="adapt":
                if not a.variant:raise ValueError("adapt needs --variant")
                train_candidate(a.variant,train,dev,a.out,"cuda",budget,batch_size=a.batch_size)
            else:
                reader_only(train,dev,a.out,"cuda",budget)
        status="complete"
    except BaseException as exc:
        atomic_json(a.out/f"{stage}_status.json",dict(status="partial",reason=str(exc),runtime=runtime(),slurm=slurm))
        receipt["reason"]=str(exc)
        raise
    finally:
        budget.close(status)
        receipt.update(status=status,end_unix=time.time())
        atomic_json(step_receipt,receipt)
    atomic_json(a.out/f"{stage}_status.json",dict(status=status,runtime=runtime(),slurm=slurm))
    if a.phase=="replay":report(a.out)


if __name__=="__main__":main()
