import argparse
from pathlib import Path
import json
import sys
import traceback
import torch
from .upstream import prepare,ROOT,UPSTREAM_COMMIT
from .common import seed_all,write_json,environment,clock,code_fingerprint


def parser():
    p=argparse.ArgumentParser(description="RealMotion memory diagnosis and controller training")
    sub=p.add_subparsers(dest="command",required=True)
    setup=sub.add_parser("setup"); setup.add_argument("--source")
    doctor=sub.add_parser("doctor"); doctor.add_argument("--device",default="cuda"); doctor.add_argument("--strict",action="store_true")
    for command in ("smoke","run"):
        q=sub.add_parser(command)
        q.add_argument("--device",default="cpu" if command=="smoke" else "cuda")
        q.add_argument("--out",default=f"runs/{command}_v2")
        q.add_argument("--seed",type=int,default=42)
        q.add_argument("--steps",type=int,default=30 if command=="smoke" else 500)
        q.add_argument("--rounds",type=int,default=1)
        q.add_argument("--dev-candidates",type=int,default=3,help="Maximum candidate policies evaluated on complete dev rollouts per round")
        q.add_argument("--threads",type=int,default=2)
        q.add_argument("--resume",action="store_true",help="Reuse completed stages only, with exact same settings")
        if command=="run":
            q.add_argument("--data-root",required=True)
            q.add_argument("--checkpoint",required=True)
            q.add_argument("--train-scenes",type=int,default=1000)
            q.add_argument("--dev-scenes",type=int,default=200)
            q.add_argument("--calib-scenes",type=int,default=200)
            q.add_argument("--eval-scenes",type=int,default=1000)
            q.add_argument("--trust-inputs",action="store_true",help="Allow Python pickle in known official checkpoint/own processed caches")
    prof=sub.add_parser("profile")
    prof.add_argument("--data-root",required=True);prof.add_argument("--checkpoint",required=True)
    prof.add_argument("--device",default="cuda");prof.add_argument("--scenes",type=int,default=5)
    prof.add_argument("--out",default="runs/profile");prof.add_argument("--trust-inputs",action="store_true")
    return p


def doctor(args):
    info=environment(args.device)
    info["target"]="Python 3.11 / PyTorch 2.9.1 / CUDA runtime 12.8"
    info["matches_target"]=sys.version_info[:2]==(3,11) and torch.__version__.split("+")[0]=="2.9.1" and torch.version.cuda=="12.8"
    if str(args.device).startswith("cuda"):
        a=torch.randn(256,256,device=args.device,requires_grad=True)
        (a@a).square().mean().backward()
        torch.cuda.synchronize(args.device)
        info["cuda_forward_backward"]=True
        info["capability"]=list(torch.cuda.get_device_capability(args.device))
    print(json.dumps(info,indent=2))
    if args.strict and not info["matches_target"]:
        raise RuntimeError("Environment differs from target; see doctor output")


def run(args):
    from .adapter import RealMotionAdapter
    from .data import make_manifest,create_synthetic,load_sequence
    from .experiment import correctness,baseline_adapter_check,calibrate,collect,evaluate,development_rollout
    from .controller import train_controller,load_controller
    from .report import report,read_jsonl
    smoke=args.command=="smoke"
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    if args.steps<1 or args.rounds<1 or args.dev_candidates<1:
        raise ValueError("steps, rounds and dev-candidates must be positive")
    out=Path(args.out).resolve()
    info=environment(args.device)
    settings={k:v for k,v in vars(args).items() if k!="resume"}
    settings["code_fingerprint"]=code_fingerprint()
    if (out/"settings.json").exists():
        if not args.resume:
            raise ValueError(f"Output exists: {out}; choose another --out or use --resume")
        if json.loads((out/"settings.json").read_text())!=settings:
            raise ValueError("Resume settings differ; use a new output directory")
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"settings.json",settings)
    info.update(smoke_only=smoke,upstream_commit=UPSTREAM_COMMIT)
    write_json(out/"environment.json",info)
    if smoke:
        create_synthetic(out/"synthetic_data")
        data_root=out/"synthetic_data"
        limits=dict(train_limit=5,dev_limit=5,calib_limit=3,eval_limit=4)
    else:
        data_root=args.data_root
        limits=dict(train_limit=args.train_scenes,dev_limit=args.dev_scenes,calib_limit=args.calib_scenes,eval_limit=args.eval_scenes)
    manifest=out/"manifest.json"
    groups=json.loads(manifest.read_text())["splits"] if args.resume and manifest.exists() else make_manifest(data_root,manifest,**limits)
    trusted=getattr(args,"trust_inputs",False)
    adapter=RealMotionAdapter(args.device,getattr(args,"checkpoint",None),trusted,random_init=smoke)
    identity=out/"checkpoint.json"
    if identity.exists() and json.loads(identity.read_text())["sha256"]!=adapter.checkpoint_hash:
        raise ValueError("Checkpoint changed since run")
    write_json(identity,{"sha256":adapter.checkpoint_hash})
    timer=clock(args.device)
    try:
        check_sequence=load_sequence(groups["calib"][0],args.device,trusted)
        write_json(out/"baseline_adapter.json",baseline_adapter_check(adapter,check_sequence))
        e0=correctness(adapter,check_sequence)
        write_json(out/"correctness.json",e0)
        cal=out/"calibration.json"
        config=json.loads(cal.read_text()) if args.resume and cal.exists() else calibrate(adapter,groups["calib"],cal,args.device,trusted,args.seed)
        current="keep"
        for r in range(args.rounds):
            prefix=out/f"round{r}"
            done=prefix.with_suffix(".complete.json")
            ctrl=prefix.with_suffix(".controller.pt")
            if args.resume and done.exists() and ctrl.exists():
                current=load_controller(ctrl,args.device)
                continue
            train_rows=collect(adapter,groups["train"],prefix.with_suffix(".train.jsonl"),config,args.device,current,trusted=trusted,seed=args.seed,round_id=r)
            dev_rows=collect(adapter,groups["dev"],prefix.with_suffix(".dev.jsonl"),config,args.device,current,trusted=trusted,seed=args.seed,round_id=r)
            dev_evaluator=lambda policy: development_rollout(adapter,groups["dev"],config,args.device,policy,trusted,args.seed)
            current=train_controller(train_rows,dev_rows,ctrl,args.device,args.steps,seed=args.seed+r,
                                     dev_evaluator=dev_evaluator,max_candidates=args.dev_candidates)
            write_json(done,{"complete":True,"round":r})
        # Confirmation is evaluated once, after the fixed number of rounds.
        collect(adapter,groups["eval"],out/"diagnostic.jsonl",config,args.device,"keep",True,trusted,args.seed)
        evaluate(adapter,groups["eval"],out/"evaluation.jsonl",config,args.device,current,trusted,args.seed)
        report(out,smoke)
        elapsed=clock(args.device)-timer
        write_json(out/"run_stats.json",{"status":"complete","seconds":elapsed,"device_hours":elapsed/3600,
                   "predictor_forward_calls":adapter.forward_calls,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else None})
        failure=out/"failure.json"
        if failure.exists(): failure.unlink()
        print(f"Complete: {out/'REPORT.md'}",flush=True)
    except Exception as exc:
        write_json(out/"failure.json",{"status":"failed","error":str(exc),"traceback":traceback.format_exc()})
        raise


def profile(args):
    from .adapter import RealMotionAdapter
    from .data import load_sequence,scene_id
    from .common import sha256
    from .experiment import rollout
    info=environment(args.device)
    torch.set_num_threads(2)
    model=RealMotionAdapter(args.device,args.checkpoint,args.trust_inputs)
    files=sorted((Path(args.data_root)/"val").glob("*.pt"))[:args.scenes]
    if not files: raise ValueError("No processed val scenes")
    timings=[]
    for path in files:
        seq=load_sequence({"path":str(path),"scene_id":scene_id(path),"sha256":sha256(path)},args.device,args.trust_inputs)
        rollout(model,seq) # warm up
        t0=clock(args.device)
        rollout(model,seq)
        timings.append(clock(args.device)-t0)
    info.update(seconds_per_scene=timings,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else None)
    write_json(Path(args.out)/"profile.json",info)
    print(json.dumps(info,indent=2))


def main():
    args=parser().parse_args()
    if args.command=="setup": print(prepare(args.source))
    elif args.command=="doctor": doctor(args)
    elif args.command=="profile": profile(args)
    else: run(args)


if __name__=="__main__":
    main()
