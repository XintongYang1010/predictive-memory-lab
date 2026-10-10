"""Resumable pilot training, matched search budgets, explicit prerequisite gates."""
import copy
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from .common import atomic_json,atomic_save,digest,BudgetExceeded
from .predictor import Predictor,Candidate,ActorBatch,batch,set_normalization


@torch.no_grad()
def evaluate(model,rows,device,batch_size=32,interventions=False):
    model.eval(); result=[]
    backbone=model.backbone if isinstance(model,Candidate) else model
    for start in range(0,len(rows),batch_size):
        part=rows[start:start+batch_size]
        actor,target,masks=batch(part,device)
        label=backbone.norm_target(target)
        pred=model(actor)
        patch_error=(pred-label).square().mean(-1)
        total=patch_error.mean((1,2)); roi=masks.sum(2).clamp(max=1)
        roi_loss=(patch_error*roi).sum((1,2))/roi.sum((1,2)).clamp_min(1e-8)
        objects=(patch_error[:,:,None,:]*masks).sum((1,3))/masks.sum((1,3)).clamp_min(1e-8)
        extra={}
        if interventions:
            if len(part)%2 or any(part[i]["metadata"]["pair_id"]!=part[i+1]["metadata"]["pair_id"] for i in range(0,len(part),2)):
                raise ValueError("Interventions require adjacent complete pairs")
            swap=torch.arange(len(part),device=device)^1
            if isinstance(model,Candidate) and hasattr(model,"writer"):
                zero=model(actor,zero_memory=True)
                swapped=model(actor,swap_index=swap)
            else:
                zero=backbone(actor,mode="recent")
                swapped=model(ActorBatch(actor.features[swap],actor.times,actor.shutters,actor.valid,actor.query_time))
            extra["zero_memory_mse" if hasattr(model,"writer") else "recent_mse"]=(zero-label).square().mean((1,2,3))
            extra["swapped_memory_mse" if hasattr(model,"writer") else "swapped_history_mse"]=(swapped-label).square().mean((1,2,3))
            extra["paired_difference_mse"]=((pred-pred[swap])-(label-label[swap])).square().mean((1,2,3))
            extra["paired_target_energy"]=(label-label[swap]).square().mean((1,2,3))
        for i,r in enumerate(part):
            result.append(dict(**r["metadata"],mse=float(total[i]),roi_mse=float(roi_loss[i]),
                               object_a_mse=float(objects[i,0]),object_b_mse=float(objects[i,1]),
                               **{k:float(v[i]) for k,v in extra.items()}))
    return result


def mean_mse(metrics):
    return sum(r["mse"] for r in metrics)/len(metrics)


def optimize(model,train,dev,folder,steps,lr,device,budget,batch_size=32,seed=42,eval_every=200,p0=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model.to(device)
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0.)
    config=dict(steps=steps,lr=lr,batch_size=batch_size,seed=seed,p0=p0,train_count=len(train),dev_count=len(dev))
    path=folder/"last.pt"
    best,step=float("inf"),0
    if path.exists():
        ck=torch.load(path,map_location=device,weights_only=True)
        if ck["config"]!=config:
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        step,best=ck["step"],ck["best"]
        torch.set_rng_state(ck["rng"].cpu())
        if device.startswith("cuda") and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
    def save(status):
        atomic_save(path,dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
                             step=step,best=best,rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
                             status=status))
    if step==steps:
        ck=torch.load(folder/"best.pt",map_location=device,weights_only=True)
        model.load_state_dict(ck["model"]);return best
    started=time.monotonic()
    try:
        for current in range(step,steps):
            budget.check();model.train()
            # Exact same episode order for every candidate and LR trial; resumes
            # do not skip or duplicate completed optimization steps.
            rng=np.random.default_rng(seed+current)
            indices=rng.integers(0,len(train),batch_size)
            actor,target,_=batch([train[int(i)] for i in indices],device)
            backbone=model.backbone if isinstance(model,Candidate) else model
            optimizer.zero_grad(set_to_none=True)
            # Dropout mask seeded by current step. No future label enters actor.
            torch.manual_seed(seed+current)
            pred=model(actor,drop_history=current%2==1) if p0 else model(actor)
            loss=(pred-backbone.norm_target(target)).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite loss")
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(grad):raise FloatingPointError("Non-finite gradients")
            optimizer.step();step=current+1
            if step%20==0:save("partial")
            if step%eval_every==0 or step==steps:
                metrics=evaluate(model,dev,device,batch_size)
                score=mean_mse(metrics)
                if score<best:
                    best=score
                    atomic_save(folder/"best.pt",dict(model=model.state_dict(),step=step,dev_mse=best,config=config))
                row=dict(step=step,train_loss=float(loss.detach()),dev_mse=score,gradient_norm=float(grad),
                         elapsed_seconds=time.monotonic()-started,trainable_parameters=sum(p.numel() for p in params))
                if hasattr(model,"memory_scale"):
                    row["memory_scale"]=model.memory_scale.detach().cpu().tolist()
                with open(folder/"learning_curves.jsonl","a") as f:f.write(json.dumps(row)+"\n")
                print(json.dumps(dict(run=folder.name,**row)),flush=True)
                save("partial")
        save("complete")
    except BaseException:
        save("partial");raise
    model.load_state_dict(torch.load(folder/"best.pt",map_location=device,weights_only=True)["model"])
    return best


def search_and_train(factory,train,dev,folder,lrs,steps,device,budget,batch_size=32,seed=42,p0=False,trial_steps=150):
    folder=Path(folder); scores=[]
    for lr in lrs:
        torch.manual_seed(seed);model=factory()
        score=optimize(model,train,dev,folder/f"lr_{lr:g}",trial_steps,lr,device,budget,
                       batch_size,seed,eval_every=trial_steps,p0=p0)
        scores.append(dict(lr=lr,dev_mse=score))
        del model
    lr=min(scores,key=lambda x:x["dev_mse"])["lr"]
    atomic_json(folder/"lr_search.json",dict(trials=scores,selected_lr=lr,
                policy="Equal short trials; final training restarts at common initialization"))
    torch.manual_seed(seed);model=factory()
    optimize(model,train,dev,folder/"fit",steps,lr,device,budget,batch_size,seed,p0=p0)
    return model


def history_gate(model,dev,device):
    rows=evaluate(model,dev,device,interventions=True)
    grouped={}
    for r in rows:
        grouped.setdefault(r["pair_id"],[]).append(r)
    deltas=np.array([np.mean([r["recent_mse"]-r["mse"] for r in p]) for p in grouped.values()])
    swap=np.array([np.mean([r["swapped_history_mse"]-r["mse"] for r in p]) for p in grouped.values()])
    rng=np.random.default_rng(171)
    ci=np.quantile([deltas[rng.integers(0,len(deltas),len(deltas))].mean() for _ in range(1000)],[.025,.975])
    family={f:float(np.mean([r["recent_mse"]-r["mse"] for r in rows if r["family"]==f])) for f in ("hold","interference","update")}
    passed=bool(ci[0]>0 and swap.mean()>0 and all(x>0 for x in family.values()))
    return dict(passed=passed,full_mse=mean_mse(rows),mean_recent_minus_full=float(deltas.mean()),
                paired_bootstrap_95ci=ci.tolist(),swapped_minus_full=float(swap.mean()),by_family=family,
                criterion="Dev paired lower CI>0, wrong history harms, all three family means positive"),rows


def save_metrics(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w") as f:
        for r in rows:f.write(json.dumps(r)+"\n")


def train_p0(train,dev,out,device,budget,steps=5000,batch_size=32):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    template=Predictor(train[0]["features"].shape[-1],train[0]["features"].shape[-2])
    set_normalization(template,train)
    factory=lambda:copy.deepcopy(template)
    small=train[:32]
    small_model=factory().to(device)
    before=mean_mse(evaluate(small_model,small,device))
    after=optimize(small_model,small,small,out/"overfit32",300,1e-3,device,budget,batch_size,eval_every=100,p0=True)
    check=dict(initial_mse=before,final_mse=after,passed=after<before*.8,
               criterion="Pilot small-sample training MSE falls at least 20%")
    atomic_json(out/"overfit_gate.json",check)
    if not check["passed"]:raise RuntimeError("Small-sample gate failed; stop and inspect optimization")
    model=search_and_train(factory,train,dev,out/"p0_training",[1e-4,3e-4,1e-3],steps,device,budget,batch_size,p0=True)
    atomic_save(out/"p0.pt",dict(config=model.config,model=model.state_dict()))
    gate,rows=history_gate(model,dev,device)
    gate["p0_sha256"]=digest(out/"p0.pt")
    atomic_json(out/"history_gate.json",gate);save_metrics(out/"p0_dev.jsonl",rows)
    if not gate["passed"]:raise RuntimeError("History gate failed; no memory comparison is authorized by this protocol yet")


def load_p0(out,device):
    out=Path(out);gate=json.loads((out/"history_gate.json").read_text())
    if not gate["passed"] or gate["p0_sha256"]!=digest(out/"p0.pt"):
        raise ValueError("A passed history gate for this exact P0 is required")
    ck=torch.load(out/"p0.pt",map_location="cpu",weights_only=True)
    model=Predictor(**ck["config"]);model.load_state_dict(ck["model"])
    return model.to(device)


def train_candidate(variant,train,dev,out,device,budget,steps=2000,batch_size=32):
    out=Path(out);base=load_p0(out,device="cpu")
    factory=lambda:Candidate(copy.deepcopy(base),variant)
    model=search_and_train(factory,train,dev,out/variant,[3e-5,1e-4,3e-4],steps,device,budget,batch_size)
    atomic_save(out/variant/"model.pt",dict(variant=variant,model=model.state_dict(),p0_sha256=digest(out/"p0.pt")))
    rows=evaluate(model,dev,device,interventions=True)
    save_metrics(out/variant/"per_episode.jsonl",rows)
    atomic_json(out/variant/"status.json",dict(status="complete",variant=variant,dev_mse=mean_mse(rows),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  checkpoint_sha256=digest(out/variant/"model.pt")))


def reader_only(train,dev,out,device,budget,steps=500):
    """C0 writer, exact same recurrent states; only Q adapter and fusion can learn."""
    out=Path(out);backbone=load_p0(out,"cpu")
    source=torch.load(out/"C0"/"model.pt",map_location="cpu",weights_only=True)
    torch.manual_seed(42);model=Candidate(backbone,"C1")
    missing=model.load_state_dict(source["model"],strict=False)
    if set(missing.missing_keys)!={"reader.down.weight","reader.up.weight"} or missing.unexpected_keys:
        raise ValueError("Unexpected reader-only initialization difference")
    model.writer.requires_grad_(False)
    # No writer weights/projections can change, so deterministic prefix replay
    # gives the same M at every optimization step without retaining huge caches.
    writer_before={k:v.clone() for k,v in model.writer.state_dict().items()}
    optimize(model,train,dev,out/"reader_only_C0",steps,1e-4,device,budget)
    if any(not torch.equal(v,model.writer.state_dict()[k].cpu()) for k,v in writer_before.items()):
        raise AssertionError("Reader-only run modified the writer")
    save_metrics(out/"reader_only_C0"/"per_episode.jsonl",evaluate(model,dev,device,interventions=True))
