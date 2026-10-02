"""Paired branch replay, train-only calibration, and deployed-policy evaluation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from tqdm import tqdm
from .adapter import ACTIONS, DIAGNOSTICS, FEATURES, FEATURE_SCHEMA, MODEL_KWARGS, evidence, observed_only
from .common import clone, clock, write_json
from .data import load_sequence
from .metrics import frame_scores


def stable_seed(scene_id,index,seed):
    return (int(hashlib.sha256(f"{scene_id}:{index}".encode()).hexdigest()[:8],16)+seed)%(2**31)


def choose(policy,frame,state,threshold):
    if state is None:
        return "keep"
    ev=evidence(frame,state)
    if isinstance(policy,str):
        if policy=="threshold":
            return "reset" if float(ev["features"][4])>threshold else "keep"
        return policy
    return policy.choose(ev["features"])


def rollout(adapter,seq,policy="keep",sigma=2.,threshold=2.,tau=1.,seed=0):
    state=None
    outs,states,edits=[],[],[]
    for i,frame in enumerate(seq):
        edit=choose(policy,frame,state,threshold)
        out,state=adapter.step(frame,state,edit,sigma=sigma,tau=tau,
                               seed=stable_seed(frame["scenario_id"][0],i,seed))
        outs.append(out)
        states.append(clone(state))
        edits.append(edit)
    return outs,states,edits


def replay(adapter,seq,start,state,edit,continuation="keep",sigma=2.,threshold=2.,tau=1.,seed=0):
    state=clone(state)
    result=[]
    t0=clock(adapter.device)
    calls=adapter.forward_calls
    for i in range(start,len(seq)):
        action=edit if i==start else choose(continuation,seq[i],state,threshold)
        out,state=adapter.step(seq[i],state,action,sigma=sigma,tau=tau,
                               seed=stable_seed(seq[i]["scenario_id"][0],i,seed))
        result.append(frame_scores(out,seq[i],common=True))
    return {"loss":float(np.mean([x["energy_score"] for x in result])),
            "final":frame_scores(out,seq[-1]),
            "curve":result,"seconds":clock(adapter.device)-t0,"forward_calls":adapter.forward_calls-calls}


@torch.no_grad()
def baseline_adapter_check(adapter, seq):
    """Compare Keep against a separate hook-free pinned model on the same backend.

    This checks the wrapper, not NATTEN parity or the published benchmark score.
    """
    reference = type(adapter.model)(**MODEL_KWARGS).to(adapter.device).eval().requires_grad_(False)
    reference.load_state_dict(adapter.model.state_dict(), strict=True)
    state, memory, max_error = None, None, 0.
    for frame in seq:
        data = observed_only(frame)
        data["memory_dict"] = clone(memory)
        expected = reference(data)
        memory = expected["memory_dict"]
        actual, state = adapter.step(frame, state, "keep")
        for key in ("y_hat", "pi"):
            torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5)
            max_error = max(max_error, float((actual[key]-expected[key]).abs().max()))
    return {"passed": True, "updates": len(seq), "max_absolute_error": max_error,
            "scope": "Keep adapter vs hook-free pinned model, both portable PyTorch backend",
            "official_NATTEN_or_benchmark_reproduction": False}


def correctness(adapter,seq):
    out,state=adapter.step(seq[0])
    expected,next_state=adapter.step(seq[1],state)
    actual,_=adapter.step(seq[1],clone(state),"keep")
    if not torch.equal(expected["y_hat"],actual["y_hat"]):
        raise AssertionError("No-op clone/replay differs")
    mutated=clone(seq[1])
    mutated["target"]=mutated["target"]+10000
    mutated["target_mask"]=~mutated["target_mask"]
    blind,_=adapter.step(mutated,state)
    if not torch.equal(actual["y_hat"],blind["y_hat"]):
        raise AssertionError("Future labels affect prediction")
    before=clone(state)
    adapter.step(seq[1],state,"rebuild")
    for key in state["memory"]:
        if isinstance(state["memory"][key],torch.Tensor) and not torch.equal(state["memory"][key],before["memory"][key]):
            raise AssertionError(f"State mutation: {key}")
    direct=adapter._forward(seq[0],None)
    if not torch.equal(direct["y_hat"],out["y_hat"]):
        raise AssertionError("Cross-scene state contamination")
    return {"noop_exact":True,"future_blind":True,"branch_isolated":True,"scene_reset":True}


def calibrate(adapter,records,out,device,trusted=False,seed=0):
    # Cached CPU sequences avoid repeated deserialization, not extra observations.
    seqs=[load_sequence(r,"cpu",trusted) for r in tqdm(records,desc="calibration data")]
    from .common import move
    residuals=[]
    for seq_cpu in seqs:
        seq=move(seq_cpu,device)
        _,states,_=rollout(adapter,seq)
        residuals.extend(float(evidence(seq[i],states[i-1])["features"][4]) for i in (1,2))
    threshold=float(np.quantile(residuals,.8))
    trials=[]
    for kind,values in (("reweight",(.5,1.,2.,4.,8.)),("decay",(.25,1.,4.))):
        for value in values:
            losses=[]
            start=clock(device)
            calls=adapter.forward_calls
            for seq_cpu in seqs:
                seq=move(seq_cpu,device)
                outs,_,_=rollout(adapter,seq,kind,sigma=value if kind=="reweight" else 2.,tau=value if kind=="decay" else 1.,seed=seed)
                losses.append(frame_scores(outs[-1],seq[-1],common=True)["energy_score"])
            trials.append({"policy":kind,"value":value,"energy_score":float(np.mean(losses)),
                           "seconds":clock(device)-start,"forward_calls":adapter.forward_calls-calls})
    sigma=min((x for x in trials if x["policy"]=="reweight"),key=lambda x:x["energy_score"])["value"]
    tau=min((x for x in trials if x["policy"]=="decay"),key=lambda x:x["energy_score"])["value"]
    config={"sigma":sigma,"tau":tau,"threshold":threshold,"threshold_quantile":.8,
            "split":"calib_from_official_train","trials":trials,"scene_ids":[r["scene_id"] for r in records]}
    write_json(out,config)
    return config


def collect(adapter,records,out,config,device,policy="keep",diagnostic=False,trusted=False,seed=0,round_id=0):
    """Labels see future only in frame_scores; causal features never see it.

    Both native update events are retained. Rebuild at step40 is necessarily a
    no-op because step30 has no earlier context. Step50 is the first nontrivial
    rebuild. Logs expose event_step so this is never hidden by aggregation.
    """
    actions=DIAGNOSTICS if diagnostic else ACTIONS
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with out.open("w") as log:
        for record in tqdm(records,desc=f"branches round{round_id}"):
            seq=load_sequence(record,device,trusted)
            _,states,_=rollout(adapter,seq,policy,config["sigma"],config["threshold"],config["tau"],seed)
            _,reference,_=rollout(adapter,seq,"keep")
            for i in (1,2):
                ev=evidence(seq[i],states[i-1])
                ref=evidence(seq[i],reference[i-1])
                branches={a:replay(adapter,seq,i,states[i-1],a,policy,config["sigma"],config["threshold"],config["tau"],seed) for a in actions}
                losses=[branches[a]["loss"] for a in ACTIONS]
                best=int(np.argmin(losses))
                gain=losses[0]-losses[best]
                # Explicit practical tie margin in ES units (meters).
                label=best if gain>.01 else 0
                row={"scene_id":record["scene_id"],"event_step":(30,40,50)[i],"observed_until":(30,40,50)[i]-1,
                     "actor_id":seq[i]["track_id"][0],"checkpoint_hash":adapter.checkpoint_hash,
                     "controller_round":round_id,"seed":seed,"features":ev["features"].cpu().tolist(),
                     "feature_schema":FEATURE_SCHEMA,"feature_names":list(FEATURES),
                     "gain_target_metric":"final.energy_score",
                     "label":label,"weight":float(1+min(max(gain,0),5)),"tie":bool(gain<=.01),
                     "reference_residual":float(ref["features"][4]),
                     "group":"high_conflict" if float(ref["features"][4])>config["threshold"] else "ordinary",
                     "evidence_count":ev["valid_count"],"source":"predicted_trajectory_bank",
                     "branches":branches,"status":"ok"}
                rows.append(row)
                log.write(json.dumps(row,allow_nan=False)+"\n")
                log.flush()
    return rows


def development_rollout(adapter, records, config, device, policy, trusted=False, seed=0):
    """Closed-loop selection on official-train development scenes only."""
    rows = []
    for record in tqdm(records, desc="development policy rollout"):
        seq = load_sequence(record, device, trusted)
        # Group assignment must stay fixed across all candidates.
        _, reference, _ = rollout(adapter, seq, "keep")
        group = "high_conflict" if float(evidence(seq[1], reference[0])["features"][4]) > config["threshold"] else "ordinary"
        outs, _, edits = rollout(adapter, seq, policy, config["sigma"], config["threshold"], config["tau"], seed)
        rows.append({"scene_id": record["scene_id"], "group": group, "edits": edits,
                     "energy_score": frame_scores(outs[-1], seq[-1])["energy_score"]})
    return rows


def evaluate(adapter,records,out,config,device,controller,trusted=False,seed=0):
    policies={name:name for name in (*DIAGNOSTICS,"threshold")}
    policies["controller"]=controller
    out=Path(out)
    rows=[]
    with out.open("w") as log:
        for record in tqdm(records,desc="held-out deployed policies"):
            seq=load_sequence(record,device,trusted)
            _,reference,_=rollout(adapter,seq,"keep")
            groups={i:("high_conflict" if float(evidence(seq[i],reference[i-1])["features"][4])>config["threshold"] else "ordinary") for i in (1,2)}
            for name,policy in policies.items():
                start=clock(device)
                calls=adapter.forward_calls
                outs,_,edits=rollout(adapter,seq,policy,config["sigma"],config["threshold"],config["tau"],seed)
                row={"scene_id":record["scene_id"],"policy":name,"edits":edits,
                     "seconds":clock(device)-start,"forward_calls":adapter.forward_calls-calls,
                     "final":frame_scores(outs[-1],seq[-1]),
                     "common_curve":[dict(frame_scores(outs[i],seq[i],True),event_step=(30,40,50)[i],group=groups[1]) for i in (1,2)],
                     "status":"ok"}
                rows.append(row)
                log.write(json.dumps(row,allow_nan=False)+"\n")
                log.flush()
    return rows
