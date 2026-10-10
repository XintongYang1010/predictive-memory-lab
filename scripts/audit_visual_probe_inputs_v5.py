#!/usr/bin/env python3
"""Offline input controls for the frozen V5 feature probe; never alter its gate."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pmr.visual_memory.common import atomic_json,digest
from pmr.visual_memory.data import episode
from pmr.visual_memory.features import load_records


DIMENSIONS=("A_x_t7","A_y_t7","B_x_t7","B_y_t7","A_dx_6_7","A_dy_6_7","B_dx_6_7","B_dy_6_7")
REPRESENTATIONS=("raw","original_pooled","raw_repooled_bf16","raw_repooled_fp32")
CONTROLS={"concat_t6_t7":(1.,1.),"t6_only":(1.,0.),"t7_only":(0.,1.),
          "repeat_t6":(2.,0.),"repeat_t7":(0.,2.)}


def pool_raw(raw,grid=4,bf16=True):
    """Same spatial layout as Encoder.__call__, but cached raw is already BF16."""
    side=int(raw.shape[1]**.5)
    if side*side!=raw.shape[1] or side%grid:
        raise ValueError("Raw cue must be a square spatial grid divisible by pooling grid")
    z=raw.float().transpose(1,2).reshape(len(raw),-1,side,side)
    pooled=F.adaptive_avg_pool2d(z,grid).flatten(2).transpose(1,2)
    return pooled.bfloat16() if bf16 else pooled


def gram_components(train,dev,extractor,chunk=8192):
    """Accumulate centered linear kernels without a huge concatenated FP64 tensor."""
    a=torch.stack([extractor(row).flatten().float() for row in train])
    b=torch.stack([extractor(row).flatten().float() for row in dev])
    if a.shape[1]!=b.shape[1]:raise ValueError("Train/Dev feature widths differ")
    gram=torch.zeros(len(a),len(a),dtype=torch.float64)
    cross=torch.zeros(len(b),len(a),dtype=torch.float64)
    for start in range(0,a.shape[1],chunk):
        x=a[:,start:start+chunk].double();z=b[:,start:start+chunk].double()
        mean=x.mean(0);x=x-mean;z=z-mean
        gram+=x@x.T;cross+=z@x.T
    return gram,cross


def ridge_prediction(gram,cross,targets,alpha=.01):
    scale=gram.diag().mean().clamp_min(1e-12)
    mean=targets.mean(0)
    # Match the frozen probe's FP32 alpha*eye term before promotion to FP64.
    coef=torch.linalg.solve(gram/scale+alpha*torch.eye(len(gram)),targets-mean)
    return (cross/scale)@coef+mean


def summarize(pred,target,train_mean):
    error=(pred-target).square().mean(0)
    baseline=(target-train_mean).square().mean(0).clamp_min(1e-12)
    r2=1-error/baseline
    return dict(r2_per_dimension=r2.tolist(),position_r2=float(r2[:4].mean()),
                displacement_r2=float(r2[4:].mean()),mse=float(error.mean()))


def frame(row,representation,index):
    if representation=="raw":return row["raw_cue"][index]
    if representation=="original_pooled":return row["features"][6+index]
    return pool_raw(row["raw_cue"],bf16=representation.endswith("bf16"))[index]


def pooling_comparison(rows):
    total=equal=0;se=absolute=maximum=stored_energy=0.
    for row in rows:
        original=row["features"][6:8].float()
        rebuilt=pool_raw(row["raw_cue"]).float()
        error=rebuilt-original
        total+=error.numel();equal+=int((error==0).sum())
        se+=float(error.square().sum());absolute+=float(error.abs().sum())
        maximum=max(maximum,float(error.abs().max()));stored_energy+=float(original.square().sum())
    return dict(elements=total,equal_fraction=equal/total,mse=se/total,mean_absolute_error=absolute/total,
                max_absolute_error=maximum,relative_rmse=(se/max(stored_energy,1e-30))**.5)


def generator_controls(train,dev):
    """Diagnostic oracle inputs only: never supplied to a scientific actor or gate."""
    targets={};pixels={};checks={}
    for split,rows in (("train",train),("dev",dev)):
        positions=[];labels=[];pair_rgb={};changed=0
        for row in rows:
            meta=row["metadata"];pair=int(meta["pair_id"].rsplit("-",1)[1])
            ep=episode(split,pair,meta["variant"])
            torch.testing.assert_close(row["probe_state"],torch.from_numpy(ep.probe_state),rtol=0,atol=0)
            if ep.times[6:8].tolist()!=[6.,7.]:raise ValueError("Cue timestamps changed")
            labels.append(ep.probe_state)
            centers=[]
            for rgb in ep.rgb[6:8]:
                center=[]
                for red,blue in ((235,60),(60,235)):
                    yy,xx=np.nonzero((rgb[:,:,0]==red)&(rgb[:,:,2]==blue))
                    if not len(xx):raise ValueError("Fixed-color centroid oracle found no visible target")
                    center.extend((float(xx.mean()/224),float(yy.mean()/224)))
                centers.append(center)
            positions.append(centers)
            pair_rgb.setdefault(pair,[]).append(ep.rgb[-4:])
            changed+=int(np.any(ep.rgb[6]!=ep.rgb[7]))
        checks[split]=dict(episodes=len(rows),independent_pairs=len(pair_rgb),
                          cue_timestamps=[6,7],visible_two_frame_changes=changed,
                          paired_recent_rgb_equal=all(len(v)==2 and np.array_equal(v[0],v[1]) for v in pair_rgb.values()))
        targets[split]=torch.from_numpy(np.stack(labels)).double()
        pixels[split]=torch.tensor(positions,dtype=torch.float64)
    ym=targets["train"].mean(0)
    oracle=torch.cat((pixels["dev"][:,1],pixels["dev"][:,1]-pixels["dev"][:,0]),1)
    results={"rgb_two_frame_centroid_oracle":summarize(oracle,targets["dev"],ym)}
    for kind,degree in (("privileged_true_t7_position",1),("privileged_true_t7_position",3),("rgb_t7_centroid",1)):
        bases={split:(targets[split][:,:4] if kind.startswith("privileged") else pixels[split][:,1]) for split in targets}
        a=torch.cat([bases["train"]**k for k in range(1,degree+1)],1)
        b=torch.cat([bases["dev"]**k for k in range(1,degree+1)],1)
        mean=a.mean(0);a=a-mean;b=b-mean
        pred=ridge_prediction(a@a.T,b@a.T,targets["train"])
        results[f"{kind}_degree_{degree}"]=summarize(pred,targets["dev"],ym)
    return dict(split_checks=checks,metrics=results,
                interpretation="Non-temporal controls measure generator priors. True positions are privileged diagnostic inputs. RGB centroids use the known renderer color rule. None enter the actor or replace the original feature gate.")


def write_csv(path,rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit(cache,gate_path,run_manifest_path,out,threads=4):
    cache,gate_path,run_manifest_path,out=map(Path,(cache,gate_path,run_manifest_path,out))
    if out.exists():raise FileExistsError("Choose a fresh diagnostic output directory")
    torch.set_num_threads(threads)
    started=time.monotonic()
    gate=json.loads(gate_path.read_text());run_manifest=json.loads(run_manifest_path.read_text())
    train,dev=load_records(cache,"train"),load_records(cache,"dev")
    for split,rows in (("train",train),("dev",dev)):
        if len(rows)!=128 or any(row["raw_cue"] is None or row["times"][6:8].tolist()!=[6.,7.] for row in rows):
            raise ValueError("Expected the unchanged complete 128-episode Train/Dev probe caches")
    y=torch.stack([r["probe_state"] for r in train]).double()
    target=torch.stack([r["probe_state"] for r in dev]).double();ym=y.mean(0)
    results={};predictions={};table=[];equivalences={}
    for representation in REPRESENTATIONS:
        components=[gram_components(train,dev,lambda row,i=i:frame(row,representation,i)) for i in (0,1)]
        results[representation]={};predictions[representation]={}
        for control,weights in CONTROLS.items():
            gram=sum(weight*part[0] for weight,part in zip(weights,components))
            cross=sum(weight*part[1] for weight,part in zip(weights,components))
            prediction=ridge_prediction(gram,cross,y)
            metric=summarize(prediction,target,ym)
            results[representation][control]=metric;predictions[representation][control]=prediction
            table.append(dict(representation=representation,control=control,alpha=.01,
                              position_r2=metric["position_r2"],displacement_r2=metric["displacement_r2"],
                              **{name:value for name,value in zip(DIMENSIONS,metric["r2_per_dimension"])}))
        equivalences[representation]={f"t{i+6}_repeat_max_abs_difference":float((predictions[representation][f"t{i+6}_only"]-predictions[representation][f"repeat_t{i+6}"]).abs().max()) for i in (0,1)}
    reproduction={}
    for mode,representation in (("raw","raw"),("pooled","original_pooled")):
        actual=results[representation]["concat_t6_t7"]
        reproduction[mode]=dict(matches=bool(np.allclose(actual["r2_per_dimension"],gate[mode]["r2_per_dimension"],rtol=1e-8,atol=1e-10)),
                               max_abs_r2_difference=float(np.max(np.abs(np.asarray(actual["r2_per_dimension"])-gate[mode]["r2_per_dimension"]))))
    if not all(x["matches"] for x in reproduction.values()):
        raise ValueError("Chunked probe does not reproduce original gate: "+json.dumps(reproduction))
    comparisons={split:pooling_comparison(rows) for split,rows in (("train",train),("dev",dev))}
    oracles=generator_controls(train,dev)
    out.mkdir(parents=True)
    write_csv(out/"input_controls.csv",table)
    per_case=[]
    for representation,controls in predictions.items():
        for control,prediction in controls.items():
            for index,row in enumerate(dev):
                for dimension,name in enumerate(DIMENSIONS):
                    per_case.append(dict(representation=representation,control=control,episode_id=row["metadata"]["episode_id"],
                                         pair_id=row["metadata"]["pair_id"],dimension=name,target=float(target[index,dimension]),
                                         prediction=float(prediction[index,dimension])))
    write_csv(out/"dev_predictions.csv",per_case)
    atomic_json(out/"generator_controls.json",oracles)
    summary=dict(schema="v5-input-audit-1",original_gate_reproduction=reproduction,metrics=results,
                 repeat_controls=equivalences,pooling_numeric_comparison=comparisons,
                 parameters=dict(alpha=.01,threads=threads,chunk_columns=8192,train_episodes=128,dev_episodes=128,
                                 independent_pairs_per_split=64,mean_fitting="Train only",kernel_normalization="Train centered mean squared feature norm",
                                 target="t7 A/B x/y position; observed displacement position(7)-position(6)",test_opened=False),
                 provenance=dict(script_sha256=digest(__file__),original_gate_sha256=digest(gate_path),
                                 cached_source=run_manifest["configuration"]["source"],
                                 cache_manifest_sha256={split:digest(cache/split/"manifest.json") for split in ("train","dev")},
                                 cache_encoder={split:json.loads((cache/split/"manifest.json").read_text())["configuration"]["encoder"] for split in ("train","dev")}),
                 limitations=["Exploratory diagnostics after the failed Dev gate; no CV winner or new gate selected.",
                              "Cached raw features are already BF16: repooling changes quantization order as well as removing the separate encoder batch.",
                              "Positive displacement R2 does not prove temporal dependence; inspect the single-frame controls.",
                              "Different output columns of multi-output ridge solve independently. Position labels do not dominate a shared loss."],
                 elapsed_seconds=time.monotonic()-started)
    atomic_json(out/"input_audit.json",summary)
    lines=["# V5 probe input audit","","The original frozen gate is reproduced from the existing Train/Dev cache. No Test data or new GPU inference is used; all controls keep alpha=0.01.","",
           "| Representation | Input | Position R2 | Displacement R2 |","|---|---|---:|---:|"]
    for row in table:lines.append(f"| {row['representation']} | {row['control']} | {row['position_r2']:.6f} | {row['displacement_r2']:.6f} |")
    lines.extend(["","`raw_repooled_bf16` spatially pools the cached BF16 raw tokens then rounds to BF16. `raw_repooled_fp32` omits that last rounding. The original pooled tokens were pooled before their storage cast and came from a separate encoder batch; these diagnostics cannot uniquely attribute differences to batch size.","",
                  "Repeating a single frame and using it once give the same trace-normalized ridge kernel; the numerical equivalence is recorded. The two-frame input contains both frames, but its score must be compared with single-frame controls before claiming temporal dependence.","",
                  "## Renderer and non-temporal controls","","| Control | Position R2 | Displacement R2 |","|---|---:|---:|"])
    for name,metric in oracles["metrics"].items():lines.append(f"| {name} | {metric['position_r2']:.6f} | {metric['displacement_r2']:.6f} |")
    lines.extend(["",oracles["interpretation"],"","These are exploratory controls, not replacement acceptance criteria. All original stage status and gate files remain unchanged.","",
                  "[Numerical results and provenance](input_audit.json) · [Controls CSV](input_controls.csv) · [Per-episode predictions](dev_predictions.csv) · [Generator controls](generator_controls.json)",""])
    (out/"README.md").write_text("\n".join(lines))
    return dict(out=str(out),original_gate_reproduced=True,controls=len(table),elapsed_seconds=summary["elapsed_seconds"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",required=True,type=Path)
    parser.add_argument("--gate",required=True,type=Path)
    parser.add_argument("--run-manifest",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args()
    if args.threads<1:parser.error("threads must be positive")
    print(json.dumps(audit(args.cache,args.gate,args.run_manifest,args.out,args.threads),indent=2))
