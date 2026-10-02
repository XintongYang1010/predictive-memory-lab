from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np
from .common import write_json


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def paired_ci(deltas,seed=0):
    # Input has one mean difference per complete scenario.
    a=np.asarray(deltas,dtype=float)
    if not len(a):
        return None
    gen=np.random.default_rng(seed)
    # Chunk draws to keep memory bounded on full validation sets.
    draws=[]
    for _ in range(20):
        draws.extend(gen.choice(a,(50,len(a)),replace=True).mean(-1).tolist())
    return {"n_scenes":len(a),"mean":float(a.mean()),"ci95":np.quantile(draws,[.025,.975]).tolist()}


def report(out,smoke=False):
    out=Path(out)
    rows=read_jsonl(out/"evaluation.jsonl")
    diagnostic=read_jsonl(out/"diagnostic.jsonl")
    baseline={r["scene_id"]:r for r in rows if r["policy"]=="keep"}
    summary={"smoke_only":smoke,"interpretation":"CPU/synthetic functional check; no research evidence" if smoke else "Exploratory frozen-backbone AV2 results; inspect baseline reproduction before claims",
             "policies":{},"diagnostic":{}}
    for policy in sorted({r["policy"] for r in rows}):
        subset=[r for r in rows if r["policy"]==policy]
        if {r["scene_id"] for r in subset}!=set(baseline):
            raise ValueError("Unpaired evaluation rows")
        if len(subset)!=len(baseline):
            raise ValueError("Duplicate evaluation rows")
        metrics={k:float(np.mean([r["final"][k] for r in subset])) for k in subset[0]["final"]}
        gains=[baseline[r["scene_id"]]["final"]["energy_score"]-r["final"]["energy_score"] for r in subset]
        metrics["paired_ES_improvement"]=paired_ci(gains)
        metrics["mean_seconds_per_scene"]=float(np.mean([r["seconds"] for r in subset]))
        metrics["forward_calls"]=sum(r["forward_calls"] for r in subset)
        metrics["actions_by_step"]={str(step):dict(Counter(r["edits"][i] for r in subset))
                                    for i,step in enumerate((30,40,50))}
        metrics["groups"]={}
        for group in ("ordinary","high_conflict"):
            group_rows=[r for r in subset if r["common_curve"][0]["group"]==group]
            metrics["groups"][group]={"n_scenes":len(group_rows),"paired_ES_improvement":paired_ci([
                baseline[r["scene_id"]]["final"]["energy_score"]-r["final"]["energy_score"] for r in group_rows])}
        summary["policies"][policy]=metrics
    for step in (40,50):
        for group in ("all","ordinary","high_conflict"):
            events=[r for r in diagnostic if r["event_step"]==step and (group=="all" or r["group"]==group)]
            stats={}
            for action in sorted({a for r in events for a in r["branches"]}):
                # At a fixed event_step there is one event per scene.
                stats[action]=paired_ci([r["branches"]["keep"]["loss"]-r["branches"][action]["loss"] for r in events])
            summary["diagnostic"][f"step{step}_{group}"]=stats
    summary["controller_selection"]=[json.loads(path.read_text()) for path in sorted(
        out.glob("round*.controller.selection.json"),key=lambda p:int(p.name.split('.')[0][5:]))]
    write_json(out/"summary.json",summary)
    fields=["policy","energy_score","minADE6","minFDE6","b_minFDE6","b_minFDE6_upstream","miss_rate","mean_seconds_per_scene"]
    with (out/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for policy,m in summary["policies"].items():
            writer.writerow({"policy":policy,**{k:m[k] for k in fields[1:]}})
    title="SMOKE CHECK — NOT RESEARCH RESULTS" if smoke else "Week-one predictive memory diagnosis"
    lines=[f"# {title}","",summary["interpretation"],"",
           "Positive paired ES improvement means lower error than Keep. CIs resample complete scenarios.","",
           "|Policy|ES ↓|minADE6 ↓|minFDE6 ↓|Paired ES improvement [95% CI]|",
           "|---|---:|---:|---:|---:|"]
    for policy,m in summary["policies"].items():
        d=m["paired_ES_improvement"]
        lines.append(f"|{policy}|{m['energy_score']:.4f}|{m['minADE6']:.4f}|{m['minFDE6']:.4f}|{d['mean']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]|")
    if summary["controller_selection"]:
        selection=summary["controller_selection"][-1]
        lines.extend(["",f"Controller selection: **{selection['mode']}** (development scenes only)."])
        if selection["mode"]=="keep_fallback":
            lines.append("No shortlisted controller passed the development gate. Deployed controller uses Keep; this is not a learned improvement.")
        else:
            choice=selection['selected']
            lines.append(f"Selected training step {choice['step']}, predicted ES gain margin {choice['gain_margin']}. Development selection is not confirmation evidence.")
        lines.extend(["","|Update|Keep|Reweight|Rebuild|Reset|","|---|---:|---:|---:|---:|"])
        for step,counts in summary['policies']['controller']['actions_by_step'].items():
            if step!='30':
                lines.append(f"|{step} frames|"+'|'.join(str(counts.get(a,0)) for a in ('keep','reweight','rebuild','reset'))+'|')
    lines.extend(["","Final metrics use the native 60-step forecast at step50. Curves compare the same absolute [50,100) target interval.",
                  "Group labels for policy curves are frozen at the reference model's step40 event.",
                  "Single-edit diagnostic results are in summary.json, split by event step. Rebuild at step40 is structurally a no-op.",
                  "b_minFDE6 selects the minimum-FDE mode then adds its Brier term; b_minFDE6_upstream minimizes the sum, matching the pinned repository.",
                  "No best-branch/oracle score is presented as deployed-controller performance."])
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,group in zip(axes,("ordinary","high_conflict")):
        for policy in ("keep","reweight","rebuild","reset","controller"):
            subset=[r for r in rows if r["policy"]==policy and r["common_curve"][0]["group"]==group]
            if subset:
                ax.plot([4,5],np.mean([[p["energy_score"] for p in r["common_curve"]] for r in subset],axis=0),marker="o",label=policy)
        ax.set(title=f"{group} (fixed step40 group)",xlabel="Observation update time (s)",ylabel="ES on common absolute target interval")
        if ax.lines:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,"No scenes in this group",ha="center",transform=ax.transAxes)
    fig.suptitle(title)
    fig.savefig(out/"update_curves.png",dpi=150)
    plt.close(fig)
    return summary
