from collections import Counter
from pathlib import Path
import csv
import json
import math
import numpy as np
from .common import write_json,sha256


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


def audit_selections(root):
    """Recompute every gate from saved complete-scene rollouts, not summaries."""
    root=Path(root)
    results=[]
    def same(a,b):
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError("Selection audit: keys differ")
            for k in a:same(a[k],b[k])
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError("Selection audit: list lengths differ")
            for x,y in zip(a,b):same(x,y)
        elif isinstance(a,(int,float)):
            if not math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10):raise ValueError(f"Selection audit mismatch: {a} != {b}")
        elif a!=b:raise ValueError(f"Selection audit mismatch: {a} != {b}")
    for path in sorted(root.rglob("*.selection.json")):
        selection=json.loads(path.read_text())
        if selection.get('schema')!=3:
            raise ValueError("Selection audit needs V3 per-scene candidate records")
        def artifact(relative):
            target=(path.parent/relative).resolve()
            if not target.is_relative_to(path.parent.resolve()):raise ValueError("Candidate artifact outside run")
            return target
        def indexed(relative):
            rows=read_jsonl(artifact(relative))
            if len(rows)!=selection['n_scenes'] or len({r['scene_id'] for r in rows})!=len(rows):
                raise ValueError("Selection audit: missing or duplicate scenes")
            if not all(math.isfinite(r['energy_score']) for r in rows):raise ValueError("Nonfinite candidate ES")
            return {r['scene_id']:r for r in rows}
        baseline=indexed(selection['baseline_rollout_file'])
        ids=sorted(baseline)
        same(float(np.mean([baseline[s]['energy_score'] for s in ids])),selection['keep_ES'])
        eligible=[]
        for candidate in selection['candidates']:
            trial=indexed(candidate['rollout_file'])
            if set(trial)!=set(baseline):raise ValueError("Selection audit: foreign scenes")
            if sha256(artifact(candidate['checkpoint_file']))!=candidate['checkpoint_sha256']:
                raise ValueError("Candidate checkpoint hash differs")
            gains=[baseline[s]['energy_score']-trial[s]['energy_score'] for s in ids]
            for s,g in zip(ids,gains):
                same(trial[s]['group'],baseline[s]['group'])
                same(trial[s]['keep_energy_score'],baseline[s]['energy_score'])
                same(trial[s]['paired_gain'],g)
            stats=paired_ci(gains)
            ordinary=[g for s,g in zip(ids,gains) if baseline[s]['group']=='ordinary']
            ordinary_gain=float(np.mean(ordinary)) if ordinary else None
            valid=stats['ci95'][0]>0 and (ordinary_gain is None or ordinary_gain>=0)
            same(stats,candidate['paired_ES_improvement'])
            same(ordinary_gain,candidate['ordinary_mean_gain'])
            same(valid,candidate['eligible'])
            same(float(np.mean([trial[s]['energy_score'] for s in ids])),candidate['mean_final_ES'])
            same(dict(Counter(a for s in ids for a in trial[s]['edits'][1:])),candidate['action_counts'])
            if valid:eligible.append(candidate)
        expected=max(eligible,key=lambda c:c['paired_ES_improvement']['mean']) if eligible else None
        same(expected,selection['selected'])
        same('gain_controller' if eligible else 'keep_fallback',selection['mode'])
        results.append({'selection':str(path.relative_to(root)),'candidates':len(selection['candidates']),'verified':True})
    if not results:raise ValueError("No selection records found")
    return {'passed':True,'selections':results,'scope':'Saved Dev rollouts and gate arithmetic; no predictor rerun or significance claim'}


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
        metrics["probe_forward_calls"]=sum(r.get("probe_forward_calls",0) for r in subset)
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
    summary['diagnostic_metrics']={'diagnostic':'common-window average ES gain (legacy)',
                                   'diagnostic_final':'final 60-frame ES gain; hindsight is not deployed performance'}
    summary['diagnostic_final']={}
    for step in (40,50):
        events=[r for r in diagnostic if r['event_step']==step]
        if not events:continue
        summary['diagnostic_final'][str(step)]={a:paired_ci([
            r['branches']['keep']['final']['energy_score']-r['branches'][a]['final']['energy_score'] for r in events])
            for a in sorted(events[0]['branches'])}
        summary['diagnostic_final'][str(step)]['hindsight_core4']=paired_ci([
            r['branches']['keep']['final']['energy_score']-min(r['branches'][a]['final']['energy_score']
            for a in ('keep','reweight','rebuild','reset')) for r in events])
    summary["controller_selection"]=[json.loads(path.read_text()) for path in sorted(
        out.glob("round*.controller.selection.json"),key=lambda p:int(p.name.split('.')[0][5:]))]
    summary["ablation_selection"]={path.parent.name:json.loads(path.read_text())
                                     for path in sorted((out/"ablations").glob("*/round0.controller.selection.json"))}
    summary["ablation_comparisons"]={}
    for left,right in (("controller_base","controller_temporal"),("controller_temporal","controller")):
        left_rows={r['scene_id']:r for r in rows if r['policy']==left}
        right_rows={r['scene_id']:r for r in rows if r['policy']==right}
        if left_rows and right_rows:
            if set(left_rows)!=set(right_rows):raise ValueError("Unpaired ablation scenes")
            summary['ablation_comparisons'][f'{right}_over_{left}']=paired_ci([
                left_rows[s]['final']['energy_score']-right_rows[s]['final']['energy_score'] for s in sorted(left_rows)])
    training_paths=sorted(out.glob("round*.controller.training.json"),key=lambda p:int(p.name.split('.')[0][5:]))
    training=[json.loads(path.read_text()) for path in training_paths]
    summary["controller_learning"]=[{
        "round":int(path.name.split('.')[0][5:]), "status":item.get("status"),
        "tiny_fit_status":item.get("tiny_fit_status"), "initial":item.get("initial"),
        "last_logged_checkpoint":item["history"][-1] if item["history"] else None,
        "dev_baselines":item.get("dev_baselines"),
    } for path,item in zip(training_paths,training)]
    write_json(out/"summary.json",summary)
    fields=["policy","energy_score","minADE6","minFDE6","b_minFDE6","b_minFDE6_upstream","miss_rate","mean_seconds_per_scene","forward_calls","probe_forward_calls"]
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
    lines.extend(["","All controller arms and keep_probe run two target-free influence probes per decision.",
                  "Keep is the native-cost baseline. Extra Rebuild forwards are included in the totals below.","",
                  "|Policy|Seconds/scene|Forwards/scene|Probe forwards/scene|","|---|---:|---:|---:|"])
    for policy,m in summary['policies'].items():
        n=m['paired_ES_improvement']['n_scenes']
        lines.append(f"|{policy}|{m['mean_seconds_per_scene']:.5f}|{m['forward_calls']/n:.2f}|{m['probe_forward_calls']/n:.2f}|")
    if summary['ablation_selection']:
        lines.extend(["","Feature ablations use identical Keep-generated Train/Dev branches and equal training/candidate budgets.",
                      "All arms are fixed on Dev before Eval; this table does not choose an Eval winner.","",
                      "|Feature set|Dev selection|","|---|---|"])
        for name,s in {**summary['ablation_selection'],'temporal_influence':summary['controller_selection'][-1]}.items():
            lines.append(f"|{name}|{s['mode']}|")
        for name,d in summary['ablation_comparisons'].items():
            lines.append(f"{name}: paired ES gain {d['mean']:.6g}, CI {d['ci95']} (exploratory).")
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
    if training and training[-1].get("dev_baselines"):
        fit=training[-1]
        last=fit["history"][-1]
        lines.extend(["", "## Learning diagnostics", "",
                      f"Tiny-set fit on Train only: **{fit['tiny_fit_status']}**.",
                      "The table describes recorded Dev single-edit branches. It does not measure complete deployed policies.",
                      "The logged controller below is the last training checkpoint, not necessarily the selected checkpoint.", "",
                      "|Predictor|Dev gain MSE ↓|Offline mean gain ↑|Intervention fraction|",
                      "|---|---:|---:|---:|"])
        metrics={**fit['dev_baselines'], 'last_logged_controller':last['dev_at_margin_001']}
        for name,m in metrics.items():
            lines.append(f"|{name}|{m['MSE']:.6g}|{m['offline_mean_gain']:.6g}|{m['intervention_fraction']:.3f}|")
        lines.extend(["", "Read `training_curves.png` and each `roundN.controller.diagnostics.json` alongside the rollout selection.",
                      "Tiny-set failure can reflect optimization or indistinguishable features with conflicting outcomes; it is not automatically an implementation bug.",
                      "A zero-weight layer has an undefined update/weight ratio (null). No universal ratio threshold is enforced."])
        lines.extend(["", "![Controller learning diagnostics](training_curves.png)"])
    lines.extend(["","Final metrics use the native 60-step forecast at step50. Curves compare the same absolute [50,100) target interval.",
                  "Group labels for policy curves are frozen at the reference model's step40 event.",
                  "Single-edit final-60-frame results are in summary.json / diagnostic_final; diagnostic retains legacy common-window gains. Rebuild at step40 is structurally a no-op.",
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
    if training and training[-1].get("initial"):
        learning_curves(training, out/"training_curves.png")
    for path in (out/"ablations").glob("*/round0.controller.training.json"):
        learning_curves([json.loads(path.read_text())],path.parent/"training_curves.png")
    return summary


def learning_curves(training, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for r,fit in enumerate(training):
        h=fit['history']
        steps=[v['step'] for v in h]
        for split in ('train','dev'):
            initial=fit['initial'][f'{split}_MSE']
            axes[0,0].plot([0]+steps,[initial]+[v[f'{split}_MSE'] for v in h],
                           marker='o',markersize=3,label=f'r{r} {split}')
        axes[0,1].plot(steps,[v['dev_at_margin_001']['offline_mean_gain'] for v in h],marker='o',markersize=3,label=f'r{r}')
        axes[1,0].plot(steps,[v['gradient_norm_before_clip'] for v in h],marker='o',markersize=3,label=f'r{r}')
        for name in ('net.0.weight','net.2.weight','net.4.weight'):
            axes[1,1].plot(steps,[v['parameter_updates'][name]['update_weight_ratio'] for v in h],
                           marker='o',markersize=3,label=f'r{r} {name}')
    for ax,title in zip(axes.flat,('Gain MSE (Train/Dev in the same units)',
                                 'Dev single-edit gain at margin 0.01 (not rollout)',
                                 'Gradient norm before clipping', 'Actual AdamW update / prior weight norm')):
        ax.set(title=title,xlabel='Training step')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    axes[0,0].set_yscale('symlog',linthresh=1e-10)
    axes[0,1].axhline(0.,color='grey',linestyle='--',linewidth=.8)
    fig.suptitle('Training diagnostics — not evidence of held-out improvement')
    fig.savefig(path,dpi=150)
    plt.close(fig)
