"""Dev-only grouped reports. High errors are candidates, not causal explanations."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from .common import atomic_json
from .memory import VARIANTS,persistent_cost,matched_fifo_frames


def report(out):
    out=Path(out); grouped=[]; ledger=[]; costs=[]
    for variant in ["full","recent","fifo",*VARIANTS,"reader_only_C0"]:
        path=out/variant/"per_episode.jsonl"
        if not path.exists():continue
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        groups=defaultdict(list)
        for r in rows:groups[(r["family"],r["gap"],r["interference_writes"])].append(r)
        for (family,gap,writes),values in groups.items():
            keys=["mse","roi_mse","object_a_mse","object_b_mse","paired_difference_mse"]
            grouped.append(dict(variant=variant,family=family,gap=gap,interference_writes=writes,n=len(values),
                                **{k:float(np.mean([r[k] for r in values])) for k in keys}))
        for family in ("hold","interference","update"):
            family_rows=sorted([r for r in rows if r["family"]==family],key=lambda r:r["mse"],reverse=True)
            for r in family_rows[:20]:
                ledger.append(dict(variant=variant,episode_id=r["episode_id"],pair_id=r["pair_id"],
                                   failure_type="high_dev_error_candidate",family=family,gap=r["gap"],
                                   mse=r["mse"],minimal_prefix=None,changed_factor=None,
                                   intervention="M=0 and paired M/history swap logged in per_episode.jsonl",
                                   outcome="causal explanation not established",next_test="minimize prefix and confirm on fresh Dev pairs"))
        if variant in VARIANTS or variant=="reader_only_C0":
            costs.append(dict(variant=variant,**persistent_cost()))
        elif variant!="full":
            costs.append(dict(variant=variant,**persistent_cost(exact=0 if variant=="recent" else matched_fifo_frames(),memory=False)))
    for name,rows in [("metrics_by_family.csv",grouped),("state_costs.csv",costs)]:
        if rows:
            with open(out/name,"w") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with open(out/"failure_ledger.jsonl","w") as f:
        for row in ledger:f.write(json.dumps(row)+"\n")
    if grouped:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for ax,family in zip(axes,("hold","interference","update")):
            subset=[r for r in grouped if r["family"]==family]
            for variant in sorted({r["variant"] for r in subset}):
                # Hold gap varies at a fixed write count; interference and update
                # vary writes at gap=16. Never marginalize the confounding axis.
                curve=sorted([r for r in subset if r["variant"]==variant and
                              (r["interference_writes"]==8 if family=="hold" else r["gap"]==16)],
                             key=lambda r:r["gap"] if family=="hold" else r["interference_writes"])
                if curve:ax.plot([r["gap"] if family=="hold" else r["interference_writes"] for r in curve],
                                 [r["object_b_mse"] if family=="update" else r["mse"] for r in curve],marker="o",label=variant)
            ax.set_title(family);ax.set_xlabel("Time gap (writes=8)" if family=="hold" else "Interference observations (gap=16)")
            ax.set_ylabel("Unchanged B ROI latent MSE" if family=="update" else "Normalized latent MSE")
            ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/"pressure_curves.png",dpi=180);plt.close(fig)
    atomic_json(out/"report_status.json",dict(status="partial" if len({r["variant"] for r in grouped})<9 else "pilot_complete",
                evaluation_split="dev",test_evaluated=False,
                limitations=["controlled synthetic RGB only","two targets; capacity and similar-address sweeps pending",
                              "ROI latent errors are not physical trajectory errors","prefix minimization and fresh confirmation remain human-guided",
                              "sharing/selector experiments deliberately not started"]))
