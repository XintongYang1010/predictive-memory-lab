"""Aggregate V6 outputs into metrics.csv, summary.json, hypotheses.json and REPORT.md.

Decisions follow prereg.py: a hypothesis passes only if every training seed
passes; intervals are per-seed paired episode bootstraps (never pooled
across seeds).  Missing inputs are reported as "missing", never as pass/fail.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import prereg
from .metrics import auroc, best_f1_threshold, f1_at, bootstrap, stat_er, stat_nhe

KEYS = ("er", "er_all", "nhe", "hidden_rmse", "visible_rmse", "reveal_err_pred", "reveal_err_unexp",
        "post3_pred", "post3_unexp", "never_recovered_pred", "never_recovered_unexp",
        "history_pair_error", "pred_separation", "true_separation", "mean_displacement",
        "relative_to_correction", "lemma4_slope", "memory_gain", "mra", "bin1_nhe", "bin9_nhe",
        "bin33_nhe", "bin65_nhe", "bin193_nhe")


def collect(root):
    rows = []
    for spec_f in Path(root).glob("**/job_spec.json"):
        spec = json.loads(spec_f.read_text())
        if spec.get("type") != "eval":
            continue
        d = spec_f.parent
        stage = spec["id"].split("/")[0]
        if spec.get("reference"):
            variant, world, seed = spec["reference"], spec["id"].split("/")[3], None
        else:
            parts = spec["id"].split("/")
            variant = spec.get("as_variant") or parts[2]
            if spec.get("as_variant"):
                variant = f"{spec['as_variant']}@lr{spec.get('ttt_lr')}"
            world, seed = parts[3], spec.get("training_seed")
        for c in spec["conditions"]:
            f = d / f"{c['name']}.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text())
            row = {"stage": stage, "variant": variant, "train_world": world, "seed": seed,
                   "condition": c["name"], "split": spec.get("split", "dev"), "dir": str(d)}
            row.update({k: s.get(k) for k in KEYS})
            rows.append(row)
    return rows


def _episodes(row):
    f = Path(row["dir"]) / f"{row['condition']}_episodes.jsonl.gz"
    if not f.exists():
        return None
    with gzip.open(f, "rt") as fh:
        return [json.loads(x) for x in fh]


def _events(row):
    f = Path(row["dir"]) / f"{row['condition']}_events.npz"
    return dict(np.load(f)) if f.exists() else None


def _pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def _per_seed(rows, variant, cond, key, stage="S2", world="l0_ctrl"):
    return {r["seed"]: r.get(key) for r in _pick(rows, stage=stage, variant=variant, condition=cond,
                                                 train_world=world, split="dev")}


def _all(vals, pred):
    if not vals or any(v is None for v in vals.values()):
        return "missing"
    return "pass" if all(pred(v) for v in vals.values()) else "fail"


def hypotheses(rows, root):
    H = {}
    # H1 -----------------------------------------------------------------
    er = _per_seed(rows, "ours", "sched_T64", "er")
    nhe = _per_seed(rows, "ours", "sched_T64", "nhe")
    lo, hi = prereg.H1["ours_er_range"]
    a = _all(er, lambda v: lo <= v <= hi)
    b = _all(nhe, lambda v: v <= prereg.H1["ours_nhe_max"])
    h1 = {"ours_er": er, "ours_nhe": nhe,
          "ours": "missing" if "missing" in (a, b) else ("pass" if a == b == "pass" else "fail")}
    for v in prereg.H1["pausing_variants"]:
        e = _per_seed(rows, v, "sched_T64", "er")
        h1[f"{v}_er"] = e
        h1[f"{v}_pauses"] = _all(e, lambda x: x <= prereg.H1["pausing_er_max"])
        h1[f"{v}_er_9T"] = _per_seed(rows, v, "sched_T576", "er")
    H["H1"] = h1
    # H2 -----------------------------------------------------------------
    h2 = {"vo_er": _per_seed(rows, "vo_rotation", "sched_T64", "er")}
    h2["vo_fails"] = _all(h2["vo_er"], lambda v: v <= prereg.H2["vo_er_max"])
    for T in (64, 192):
        res = {}
        for s in (0, 1, 2):
            a = _pick(rows, stage="S2", variant="ours_frame", condition=f"sched_T{T}", train_world="l0_ctrl", seed=s)
            b = _pick(rows, stage="S2", variant="ours", condition=f"sched_T{T}", train_world="l0_ctrl", seed=s)
            if a and b:
                ea, eb = _episodes(a[0]), _episodes(b[0])
                if ea and eb:
                    res[s] = bootstrap(ea, stat_er, rows_b=eb, draws=500)
        h2[f"frame_minus_ours_er_T{T}"] = res
        h2[f"frame_equivalent_T{T}"] = ("missing" if len(res) < 3 else
                                        "pass" if all(r["ci95"][0] <= 0 <= r["ci95"][1] for r in res.values()) else "fail")
    for T, key, cmp in ((64, "tpa_nhe_ratio_at_T_max", lambda x, y: x <= y), (576, "tpa_nhe_ratio_at_9T_min", lambda x, y: x >= y)):
        t = _per_seed(rows, "tpa_readout", f"sched_T{T}", "nhe")
        o = _per_seed(rows, "ours", f"sched_T{T}", "nhe")
        ratio = {s: (t[s] / o[s] if t.get(s) and o.get(s) else None) for s in set(t) | set(o)}
        h2[f"tpa_over_ours_nhe_T{T}"] = ratio
        h2[f"tpa_check_T{T}"] = _all(ratio, lambda r, k=key, c=cmp: c(r, prereg.H2[k]))
    H["H2"] = h2
    # H3 -----------------------------------------------------------------
    kf = _pick(rows, stage="S2", variant="analytic_kf", condition="ctrl_T64", split="dev")
    kf3 = kf[0]["post3_unexp"] if kf else None
    p3 = _per_seed(rows, "ours", "ctrl_T64", "post3_unexp")
    nr = _per_seed(rows, "ours", "ctrl_T64", "never_recovered_unexp")
    sl = _per_seed(rows, "ours", "change_T64", "lemma4_slope")
    h3 = {"ours_post3_unexp": p3, "kf_post3_unexp": kf3, "ours_never_recovered_unexp": nr, "lemma4_slope": sl,
          "post3_vs_kf": "missing" if kf3 is None else _all(p3, lambda v: v <= prereg.H3["post3_vs_kf_max_ratio"] * kf3),
          "never_recovered": _all(nr, lambda v: v <= prereg.H3["never_recovered_max"]),
          "lemma4": _all(sl, lambda v: abs(v - 1) <= prereg.H3["lemma4_slope_tol"])}
    for rho in (0.0, 0.25, 0.5):
        rel = {r["seed"]: r.get("relative_to_correction") for r in
               _pick(rows, stage="S3", variant="ours", condition=f"change_rho{rho}", train_world="l0_ctrl")}
        h3[f"interference_rel_rho{rho}"] = rel
        h3[f"interference_ok_rho{rho}"] = _all(rel, lambda v: v <= prereg.H3["interference_rel_max_at_low_rho"])
    H["H3"] = h3
    # H4 -----------------------------------------------------------------
    h4 = {}
    for v in ("ours", "gru", "slot_known_id", "tpa_readout", "lact", "m2rnn"):
        a, b = _per_seed(rows, v, "sched_T64", "nhe"), _per_seed(rows, v, "sched_T576", "nhe")
        h4[f"{v}_nhe_9T_over_T"] = {s: (b[s] / a[s] if a.get(s) and b.get(s) else None) for s in set(a) | set(b)}
    h4["ours_extrapolates"] = _all(h4["ours_nhe_9T_over_T"], lambda r: r <= prereg.H4["ours_nhe_9T_over_T_max"])
    h4["recurrent_drift_reproduced"] = {v: _all(h4[f"{v}_nhe_9T_over_T"], lambda r: r >= prereg.H4["recurrent_nhe_9T_over_T_min"])
                                        for v in ("gru", "slot_known_id")}
    num = Path(root) / "S1" / "numerics" / "status.json"
    h4["numerics"] = json.loads(num.read_text()) if num.exists() else "missing"
    H["H4"] = h4
    # H5 -----------------------------------------------------------------
    h5 = {}
    an = Path(root) / "S3" / "analog" / "analog.json"
    analog = json.loads(an.read_text()) if an.exists() else None
    if analog:
        k1, k3 = analog["count_static"]["k1"]["lfp_mra"], analog["count_static"]["k3"]["lfp_mra"]
        h5["lfp_mra_k1_k3"] = (k1, k3)
        h5["lfp_fails_on_copies"] = "pass" if k1 > 0 and (k1 - k3) / k1 >= prereg.H5["lfp_mra_drop_k3_min_frac"] else "fail"
        h5["auroc_lfp"] = analog["events_T64"].get("auroc_lfp")
        h5["auroc_nn_novelty"] = analog["events_T64"].get("auroc_nn_novelty")
    for v in ("ours", "static_delta"):
        m1 = _per_seed(rows, v, "count_static_k1", "mra", stage="S3")
        m3 = _per_seed(rows, v, "count_static_k3", "mra", stage="S3")
        h5[f"{v}_mra_k1_k3"] = {s: (m1.get(s), m3.get(s)) for s in m1}
        h5[f"{v}_count_stable"] = _all({s: (m1[s] - m3[s]) if m3.get(s) is not None else None for s in m1},
                                       lambda d: d <= prereg.H5["memory_mra_drop_k3_max_points"])
        au, f1t = {}, {}
        for s in (0, 1, 2):
            r64 = _pick(rows, stage="S2", variant=v, condition="ctrl_T64", train_world="l0_ctrl", seed=s)
            ev = _events(r64[0]) if r64 else None
            if ev is None or ev["label"].sum() == 0:
                continue
            au[s] = auroc(ev["surprise"], ev["label"])
            thr, f64 = best_f1_threshold(ev["surprise"], ev["label"])
            drops = []
            for T in (192, 576):
                rT = _pick(rows, stage="S2", variant=v, condition=f"ctrl_T{T}", train_world="l0_ctrl", seed=s)
                evT = _events(rT[0]) if rT else None
                if evT is not None:
                    drops.append(f64 - f1_at(evT["surprise"], evT["label"], thr))
            f1t[s] = max(drops) if drops else None
        h5[f"{v}_auroc"] = au
        h5[f"{v}_f1_transfer_drop"] = f1t
    h5["ours_auroc_ok"] = _all(h5.get("ours_auroc", {}), lambda a: a >= prereg.H5["auroc_ours_min"])
    h5["static_auroc_low"] = _all(h5.get("static_delta_auroc", {}), lambda a: a <= prereg.H5["auroc_static_max"])
    h5["ours_f1_transfer"] = _all(h5.get("ours_f1_transfer_drop", {}), lambda d: d <= prereg.H5["f1_transfer_drop_max"])
    H["H5"] = h5
    # H6 -----------------------------------------------------------------
    h6 = {}
    base = _per_seed(rows, "ours", "drift_T192", "hidden_rmse", stage="S4", world="l0_drift")
    amort = _per_seed(rows, "amortized_dL", "drift_T192", "hidden_rmse", stage="S4", world="l0_drift")
    best_lr, best_val = None, math.inf
    for lr in prereg.H6["ttt_lrs"]:
        v = _per_seed(rows, f"ours_ttt@lr{lr}", "drift_T64", "hidden_rmse", stage="S4", world="l0_drift")
        if v and all(x is not None for x in v.values()) and np.mean(list(v.values())) < best_val:
            best_lr, best_val = lr, float(np.mean(list(v.values())))
    h6["selected_ttt_lr_on_drift_T64"] = best_lr
    if best_lr is not None:
        ttt = _per_seed(rows, f"ours_ttt@lr{best_lr}", "drift_T192", "hidden_rmse", stage="S4", world="l0_drift")
        h6["reduction_vs_ours"] = {s: (1 - ttt[s] / base[s]) if base.get(s) and ttt.get(s) else None for s in ttt}
        h6["ttt_effective"] = _all(h6["reduction_vs_ours"], lambda r: r >= prereg.H6["hidden_error_reduction_min"])
        h6["ttt_vs_amortized"] = {s: (ttt.get(s), amort.get(s)) for s in ttt}
    H["H6"] = h6
    # L1 -----------------------------------------------------------------
    hl = {}
    for v in ("ours", "static_delta", "gdn_neg", "lact", "tpa_readout"):
        hl[f"{v}_er_l1_dev"] = _per_seed(rows, v, "l1_dev", "er", stage="S5", world="l1_train")
        hl[f"{v}_nhe_l1_dev"] = _per_seed(rows, v, "l1_dev", "nhe", stage="S5", world="l1_train")
    o, s_ = hl["ours_er_l1_dev"], hl["static_delta_er_l1_dev"]
    hl["direction_ours_gt_static"] = _all({k: (o[k] - s_[k]) if o.get(k) is not None and s_.get(k) is not None else None
                                           for k in o}, lambda d: d > 0)
    H["L1"] = hl
    # H7 -----------------------------------------------------------------
    h7 = {}
    for v in ("ours", "static_delta", "gdn_neg", "tpa_readout", "gru", "fifo"):
        h7[f"{v}_er"] = _per_seed(rows, v, "nusc_dev", "er", stage="S7", world="real_nuscenes_train")
        h7[f"{v}_rmse"] = _per_seed(rows, v, "nusc_dev", "hidden_rmse", stage="S7", world="real_nuscenes_train")
    refs = {r["variant"]: r for r in _pick(rows, stage="S7", condition="nusc_dev")
            if r["variant"] in ("pause", "cv_kf", "interp_oracle")}
    if all(k in refs for k in ("pause", "cv_kf", "interp_oracle")):
        p, kf_, it = refs["pause"]["hidden_rmse"], refs["cv_kf"]["hidden_rmse"], refs["interp_oracle"]["hidden_rmse"]
        h7["refs_rmse"] = {"pause": p, "cv_kf": kf_, "interp_oracle": it}
        h7["ours_gap_closure"] = {s: ((p - v) / (p - it) if v is not None and p != it else None)
                                  for s, v in h7["ours_rmse"].items()}
        h7["ours_vs_cv_kf"] = _all(h7["ours_rmse"], lambda v: v <= prereg.H7["ade_vs_cv_kf_max_ratio"] * kf_)
    lo, hi = prereg.H7["ours_er_range"]
    h7["ours_er_ok"] = _all(h7["ours_er"], lambda v: lo <= v <= hi)
    h7["static_pauses"] = _all(h7["static_delta_er"], lambda v: v <= prereg.H7["static_er_max"])
    H["H7"] = h7
    return H


def write_report(root):
    root = Path(root)
    rows = collect(root)
    cols = ["stage", "variant", "train_world", "seed", "condition", "split"] + list(KEYS)
    with open(root / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["stage"], r["condition"], r["variant"], str(r["seed"]))):
            w.writerow(r)
    H = hypotheses(rows, root)
    (root / "hypotheses.json").write_text(json.dumps(H, indent=2, default=lambda x: x if not isinstance(x, float) else float(x)))
    ledger = root / "budget_ledger.json"
    lines = ["# V6 predictive-memory report", "",
             f"Rows: {len(rows)}. Pre-registration status in code: `{prereg.STATUS}`.", "",
             "Budget ledger: " + (ledger.read_text() if ledger.exists() else "missing"), "",
             "## Hypothesis decisions (per prereg.py; pass requires every seed)", ""]
    for h, d in H.items():
        flags = {k: v for k, v in d.items() if isinstance(v, str) and v in ("pass", "fail", "missing")}
        lines.append(f"- **{h}**: " + ", ".join(f"{k}={v}" for k, v in flags.items()))
    lines += ["", "## Key table (dev split, mean over seeds)", "",
              "| stage | condition | variant | ER | NHE | hidden RMSE | post3 unexp | memory gain | MRA |",
              "|---|---|---|---|---|---|---|---|---|"]
    agg = defaultdict(list)
    for r in rows:
        if r["split"] == "dev":
            agg[(r["stage"], r["condition"], r["variant"])].append(r)
    fmt = lambda xs: "" if not [x for x in xs if x is not None] else f"{np.mean([x for x in xs if x is not None]):.3f}"
    for (st, c, v), rs in sorted(agg.items()):
        lines.append(f"| {st} | {c} | {v} | {fmt([r['er'] for r in rs])} | {fmt([r['nhe'] for r in rs])} | "
                     f"{fmt([r['hidden_rmse'] for r in rs])} | {fmt([r['post3_unexp'] for r in rs])} | "
                     f"{fmt([r['memory_gain'] for r in rs])} | {fmt([r['mra'] for r in rs])} |")
    lines += ["", "Full numbers: `metrics.csv`, `hypotheses.json`; per-episode rows next to each condition.",
              "Do not select variants or thresholds on these dev numbers and then reuse them as confirmation."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    _spectra(root)
    return H


def _spectra(root):
    """Learned generator / gain heatmaps for every 'ours' checkpoint."""
    try:
        import torch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .train import load_model
    except Exception:
        return
    figs = Path(root) / "figures"
    figs.mkdir(exist_ok=True)
    for ck in Path(root).glob("S*/train/ours/*/s*/best.pt"):
        model, man = load_model(ck, "cpu")
        L = model._structured_dyn().detach().numpy()
        G = np.concatenate([model.K_p.detach().numpy(), model.K_q.detach().numpy(), model.K_z.detach().numpy()])
        eig = np.linalg.eigvals(L)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        ax[0].imshow(L, cmap="RdBu_r", vmin=-np.abs(L).max(), vmax=np.abs(L).max()); ax[0].set_title("L_dyn")
        ax[1].imshow(G, cmap="RdBu_r", vmin=-np.abs(G).max(), vmax=np.abs(G).max()); ax[1].set_title("gain rows p,q,z")
        ax[2].scatter(eig.real, eig.imag); ax[2].axvline(0, color="k", lw=.5); ax[2].set_title("spectrum of L")
        tag = "_".join(ck.parts[-4:-1])
        fig.tight_layout()
        fig.savefig(figs / f"ours_{tag}.png", dpi=120)
        plt.close(fig)
