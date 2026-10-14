"""S0 CPU gates, S1 numerics, S3 Cambrian-S analogue job."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import prereg
from .data import make_world, geometry_of, stable_seed
from .metrics import score_episodes, summarize, mra, auroc, best_f1_threshold, f1_at, tracking_masks
from .numpy_reference import ideal_params, rollout as np_rollout
from .references import run_reference


def _l0_truth_ok(spec, T=64, n=64):
    d = make_world(spec, n, T, stable_seed("s0", spec))
    step = d["pos_unwrapped"][:, 1:] - d["pos_unwrapped"][:, :-1]
    tele = d["change_step"][:, 1:] == 2
    return bool(np.allclose(step[~tele], d["vel"][:, :-1][~tele]))


def kf_gate():
    d = make_world("l0_sched", 512, 64, stable_seed("s0", "kf"))
    g = geometry_of(d)
    p = run_reference("analytic_kf", d, g, obs_noise=0.0, vel_scale=2.0)
    m = tracking_masks(d)
    sel = m["hidden"] & m["identifiable"] & ~m["changed"]
    err = np.sqrt((g.diff(p, d["pos"])[sel] ** 2).sum(-1).mean())
    return float(err)


def ideal_reference_check():
    d = make_world("l0_sched", 128, 64, stable_seed("s0", "ideal"))
    g = geometry_of(d)
    out = {}
    for tr in ("ours", "static"):
        p = ideal_params(transition=tr)
        pred = np_rollout(p, d["app"].astype(np.float64), g.normalize(d["obs"]), d["vis"].astype(float),
                          np.zeros(d["u"].shape), d["dt"], "torus")
        s = summarize(score_episodes(d, g.denormalize(pred), g))
        out[tr] = {"er": s["er"], "nhe": s["nhe"]}
    return out


def s0(args) -> dict:
    t0 = time.time()
    rep = {"stage": "S0", "gates": {}}
    r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/predictive_memory_v6", "-t", "."],
                       capture_output=True, text=True)
    rep["unit_tests"] = {"returncode": r.returncode, "tail": r.stderr[-3000:]}
    rep["gates"]["unit_tests_pass"] = r.returncode == 0
    rep["gates"]["l0_truth_consistent"] = all(_l0_truth_ok(w) for w in
                                              ("l0_main", "l0_sched", "l0_ctrl", "l0_window_ctrl", "l0_blackout", "l0_drift"))
    kf = kf_gate()
    rep["kf_hidden_rmse"] = kf
    rep["gates"]["kf_oracle"] = kf < prereg.GATES["kf_hidden_rmse_max"]
    rep["gates"]["mra_cases"] = all(abs(mra(p, g) - v) < 1e-12 for (p, g), v in prereg.GATES["mra_cases"].items())
    ideal = ideal_reference_check()
    rep["ideal_numpy_reference"] = ideal
    rep["gates"]["ideal_ours_exact"] = abs(ideal["ours"]["er"] - 1) < 1e-6 and ideal["ours"]["nhe"] < 1e-6
    rep["gates"]["ideal_static_pauses"] = abs(ideal["static"]["er"]) < 1e-6
    try:
        import torch  # noqa: F401
        rep["torch"] = __import__("torch").__version__
    except ImportError:
        rep["torch"] = None
        rep["gates"]["torch_available"] = False
    if getattr(args, "nuscenes_meta", None):
        from .worlds_real import prepare_nuscenes
        st = prepare_nuscenes(args.nuscenes_meta, args.cache)
        rep["nuscenes"] = st
        n_ev = st.get("event_tracks", 0)
        rep["gates"]["nuscenes_events_ge_200"] = n_ev >= prereg.H7["min_natural_events"]
    if getattr(args, "mot_roots", None):
        from .worlds_real import prepare_mot
        rep["mot"] = prepare_mot(args.mot_roots, args.cache)
    if getattr(args, "miniworld", None):
        from . import worlds_l1
        rep["l1_smoke_generation"] = worlds_l1.generate(args.miniworld, args.cache, "dev", n=4, procs=4)
        rep["l1_convert"] = worlds_l1.convert_split(args.cache, "dev")
        rep["l1_dynamics"] = worlds_l1.check_dynamics(args.cache, "dev")
        rep["l1_depth"] = worlds_l1.depth_check(args.cache, "dev", episodes=4)
        rep["gates"]["l1_fold_model"] = rep["l1_dynamics"]["fold_consistent_fraction"] > 0.99
        rep["gates"]["l1_depth_agreement"] = rep["l1_depth"]["agreement"] >= prereg.GATES["l1_depth_agreement_min"]
    rep["all_gates_pass"] = all(rep["gates"].values())
    rep["seconds"] = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "S0_report.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


# ------------------------------------------------------------- S1 numerics
def numerics_job(spec) -> dict:
    import torch
    from . import memories
    from .data import to_tbatch
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    model = memories.build("ours", n_objects=5, geom="torus")
    L = model._structured_dyn().detach().double()
    abscissa = float(torch.linalg.eigvals(L).real.max())
    d = make_world("l0_sched", 64, 576, stable_seed("s1", "numerics"))
    b32 = to_tbatch(d, "cpu", torch.float32)
    b64 = to_tbatch(d, "cpu", torch.float64)
    with torch.no_grad():
        p32 = model.float().rollout(b32)["pred"].double().numpy()
        p64 = model.double().rollout(b64)["pred"].numpy()
    g = geometry_of(d)
    dev = float(np.abs(g.diff(g.denormalize(p32), g.denormalize(p64))).max())
    res = {"status": "complete", "spectral_abscissa": abscissa, "fp32_vs_fp64_max_world_units": dev,
           "gates": {"spectral": abscissa <= prereg.GATES["spectral_abscissa_max"],
                     "fp64_replay": dev <= prereg.H4["fp64_replay_max_dev"]}}
    (out / "status.json").write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------- S3 Cambrian-S analogues
def analog_job(spec) -> dict:
    import torch
    from .lfp import train_lfp, lfp_surprise
    from .surprise import segment_count, true_count, nn_novelty, reveal_events, novelty_count
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"status": "complete"}
    small = bool(spec.get("small"))
    n_cal, n_cnt, n_ev = (8, 8, 16) if small else (256, 512, 1024)
    Ts = (16, 32, 48) if small else (64, 192, 576)
    log = open(out / "analog.jsonl", "a")
    wlog = lambda r: (log.write(json.dumps(r) + "\n"), log.flush())
    # ---- counting (VSC-Repeat analogue)
    for kind in ("static", "dynamic"):
        lfp = train_lfp(f"l0_tour_{kind}:copies=1", 50, spec.get("lfp_steps", 3000), device, log=wlog)
        cal = make_world(f"l0_tour_{kind}:copies=1", n_cal, 50, stable_seed("devselect", "count", kind))
        s_cal = lfp_surprise(lfp, cal, device)
        gt = true_count(cal["vis"])
        cands = np.unique(np.quantile(s_cal[:, 1:], np.linspace(0.5, 0.999, 60)))
        best = max(cands, key=lambda th: np.mean([mra(p, g) for p, g in zip(segment_count(s_cal, cal["vis"], th), gt)]))
        nn_thr = 0.5 * math.sqrt(2.0)
        res[f"count_{kind}"] = {"lfp_threshold": float(best)}
        for k in (1, 2, 3, 6):
            d = make_world(f"l0_tour_{kind}:copies={k}", n_cnt, 50 * k, stable_seed("dev", "count", kind, k))
            s = lfp_surprise(lfp, d, device)
            gt = true_count(d["vis"])
            lfp_c = segment_count(s, d["vis"], best)
            nn_c = novelty_count(nn_novelty(d, geometry_of(d), use_position=False, admit_thr=nn_thr), d["vis"], nn_thr)
            res[f"count_{kind}"][f"k{k}"] = {
                "lfp_mra": float(np.mean([mra(p, g) for p, g in zip(lfp_c, gt)])),
                "nn_novelty_mra": float(np.mean([mra(p, g) for p, g in zip(nn_c, gt)])),
                "true_mean": float(gt.mean()), "lfp_mean": float(lfp_c.mean()), "nn_mean": float(nn_c.mean())}
    # ---- reveal-event AUROC (expected vs unexpected change)
    lfp = train_lfp("l0_ctrl", Ts[0], spec.get("lfp_steps", 3000), device, log=wlog)
    thr = {}
    for T in Ts:
        d = make_world("l0_ctrl", n_ev if T != Ts[-1] else max(4, n_ev // 2), T, stable_seed("dev", "events", T))
        g = geometry_of(d)
        ev = reveal_events(d, {"lfp": lfp_surprise(lfp, d, device),
                               "nn_novelty": nn_novelty(d, g, use_position=True)})
        row = {"events": int(len(ev["label"])), "positives": int(ev["label"].sum())}
        for m in ("lfp", "nn_novelty"):
            sc = np.nan_to_num(ev[m], posinf=1e9)
            row[f"auroc_{m}"] = auroc(sc, ev["label"])
            if T == Ts[0]:
                thr[m], row[f"f1_{m}"] = best_f1_threshold(sc, ev["label"])
            else:
                row[f"f1_{m}"] = f1_at(sc, ev["label"], thr[m])
        label = dict(zip(Ts, (64, 192, 576)))[T]   # report keys use the full-size names
        res[f"events_T{label}"] = row
    (out / "analog.json").write_text(json.dumps(res, indent=2, default=float))
    (out / "status.json").write_text(json.dumps({"status": "complete"}))
    return res
