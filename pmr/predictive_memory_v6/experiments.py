"""Stage definitions (S1-S7, FINAL) and a budgeted multi-GPU job runner.

Each job is a JSON spec executed in its own process:
``python -m pmr.predictive_memory_v6 job --spec <file>``.  The runner packs
``procs_per_gpu`` jobs per visible GPU, skips completed jobs (resume), and
charges GPU-hours as (stage wall time) x (number of GPUs held).  When the
stage cap or the cumulative 160 GPUh cap would be exceeded it stops launching,
lets running jobs hit their own ``max_seconds``, writes a partial status and
exits non-zero.  It never raises a cap on its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import prereg

PRIMARY = ["static_delta", "gdn", "gdn_neg", "vo_rotation", "tpa_readout", "ours", "ours_GCt",
           "ours_frame", "fifo", "gru", "slot_known_id", "lact"]
SECONDARY = ["m2rnn", "kdn_diag", "ours_dense", "flowm_token"]
H6_TRAINED = ["ours", "static_delta", "tpa_readout", "gru", "amortized_dL", "oracle_L"]
REFERENCES = ["pause", "cv_kf", "cv_kf_oru", "analytic_kf", "interp_oracle"]
SEEDS = [0, 1, 2]
REAL_VARIANTS = ["ours", "static_delta", "gdn_neg", "tpa_readout", "gru", "fifo"]
MOT_VARIANTS = ["ours", "static_delta", "gdn_neg", "tpa_readout"]


def C(name, kind, world, T, n, **kw):
    d = {"name": name, "kind": kind, "world": world, "T": int(T), "n": int(n)}
    d.update(kw)
    return d


SUITE_MAIN = [C("main_T64", "standard", "l0_main", 64, 2048), C("main_T192", "standard", "l0_main", 192, 2048),
              C("main_T576", "standard", "l0_main", 576, 1024),
              C("flowm_protocol", "standard", "l0_main:blind_after=50", 200, 1024),
              C("history_T64", "history_pair", "l0_sched", 64, 1024)]
SUITE_CTRL = [C("ctrl_T64", "standard", "l0_ctrl", 64, 2048), C("ctrl_T192", "standard", "l0_ctrl", 192, 2048),
              C("ctrl_T576", "standard", "l0_ctrl", 576, 1024),
              C("sched_T64", "standard", "l0_sched", 64, 2048), C("sched_T192", "standard", "l0_sched", 192, 2048),
              C("sched_T576", "standard", "l0_sched", 576, 1024),
              C("change_T64", "change_pair", "l0_ctrl", 64, 2048),
              C("history_T64", "history_pair", "l0_sched", 64, 1024),
              C("blackout_T64", "blackout", "l0_blackout", 64, 1024)]
SUITE_DRIFT = [C("drift_T64", "standard", "l0_drift", 64, 2048), C("drift_T192", "standard", "l0_drift", 192, 2048),
               C("drift_T576", "standard", "l0_drift", 576, 1024)]


def suite_sweeps(variant):
    out = [C(f"change_rho{r}", "change_pair", f"l0_ctrl:rho={r}", 64, 1024) for r in (0.0, 0.25, 0.5, 0.75, 0.9)]
    out += [C(f"ctrl_noise{s}", "standard", f"l0_ctrl:obs_noise={s}", 64, 1024) for s in (0.25, 0.5, 1.0)]
    if variant != "gru":  # GRU input width is tied to N
        out += [C(f"ctrl_N{n}", "standard", f"l0_ctrl:n_objects={n}", 64, 1024) for n in (3, 5, 8, 12, 24)]
    out += [C(f"count_static_k{k}", "count", f"l0_tour_static:copies={k}", 50 * k, 512) for k in (1, 2, 3, 6)]
    out += [C(f"count_dynamic_k{k}", "count", f"l0_tour_dynamic:copies={k}", 50 * k, 512) for k in (1, 2, 3, 6)]
    return out


def suite_l1(cache):
    return [C("l1_dev_T64", "standard", "l1_dev", 64, 2048, cache=cache),
            C("l1_dev", "standard", "l1_dev", 500, 1000, cache=cache),
            C("l1_peekaboo", "standard", "l1_peekaboo", 500, 1000, cache=cache),
            C("l1_backforth", "standard", "l1_backforth", 500, 1000, cache=cache)]


def suite_nusc(cache):
    return [C("nusc_dev", "standard", "real_nuscenes_dev", 38, -1, cache=cache),
            C("nusc_dev_pseudo", "standard", "real_nuscenes_dev_pseudo", 38, -1, cache=cache)]


def suite_mot(fold, cache):
    return [C(f"mot{fold}_dev", "standard", f"real_mot_{fold}_dev", 64, -1, cache=cache)]


# ------------------------------------------------------------------- jobs
def _train(stage, root, variant, world, seed, steps=20000, **kw):
    jid = f"{stage}/train/{variant}/{world.replace(':', '@').replace(',', '+')}/s{seed}"
    d = {"type": "train", "id": jid, "out": str(Path(root) / jid), "variant": variant, "world": world,
         "seed": seed, "steps": steps}
    d.update(kw)
    return d


def _eval(stage, root, train_job, conditions, split="dev", as_variant=None, tag=None, **kw):
    base = train_job["id"].replace("/train/", "/eval/")
    jid = base + (f"/{tag}" if tag else "") + (f"/{split}" if split != "dev" else "")
    d = {"type": "eval", "id": jid, "out": str(Path(root) / jid), "deps": [train_job["id"]],
         "checkpoint": str(Path(train_job["out"]) / "best.pt"), "conditions": conditions, "split": split,
         "training_seed": train_job["seed"]}
    if as_variant:
        d["as_variant"] = as_variant
    d.update(kw)
    return d


def _ref(stage, root, name, conditions, tag, split="dev", **kw):
    jid = f"{stage}/ref/{name}/{tag}" + (f"/{split}" if split != "dev" else "")
    d = {"type": "eval", "id": jid, "out": str(Path(root) / jid), "reference": name, "conditions": conditions,
         "split": split, "deps": []}
    d.update(kw)
    return d


def stage_jobs(stage, root, cache=None):
    jobs = []
    if stage == "S1":
        for v in PRIMARY + SECONDARY + ["amortized_dL"]:
            world = "l0_drift" if v in ("amortized_dL",) else "l0_sched"
            jobs.append(_train("S1", root, v, world, 0, steps=300, overfit=16, batch=16, dev_every=300,
                               dev_episodes=16, log_every=10))
        for v in PRIMARY:
            jobs.append(_train("S1", root, v, "l0_main", 0, steps=30, dev_every=10 ** 9, dev_episodes=64,
                               pool_size=2048, tag="profile"))
            jobs[-1]["id"] += "/profile"
            jobs[-1]["out"] += "/profile"
        jobs.append({"type": "numerics", "id": "S1/numerics", "out": str(Path(root) / "S1/numerics"), "deps": []})
    elif stage == "S2":
        for v in PRIMARY:
            for w, suite in (("l0_main", SUITE_MAIN), ("l0_ctrl", SUITE_CTRL)):
                for s in SEEDS:
                    tj = _train("S2", root, v, w, s)
                    jobs += [tj, _eval("S2", root, tj, suite)]
        for r in REFERENCES:
            jobs.append(_ref("S2", root, r, SUITE_MAIN, "l0_main"))
            jobs.append(_ref("S2", root, r, SUITE_CTRL, "l0_ctrl"))
    elif stage == "S3":
        for v in ("ours", "static_delta", "gdn_neg", "tpa_readout", "slot_known_id"):
            for rho in (0.5, 0.9):
                tj = _train("S3", root, v, f"l0_ctrl:rho={rho}", 0)
                jobs += [tj, _eval("S3", root, tj, [C(f"change_rho{rho}", "change_pair", f"l0_ctrl:rho={rho}", 64, 1024),
                                                     C(f"ctrl_rho{rho}", "standard", f"l0_ctrl:rho={rho}", 64, 1024)])]
        for v in PRIMARY:
            for s in SEEDS:
                tj = _train("S2", root, v, "l0_ctrl", s)          # reuse S2 checkpoints
                ej = _eval("S3", root, tj, suite_sweeps(v), tag="sweeps")
                ej["id"] = ej["id"].replace("S2/eval", "S3/eval")
                ej["out"] = str(Path(root) / ej["id"])
                ej["deps"] = []
                jobs.append(ej)
        for r in ("pause", "cv_kf", "interp_oracle"):
            jobs.append(_ref("S3", root, r, [c for c in suite_sweeps("ref") if c["kind"] != "count"], "sweeps"))
        jobs.append({"type": "analog", "id": "S3/analog", "out": str(Path(root) / "S3/analog"), "deps": []})
    elif stage == "S4":
        for v in H6_TRAINED:
            for s in SEEDS:
                tj = _train("S4", root, v, "l0_drift", s)
                jobs += [tj, _eval("S4", root, tj, SUITE_DRIFT)]
                if v == "ours":
                    for lr in prereg.H6["ttt_lrs"]:
                        jobs.append(_eval("S4", root, tj, SUITE_DRIFT, as_variant="ours_ttt", tag=f"ttt_lr{lr}",
                                          ttt_lr=lr))
        for r in ("pause", "cv_kf", "interp_oracle"):
            jobs.append(_ref("S4", root, r, SUITE_DRIFT, "l0_drift"))
    elif stage == "S5":
        if cache is None:
            raise ValueError("S5 needs --cache with prepared L1 tokens")
        for v in PRIMARY:
            for s in SEEDS:
                tj = _train("S5", root, v, "l1_train", s, dev_world="l1_dev", cache=cache)
                jobs += [tj, _eval("S5", root, tj, suite_l1(cache))]
        for r in REFERENCES:
            jobs.append(_ref("S5", root, r, suite_l1(cache), "l1"))
    elif stage == "S6":
        for v in SECONDARY:
            for w, suite in (("l0_main", SUITE_MAIN), ("l0_ctrl", SUITE_CTRL)):
                for s in SEEDS:
                    tj = _train("S6", root, v, w, s)
                    jobs += [tj, _eval("S6", root, tj, suite)]
    elif stage == "S7":
        if cache is None:
            raise ValueError("S7 needs --cache with prepared real tokens")
        for v in REAL_VARIANTS:
            for s in SEEDS:
                tj = _train("S7", root, v, "real_nuscenes_train", s, steps=5000, T=38, batch=128,
                            dev_world="real_nuscenes_dev", cache=cache, dev_episodes=256)
                jobs += [tj, _eval("S7", root, tj, suite_nusc(cache))]
        for r in REFERENCES:
            jobs.append(_ref("S7", root, r, suite_nusc(cache), "nuscenes"))
        for f in range(7):
            for v in MOT_VARIANTS:
                tj = _train("S7", root, v, f"real_mot_{f}_train", 0, steps=5000, T=64, batch=128,
                            dev_world=f"real_mot_{f}_dev", cache=cache, dev_episodes=128)
                jobs += [tj, _eval("S7", root, tj, suite_mot(f, cache))]
            for r in REFERENCES:
                jobs.append(_ref("S7", root, r, suite_mot(f, cache), f"mot{f}"))
    elif stage == "SMOKE":  # CPU end-to-end software check (CI); never a scientific result
        small = dict(steps=4, batch=4, T=16, dev_episodes=4, pool_size=16, dev_every=2, refresh=100)
        ctrl = [C("std", "standard", "l0_ctrl", 16, 4), C("hist", "history_pair", "l0_sched", 16, 4),
                C("chg", "change_pair", "l0_ctrl", 32, 8), C("blk", "blackout", "l0_blackout", 48, 4),
                C("cnt", "count", "l0_tour_static:copies=2", 100, 4),
                C("proto", "standard", "l0_main:blind_after=8", 16, 4)]
        drift = [C("drift", "standard", "l0_drift", 16, 4)]
        for v in PRIMARY + SECONDARY + ["amortized_dL", "oracle_L"]:
            w = "l0_drift" if v in ("amortized_dL", "oracle_L") else "l0_ctrl"
            tj = _train("SMOKE", root, v, w, 0, **small)
            jobs += [tj, _eval("SMOKE", root, tj, drift if w == "l0_drift" else ctrl)]
            if v == "ours":
                jobs.append(_eval("SMOKE", root, tj, drift, as_variant="ours_ttt", tag="ttt", ttt_lr=0.3))
        for r in REFERENCES:
            jobs.append(_ref("SMOKE", root, r, [c for c in ctrl if c["kind"] != "count"], "l0_ctrl"))
        jobs.append({"type": "numerics", "id": "SMOKE/numerics", "out": str(Path(root) / "SMOKE/numerics"), "deps": []})
        jobs.append({"type": "analog", "id": "SMOKE/analog", "out": str(Path(root) / "SMOKE/analog"), "deps": [],
                     "small": True, "lfp_steps": 3})
    elif stage == "FINAL":
        jobs = final_jobs(root, cache)
    else:
        raise ValueError(stage)
    return jobs


def to_test(cond):
    c = dict(cond)
    w = c["world"]
    for a, b in (("l1_dev", "l1_test"), ("real_nuscenes_dev", "real_nuscenes_test")):
        w = w.replace(a, b)
    if w.startswith("real_mot_") and w.endswith("_dev"):
        w = w[:-4] + "_test"
    c["world"] = w
    c["name"] = c["name"].replace("_dev", "_test")
    return c


def final_jobs(root, cache):
    """Re-evaluate every completed dev evaluation on the unopened test split."""
    jobs = []
    science = lambda p: Path(p).relative_to(root).parts[0] not in ("SMOKE", "S1", "FINAL")
    for st in filter(science, Path(root).glob("S*/eval/**/status.json")):
        spec_f = st.parent / "job_spec.json"
        if not spec_f.exists():
            continue
        spec = json.loads(spec_f.read_text())
        if spec.get("split") != "dev":
            continue
        t = dict(spec)
        t["split"], t["open_test"] = "test", True
        t["conditions"] = [to_test(c) for c in spec["conditions"]]
        t["id"] = spec["id"].replace("/eval/", "/test/").replace("/ref/", "/test-ref/")
        t["out"] = str(Path(root) / "FINAL" / t["id"])
        t["deps"] = []
        jobs.append(t)
    for st in Path(root).glob("S*/ref/**/job_spec.json"):
        spec = json.loads(st.read_text())
        if spec.get("split") != "dev":
            continue
        t = dict(spec)
        t["split"], t["open_test"] = "test", True
        t["conditions"] = [to_test(c) for c in spec["conditions"]]
        t["id"] = "FINAL/" + spec["id"]
        t["out"] = str(Path(root) / t["id"])
        jobs.append(t)
    return jobs


# ----------------------------------------------------------------- runner
def _done(job):
    st = Path(job["out"]) / "status.json"
    if not st.exists():
        return False
    s = json.loads(st.read_text())
    return s.get("status") == "complete"


def _ledger(root):
    f = Path(root) / "budget_ledger.json"
    return json.loads(f.read_text()) if f.exists() else {"stages": {}, "total_gpuh": 0.0}


def run_stage(stage, root, gpus, procs_per_gpu=4, max_gpuh=None, cache=None, prereg_commit=None,
              job_filter=None, dry_run=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if stage not in ("S1", "SMOKE") and not prereg_commit:
        raise SystemExit(f"{stage} requires --prereg-commit <hash> (freeze docs/PREREGISTRATION_V6.md first)")
    cap = max_gpuh if max_gpuh is not None else prereg.BUDGET_GPUH.get(stage, prereg.BUDGET_GPUH["reserve"])
    ledger = _ledger(root)
    hard = prereg.BUDGET_GPUH["hard_cap"]
    jobs = [j for j in stage_jobs(stage, root, cache) if job_filter is None or job_filter in j["id"]]
    for j in jobs:
        j.setdefault("cache", cache)
    pending = [j for j in jobs if not _done(j)]
    print(f"[{stage}] {len(jobs)} jobs, {len(pending)} pending, cap {cap} GPUh, "
          f"ledger {ledger['total_gpuh']:.2f}/{hard} GPUh")
    if dry_run:
        for j in pending:
            print("  ", j["type"], j["id"])
        return 0
    slots = [g for g in gpus for _ in range(procs_per_gpu)] if gpus else [None] * procs_per_gpu
    running = {}
    start = time.time()
    done_ids = {j["id"] for j in jobs if _done(j)}
    failed, stopped = [], False
    used_before = ledger["stages"].get(stage, 0.0)

    def charged():
        return used_before + (time.time() - start) / 3600.0 * max(1, len(gpus))

    while pending or running:
        for slot, (proc, job, t0) in list(running.items()):
            if proc.poll() is not None:
                del running[slot]
                (done_ids.add(job["id"]) if _done(job) else failed.append(job["id"]))
        budget_left = min(cap - charged(), hard - ledger["total_gpuh"] - (charged() - used_before))
        if budget_left <= 0 and pending:
            stopped = True
            pending = []
        free = [i for i in range(len(slots)) if i not in running]
        for i in free:
            ready = [j for j in pending if all(d in done_ids for d in j.get("deps", []))]
            if not ready:
                break
            job = ready[0]
            pending.remove(job)
            job = dict(job)
            per_job_hours = max(0.05, budget_left / max(1, len(gpus)))
            job.setdefault("max_seconds", per_job_hours * 3600.0)
            Path(job["out"]).mkdir(parents=True, exist_ok=True)
            spec = Path(job["out"]) / "job_spec.json"
            spec.write_text(json.dumps(job, indent=2))
            env = dict(os.environ)
            if slots[i] is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(slots[i])
            log = open(Path(job["out"]) / "job.log", "a")
            proc = subprocess.Popen([sys.executable, "-m", "pmr.predictive_memory_v6", "job", "--spec", str(spec)],
                                    stdout=log, stderr=subprocess.STDOUT, env=env)
            running[i] = (proc, job, time.time())
        if not any(all(d in done_ids for d in j.get("deps", [])) for j in pending) and pending and not running:
            failed += [j["id"] for j in pending]  # dependencies failed
            pending = []
        time.sleep(5)
    used = charged()
    ledger["stages"][stage] = used
    ledger["total_gpuh"] = sum(ledger["stages"].values())
    ledger["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (root / "budget_ledger.json").write_text(json.dumps(ledger, indent=2))
    status = {"stage": stage, "status": "complete" if not failed and not stopped else "partial",
              "failed_jobs": failed, "stopped_by_budget": stopped, "gpuh_charged_stage": used,
              "gpus": gpus, "procs_per_gpu": procs_per_gpu, "prereg_commit": prereg_commit}
    (root / f"{stage}_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "complete" else 2
