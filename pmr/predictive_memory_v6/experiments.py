"""Stage DAGs (S1-S6, FINAL, SMOKE) and the budgeted multi-GPU job runner.

Every job is a JSON spec run in its own process
(``python -m pmr.predictive_memory_v6 job --spec <file>``).  The runner packs
``procs_per_gpu`` jobs per visible GPU, resolves dependencies, skips jobs whose
``status.json`` says complete (resume), and charges GPU-hours as stage wall
time x GPUs held.  It stops launching when the stage cap or the 160 GPUh hard
cap would be exceeded, never raises a cap, and refuses:
  * S2+ without ``--prereg-commit`` whose committed prereg.py equals the
    working copy;
  * a stage whose predecessor gate (written by ``report``) says stop;
  * FINAL without ``--final-ack`` (the user's go-ahead after the human checks).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import prereg

PKG = "pmr.predictive_memory_v6"
TRACKS = {
    "T0d": {"name": "T0d", "sbar_inj": 0.0},
    "T1": {"name": "T1", "sbar_inj": prereg.CALIB["t1_sbar_inj"]},
    "T2": {"name": "T2", "sbar_inj": 0.0},
    "T3": {"name": "T3", "sbar_inj": 0.0},
}
for _s in prereg.CALIB["t0_sbar_inj"]:
    TRACKS[f"T0_s{_s:g}"] = {"name": "T0", "sbar_inj": _s}
CONDS = ["STATIC", "CONT", "EXACT", "NEWPATH", "CHANGE"]
GATE_ARMS = ("A2", "A2P", "A3", "A4")
PROBE_ARMS = ("A0", "A1_best", "A6", "A2", "A8", "A8g")
SEEDS = list(range(prereg.RULES["phi_seeds"]))
NOISE_SEEDS = list(range(prereg.RULES["t0_noise_seeds"]))


class Ctx:
    def __init__(self, root, cache, data=None, small=False, n_items=None, vsc=None):
        self.root, self.cache, self.data, self.small = Path(root), cache, data, small
        self.n_items = n_items or prereg.DATA["dev_items_per_condition"]
        self.vsc = vsc or {}

    def p(self, *parts):
        return str(self.root.joinpath(*parts))


def J(ctx, jid, type_, deps=(), **kw):
    d = {"type": type_, "id": jid, "out": ctx.p(jid), "deps": list(deps), "cache": ctx.cache, "data": ctx.data}
    d.update(kw)
    return d


def _small(ctx, **kw):
    return kw if ctx.small else {}


# --------------------------------------------------------------- S1 (profile)
def stage_s1(ctx):
    t = TRACKS["T1"]
    js = []
    js.append(J(ctx, "S1/fit_slow/L0", "fit_slow", track=t, steps=1000, pool=500, dev_pool=50))
    js.append(J(ctx, "S1/fit_slow/L0_overfit", "fit_slow", track=t, steps=300, pool=16, dev_pool=16, overfit=16))
    slow = ctx.p("S1/fit_slow/L0/slow.pt")
    js.append(J(ctx, "S1/calib/T1_oracle", "calib", ["S1/fit_slow/L0"], track=t, slow=slow, n_eps=64))
    cal = ctx.p("S1/calib/T1_oracle/calib.json")
    js.append(J(ctx, "S1/fit_phi/overfit", "fit_phi", ["S1/calib/T1_oracle"], track=t, slow=slow, calib=cal,
                dk=128, steps=300, pool=16, overfit=16, dev_pool=16, dev_every=50))
    js.append(J(ctx, "S1/fit_phi/profile", "fit_phi", ["S1/calib/T1_oracle"], track=t, slow=slow, calib=cal,
                dk=256, steps=50, pool=256, dev_pool=16, dev_every=25))
    for a in GATE_ARMS:
        js.append(J(ctx, f"S1/fit_gate/{a}", "fit_gate", ["S1/calib/T1_oracle"], track=t, slow=slow, calib=cal,
                    arm=a, steps=20 if a != "A3" else 20, pool=128))
    js.append(J(ctx, "S1/eval/T1_oracle", "eval", ["S1/calib/T1_oracle"] + [f"S1/fit_gate/{a}" for a in GATE_ARMS],
                track=t, slow=slow, calib=cal, n_items=32, conditions=CONDS,
                gates={a: ctx.p(f"S1/fit_gate/{a}/gate.pt") for a in GATE_ARMS}))
    return js


# --------------------------------------------------------- S2 (T0d, T0)
def stage_s2(ctx):
    js = []
    sm = ctx.small
    js.append(J(ctx, "S2/fit_slow/L0", "fit_slow", track=TRACKS["T1"], **_small(ctx, steps=30, pool=8, dev_pool=4)))
    slow = ctx.p("S2/fit_slow/L0/slow.pt")
    small_cal = _small(ctx, n_eps=4, grid=[0.25, 0.5, 0.75])
    js.append(J(ctx, "S2/calib/T0d", "calib", ["S2/fit_slow/L0"], track=TRACKS["T0d"], slow=slow, **small_cal))
    js.append(J(ctx, "S2/eval/T0d", "eval", ["S2/calib/T0d"], track=TRACKS["T0d"], slow=slow,
                calib=ctx.p("S2/calib/T0d/calib.json"), n_items=ctx.n_items, conditions=CONDS))
    for s in prereg.CALIB["t0_sbar_inj"]:
        tk = f"T0_s{s:g}"
        js.append(J(ctx, f"S2/calib/{tk}", "calib", ["S2/fit_slow/L0"], track=TRACKS[tk], slow=slow, **small_cal))
        cal = ctx.p(f"S2/calib/{tk}/calib.json")
        gdeps, gates = [], {}
        if s == prereg.CALIB["t1_sbar_inj"]:
            # learned gates on oracle keys: descriptive only in T0 (one seed)
            for a in GATE_ARMS:
                jid = f"S2/fit_gate/{tk}/{a}"
                js.append(J(ctx, jid, "fit_gate", [f"S2/calib/{tk}"], track=TRACKS[tk], slow=slow, calib=cal, arm=a,
                            seed=0, **_small(ctx, steps=2, pool=8)))
                gdeps.append(jid)
                gates[a] = ctx.p(jid, "gate.pt")
        for ns in (NOISE_SEEDS[:1] if sm else NOISE_SEEDS):
            js.append(J(ctx, f"S2/eval/{tk}/ns{ns}", "eval", [f"S2/calib/{tk}"] + (gdeps if ns == 0 else []),
                        track=TRACKS[tk], slow=slow, calib=cal, n_items=ctx.n_items, conditions=CONDS,
                        noise_seed=ns, gates=gates if ns == 0 else {}))
    return js


# ------------------------------------------------------------- S3 (T1)
def _learned_track(ctx, stage, tk, slow, oracle_cal, dk, conds, seeds, extra_eval=None, probes=True,
                   track_deps=()):
    """phi x seeds -> calib -> gates -> eval -> probes for one learned-key track."""
    js = []
    t = TRACKS[tk]
    for sd in seeds:
        phi_id = f"{stage}/fit_phi/{tk}/s{sd}"
        js.append(J(ctx, phi_id, "fit_phi", list(track_deps), track=t, slow=slow, calib=oracle_cal, dk=dk, seed=sd,
                    **_small(ctx, steps=5, pool=8, dev_pool=4, dev_every=2)))
        phi = ctx.p(phi_id, "phi.pt")
        cal_id = f"{stage}/calib/{tk}/s{sd}"
        js.append(J(ctx, cal_id, "calib", [phi_id], track=t, slow=slow, phi=phi, seed=sd,
                    **_small(ctx, n_eps=4, grid=[0.25, 0.5, 0.75])))
        cal = ctx.p(cal_id, "calib.json")
        gates, gdeps = {}, []
        for a in GATE_ARMS:
            gid = f"{stage}/fit_gate/{tk}/{a}/s{sd}"
            js.append(J(ctx, gid, "fit_gate", [cal_id], track=t, slow=slow, phi=phi, calib=cal, arm=a, seed=sd,
                        **_small(ctx, steps=2, pool=8)))
            gates[a] = ctx.p(gid, "gate.pt")
            gdeps.append(gid)
        ev_id = f"{stage}/eval/{tk}/s{sd}"
        js.append(J(ctx, ev_id, "eval", [cal_id] + gdeps, track=t, slow=slow, phi=phi, calib=cal, gates=gates,
                    n_items=ctx.n_items, conditions=conds, seed=sd, **(extra_eval or {})))
        if probes:
            for a in PROBE_ARMS:
                js.append(J(ctx, f"{stage}/probe/{tk}/{a}/s{sd}", "probe", [ev_id], track=t, slow=slow, phi=phi,
                            calib=cal, gates={k: v for k, v in gates.items() if k == "A2"}, arm=a, seed=sd,
                            select_from=ctx.p(ev_id, "summary.json"),
                            **_small(ctx, steps=5, override={"train_episodes": 4, "states_per_episode": 3,
                                                             "eval_episodes": 4, "eval_stride": 100})))
    return js


def stage_s3(ctx):
    js = []
    slow = ctx.p("S2/fit_slow/L0/slow.pt")
    ocal = ctx.p("S2/calib/T0_s0.5/calib.json")
    sweep = prereg.CALIB["dk_grid"] if not ctx.small else (16, 32)
    for dk in sweep:
        jid = f"S3/fit_phi/T1_dk{dk}"
        js.append(J(ctx, jid, "fit_phi", [], track=TRACKS["T1"], slow=slow, calib=ocal, dk=dk, seed=0,
                    **_small(ctx, steps=5, pool=8, dev_pool=4, dev_every=2)))
    js.append(J(ctx, "S3/dk_select", "dk_select", [f"S3/fit_phi/T1_dk{dk}" for dk in sweep], track=TRACKS["T1"],
                slow=slow, calib=ocal, dks=list(sweep), phis={str(dk): ctx.p(f"S3/fit_phi/T1_dk{dk}/phi.pt")
                                                              for dk in sweep},
                oracle_summary=ctx.p("S2/eval/T0_s0.5/ns0/summary.json"), n_items=min(200, ctx.n_items)))
    dk = {"from": ctx.p("S3/dk_select/dk.json")}
    js += _learned_track(ctx, "S3", "T1", slow, ocal, dk, CONDS, SEEDS[:2] if ctx.small else SEEDS,
                         track_deps=("S3/dk_select",))
    return js


# ------------------------------------------------------------- S4 (T2)
def stage_s4(ctx):
    js = []
    js.append(J(ctx, "S4/fit_slow/L1", "fit_slow", [], track=TRACKS["T2"], sigma_from=ctx.p("S2/fit_slow/L0/slow.pt"),
                **_small(ctx, steps=30, pool=8, dev_pool=4)))
    slow = ctx.p("S4/fit_slow/L1/slow.pt")
    js.append(J(ctx, "S4/calib/T2_oracle", "calib", ["S4/fit_slow/L1"], track=TRACKS["T2"], slow=slow,
                **_small(ctx, n_eps=4, grid=[0.25, 0.5, 0.75])))
    dk = {"from": ctx.p("S3/dk_select/dk.json")}
    js += _learned_track(ctx, "S4", "T2", slow, ctx.p("S4/calib/T2_oracle/calib.json"), dk, CONDS + ["MOVE"],
                         SEEDS[:1] if ctx.small else SEEDS, probes=False, track_deps=("S4/calib/T2_oracle",))
    return js


# ------------------------------------------------------------- S5 (T3)
def stage_s5(ctx, n_shards=8):
    js = []
    enc = {"smoke_encoder": True} if ctx.small else {}
    js.append(J(ctx, "S5/fit_pca", "fit_pca", [], **enc, **_small(ctx, pca_episodes=2)))
    sh = 2 if ctx.small else n_shards
    pdeps = []
    for split in ("train", "dev"):
        for i in range(sh):
            jid = f"S5/project/{split}/sh{i}"
            js.append(J(ctx, jid, "project", ["S5/fit_pca"], split=split, shard=i, n_shards=sh, **enc))
            pdeps.append(jid)
    js.append(J(ctx, "S5/fit_slow/R", "fit_slow", pdeps, track=TRACKS["T3"],
                **_small(ctx, steps=30, pool=8, dev_pool=4)))
    slow = ctx.p("S5/fit_slow/R/slow.pt")
    js.append(J(ctx, "S5/r_gate", "r_gate", ["S5/fit_slow/R"], slow_status=ctx.p("S5/fit_slow/R/status.json"),
                **_small(ctx, n_train=8, n_dev=4)))
    js.append(J(ctx, "S5/calib/T3_pose_oracle", "calib", ["S5/r_gate"], track=TRACKS["T3"], slow=slow,
                require_gate=None if ctx.small else ctx.p("S5/r_gate/r_gate.json"),
                **_small(ctx, n_eps=4, grid=[0.25, 0.5, 0.75])))
    dk = {"from": ctx.p("S3/dk_select/dk.json")}
    js += _learned_track(ctx, "S5", "T3", slow, ctx.p("S5/calib/T3_pose_oracle/calib.json"), dk, CONDS,
                         SEEDS[:1] if ctx.small else SEEDS, track_deps=("S5/calib/T3_pose_oracle",))
    return js


# --------------------------------------------------------- S6 / FINAL (VSC)
# S6 runs the official base pass once with logging; only the designated
# calibration videos (n_calib lowest sha1 of the video key) are read by
# vsc_calib.  FINAL runs the variants and scores the remaining videos once.
VSC_VARIANTS = ("repeat1", "repeat2", "repeat5", "phase", "reverse")


def stage_s6(ctx):
    v = ctx.vsc
    js = [J(ctx, "S6/vsc_infer/base", "vsc_infer", [], cache_name="cambrians_vsc",
            task=v.get("task", "cambrians_vsc_10mins"), dump=ctx.p("S6/vsc_infer/base/dumps"))]
    js.append(J(ctx, "S6/vsc_calib", "vsc_calib", ["S6/vsc_infer/base"], dumps=ctx.p("S6/vsc_infer/base/dumps"),
                n_calib=v.get("n_calib", 10)))
    return js


def stage_final(ctx):
    js = []
    js.append(J(ctx, "FINAL/prepare_test", "prepare_test", [], open_test=True))
    for tk, stage in (("T1", "S3"), ("T2", "S4"), ("T3", "S5")):
        for sd in SEEDS:
            src = Path(ctx.p(f"{stage}/eval/{tk}/s{sd}/job_spec.json"))
            if not src.exists():
                continue
            spec = json.loads(src.read_text())
            spec.update(id=f"FINAL/eval/{tk}/s{sd}", out=ctx.p(f"FINAL/eval/{tk}/s{sd}"), deps=["FINAL/prepare_test"],
                        split="test", open_test=True, n_items=prereg.DATA["test_items_per_condition"])
            spec.pop("max_seconds", None)
            js.append(spec)
    v = ctx.vsc
    for var in VSC_VARIANTS:
        js.append(J(ctx, f"FINAL/vsc_infer/{var}", "vsc_infer", [], cache_name=f"cambrians_vsc_{var}",
                    task=v.get("task", "cambrians_vsc_10mins"), dump=ctx.p(f"FINAL/vsc_infer/{var}/dumps"),
                    open_test=True))
    dumps = {"base": ctx.p("S6/vsc_infer/base/dumps")}
    dumps.update({var: ctx.p(f"FINAL/vsc_infer/{var}/dumps") for var in VSC_VARIANTS})
    js.append(J(ctx, "FINAL/vsc_eval", "vsc_eval", [f"FINAL/vsc_infer/{var}" for var in VSC_VARIANTS],
                cal=ctx.p("S6/vsc_calib/vsc_cal.npz"), calib_names=ctx.p("S6/vsc_calib/calib_names.json"),
                dumps=dumps, gt_json=v.get("gt_json"), open_test=True))
    return js


# -------------------------------------------------------------- SMOKE (CPU)
def stage_smoke(ctx):
    """CPU end-to-end software check on synthetic Memory-Maze-format data:
    S2 -> S3 -> S4 -> S5 chained, tiny sizes, plus the offline VSC path on
    synthetic logs.  Numbers are meaningless; only the plumbing is tested."""
    js = [J(ctx, "SMOKE/fake_data", "fake_data", [], n_train=24, n_eval=4)]
    prev = ["SMOKE/fake_data"]
    for st in (stage_s2(ctx), stage_s3(ctx), stage_s4(ctx), stage_s5(ctx)):
        for j in st:
            j["deps"] = list(prev) + j["deps"]
        js += st
        prev = [j["id"] for j in st]
    js.append(J(ctx, "SMOKE/vsc_synthetic", "vsc_synthetic", []))
    return js


def stage_jobs(stage, ctx):
    return {"S1": stage_s1, "S2": stage_s2, "S3": stage_s3, "S4": stage_s4, "S5": stage_s5, "S6": stage_s6,
            "FINAL": stage_final, "SMOKE": stage_smoke}[stage](ctx)


# ------------------------------------------------------------------ gates
GATE_REQUIRES = {"S3": ("S2", "H1_T0"), "S4": ("S3", "H1_T1"), "S5": ("S3", "H1_T1"), "S6": ("S3", "H1_T1"),
                 "FINAL": ("S3", "H1_T1")}


def check_stage_gate(stage, root):
    if stage not in GATE_REQUIRES:
        return True, "no gate"
    _, key = GATE_REQUIRES[stage]
    f = Path(root) / "report" / "gates.json"
    if not f.exists():
        return False, f"{f} missing: run `report` after the previous stage"
    g = json.loads(f.read_text())
    v = g.get(key)
    if v is True:
        return True, f"{key} passed"
    return False, f"{key} = {v!r}: protocol says stop (AGENTS.md rule 5, docs/V6_BF_DELTA.md 'ladder')"


def check_prereg(commit):
    here = Path(__file__).with_name("prereg.py")
    try:
        repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                              cwd=here.parent, check=True).stdout.strip()
        rel = here.relative_to(repo)
        committed = subprocess.run(["git", "show", f"{commit}:{rel}"], capture_output=True, text=True,
                                   cwd=repo, check=True).stdout
    except Exception as exc:
        return False, f"cannot read prereg.py at {commit}: {exc}"
    return (committed == here.read_text(), "prereg.py matches commit" if committed == here.read_text()
            else "working prereg.py differs from the committed one")


# ----------------------------------------------------------------- runner
def _done(job):
    st = Path(job["out"]) / "status.json"
    return st.exists() and json.loads(st.read_text()).get("status") == "complete"


def _ledger(root):
    f = Path(root) / "budget_ledger.json"
    return json.loads(f.read_text()) if f.exists() else {"stages": {}, "total_gpuh": 0.0}


def run_stage(stage, root, gpus, procs_per_gpu=4, max_gpuh=None, cache=None, data=None, prereg_commit=None,
              job_filter=None, dry_run=False, final_ack=False, small=False, vsc=None, n_items=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if stage not in ("S1", "SMOKE"):
        if not prereg_commit:
            raise SystemExit(f"{stage} requires --prereg-commit <hash> (docs/PREREGISTRATION_V6.1.md)")
        ok, msg = check_prereg(prereg_commit)
        if not ok:
            raise SystemExit(msg)
        ok, msg = check_stage_gate(stage, root)
        if not ok:
            raise SystemExit(f"{stage} blocked: {msg}")
    if stage == "FINAL" and not final_ack:
        raise SystemExit("FINAL opens the test split once; it needs --final-ack after the user signs off "
                         "on the checks in docs/PREREGISTRATION_V6.1.md section 'Before FINAL'")
    if stage == "SMOKE":
        small = True
        cache = str(root / "SMOKE" / "cache")
        data = str(root / "SMOKE" / "data")
    ctx = Ctx(root, cache, data, small=small, n_items=n_items or (4 if small else None), vsc=vsc)
    cap = max_gpuh if max_gpuh is not None else prereg.BUDGET_GPUH.get(stage, prereg.BUDGET_GPUH["reserve"])
    cap = min(cap, prereg.BUDGET_GPUH.get(stage, cap)) if stage in prereg.BUDGET_GPUH else cap
    ledger = _ledger(root)
    hard = prereg.BUDGET_GPUH["hard_cap"]
    jobs = [j for j in stage_jobs(stage, ctx) if job_filter is None or job_filter in j["id"]]
    pending = [j for j in jobs if not _done(j)]
    print(f"[{stage}] {len(jobs)} jobs, {len(pending)} pending, cap {cap} GPUh, "
          f"ledger {ledger['total_gpuh']:.2f}/{hard} GPUh")
    if dry_run:
        for j in pending:
            print("  ", j["type"], j["id"], "<-", ",".join(j["deps"]))
        return 0
    slots = [g for g in gpus for _ in range(procs_per_gpu)] if gpus else [None] * procs_per_gpu
    running, failed, stopped = {}, [], False
    done_ids = {j["id"] for j in jobs if _done(j)}
    start = time.time()
    used_before = ledger["stages"].get(stage, 0.0)

    def charged():
        return used_before + (time.time() - start) / 3600.0 * max(1, len(gpus))

    while pending or running:
        for slot, (proc, job) in list(running.items()):
            if proc.poll() is not None:
                del running[slot]
                (done_ids.add(job["id"]) if _done(job) else failed.append(job["id"]))
        budget_left = min(cap - charged(), hard - ledger["total_gpuh"] - (charged() - used_before))
        if budget_left <= 0 and pending:
            stopped, pending = True, []
        for i in [i for i in range(len(slots)) if i not in running]:
            ready = [j for j in pending if all(d in done_ids for d in j.get("deps", []))]
            if not ready:
                break
            job = dict(ready[0])
            pending.remove(ready[0])
            job.setdefault("max_seconds", max(180.0, budget_left / max(1, len(gpus)) * 3600.0))
            Path(job["out"]).mkdir(parents=True, exist_ok=True)
            spec = Path(job["out"]) / "job_spec.json"
            spec.write_text(json.dumps(job, indent=2))
            env = dict(os.environ)
            if slots[i] is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(slots[i])
            log = open(Path(job["out"]) / "job.log", "a")
            running[i] = (subprocess.Popen([sys.executable, "-m", PKG, "job", "--spec", str(spec)],
                                           stdout=log, stderr=subprocess.STDOUT, env=env), job)
        blocked = [j for j in pending if any(d in failed for d in j.get("deps", []))]
        for j in blocked:
            pending.remove(j)
            failed.append(j["id"] + " (dependency failed)")
        if pending and not running and not any(all(d in done_ids for d in j.get("deps", [])) for j in pending):
            failed += [j["id"] + " (unsatisfiable)" for j in pending]
            pending = []
        time.sleep(0.5 if small else 5)
    used = charged() if gpus else used_before
    ledger["stages"][stage] = used
    ledger["total_gpuh"] = sum(ledger["stages"].values())
    ledger["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (root / "budget_ledger.json").write_text(json.dumps(ledger, indent=2))
    status = {"stage": stage, "status": "complete" if not failed and not stopped else "partial",
              "failed_jobs": failed, "stopped_by_budget": stopped, "gpuh_charged_stage": used, "gpus": gpus,
              "procs_per_gpu": procs_per_gpu, "prereg_commit": prereg_commit}
    (root / f"{stage}_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))
    return 0 if status["status"] == "complete" else 2
