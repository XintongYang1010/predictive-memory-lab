"""CLI for V6.

  python -m pmr.predictive_memory_v6 s0 --out runs/v6 [--cache DIR --nuscenes-meta DIR --mot-roots A B --miniworld DIR]
  python -m pmr.predictive_memory_v6 run --stage S1 --out runs/v6 --gpus 0,1,2,3,4,5,6,7 --procs-per-gpu 4
  python -m pmr.predictive_memory_v6 run --stage S2 --out runs/v6 --gpus 0,...,7 --prereg-commit <hash>
  python -m pmr.predictive_memory_v6 prepare-l1 --miniworld DIR --cache DIR --procs 48
  python -m pmr.predictive_memory_v6 prepare-real --cache DIR --nuscenes-meta DIR --mot-roots MOT17/train MOT20/train
  python -m pmr.predictive_memory_v6 report --out runs/v6
  python -m pmr.predictive_memory_v6 job --spec job_spec.json        (used by the runner)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pmr.predictive_memory_v6")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--spec", required=True)
    s0 = sub.add_parser("s0")
    s0.add_argument("--out", required=True)
    s0.add_argument("--cache")
    s0.add_argument("--nuscenes-meta")
    s0.add_argument("--mot-roots", nargs="*")
    s0.add_argument("--miniworld")
    r = sub.add_parser("run")
    r.add_argument("--stage", required=True, choices=["SMOKE", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "FINAL"])
    r.add_argument("--out", required=True)
    r.add_argument("--gpus", default="")
    r.add_argument("--procs-per-gpu", type=int, default=4)
    r.add_argument("--max-gpuh", type=float)
    r.add_argument("--cache")
    r.add_argument("--prereg-commit")
    r.add_argument("--filter")
    r.add_argument("--dry-run", action="store_true")
    p1 = sub.add_parser("prepare-l1")
    p1.add_argument("--miniworld", required=True)
    p1.add_argument("--cache", required=True)
    p1.add_argument("--procs", type=int, default=48)
    p1.add_argument("--splits", nargs="*", default=["dev", "train", "test", "peekaboo", "backforth"])
    p1.add_argument("--n", type=int)
    pr = sub.add_parser("prepare-real")
    pr.add_argument("--cache", required=True)
    pr.add_argument("--nuscenes-meta")
    pr.add_argument("--mot-roots", nargs="*")
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "job":
        spec = json.loads(Path(a.spec).read_text())
        t = spec["type"]
        if t == "train":
            from .train import train_job
            res = train_job(spec)
        elif t == "eval":
            from .evaluate import eval_job
            res = eval_job(spec)
        elif t == "numerics":
            from .checks import numerics_job
            res = numerics_job(spec)
        elif t == "analog":
            from .checks import analog_job
            res = analog_job(spec)
        else:
            raise SystemExit(f"unknown job type {t}")
        print(json.dumps(res, default=str)[:2000])
        ok = isinstance(res, dict) and res.get("status") == "complete"
        return 0 if ok else 3
    if a.cmd == "s0":
        from .checks import s0
        rep = s0(a)
        print(json.dumps({"gates": rep["gates"], "all_gates_pass": rep["all_gates_pass"]}, indent=2))
        return 0 if rep["all_gates_pass"] else 4
    if a.cmd == "run":
        from .experiments import run_stage
        gpus = [g for g in a.gpus.split(",") if g != ""]
        return run_stage(a.stage, a.out, gpus, a.procs_per_gpu, a.max_gpuh, a.cache, a.prereg_commit,
                         a.filter, a.dry_run)
    if a.cmd == "prepare-l1":
        from . import worlds_l1
        for sp in a.splits:
            print(worlds_l1.generate(a.miniworld, a.cache, sp, n=a.n, procs=a.procs)["failures"], "failures in", sp)
            print(worlds_l1.convert_split(a.cache, sp))
        print(worlds_l1.check_dynamics(a.cache, "dev"))
        print(worlds_l1.depth_check(a.cache, "dev"))
        return 0
    if a.cmd == "prepare-real":
        from .worlds_real import prepare_nuscenes, prepare_mot
        if a.nuscenes_meta:
            print(json.dumps(prepare_nuscenes(a.nuscenes_meta, a.cache), indent=2)[:3000])
        if a.mot_roots:
            print(json.dumps(prepare_mot(a.mot_roots, a.cache), indent=2)[:3000])
        return 0
    if a.cmd == "report":
        from .report import write_report
        H = write_report(a.out)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if vv in ("pass", "fail", "missing")}
                          for k, v in H.items()}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
