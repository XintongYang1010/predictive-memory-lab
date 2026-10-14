"""CLI for V6.1 (BF-Delta).

  python -m pmr.predictive_memory_v6 s0      --data DIR --cache DIR --out RUN
  python -m pmr.predictive_memory_v6 run     --stage S1 --out RUN --cache DIR --data DIR --gpus 0,1,...
  python -m pmr.predictive_memory_v6 run     --stage S2 ... --prereg-commit <hash>
  python -m pmr.predictive_memory_v6 report  --out RUN
  python -m pmr.predictive_memory_v6 run     --stage SMOKE --out RUN            (CPU, synthetic data)
  python -m pmr.predictive_memory_v6 job     --spec job_spec.json              (used by the runner)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_job(spec):
    t = spec["type"]
    if t == "fit_slow":
        from .train import fit_slow as f
    elif t == "calib":
        from .train import calib as f
    elif t == "fit_phi":
        from .train import fit_phi as f
    elif t == "fit_gate":
        from .train import fit_gate as f
    elif t == "eval":
        from .evaluate import evaluate as f
    elif t == "probe":
        from .probes import probe as f
    elif t == "fit_pca":
        from .features_r import fit_pca as f
    elif t == "project":
        from .features_r import project as f
    elif t == "r_gate":
        from .features_r import r_gate as f
    elif t in ("dk_select", "fake_data", "prepare_test", "vsc_infer", "vsc_calib", "vsc_eval", "vsc_synthetic"):
        from . import jobs
        f = getattr(jobs, t)
    else:
        raise SystemExit(f"unknown job type {t}")
    return f(spec)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pmr.predictive_memory_v6")
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--spec", required=True)
    s0 = sub.add_parser("s0")
    s0.add_argument("--out", required=True)
    s0.add_argument("--data")
    s0.add_argument("--cache")
    s0.add_argument("--procs", type=int)
    s0.add_argument("--skip-tests", action="store_true")
    pr = sub.add_parser("prepare")
    pr.add_argument("--data", required=True)
    pr.add_argument("--cache", required=True)
    pr.add_argument("--splits", nargs="*", default=["train", "dev"])
    pr.add_argument("--procs", type=int)
    r = sub.add_parser("run")
    r.add_argument("--stage", required=True, choices=["SMOKE", "S1", "S2", "S3", "S4", "S5", "S6", "FINAL"])
    r.add_argument("--out", required=True)
    r.add_argument("--cache")
    r.add_argument("--data")
    r.add_argument("--gpus", default="")
    r.add_argument("--procs-per-gpu", type=int, default=4)
    r.add_argument("--max-gpuh", type=float)
    r.add_argument("--prereg-commit")
    r.add_argument("--final-ack", action="store_true")
    r.add_argument("--filter")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--vsc-task", default="cambrians_vsc_10mins")
    r.add_argument("--vsc-calib-cache", default="cambrians_vsc_calib")
    r.add_argument("--vsc-gt-json")
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "job":
        spec = json.loads(Path(a.spec).read_text())
        try:
            res = run_job(spec)
        except SystemExit:
            raise
        except Exception as exc:  # recorded, then re-raised for the log
            Path(spec["out"]).mkdir(parents=True, exist_ok=True)
            (Path(spec["out"]) / "status.json").write_text(json.dumps({"status": "failed", "error": repr(exc)}))
            raise
        print(json.dumps(res, default=str)[:2000])
        return 0 if isinstance(res, dict) and res.get("status") == "complete" else 3
    if a.cmd == "s0":
        from .checks import s0 as run_s0
        rep = run_s0(a.data, a.cache, a.out, a.procs, a.skip_tests)
        print(json.dumps({"gates": rep["gates"], "all_gates_pass": rep["all_gates_pass"]}, indent=2))
        return 0 if rep["all_gates_pass"] else 4
    if a.cmd == "prepare":
        from .prepare import default_procs, prepare
        print(json.dumps(prepare(a.data, a.cache, tuple(a.splits), a.procs or default_procs()), indent=2,
                         default=str)[:4000])
        return 0
    if a.cmd == "run":
        from .experiments import run_stage
        gpus = [g for g in a.gpus.split(",") if g != ""]
        vsc = {"task": a.vsc_task, "calib_cache": a.vsc_calib_cache, "gt_json": a.vsc_gt_json}
        return run_stage(a.stage, a.out, gpus, a.procs_per_gpu, a.max_gpuh, a.cache, a.data, a.prereg_commit,
                         a.filter, a.dry_run, a.final_ack, vsc=vsc)
    if a.cmd == "report":
        from .report import write_report
        H = write_report(a.out)
        print((Path(a.out) / "report" / "REPORT.md").read_text()[:4000])
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
