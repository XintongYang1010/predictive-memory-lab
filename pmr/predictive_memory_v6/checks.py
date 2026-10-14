"""S0: CPU gates before any GPU stage (unit tests, data checks, token cache)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import prepare, prereg


def s0(data=None, cache=None, out="runs/v6_1", procs=None, skip_tests=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"gates": {}}
    if not skip_tests:
        root = Path(__file__).resolve().parents[2]
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/predictive_memory_v6", "-t", "."],
                           cwd=root, capture_output=True, text=True)
        rep["unit_tests_tail"] = (r.stdout + r.stderr)[-3000:]
        rep["gates"]["unit_tests"] = r.returncode == 0
    if data:
        files = prepare.discover(data)
        rep["counts"] = {s: sum(f["split"] == s for f in files) for s in ("train", "dev", "test")}
        res = prepare.prepare(data, cache, splits=("train", "dev"), procs=procs or prepare.default_procs())
        cal = json.loads((Path(cache) / "calibration.json").read_text())
        rep["calibration"] = cal
        rep["prepare"] = res
        checks = cal["data_checks"]
        rep["gates"].update({
            "axis_convention_fit": bool(cal["axis_ok"]),
            "visibility_calibration": bool(cal["visibility_ok"]),
            "npz_len_1001": all(c["len_1001"] for c in checks),
            "layout_and_targets_static": all(c["layout_static"] and c["targets_static"] for c in checks),
            "agent_on_corridor": all(c["agent_on_corridor_frac"] >= 0.99 for c in checks),
            "forward_action_moves_forward": all(c["forward_moves_along_dir"] > 0 for c in checks),
            "cache_complete": all(res[s].get("n_ok", 0) == res[s].get("n", -1) for s in ("train", "dev")),
            "dev_size_ok": res["dev"].get("n", 0) >= 500,
        })
    rep["all_gates_pass"] = all(rep["gates"].values()) if rep["gates"] else False
    rep["thresholds"] = {"vis_agreement_min": prereg.GATES["s0_vis_agreement_min"],
                         "axis_fit_max_err_cells": prereg.GATES["axis_fit_max_err_cells"]}
    (out / "S0.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep
