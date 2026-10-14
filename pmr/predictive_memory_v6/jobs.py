"""Small job types: dk selection, fake data, test preparation, VSC wrappers."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from . import prereg


def _status(out, status, **kw):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "status.json").write_text(json.dumps({"status": status, **kw}, indent=2, default=str))
    return {"status": status, **kw}


def dk_select(spec):
    """Capacity rule: smallest dk whose A6 gain over A0 on dev STAT reaches
    80% of the oracle-key A6 gain, all on the same items and noise."""
    from . import evaluate as E
    out = Path(spec["out"])
    gains = {}

    def gain(phi, tag):
        sub = dict(spec, out=str(out / tag), phi=phi, conditions=["STATIC"], arms=["A0", "A6"],
                   n_items=spec.get("n_items", 200), memory_gain=False)
        E.evaluate(sub)
        s = json.loads((out / tag / "summary.json").read_text())["summary"]["STATIC"]
        return s["A0|STAT"] - s["A6|STAT"]
    oracle = gain(None, "oracle")
    for dk in spec["dks"]:
        gains[int(dk)] = gain(spec["phis"][str(dk)], f"dk{dk}")
    frac = prereg.CALIB["dk_frac_of_oracle_gain"]
    ok = [dk for dk in sorted(gains) if gains[dk] >= frac * oracle]
    chosen = ok[0] if ok else max(gains)
    res = {"dk": chosen, "gains": gains, "oracle_gain": oracle, "rule_met": bool(ok),
           "note": None if ok else "capacity rule unmet (F9): largest dk used; one diagnostic round allowed"}
    (out / "dk.json").write_text(json.dumps(res, indent=2))
    return _status(out, "complete", **res)


def fake_data(spec):
    from . import fake_maze, prepare
    out = Path(spec["out"])
    data = Path(spec["data"])
    fake_maze.write_fake_dataset(data, n_train=spec.get("n_train", 24), n_eval=spec.get("n_eval", 4))
    res = prepare.prepare(data, spec["cache"], splits=("train", "dev"), procs=2)
    # the fake set is tiny: make sure dev exists even if no basename hashes into it
    if res.get("dev", {}).get("n", 0) < 4:
        files = prepare.discover(data)
        rows = [r for r in files if r["split"] == "train"][-4:]
        cal = json.loads((Path(spec["cache"]) / "calibration.json").read_text())
        import shutil
        shutil.rmtree(Path(spec["cache"]) / "dev", ignore_errors=True)
        prepare.build_split(rows, Path(spec["cache"]) / "dev", cal, procs=2)
    return _status(out, "complete", prepare=res)


def prepare_test(spec):
    if not spec.get("open_test"):
        raise SystemExit("prepare_test is FINAL-only")
    from . import prepare
    res = prepare.prepare(spec["data"], spec["cache"], splits=("test",), procs=prepare.default_procs())
    return _status(spec["out"], "complete", prepare=res)


def vsc_infer(spec):
    """One Cambrian-S-7B-LFP VSC pass with the logging patch (separate env)."""
    root = Path(__file__).resolve().parents[2]
    dump = Path(spec["dump"])
    dump.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PMR_VSC_DUMP=str(dump), CAMBRIANS_VSC_CACHE_NAME=spec["cache_name"])
    cmd = ["bash", str(root / "scripts" / "vsc" / "run_vsc.sh"), spec["task"]]
    log = open(Path(spec["out"]) / "vsc.log", "a")
    code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    n = len(list(dump.glob("*.npz")))
    return _status(spec["out"], "complete" if code == 0 and n > 0 else "failed", exit_code=code, dumps=n)


def _sha(name):
    import hashlib
    return int(hashlib.sha1(name.encode()).hexdigest(), 16)


def vsc_calib(spec):
    """Calibrate on the designated calibration videos only (the n_calib lowest
    sha1 of the video key); every other base dump stays unread until FINAL."""
    from . import vsc
    files = sorted(Path(spec["dumps"]).glob("*.npz"), key=lambda p: _sha(p.stem))
    names = [p.stem for p in files[: spec.get("n_calib", 10)]]
    dumps = [vsc.load_dump(Path(spec["dumps"]) / f"{n}.npz") for n in names]
    cal = vsc.calibrate(dumps)
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    vsc.save_cal(cal, out / "vsc_cal.npz")
    (out / "calib_names.json").write_text(json.dumps(names))
    return _status(out, "complete", sbar=cal["sbar"], key_overlap=cal["key_overlap"],
                   use_ell=cal["use_ell"], thr=cal["thr"], n_calib=len(names))


def vsc_eval(spec):
    """FINAL only: score every non-calibration video on base and all variants."""
    from . import metrics as Me
    from . import vsc
    if not spec.get("open_test"):
        raise SystemExit("vsc_eval is FINAL-only")
    cal = vsc.load_cal(spec["cal"])
    skip = set(json.loads(Path(spec["calib_names"]).read_text()))
    gts = json.loads(Path(spec["gt_json"]).read_text())
    out = {}
    base_len = {}
    for var in ("base",) + tuple(k for k in spec["dumps"] if k != "base"):
        files = [p for p in sorted(Path(spec["dumps"][var]).glob("*.npz")) if p.stem not in skip]
        dumps = [vsc.load_dump(p) for p in files]
        if var == "base":
            base_len = {d["name"]: len(d["s"]) for d in dumps}
        g = {d["name"]: float(gts.get(d["name"], np.nan)) for d in dumps}
        rows = vsc.evaluate_variant(dumps, g, cal, base_len=None if var == "base" else base_len)
        out[var] = {"n": len(rows), "rows": rows,
                    "mra": {a: Me.mean_mra([r[f"{a}|pred"] for r in rows], [r["gt"] for r in rows])
                            for a in vsc.ARMS},
                    "median_pred_over_gt": {a: float(np.median([r[f"{a}|pred"] / r["gt"] for r in rows if r["gt"]]))
                                            for a in vsc.ARMS},
                    "veto_rate": {a: float(np.mean([r[f"{a}|veto"] for r in rows])) for a in vsc.ARMS}}
    Path(spec["out"]).mkdir(parents=True, exist_ok=True)
    (Path(spec["out"]) / "vsc_results.json").write_text(json.dumps(out, indent=2, default=float))
    return _status(spec["out"], "complete", mra={k: v["mra"] for k, v in out.items()})


def vsc_synthetic(spec):
    """SMOKE: synthetic logs with a repeated second half exercise the whole
    offline VSC path (calibration, familiarity, aggregation, MRA)."""
    from . import vsc
    rng = np.random.default_rng(0)
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)

    def stream(T, rooms, repeat=1):
        protos = rng.standard_normal((rooms, 1152))
        seq, segs, counts = [], [], []
        for _ in range(repeat):
            for rm in range(rooms):
                n = int(rng.integers(150, 200))
                base = len(seq)
                for _t in range(n):
                    seq.append(protos[rm] + 0.3 * rng.standard_normal(1152))
                segs.append((base, base + n))
                counts.append(float(rng.integers(1, 4)))
        d = np.array(seq, dtype=np.float32)
        p = np.vstack([np.full((1, 1152), np.nan), d[:-1] + 0.1 * rng.standard_normal((len(d) - 1, 1152))])
        return {"s": np.zeros(len(d)), "d_pool": d, "p_pool": p.astype(np.float32),
                "seg_frames": np.array(segs), "seg_counts": np.array(counts)}
    for i in range(3):
        np.savez(out / f"calib_{i}.npz", **stream(0, 3))
    np.savez(out / "rep_0.npz", **stream(0, 3, repeat=2))
    cal = vsc.calibrate([vsc.load_dump(p) for p in sorted(out.glob("calib_*.npz"))])
    d = vsc.load_dump(out / "rep_0.npz")
    rows = vsc.evaluate_variant([d], {d["name"]: 6.0}, cal, base_len={d["name"]: len(d["s"]) // 2})
    return _status(out, "complete", row=rows[0])
