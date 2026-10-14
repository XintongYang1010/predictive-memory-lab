"""Hypothesis evaluation against the pre-registration; writes report/.

  python -m pmr.predictive_memory_v6 report --out <run root>

Outputs report/REPORT.md, report/hypotheses.json and report/gates.json (read
by the runner to enforce the ladder: T0 H1 must pass before T1, T1 H1 before
T2/R/VSC).  Missing evidence is reported as ``missing``, never as pass.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from . import metrics as Me
from . import prereg

B0 = prereg.CALIB["a1_beta0"]


def load_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def _split(rows):
    by = {}
    for r in rows:
        by.setdefault(r["cond"], []).append(r)
    return by


def nu_pair(by, arm):
    return (Me.nu_c(by.get("CHANGE", []), arm), Me.nu_r(by.get("EXACT", []) + by.get("NEWPATH", []), arm))


def boot2(a, b, stat, draws=None, seed=0):
    """Stratified percentile bootstrap: resample the two row sets independently."""
    draws = draws or prereg.RULES["bootstrap_draws"]
    rng = np.random.default_rng(seed)
    point = stat(a, b)
    vals = []
    for _ in range(draws):
        aa = [a[i] for i in rng.integers(0, len(a), len(a))] if a else a
        bb = [b[i] for i in rng.integers(0, len(b), len(b))] if b else b
        v = stat(aa, bb)
        if v is not None:
            vals.append(v)
    ci = np.quantile(vals, [0.025, 0.975]).tolist() if vals else [None, None]
    return {"point": point, "ci95": ci}


def _ok(b, thr, direction=">="):
    return Me.passes(b, thr, direction)


# --------------------------------------------------------------------- H1
def h1_seed(rows, t0=False):
    by = _split(rows)
    arm = prereg.H1["applies_to"]
    pts = {b: nu_pair(by, f"A1@{b:g}") for b in B0}
    bfp = {b: nu_pair(by, f"A1BF@{b:g}") for b in B0}
    me = nu_pair(by, arm)
    res = {"arm": arm, "point": me, "A8": nu_pair(by, "A8"), "A1": pts, "A1BF": bfp}
    if None in me:
        res["pass"] = None
        return res
    res["i_not_dominated"] = not Me.dominated(me, list(pts.values()))
    pchg, pprot = by.get("CHANGE", []), by.get("EXACT", []) + by.get("NEWPATH", [])

    def margin(c, p):
        F = Me.frontier([(Me.nu_c(c, f"A1@{b:g}"), Me.nu_r(p, f"A1@{b:g}")) for b in B0])
        x, y = Me.nu_c(c, arm), Me.nu_r(p, arm)
        return None if x is None or y is None or F(x) is None else y - F(x)
    mb = boot2(pchg, pprot, margin)
    res["ii_margin"] = mb
    res["ii_pass"] = _ok(mb, prereg.H1["nuR_margin_at_matched_nuC"])
    tol = prereg.H1["a1bf_tie_tol"]
    cnt = sum(1 for b in B0 if None not in pts[b] + bfp[b]
              and bfp[b][0] >= pts[b][0] - tol and bfp[b][1] >= pts[b][1] - tol)
    res["iii_count"] = cnt
    res["iii_pass"] = cnt >= prereg.H1["a1bf_ge_a1_min_count"]
    ok = res["i_not_dominated"] and bool(res["ii_pass"]) and res["iii_pass"]
    if t0:
        bc = boot2(pchg, [], lambda c, p: Me.nu_c(c, arm))
        br = boot2([], pprot, lambda c, p: Me.nu_r(p, arm))
        res["iv_nuC"], res["iv_nuR"] = bc, br
        res["iv_pass"] = bool(_ok(bc, prereg.H1["t0_nuC_min"]) and _ok(br, prereg.H1["t0_nuR_min"]))
        ok = ok and res["iv_pass"]
    res["pass"] = bool(ok)
    return res


# --------------------------------------------------------------------- H2
def h2_seed(rows):
    by = _split(rows)
    c, p = by.get("CHANGE", []), by.get("EXACT", []) + by.get("NEWPATH", [])
    cand = {a: Me.nu_c(c, a) for a in ("A8", "A8g")}
    if any(v is None for v in cand.values()) or Me.nu_c(c, "A2") is None:
        return {"pass": None, "reason": "missing arms"}
    best = max(cand, key=cand.get)
    g1 = boot2(c, p, lambda cc, pp: _sub(Me.nu_c(cc, best), Me.nu_c(cc, "A2")))
    g2 = boot2(c, p, lambda cc, pp: _sub(Me.nu_c(cc, best), Me.nu_c(cc, "A2P")))
    g3 = boot2(c, p, lambda cc, pp: _sub(Me.nu_r(pp, best), Me.nu_r(pp, "A2P")))
    r = {"best": best, "nuC_gap_vs_A2": g1, "nuC_vs_A2P": g2, "nuR_vs_A2P": g3,
         "pass": bool(_ok(g1, prereg.H2["nuC_gap_vs_A2_min"]) and _ok(g2, prereg.H2["nuC_vs_A2plus_min_diff"])
                      and _ok(g3, prereg.H2["nuR_vs_A2plus_min_diff"]))}
    return r


def _sub(a, b):
    return None if a is None or b is None else a - b


# --------------------------------------------------------------------- H3
def h3_seed(rows):
    by = _split(rows)
    st = by.get("STATIC", [])
    if not st or "A4|clean_stat" not in st[0]:
        return {"pass": None, "reason": "missing L1 STATIC rows"}

    def flicker_ratio(rs, _):
        ce = sum(r.get("A8|clean_events", 0) for r in rs)
        cs = sum(r.get("clean_steps", 0) for r in rs)
        if cs == 0:
            return None
        thr = Me.threshold_at_rate(np.concatenate([r["A4|clean_stat"] for r in rs]), ce / cs)
        fs = sum(r.get("flk_steps", 0) for r in rs)
        if fs == 0:
            return None
        a8 = sum(r.get("A8|flk_events", 0) for r in rs) / fs
        a4 = float(np.mean(np.concatenate([np.asarray(r["A4|flk_stat"]) > thr for r in rs if r["A4|flk_stat"]])))
        return None if a4 == 0 else a8 / a4
    fr = boot2(st, [], flicker_ratio, draws=200)
    pos = by.get("CHANGE", [])
    neg = by.get("EXACT", []) + by.get("NEWPATH", [])

    def au(arm):
        return lambda p, n: Me.auroc([r.get(f"{arm}|score", np.nan) for r in p + n], [1] * len(p) + [0] * len(n))
    gap = boot2(pos, neg, lambda p, n: _sub(au("A8")(p, n), au("A4")(p, n)))
    ok_fr = fr["point"] is not None and fr["point"] <= prereg.H3["flicker_false_change_ratio_max"] \
        and fr["ci95"][1] is not None and fr["ci95"][1] < 1.0
    return {"flicker_ratio": fr, "auroc_A8": au("A8")(pos, neg), "auroc_A4": au("A4")(pos, neg),
            "auroc_gap": gap, "pass": bool(ok_fr and _ok(gap, prereg.H3["auroc_gap_vs_A4_min"]))}


# --------------------------------------------------------------------- H4
def h4_seed(rows):
    mv = _split(rows).get("MOVE", [])
    if not mv:
        return {"pass": None, "reason": "no MOVE rows"}

    def nur(rs, arm):
        return Me.nu_r(rs, arm, win="MPOST_UNCH")
    gap = boot2(mv, [], lambda a, _: _sub(nur(a, "A8"), nur(a, "A8g")))

    def rev(rs, arm):
        v = [np.mean([r[f"{arm}|reveal{o}"] for o in (1, 3, 10) if f"{arm}|reveal{o}" in r]) for r in rs
             if f"{arm}|reveal1" in r]
        return float(np.mean(v)) if v else None
    ratio = None if not rev(mv, "A8g") else rev(mv, "A8") / rev(mv, "A8g")
    nr = {a: float(np.mean([r[f"{a}|never_recovered"] for r in mv if f"{a}|never_recovered" in r] or [np.nan]))
          for a in ("A6", "A8", "A8g", "OL")}
    return {"nuR_unch_gap": gap, "reveal_ratio_A8_over_A8g": ratio, "never_recovered": nr,
            "pass": bool(_ok(gap, prereg.H4["nuR_unch_gap_vs_A8g_min"]) and ratio is not None
                         and ratio <= prereg.H4["moved_reveal_err_ratio_max"])}


# --------------------------------------------------------------------- H5
def h5_seed(probes):
    if not probes or "A8" not in probes:
        return {"pass": None, "reason": "missing probes"}
    refs = [a for a in ("A1_best", "A6", "A2") if a in probes]
    if not refs or "A0" not in probes:
        return {"pass": None, "reason": "missing reference probes"}
    best_acc = max(probes[a]["walls_acc"] for a in refs)
    best_mse = min(probes[a]["obj_mse"] for a in refs)
    out = {"refs": refs, "best_ref_walls_acc": best_acc, "best_ref_obj_mse": best_mse}
    ok = True
    for arm in ("A8", "A8g"):
        if arm not in probes:
            continue
        a, m = probes[arm]["walls_acc"], probes[arm]["obj_mse"]
        out[arm] = {"walls_acc": a, "obj_mse": m}
        ok = ok and (100 * (best_acc - a) <= prereg.H5["walls_pts_tol"]) and (m <= best_mse * (1 + prereg.H5["objmse_rel_tol"]))
    fast = [k for k in probes if k != "A0"]
    beat = all(probes[k]["walls_acc"] > probes["A0"]["walls_acc"] and probes[k]["obj_mse"] < probes["A0"]["obj_mse"]
               for k in fast)
    out["all_fast_beat_A0"] = beat
    out["pass"] = bool(ok and beat)
    return out


# ----------------------------------------------------------- aggregation
def seed_rule(results, need):
    vals = [r.get("pass") for r in results]
    known = [v for v in vals if v is not None]
    if not known:
        return "missing"
    return "pass" if sum(known) >= need else "fail"


def _find_rows(root, pattern):
    return {str(p.parent.relative_to(root)): load_rows(p) for p in sorted(Path(root).glob(pattern))}


def _probes(root, stage, tk):
    out = {}
    for p in sorted(Path(root).glob(f"{stage}/probe/{tk}/*/s*/probe.json")):
        arm, seed = p.parent.parent.name, p.parent.name
        out.setdefault(seed, {})[arm] = json.loads(p.read_text())
    return out


def hypotheses(root):
    root = Path(root)
    H, gates = {}, {}
    # T0d sanity and T0 H1
    t0 = {}
    for s in prereg.CALIB["t0_sbar_inj"]:
        rs = _find_rows(root, f"S2/eval/T0_s{s:g}/ns*/rows.jsonl.gz")
        t0[str(s)] = {k: h1_seed(v, t0=True) for k, v in rs.items()}
    per_level = {s: seed_rule(list(v.values()), prereg.RULES["t0_noise_seed_pass_min"]) for s, v in t0.items()}
    H["H1_T0"] = {"levels": per_level, "detail": t0}
    lv = list(per_level.values())
    gates["H1_T0"] = (all(v == "pass" for v in lv) if lv and "missing" not in lv else None)
    t0d = _find_rows(root, "S2/eval/T0d/rows.jsonl.gz")
    if t0d:
        by = _split(next(iter(t0d.values())))
        H["T0d_sanity"] = {a: nu_pair(by, a) for a in ("A8", "A8g", "A1@1", "A6", "OC")}
    need = prereg.RULES["phi_seed_pass_min"]
    for tk, stage in (("T1", "S3"), ("T2", "S4"), ("T3", "S5")):
        rs = _find_rows(root, f"{stage}/eval/{tk}/s*/rows.jsonl.gz")
        if not rs:
            continue
        h1 = {k: h1_seed(v) for k, v in rs.items()}
        H[f"H1_{tk}"] = {"verdict": seed_rule(list(h1.values()), need), "detail": h1}
        if tk == "T1":
            h2 = {k: h2_seed(v) for k, v in rs.items()}
            H["H2_T1"] = {"verdict": seed_rule(list(h2.values()), need), "detail": h2}
        if tk == "T2":
            h3 = {k: h3_seed(v) for k, v in rs.items()}
            h4 = {k: h4_seed(v) for k, v in rs.items()}
            H["H3_T2"] = {"verdict": seed_rule(list(h3.values()), need), "detail": h3}
            H["H4_T2"] = {"verdict": seed_rule(list(h4.values()), need), "detail": h4}
        if tk in ("T1", "T3"):
            pr = _probes(root, stage, tk)
            h5 = {s: h5_seed(v) for s, v in pr.items()}
            if h5:
                H[f"H5_{tk}"] = {"verdict": seed_rule(list(h5.values()), need), "detail": h5}
        gates[f"H1_{tk}"] = {"pass": True, "fail": False}.get(H[f"H1_{tk}"]["verdict"])
    H["diagnostics"] = diagnostics(root)
    return H, gates


def diagnostics(root):
    d = {}
    cal = {}
    for p in sorted(Path(root).glob("S*/calib/**/calib.json")):
        cal[str(p.parent.relative_to(root))] = json.loads(p.read_text())["sbar"]
    d["sbar"] = cal
    nat = cal.get("S2/calib/T0d")
    if nat is not None:
        c2 = {}
        for s in prereg.CALIB["t0_sbar_inj"]:
            got = cal.get(f"S2/calib/T0_s{s:g}")
            if got is not None:
                exp = s + (1 - s) * nat
                c2[str(s)] = {"recovered": got, "expected": exp,
                              "pass": abs(got - exp) <= prereg.DIAG["C2_sbar_tol"] + 1e-9}
        d["C2"] = c2
    fa = {}
    for p in sorted(Path(root).glob("S*/eval/**/rows.jsonl.gz")):
        rows = [r for r in load_rows(p) if r["cond"] == "STATIC"]
        if rows and "A8|fa" in rows[0]:
            fa[str(p.parent.relative_to(root))] = float(np.mean([r["A8|fa"] for r in rows]))
    d["C1_fa_per_episode"] = {k: {"value": v, "pass": v <= prereg.DIAG["C1_fa_per_episode_max"]} for k, v in fa.items()}
    dg = {}
    for p in sorted(Path(root).glob("S*/eval/**/diag.json")):
        dg[str(p.parent.relative_to(root))] = json.loads(p.read_text())
    d["eval_diag"] = dg
    dk = Path(root) / "S3/dk_select/dk.json"
    if dk.exists():
        d["dk_select"] = json.loads(dk.read_text())
    return d


def _fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def write_report(root):
    root = Path(root)
    H, gates = hypotheses(root)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hypotheses.json").write_text(json.dumps(H, indent=2, default=str))
    (out / "gates.json").write_text(json.dumps(gates, indent=2))
    L = ["# V6.1 BF-Delta — automatic report", "",
         "Verdicts follow docs/PREREGISTRATION_V6.1.md; `missing` means no evidence, not a pass.", ""]
    L.append("| Hypothesis | Verdict |")
    L.append("|---|---|")
    if "H1_T0" in H:
        L.append(f"| H1 on T0 (oracle keys, per noise level) | {H['H1_T0']['levels']} |")
    for k in ("H1_T1", "H2_T1", "H5_T1", "H1_T2", "H3_T2", "H4_T2", "H1_T3", "H5_T3"):
        if k in H:
            L.append(f"| {k} | {H[k]['verdict']} |")
    L += ["", "## Stability-plasticity points (nu_C, nu_R) per seed", ""]
    for k in ("H1_T1", "H1_T2", "H1_T3"):
        if k not in H:
            continue
        L.append(f"### {k}")
        for s, r in H[k]["detail"].items():
            pts = " ".join(f"A1@{b:g}=({_fmt(v[0])},{_fmt(v[1])})" for b, v in r.get("A1", {}).items())
            L.append(f"- {s}: {r.get('arm')}=({_fmt(r['point'][0])},{_fmt(r['point'][1])}), "
                     f"A8=({_fmt(r['A8'][0])},{_fmt(r['A8'][1])}); {pts}; pass={r.get('pass')}")
        L.append("")
    if "T0d_sanity" in H:
        L += ["## T0d (deterministic) sanity", "", json.dumps(H["T0d_sanity"]), ""]
    L += ["## Gates for the runner", "", "```json", json.dumps(gates, indent=2), "```", ""]
    (out / "REPORT.md").write_text("\n".join(L))
    return H
