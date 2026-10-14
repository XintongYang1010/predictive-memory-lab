"""Evaluation job: every arm on the same residuals, keys and state budget.

Writes ``rows.jsonl`` (one row per item with every arm's window sums and
scores; rows are the bootstrap unit), ``summary.json`` and ``diag.json``.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from . import data as Dm
from . import maze as M
from . import metrics as Me
from . import models as Mo
from . import pipeline as P
from . import prereg

BF_ARMS = ("A7", "A8", "A8g")


def _load_gates(spec, dev):
    gates = {}
    for arm, path in (spec.get("gates") or {}).items():
        if path and Path(path).exists():
            g, _ = Mo.load(path, dev)
            gates[arm] = g
    return gates


def _windows(meta, T, cond):
    """Transition-index masks [T] per window name."""
    w = {}
    ts = meta.get("ts")
    if ts is not None:
        w["POST"] = Me.window_mask(T, ts + 1, ts + 1 + prereg.DATA["post_len"])
        w["PRE"] = Me.window_mask(T, 0, ts)
        w["DET"] = Me.window_mask(T, ts + 1, ts + 21)
    if cond == "CONT":
        t0 = prereg.DATA["splice_ts"]
        w["POST"] = Me.window_mask(T, t0, t0 + prereg.DATA["post_len"])
    if cond == "STATIC":
        lo, hi = prereg.DATA["stat_window"]
        w["STAT"] = Me.window_mask(T, lo - 1, hi)
        rev = np.zeros(T, dtype=bool)
        for (j, _lf, _v) in meta.get("revisits", []):
            rev[max(0, j - 1):min(T, j - 1 + prereg.DATA["rev_window"])] = True
        w["REV"] = rev
    if cond == "MOVE" and meta.get("tr") is not None:
        tr = meta["tr"]
        w["MPOST"] = Me.window_mask(T, tr, tr + prereg.DATA["post_len"])
    return w


def _item_sums(r, m, o, wr, windows, rowmask=None):
    out = {}
    oo = o if rowmask is None else o * rowmask[None, :]
    for name, mask in windows.items():
        out[name] = Me.mpe_sums(r, m, oo, wr, mask)
    return out


def _score(arm, res, R, O, Dt, scos, alpha_a2):
    """Per-step change score used for detection AUROC (higher = change)."""
    if arm in BF_ARMS or arm.startswith("A1BF"):
        return res["ell"]
    if arm == "A3":
        return scos
    if arm == "A2":
        return 1.0 - alpha_a2
    if arm == "A2P" and "alpha" in res:
        return 1.0 - res["alpha"]
    e = (R - res["m"]) * O
    return (e * e).sum(-1) / Dt


def evaluate(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    P.seed_all(0)
    timer = P.Timer(spec.get("max_seconds"))
    split = spec.get("split", "dev")
    if split == "test" and not spec.get("open_test"):
        raise SystemExit("test split requires FINAL with --final-ack")
    track = spec["track"]
    kind = P.track_kind(track)
    store = Dm.Store(spec["cache"], split)
    slow, sx = P.load_slow(spec["slow"], dev)
    sigma_vec = None if sx.get("sigma_vec") is None else np.asarray(sx["sigma_vec"], dtype=np.float32)
    keynet = Mo.load(spec["phi"], dev)[0] if spec.get("phi") else None
    dko = None if keynet is not None else P.oracle_dk(store)
    sbar = float(P.load_json(spec["calib"])["sbar"]) if spec.get("calib") else float(spec["sbar"])
    pi = float(spec.get("pi", prereg.CALIB["pi"]))
    s_inj = float(track.get("sbar_inj", 0.0))
    gates = _load_gates(spec, dev)
    arms = [(a, p) for (a, p) in P.arm_list(track) if (a not in ("A2", "A2P", "A3", "A4") or a in gates)]
    if spec.get("arms"):
        arms = [(a, p) for (a, p) in arms if P.arm_name(a, p) in spec["arms"] or a in spec["arms"]]
    conds = spec.get("conditions", ["STATIC", "CONT", "EXACT", "NEWPATH", "CHANGE"])
    n_items = spec.get("n_items", prereg.DATA["dev_items_per_condition"])
    salt = f"{split}/ns{spec.get('noise_seed', 0)}"
    bs = spec.get("batch", 100)
    rows, diag = [], {"C4_soft": {}, "C5": {}, "keys": {}}
    sig = slow.sigma
    for cond in conds:
        specs = Dm.plan(store, cond, n_items, salt=split)
        for b0 in range(0, len(specs), bs):
            items = [Dm.build_item(store, s, {"kind": kind}, rng_salt=salt, sigma_vec=sigma_vec, sbar_inj=s_inj)
                     for s in specs[b0:b0 + bs]]
            st = Dm.stack_items(items)
            bt = P.batch_tensors(st, dev)
            with torch.no_grad():
                R = P.residuals(bt, slow, s_inj)
                K = P.keys(bt, keynet, dko)
                yhat = slow(bt["Z"])
                scos = P.slow_surprise(bt, slow)
                Dt = bt["O"].sum(-1).clamp_min(1.0)
                a2_alpha = gates["A2"](bt["Z"])[..., 0] if "A2" in gates else None
            B, T, D = R.shape
            metas = st["meta"]
            reset = torch.zeros(B, T, device=dev)
            for i, mt in enumerate(metas):
                if mt.get("ts") is not None:
                    reset[i, mt["ts"]] = 1.0
                if cond == "MOVE" and mt.get("tr") is not None:
                    reset[i, mt["tr"] - 1] = 1.0
            Rn, On, Wn = R.cpu().numpy(), bt["O"].cpu().numpy(), bt["W"].cpu().numpy()
            brows = [{"uid": mt["uid"], "cond": cond, "A": mt["A"], "B": mt.get("B")} for mt in metas]
            wins = [_windows(mt, T, cond) for mt in metas]
            a6_m = None
            for (arm, p) in arms:
                name = P.arm_name(arm, p)
                with torch.no_grad():
                    res = P.run_arm(arm, p, K, R, bt["O"], bt["W"], sbar, pi, gates, Z=bt["Z"], slow_cos=scos,
                                    reset=reset if arm == "OC" else None)
                m = res["m"]
                if arm == "A6":
                    a6_m = m
                mn = m.cpu().numpy()
                score = _score(arm, res, R, bt["O"], Dt, scos, a2_alpha)
                score = score.cpu().numpy() if torch.is_tensor(score) else None
                g = res.get("g")
                gn = g.cpu().numpy() if (g is not None and arm in BF_ARMS + ("A1BF",)) else None
                for i, mt in enumerate(metas):
                    row = brows[i]
                    rowmask = None
                    sums = _item_sums(Rn[i], mn[i], On[i], Wn[i], wins[i])
                    if cond == "MOVE" and mt.get("tr") is not None:
                        rm = np.ones(D, dtype=np.float32)
                        rm[M.object_slots(store.G)[mt["g"]]] = 0.0
                        un = _item_sums(Rn[i], mn[i], On[i], Wn[i], {"MPOST_UNCH": wins[i]["MPOST"]}, rm)
                        sums.update(un)
                        rowmask = rm
                    for wname, (num, den) in sums.items():
                        row[f"{name}|{wname}|num"] = num
                        row[f"{name}|{wname}|den"] = den
                    if score is not None and "DET" in wins[i]:
                        row[f"{name}|score"] = float(np.mean(score[i][wins[i]["DET"]]))
                    if gn is not None and cond == "STATIC":
                        wr = Wn[i] > 0
                        row[f"{name}|fa"] = int((gn[i][wr] > 0.5).sum())
                        soft = (gn[i][wr] > prereg.DIAG["C4_soft_band"][0]) & (gn[i][wr] < prereg.DIAG["C4_soft_band"][1])
                        row[f"{name}|soft"] = float(soft.mean())
                    if cond == "MOVE" and mt.get("tr") is not None:
                        _reveal(row, name, mt, yhat[i], sig, m[i], st["tvec"][i], store.G, T)
                    if kind == "l1" and cond == "STATIC":
                        _flicker(row, name, mt, st, i, gn, score)
                if arm == "OC" and cond == "MOVE":
                    # Oracle L: moved rows from the reset run, other rows from A6.
                    ol = a6_m.clone() if a6_m is not None else m.clone()
                    for i, mt in enumerate(metas):
                        if mt.get("tr") is None:
                            continue
                        sl = torch.as_tensor(M.object_slots(store.G)[mt["g"]], device=ol.device)
                        t0 = mt["tr"] - 1
                        ol[i, t0:, sl] = m[i, t0:, sl]
                    oln = ol.cpu().numpy()
                    for i, mt in enumerate(metas):
                        if mt.get("tr") is None:
                            continue
                        rm = np.ones(D, dtype=np.float32)
                        rm[M.object_slots(store.G)[mt["g"]]] = 0.0
                        s2 = _item_sums(Rn[i], oln[i], On[i], Wn[i], {"MPOST": wins[i]["MPOST"]})
                        s2.update(_item_sums(Rn[i], oln[i], On[i], Wn[i], {"MPOST_UNCH": wins[i]["MPOST"]}, rm))
                        for wname, (num, den) in s2.items():
                            brows[i][f"OL|{wname}|num"] = num
                            brows[i][f"OL|{wname}|den"] = den
                        _reveal(brows[i], "OL", mt, yhat[i], sig, ol[i], st["tvec"][i], store.G, T)
                if arm in ("A6", "A8", "A8g") and b0 == 0:
                    diag["C5"].setdefault(cond, {})[name] = _c5(res)
                if gn is not None:
                    wr = Wn > 0
                    band = prereg.DIAG["C4_soft_band"]
                    diag["C4_soft"].setdefault(cond, {}).setdefault(name, []).append(
                        float(((gn > band[0]) & (gn < band[1]))[wr].mean()))
            if cond == "STATIC" and b0 == 0:
                diag["keys"] = _key_overlap(K, st, store)
            if cond == "STATIC" and spec.get("memory_gain", True):
                _memory_gain(brows, metas, K, R, bt, sbar, pi)
            rows.extend(brows)
            if timer.over():
                break
        if timer.over():
            break
    with gzip.open(out / "rows.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = summarize(rows, [P.arm_name(a, p) for a, p in arms])
    P.write_json(out / "summary.json", {"track": track, "split": split, "sbar": sbar, "pi": pi,
                                        "n_rows": len(rows), "summary": summ})
    for c in diag["C4_soft"].values():
        for k in list(c):
            c[k] = float(np.mean(c[k]))
    P.write_json(out / "diag.json", diag)
    st = "complete" if not timer.over() else "partial"
    P.write_json(out / "status.json", {"status": st, "rows": len(rows), "seconds": timer.elapsed()})
    return {"status": st, "rows": len(rows)}


def _reveal(row, name, mt, yhat_i, sig, m_i, tvec_i, G, T):
    g, tr = mt["g"], mt["tr"]
    sl = [int(x) for x in M.object_slots(G)[g][1:]]
    errs = {}
    for off in (0, 1, 3, 10):
        t = tr - 1 + off
        if 0 <= t < T:
            pred = (yhat_i[t, sl] + sig[sl] * m_i[t, sl]).detach().cpu().numpy()
            errs[off] = float(np.linalg.norm(pred - tvec_i[t, g]))
    for off, v in errs.items():
        row[f"{name}|reveal{off}"] = v
    if 10 in errs:
        row[f"{name}|never_recovered"] = int(errs[10] > 1.0)


def _flicker(row, name, mt, st, i, gn, score):
    f = mt.get("flicker")
    if f is None or "shown" not in st:
        return
    vis = st["vis"][i][:, f] & st["shown"][i][:, f]
    row["flk_steps"] = int(vis.sum())
    if gn is not None:
        row[f"{name}|flk_events"] = int((gn[i][vis] > 0.5).sum())
        clean = ~vis & (st["w"][i] > 0)
        row[f"{name}|clean_events"] = int((gn[i][clean] > 0.5).sum())
        row["clean_steps"] = int(clean.sum())
    if name == "A4" and score is not None:
        row["A4|flk_stat"] = score[i][vis].tolist()
        clean = ~vis & (st["w"][i] > 0)
        row["A4|clean_stat"] = np.round(score[i][clean], 5).tolist()


def _c5(res):
    S = res["S_final"]
    q = torch.nn.functional.normalize(torch.randn(S.shape[0], S.shape[-1], device=S.device), dim=-1)
    out = torch.einsum("bdk,bk->bd", S, q).norm(dim=-1)
    return {"S_fro_mean": float(S.flatten(1).norm(dim=-1).mean()), "rand_query_out_mean": float(out.mean())}


def _key_overlap(K, st, store, n_pairs=4000):
    rng = np.random.default_rng(0)
    Kn = K.detach().cpu().numpy()
    B, T, dk = Kn.shape
    pos = st["pos"]
    vals = []
    for _ in range(n_pairs):
        b = rng.integers(B)
        t, u = rng.integers(T, size=2)
        if abs(int(t) - int(u)) < 50:
            continue
        if (np.floor(pos[b, t]) == np.floor(pos[b, u])).all():
            continue
        vals.append(float(abs(Kn[b, t] @ Kn[b, u])))
    v = np.array(vals) if vals else np.zeros(1)
    return {"E_abs_kk": float(v.mean()), "aliasing_frac": float((v > prereg.DIAG["aliasing_cos"]).mean()),
            "n_pairs": len(vals)}


def _memory_gain(brows, metas, K, R, bt, sbar, pi):
    """REV windows: full memory vs memory cleared when the agent left the
    place (R2M-Bench pair).  A6 and A8 only (descriptive)."""
    from . import bf
    ev = []
    for i, mt in enumerate(metas):
        for (j, lf, _v) in mt.get("revisits", []):
            if j + 9 < R.shape[1]:
                ev.append((i, lf, j))
    if not ev:
        return
    L = max(j + 9 - lf for (_, lf, j) in ev)  # fresh run starts at transition lf (left the place)
    for arm in ("A6", "A8"):
        for c0 in range(0, len(ev), 256):
            chunk = ev[c0:c0 + 256]
            n = len(chunk)
            Kc = torch.zeros(n, L, K.shape[-1], device=K.device)
            Rc = torch.zeros(n, L, R.shape[-1], device=R.device)
            Oc = torch.zeros_like(Rc)
            Wc = torch.zeros(n, L, device=R.device)
            for q, (i, lf, j) in enumerate(chunk):
                a, b = lf, j + 9
                Kc[q, :b - a] = K[i, a:b]
                Rc[q, :b - a] = R[i, a:b]
                Oc[q, :b - a] = bt["O"][i, a:b]
                Wc[q, :b - a] = bt["W"][i, a:b]
            with torch.no_grad():
                res = bf.scan(arm, Kc, Rc, Oc, Wc, sbar=sbar, pi=pi, collect=("m",))
            m = res["m"]
            for q, (i, lf, j) in enumerate(chunk):
                a = lf
                t0, t1 = j - 1 - a, j - 1 - a + prereg.DATA["rev_window"]
                e = ((Rc[q, t0:t1] - m[q, t0:t1]) * Oc[q, t0:t1]) ** 2
                row = brows[i]
                row[f"{arm}|REVCLR|num"] = row.get(f"{arm}|REVCLR|num", 0.0) + float(e.sum())
                row[f"{arm}|REVCLR|den"] = row.get(f"{arm}|REVCLR|den", 0.0) + float(Oc[q, t0:t1].sum())


def summarize(rows, arms):
    out = {}
    conds = sorted({r["cond"] for r in rows})
    for c in conds:
        rs = [r for r in rows if r["cond"] == c]
        d = {}
        for a in arms + ["OL"]:
            for w in ("POST", "PRE", "STAT", "REV", "REVCLR", "MPOST", "MPOST_UNCH"):
                v = Me.ratio(rs, f"{a}|{w}")
                if v is not None:
                    d[f"{a}|{w}"] = v
        out[c] = d
    return out
