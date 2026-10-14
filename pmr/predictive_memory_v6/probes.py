"""H5: official-style Memory Maze probes reading only the memory state S.

Probe at frame t reads the pre-write state of transition t (everything
written up to and including frame t) plus map coordinates of frame t.
Walls: BCE on the full layout; objects: MSE on the egocentric target vectors.
Training uses train-split episodes only; scoring uses dev steps 500..999.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import bf
from . import data as Dm
from . import maze as M
from . import metrics as Me
from . import models as Mo
from . import pipeline as P
from . import prereg


def _resolve_arm(spec):
    arm = spec["arm"]
    if arm == "A1_best":
        summ = P.load_json(spec["select_from"])["summary"]["STATIC"]
        cands = {k.split("|")[0]: v for k, v in summ.items() if k.startswith("A1@") and k.endswith("|STAT")}
        name = min(cands, key=cands.get)
        return "A1", float(name.split("@")[1]), name
    if "@" in arm:
        a, p = arm.split("@")
        return a, float(p), arm
    return arm, None, arm


def _features(st, bt, arm, p, K, R, sbar, pi, gates, steps, kind, S_maze):
    B = R.shape[0]
    if arm == "A0":
        feats = {t: torch.cat([bt["Y"][:, t - 1] * bt["O"][:, t - 1], bt["O"][:, t - 1]], -1) for t in steps}
    else:
        if arm in ("A2", "A2P", "A3", "A4"):
            scos = P.slow_surprise(bt, gates["_slow"]) if arm == "A3" else None
            res = P.run_arm(arm, p, K, R, bt["O"], bt["W"], sbar, pi, gates, Z=bt["Z"], slow_cos=scos,
                            state_at=steps)
        else:
            res = P.run_arm(arm, p, K, R, bt["O"], bt["W"], sbar, pi, gates, state_at=steps)
        feats = {t: res["states"][t].flatten(1) for t in steps}
    pos = torch.as_tensor(st["pos"], device=R.device, dtype=torch.float32)
    dr = torch.as_tensor(st["dir"], device=R.device, dtype=torch.float32)
    out = []
    for t in steps:
        mc = torch.cat([pos[:, t - 1] / S_maze * 2 - 1, dr[:, t - 1]], -1)
        out.append(torch.cat([feats[t], mc], -1))
    return out


def probe(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    seed = spec.get("seed", 0)
    P.seed_all(seed)
    cfg = dict(prereg.TRAIN["probe"], **spec.get("override", {}))
    timer = P.Timer(spec.get("max_seconds"))
    track = spec["track"]
    kind = P.track_kind(track)
    tr_store, dv_store = Dm.Store(spec["cache"], "train"), Dm.Store(spec["cache"], spec.get("split", "dev"))
    slow, sx = P.load_slow(spec["slow"], dev)
    sigma_vec = None if sx.get("sigma_vec") is None else np.asarray(sx["sigma_vec"], dtype=np.float32)
    keynet = Mo.load(spec["phi"], dev)[0] if spec.get("phi") else None
    dko = None if keynet is not None else P.oracle_dk(tr_store)
    sbar = float(P.load_json(spec["calib"])["sbar"])
    pi = prereg.CALIB["pi"]
    s_inj = float(track.get("sbar_inj", 0.0))
    gates = {a: Mo.load(pth, dev)[0] for a, pth in (spec.get("gates") or {}).items() if pth}
    gates["_slow"] = slow
    arm, p, name = _resolve_arm(spec)
    S_maze, G = tr_store.S, tr_store.G
    rng = np.random.default_rng(seed)

    def collect(store, n_eps, steps_fn, salt):
        X, YW, YO, SEEN, VIS = [], [], [], [], []
        idx = list(store.order[:n_eps])
        for b0 in range(0, len(idx), 32):
            items = [Dm.build_item(store, {"cond": "STATIC", "A": int(i)}, {"kind": kind}, rng_salt=salt,
                                   sigma_vec=sigma_vec, sbar_inj=s_inj, want_revisits=False)
                     for i in idx[b0:b0 + 32]]
            st = Dm.stack_items(items)
            bt = P.batch_tensors(st, dev)
            with torch.no_grad():
                R = P.residuals(bt, slow, s_inj)
                K = P.keys(bt, keynet, dko)
                steps = steps_fn()
                feats = _features(st, bt, arm, p, K, R, sbar, pi, gates, steps, kind, S_maze)
            lay = np.stack([store.episode(int(i))["layout"] for i in idx[b0:b0 + 32]])
            for t, f in zip(steps, feats):
                X.append(f.half())
                YW.append(torch.as_tensor(lay.reshape(len(lay), -1), device=dev, dtype=torch.float32))
                YO.append(torch.as_tensor(st["tvec"][:, t - 1].reshape(len(lay), -1), device=dev,
                                          dtype=torch.float32))
            if salt == "probe-dev":
                for j in range(len(lay)):
                    gm = st["o"][j, :, :M.N_GRID]
                    sm = M.seen_maps(st["pos"][j], st["dir"][j], gm, S_maze)
                    SEEN.append(np.stack([sm[t - 1].ravel() for t in steps]))
                    VIS.append(np.stack([M.visage(st["vis"][j])[t - 1] for t in steps]))
            if timer.over():
                break
        return torch.cat(X), torch.cat(YW), torch.cat(YO), SEEN, VIS

    lo = 500
    Xtr, YWtr, YOtr, _, _ = collect(tr_store, cfg["train_episodes"],
                                    lambda: sorted(rng.choice(np.arange(lo, 1000), cfg["states_per_episode"],
                                                              replace=False).tolist()), "probe-train")
    mu, sd = Xtr.float().mean(0), Xtr.float().std(0).clamp_min(1e-4)
    pw = Mo.Probe(Xtr.shape[1], YWtr.shape[1], cfg["layers"], cfg["hidden"]).to(dev)
    po = Mo.Probe(Xtr.shape[1], YOtr.shape[1], cfg["layers"], cfg["hidden"]).to(dev)
    opt = torch.optim.Adam(list(pw.parameters()) + list(po.parameters()), lr=cfg["lr"], eps=cfg["eps"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(spec.get("steps", cfg["steps"])):
        i = torch.randint(0, Xtr.shape[0], (cfg["batch"],), generator=g).to(dev)
        x = (Xtr[i].float() - mu) / sd
        loss = F.binary_cross_entropy_with_logits(pw(x), YWtr[i]) + F.mse_loss(po(x), YOtr[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if timer.over():
            break
    eval_steps = list(range(lo, 1000, cfg["eval_stride"]))
    Xd, YWd, YOd, SEEN, VIS = collect(dv_store, cfg["eval_episodes"], lambda: eval_steps, "probe-dev")
    with torch.no_grad():
        x = (Xd.float() - mu) / sd
        lw = pw(x).cpu().numpy()
        lo_ = po(x).cpu().numpy()
    YWn, YOn = YWd.cpu().numpy(), YOd.cpu().numpy()
    # rows are ordered step-major within each batch of 32 episodes; regroup per episode
    n_ep = len(SEEN)
    nst = len(eval_steps)
    per_ep = []
    k = 0
    b_sizes = [min(32, n_ep - b0) for b0 in range(0, n_ep, 32)]
    for bsz in b_sizes:
        for ti in range(nst):
            for j in range(bsz):
                per_ep.append((len(per_ep) // 1, k + j, ti))
        k += bsz
    rows = {}
    for r_i, (_, ep_j, ti) in enumerate(per_ep):
        acc = (lw[r_i] > 0) == (YWn[r_i] > 0.5)
        seen = SEEN[ep_j][ti]
        obj = ((lo_[r_i] - YOn[r_i]) ** 2).reshape(G, 2).sum(-1)
        vis_age = VIS[ep_j][ti]
        d = rows.setdefault(ep_j, {"acc_all": [], "acc_seen_n": 0, "acc_seen_k": 0, "obj": [], "obj_bins": {}})
        d["acc_all"].append(acc.mean())
        d["acc_seen_k"] += int(acc[seen].sum())
        d["acc_seen_n"] += int(seen.sum())
        d["obj"].append(obj.mean())
        for gg in range(G):
            b = int(Me.visage_bins(np.array([vis_age[gg]]))[0])
            d["obj_bins"].setdefault(str(b), []).append(float(obj[gg]))
    ep_rows = []
    for ep_j, d in sorted(rows.items()):
        ep_rows.append({"ep": ep_j, "walls_acc": float(np.mean(d["acc_all"])),
                        "walls_acc_seen": d["acc_seen_k"] / max(1, d["acc_seen_n"]),
                        "obj_mse": float(np.mean(d["obj"])),
                        "obj_mse_bins": {b: float(np.mean(v)) for b, v in d["obj_bins"].items()}})
    res = {"arm": name, "track": track, "seed": seed, "n_eval_episodes": len(ep_rows),
           "walls_acc": float(np.mean([r["walls_acc"] for r in ep_rows])),
           "walls_acc_seen": float(np.mean([r["walls_acc_seen"] for r in ep_rows])),
           "obj_mse": float(np.mean([r["obj_mse"] for r in ep_rows])), "episodes": ep_rows}
    P.write_json(out / "probe.json", res)
    st = "complete" if not timer.over() else "partial"
    P.write_json(out / "status.json", {"status": st, "arm": name, "walls_acc": res["walls_acc"],
                                       "obj_mse": res["obj_mse"]})
    return {"status": st}
