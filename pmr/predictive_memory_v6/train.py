"""Fitting jobs: f_theta (slow), ML-II calibration, phi under A6, learned gates.

Every job writes ``status.json`` (complete / partial / failed) and never
reports a falling loss as scientific evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import bf
from . import data as Dm
from . import models as Mo
from . import pipeline as P
from . import prereg


def _stores(spec):
    return Dm.Store(spec["cache"], "train"), Dm.Store(spec["cache"], "dev")


def _sigma_vec(spec, store):
    """L1 object-vector noise s.d. [G,2] from the clean T1 slow model."""
    if not spec.get("sigma_from"):
        return None
    ck = torch.load(spec["sigma_from"], map_location="cpu", weights_only=False)
    sig = ck["state"]["sigma"].numpy()
    G = store.G
    frac = prereg.L1["obs_noise_frac"]
    out = np.zeros((G, 2), dtype=np.float32)
    for g in range(G):
        b = 25 + 3 * g
        out[g] = sig[b + 1:b + 3] * math.sqrt(frac / (1 - frac))
    return out


def _flat(pool, dev):
    sel = pool["w"] > 0
    Z = P.to_t(pool["zin"][sel], dev)
    Y = P.to_t(pool["y"][sel], dev)
    O = P.to_t(pool["o"][sel], dev)
    return Z, Y, O


def _status(out, status, **kw):
    P.write_json(Path(out) / "status.json", {"status": status, **kw})
    return {"status": status, **kw}


# ------------------------------------------------------------------ slow
def fit_slow(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    P.seed_all(spec.get("seed", 0))
    timer = P.Timer(spec.get("max_seconds"))
    cfg = dict(prereg.TRAIN["slow"], **spec.get("override", {}))
    track = spec["track"]
    kind = P.track_kind(track)
    tr_store, dv_store = _stores(spec)
    sigma_vec = _sigma_vec(spec, tr_store) if kind == "l1" else None
    rng = np.random.default_rng(spec.get("seed", 0))
    pool = P.build_pool(tr_store, spec.get("pool", 3000), kind, rng, sigma_vec)
    dpool = P.build_pool(dv_store, spec.get("dev_pool", 200), kind, rng, sigma_vec)
    Z, Y, O = _flat(pool, dev)
    Zd, Yd, Od = _flat(dpool, dev)
    if spec.get("overfit"):
        n = spec["overfit"] * 1000
        Z, Y, O = Z[:n], Y[:n], O[:n]
    model = Mo.SlowModel(Z.shape[1], Y.shape[1], cfg["hidden"]).to(dev)
    model.norm.fit(Z[: 200000])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = spec.get("steps", cfg["steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log = []
    g = torch.Generator(device="cpu").manual_seed(spec.get("seed", 0))
    first = None

    def loss_fn(z, y, o):
        yh = model(z)
        l = ((yh - y) ** 2 * o).sum() / o.sum().clamp_min(1.0)
        if kind == "r":
            l = l + 0.1 * (1 - F.cosine_similarity(yh, y, dim=-1)).mean()
        return l
    for step in range(steps):
        idx = torch.randint(0, Z.shape[0], (min(cfg["batch"], Z.shape[0]),), generator=g).to(dev)
        loss = loss_fn(Z[idx], Y[idx], O[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if first is None:
            first = float(loss)
        if step % max(1, steps // 50) == 0 or step == steps - 1:
            with torch.no_grad():
                dl = float(loss_fn(Zd[:200000], Yd[:200000], Od[:200000]))
            log.append({"step": step, "train": float(loss), "dev": dl})
        if timer.over():
            break
    with torch.no_grad():
        res = []
        for s in range(0, Z.shape[0], 200000):
            res.append(((Y[s:s + 200000] - model(Z[s:s + 200000])) ** 2 * O[s:s + 200000]).sum(0))
        ss = torch.stack(res).sum(0)
        cnt = O.sum(0).clamp_min(1.0)
        sig = (ss / cnt).sqrt()
        ystd = ((Y ** 2 * O).sum(0) / cnt - ((Y * O).sum(0) / cnt) ** 2).clamp_min(0).sqrt()
        sig = torch.maximum(sig, torch.maximum(0.05 * ystd, torch.full_like(sig, 1e-3)))
        model.sigma.copy_(sig)
        yh = model(Zd[:200000])
        mse_dim = ((Yd[:200000] - yh) ** 2 * Od[:200000]).sum(0) / Od[:200000].sum(0).clamp_min(1)
        var_dim = ((Yd[:200000] ** 2 * Od[:200000]).sum(0) / Od[:200000].sum(0).clamp_min(1)
                   - ((Yd[:200000] * Od[:200000]).sum(0) / Od[:200000].sum(0).clamp_min(1)) ** 2)
        r2 = float(1 - mse_dim.sum() / var_dim.clamp_min(1e-9).sum())
    extra = {"track": track, "kind": kind, "sigma_vec": None if sigma_vec is None else sigma_vec.tolist(),
             "dev_r2": r2, "first_loss": first, "last_loss": log[-1]["train"] if log else None}
    Mo.save(out / "slow.pt", model, extra)
    (out / "train_log.json").write_text(json.dumps(log))
    ratio = (log[-1]["train"] / first) if (log and first) else None
    return _status(out, "complete" if not timer.over() else "partial", dev_r2=r2, overfit_ratio=ratio,
                   seconds=timer.elapsed(), steps=step + 1)


# ----------------------------------------------------------------- ML-II
def _key_source(spec, dev, store):
    if spec.get("phi"):
        net, extra = Mo.load(spec["phi"], dev)
        return net, None
    return None, P.oracle_dk(store)


def calib(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    if spec.get("require_gate"):
        g = P.load_json(spec["require_gate"])
        if not g.get("pass"):
            return _status(out, "blocked", reason=f"gate {spec['require_gate']} failed: {g}")
    dev = P.device()
    P.seed_all(spec.get("seed", 0))
    track = spec["track"]
    kind = P.track_kind(track)
    tr_store, _ = _stores(spec)
    slow, sx = P.load_slow(spec["slow"], dev)
    sigma_vec = None if sx.get("sigma_vec") is None else np.asarray(sx["sigma_vec"], dtype=np.float32)
    rng = np.random.default_rng(1234 + spec.get("seed", 0))
    pool = P.build_pool(tr_store, spec.get("n_eps", prereg.CALIB["mlii_train_episodes"]), kind, rng, sigma_vec)
    keynet, dko = _key_source(spec, dev, tr_store)
    s_inj = float(track.get("sbar_inj", 0.0))
    grid = spec.get("grid", prereg.CALIB["sbar_grid"])
    lls = {float(s): 0.0 for s in grid}
    gen = torch.Generator(device="cpu").manual_seed(99 + spec.get("seed", 0))
    bs = spec.get("batch", 64)
    n = pool["y"].shape[0]
    with torch.no_grad():
        for s in range(0, n, bs):
            st = {k: v[s:s + bs] for k, v in pool.items() if k in ("zin", "y", "o", "w", "key", "zkey")}
            bt = P.batch_tensors(st, dev)
            if s_inj > 0:
                bt["EPS"] = torch.randn(bt["Y"].shape, generator=gen).to(dev)
            R = P.residuals(bt, slow, s_inj)
            K = P.keys(bt, keynet, dko)
            for sb in grid:
                lls[float(sb)] += bf.ml2_loglik(K, R, bt["O"], bt["W"], float(sb))
    best = max(lls, key=lls.get)
    res = {"sbar": best, "loglik": lls, "n_eps": int(n), "sbar_inj": s_inj,
           "keys": spec.get("phi") or "oracle"}
    P.write_json(out / "calib.json", res)
    return _status(out, "complete", sbar=best)


# ------------------------------------------------------------------- phi
def resolve_dk(dk):
    if isinstance(dk, dict):
        return int(P.load_json(dk["from"])["dk"])
    return int(dk)


def _sbar_of(spec):
    if "sbar" in spec:
        return float(spec["sbar"])
    return float(P.load_json(spec["calib"])["sbar"])


def fit_phi(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    seed = spec.get("seed", 0)
    P.seed_all(seed)
    timer = P.Timer(spec.get("max_seconds"))
    cfg = dict(prereg.TRAIN["phi"], **spec.get("override", {}))
    track = spec["track"]
    kind = P.track_kind(track)
    tr_store, dv_store = _stores(spec)
    slow, sx = P.load_slow(spec["slow"], dev)
    sigma_vec = None if sx.get("sigma_vec") is None else np.asarray(sx["sigma_vec"], dtype=np.float32)
    sbar = _sbar_of(spec)
    s_inj = float(track.get("sbar_inj", 0.0))
    rng = np.random.default_rng(seed)
    pool = P.build_pool(tr_store, spec.get("pool", 4096), kind, rng, sigma_vec)
    if spec.get("overfit"):
        pool = {k: (v[: spec["overfit"]] if k != "meta" else v) for k, v in pool.items()}
    dpool = P.build_pool(dv_store, spec.get("dev_pool", 64), kind, rng, sigma_vec)
    zdim = pool["zkey"].shape[-1] if "zkey" in pool else pool["zin"].shape[-1]
    dk = resolve_dk(spec["dk"])
    net = Mo.KeyNet(zdim, dk, cfg["hidden"]).to(dev)
    zfit = pool["zkey"] if "zkey" in pool else pool["zin"]
    net.norm.fit(P.to_t(zfit[: 200].reshape(-1, zdim), dev))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = spec.get("steps", cfg["steps"])
    warm = min(cfg["warmup"], max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    win = min(cfg["window"], pool["y"].shape[1])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dgen = torch.Generator(device="cpu").manual_seed(7)
    dbt = P.batch_tensors({k: v for k, v in dpool.items() if k in ("zin", "y", "o", "w", "key", "zkey")}, dev)
    if s_inj > 0:
        dbt["EPS"] = torch.randn(dbt["Y"].shape, generator=dgen).to(dev)
    dR = P.residuals(dbt, slow, s_inj)
    lo, hi = prereg.DATA["stat_window"]
    stat = torch.zeros(dR.shape[1], device=dev)
    stat[lo - 1:hi] = 1.0

    def dev_mpe():
        with torch.no_grad():
            K = P.keys(dbt, net)
            M = bf.a6_chunk(K, dR, dbt["O"], dbt["W"], sbar, cfg["chunk"])
            m = dbt["W"] * stat
            num = (((dR - M) * dbt["O"]) ** 2).sum(-1) * m
            den = dbt["O"].sum(-1) * m
            return float(num.sum() / den.sum().clamp_min(1))
    best, best_step, log, first = float("inf"), -1, [], None
    for step in range(steps):
        bt = P.pool_batch(pool, rng, cfg["batch"], dev)
        if win < bt["Y"].shape[1]:
            s0 = int(rng.integers(0, bt["Y"].shape[1] - win + 1))
            bt = {k: v[:, s0:s0 + win] for k, v in bt.items()}
        R = P.residuals(bt, slow, s_inj)
        K = P.keys(bt, net)
        M = bf.a6_chunk(K, R, bt["O"], bt["W"], sbar, cfg["chunk"])
        wm = bt["W"][..., None] * bt["O"]
        loss = (((R - M) ** 2) * wm).sum() / wm.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
        opt.step()
        sched.step()
        if first is None:
            first = float(loss)
        if step % max(1, spec.get("dev_every", 500)) == 0 or step == steps - 1:
            d = dev_mpe()
            log.append({"step": step, "train": float(loss), "dev_stat_mpe": d})
            if d < best:
                best, best_step = d, step
                Mo.save(out / "phi.pt", net, {"dk": dk, "sbar_train": sbar, "dev_stat_mpe": d,
                                              "step": step, "track": track})
        if timer.over():
            break
    (out / "train_log.json").write_text(json.dumps(log))
    ratio = log[-1]["train"] / first if (log and first) else None
    return _status(out, "complete" if not timer.over() else "partial", best_dev_stat_mpe=best,
                   best_step=best_step, overfit_ratio=ratio, seconds=timer.elapsed())


# ----------------------------------------------------------------- gates
def fit_gate(spec):
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    dev = P.device()
    seed = spec.get("seed", 0)
    P.seed_all(seed)
    timer = P.Timer(spec.get("max_seconds"))
    cfg = dict(prereg.TRAIN["gate"], **spec.get("override", {}))
    track = spec["track"]
    kind = P.track_kind(track)
    arm = spec["arm"]
    tr_store, _ = _stores(spec)
    slow, sx = P.load_slow(spec["slow"], dev)
    sigma_vec = None if sx.get("sigma_vec") is None else np.asarray(sx["sigma_vec"], dtype=np.float32)
    keynet, dko = _key_source(spec, dev, tr_store)
    sbar = _sbar_of(spec)
    s_inj = float(track.get("sbar_inj", 0.0))
    rng = np.random.default_rng(seed)
    pool = None if arm == "A2P" else P.build_pool(tr_store, spec.get("pool", 2048), kind, rng, sigma_vec)
    zdim = (pool["zin"].shape[-1] if pool is not None else None)
    if arm == "A2":
        gate = Mo.GateA2(zdim, cfg["hidden"]).to(dev)
        gate.norm.fit(P.to_t(pool["zin"][:200].reshape(-1, zdim), dev))
    elif arm == "A2P":
        gate = Mo.GateA2P(cfg["hidden"]).to(dev)
    else:
        gate = Mo.ScalarGate().to(dev)
    steps = spec.get("steps", cfg["A2_steps"] if arm == "A2" else cfg["A2P_steps"] if arm == "A2P"
                     else cfg["A3A4_steps"])
    opt = torch.optim.Adam(gate.parameters(), lr=cfg["lr"])
    tb = cfg["tbptt"]
    log, first = [], None
    for step in range(steps):
        if arm == "A2P":
            st = Dm.train_spliced(tr_store, rng, cfg["batch"], kind, sigma_vec)
            bt = P.batch_tensors(st, dev)
        else:
            bt = P.pool_batch(pool, rng, cfg["batch"], dev)
        with torch.no_grad():
            R = P.residuals(bt, slow, s_inj)
            K = P.keys(bt, keynet, dko)
            scos = P.slow_surprise(bt, slow) if arm == "A3" else None
        B, T, D = R.shape
        S, n = bf.init_state(torch, R, B, D, K.shape[-1], sbar)
        tot = 0.0
        for s0 in range(0, T, tb):
            sl = slice(s0, s0 + tb)
            if arm == "A2":
                g = gate(bt["Z"][:, sl])
            elif arm == "A3":
                g = torch.sigmoid(gate.a * scos[:, sl] + gate.b)
            elif arm == "A4":
                g = (gate.a, gate.b)
            else:
                g = gate
            res = bf.scan(arm, K[:, sl], R[:, sl], bt["O"][:, sl], bt["W"][:, sl], sbar=sbar, gate=g,
                          collect=("m",), S0=S, n0=n)
            wm = bt["W"][:, sl, None] * bt["O"][:, sl]
            loss = (((R[:, sl] - res["m"]) ** 2) * wm).sum() / wm.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()
            S, n = res["S_final"].detach(), res["n_final"].detach()
            tot += float(loss)
        if first is None:
            first = tot
        if step % max(1, steps // 40) == 0 or step == steps - 1:
            log.append({"step": step, "loss": tot})
        if timer.over():
            break
    extra = {"arm": arm, "track": track, "sbar": sbar}
    if arm in ("A3", "A4"):
        extra.update(a=float(gate.a), b=float(gate.b))
    Mo.save(out / "gate.pt", gate, extra)
    (out / "train_log.json").write_text(json.dumps(log))
    return _status(out, "complete" if not timer.over() else "partial", seconds=timer.elapsed(),
                   overfit_ratio=(log[-1]["loss"] / first) if first else None, **{k: v for k, v in extra.items()
                                                                                   if k in ("a", "b")})
