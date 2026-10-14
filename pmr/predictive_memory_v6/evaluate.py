"""Evaluation suites for the BJP protocol (torch for learned models).

A condition is a dict ``{name, kind, world, T, n}``; kinds:

* ``standard``      hidden/correction/extrapolation metrics + reveal events;
* ``history_pair``  forgetting (identical inputs after an early window);
* ``change_pair``   non-interference (+ lemma-4 prediction for 'ours' family);
* ``blackout``      MemoryGain (intact memory vs memory reset at lights-off);
* ``count``         VSC-Repeat analogue: k identical copies of a static tour.

Every condition is generated from a fixed seed in the ``dev`` or ``test``
namespace; ``test`` is only evaluated when the caller passes ``split='test'``.
"""
from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path

import numpy as np

from . import worlds_l0
from .data import make_world, geometry_of, stable_seed, l0_config
from .metrics import (score_episodes, summarize, history_pair_scores, interference_scores,
                      interference_summary, memory_gain, mra, tracking_masks)
from .references import run_reference
from .surprise import novelty_count, true_count


# ------------------------------------------------------------ prediction
def to_world(pred_norm, data):
    geom = geometry_of(data)
    if data["meta"].get("frame") == "ego":
        return np.mod(pred_norm * geom.period + data["ego"][:, :, None, :], geom.period)
    return geom.denormalize(pred_norm)


def predict_model(model, data, device, chunk=256, reset_at=None, keep=()):
    import torch
    from .data import to_tbatch
    n = len(data["vis"])
    acc = {"pred": [], "surprise": [], "presence": []}
    kept = {k: [] for k in keep}
    keys = []
    for s in range(0, n, chunk):
        d = {k: (v[s:s + chunk] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
        b = to_tbatch(d, device)
        with torch.no_grad():
            o = model.rollout(b, reset_at=reset_at, keep=keep)
            if "write" in keep and hasattr(model, "memory_readout"):
                keys.append(model.init_state(b)["k"].double().cpu().numpy())
                gens = torch.stack([model.generator(b, t) for t in range(b.shape[1])], 1)
                kept.setdefault("K", []).append(torch.linalg.matrix_exp(gens).double().cpu().numpy())
        acc["pred"].append(o["pred"].double().cpu().numpy())
        acc["surprise"].append(o["surprise"].double().cpu().numpy())
        if o.get("presence") is not None:
            acc["presence"].append(o["presence"].double().cpu().numpy())
        for k in keep:
            if o.get(k) is not None:
                kept[k].append(o[k].double().cpu().numpy())
    res = {"pred": to_world(np.concatenate(acc["pred"]), data),
           "surprise": np.concatenate(acc["surprise"]),
           "presence": np.concatenate(acc["presence"]) if acc["presence"] else None}
    for k, v in kept.items():
        if v:
            res[k] = np.concatenate(v)
    if keys:
        res["keys"] = np.concatenate(keys)
    return res


def predict_reference(name, data, **kw):
    geom = geometry_of(data)
    cfg = data["meta"].get("config", {})
    p = run_reference(name, data, geom, obs_noise=cfg.get("obs_noise", 0.0),
                      vel_scale=cfg.get("vel_max", 2.0)) if name not in ("pause", "interp_oracle") \
        else run_reference(name, data, geom)
    if data["meta"].get("frame") == "ego":
        raise ValueError("references expect allocentric observations")
    return {"pred": p, "surprise": None, "presence": None}


# ------------------------------------------------------------- conditions
def cond_seed(split, cond):
    return stable_seed(split, cond["name"], cond["world"], cond["T"], cond.get("n"))


def _write_rows(path, rows):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def run_condition(predict, cond, split, out_dir: Path, training_seed=None, is_assoc=False):
    kind, world, T, n = cond["kind"], cond["world"], int(cond["T"]), int(cond["n"])
    seed = cond_seed(split, cond)
    t0 = time.time()
    summary = {"condition": cond, "split": split}
    if kind == "standard":
        data = make_world(world, n, T, seed, cond.get("cache"))
        geom = geometry_of(data)
        res = predict(data)
        rows = score_episodes(data, res["pred"], geom, training_seed=training_seed)
        summary.update(summarize(rows))
        _write_rows(out_dir / f"{cond['name']}_episodes.jsonl.gz", rows)
        if res.get("surprise") is not None:
            m = tracking_masks(data)
            sel = m["reveal"] & (m["since"] >= 2)
            np.savez_compressed(out_dir / f"{cond['name']}_events.npz", label=m["changed"][sel],
                                surprise=res["surprise"][sel], gap=m["since"][sel])
    elif kind == "history_pair":
        cfg, _ = l0_config(world, T)
        a, b = worlds_l0.history_pair(cfg, n, T, seed)
        a["meta"]["spec"] = b["meta"]["spec"] = world
        ra, rb = predict(a), predict(b)
        geom = geometry_of(a)
        rows = history_pair_scores(a, b, ra["pred"], rb["pred"], geom, training_seed=training_seed)
        summary.update({k: float(np.mean([r[k] for r in rows])) for k in
                        ("history_pair_error", "pred_separation", "true_separation")})
        _write_rows(out_dir / f"{cond['name']}_episodes.jsonl.gz", rows)
    elif kind == "change_pair":
        cfg, _ = l0_config(world, T)
        a, b = worlds_l0.change_pair(cfg, n, T, seed)
        geom = geometry_of(a)
        keep = ("write",) if is_assoc else ()
        ra, rb = predict(a, keep=keep), predict(b, keep=keep)
        key_cos, pred_delta = None, None
        if is_assoc and "write" in ra and "keys" in ra:
            k = ra["keys"]
            key_cos = np.einsum("bik,bjk->bij", k, k)
            i = int(a["probe_object"])
            B = len(k)
            pred_delta = np.zeros((B, k.shape[1]))
            for bb in range(B):
                r0 = int(a["probe_reveal"][bb])
                if not a["pair_valid"][bb] or r0 + 1 >= T:
                    continue
                dv = ra["write"][bb, r0, i] - rb["write"][bb, r0, i]
                dpos = (ra["K"][bb, r0 + 1] @ dv)[2:4] * geom.unit
                pred_delta[bb] = np.linalg.norm(dpos) * np.abs(key_cos[bb, i])
        rows = interference_scores(a, b, ra["pred"], rb["pred"], geom, key_cos=key_cos,
                                   predicted_delta=pred_delta, training_seed=training_seed)
        summary.update(interference_summary(rows))
        _write_rows(out_dir / f"{cond['name']}_episodes.jsonl.gz", rows)
    elif kind == "blackout":
        data = make_world(world, n, T, seed)
        geom = geometry_of(data)
        cfgd = data["meta"]["config"]
        start, length = cfgd["blackout_start"], cfgd["blackout_len"]
        r = predict(data)
        r0 = predict(data, reset_at=start)
        mg = memory_gain(data, r["pred"], r0["pred"], geom, start, length)
        summary.update({"memory_gain": float(mg.mean()), "memory_gain_ci95": _ci(mg)})
        rows = [{"episode_id": str(data["episode_id"][i]), "training_seed": training_seed,
                 "memory_gain": float(mg[i])} for i in range(len(mg))]
        _write_rows(out_dir / f"{cond['name']}_episodes.jsonl.gz", rows)
    elif kind == "count":
        data = make_world(world, n, T, seed)
        res = predict(data)
        if res.get("presence") is None:
            summary["skipped"] = "variant has no presence channel"
        else:
            nov = 1.0 - np.clip(res["presence"], 0.0, 1.0)
            pred_c = novelty_count(nov, data["vis"].astype(bool))
            gt = true_count(data["vis"].astype(bool))
            scores = np.array([mra(p, g) for p, g in zip(pred_c, gt)])
            summary.update({"mra": float(scores.mean()), "mra_ci95": _ci(scores),
                            "mean_pred_count": float(pred_c.mean()), "mean_true_count": float(gt.mean())})
    else:
        raise ValueError(kind)
    summary["seconds"] = time.time() - t0
    (out_dir / f"{cond['name']}.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def _ci(x, draws=1000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (draws, len(x)))].mean(1)
    return np.quantile(m, [0.025, 0.975]).tolist()


def eval_job(spec: dict) -> dict:
    """spec: {variant|reference, checkpoint?, conditions, split, out, device, training_seed}"""
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    split = spec.get("split", "dev")
    if split == "test" and not spec.get("open_test", False):
        raise PermissionError("test split requires open_test=True (final report only)")
    is_assoc = False
    if spec.get("reference"):
        name = spec["reference"]
        predict = lambda data, **kw: predict_reference(name, data)
    else:
        import torch
        from .train import load_model
        device = torch.device(spec.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        model, man = load_model(spec["checkpoint"], device)
        if spec.get("as_variant"):  # e.g. 'ours' weights evaluated with reveal-error TTT
            from . import memories
            geom = man["geometry"]
            other = memories.build(spec["as_variant"], n_objects=man["n_objects"], geom=geom["kind"],
                                   app_dim=man["app_dim"])
            other.load_state_dict(model.state_dict())
            model = other.to(device).eval()
        if spec.get("ttt_lr") is not None and hasattr(model, "ttt_lr"):
            model.ttt_lr = float(spec["ttt_lr"])
        from .memories import AssocMemory
        is_assoc = isinstance(model, AssocMemory) and not model.frame_form and model.adapt is None
        predict = lambda data, **kw: predict_model(model, data, device, **kw)
    results, failures = {}, {}
    for cond in spec["conditions"]:
        target = out / f"{cond['name']}.json"
        if target.exists() and not spec.get("overwrite"):
            results[cond["name"]] = json.loads(target.read_text())
            continue
        try:
            results[cond["name"]] = run_condition(predict, cond, split, out, spec.get("training_seed"), is_assoc)
        except Exception as exc:  # recorded, never silently dropped
            failures[cond["name"]] = repr(exc)
    status = {"status": "complete" if not failures else "partial", "failures": failures,
              "conditions": list(results)}
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status
