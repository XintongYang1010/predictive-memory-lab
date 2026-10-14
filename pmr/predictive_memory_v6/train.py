"""Training of one memory variant on one world (torch).

Prequential objective only: squared pre-write error on visible tokens of
objects seen before (``memories.prequential_loss``).  Hidden ground truth is
never used.  Checkpoints are selected on a fixed dev-select set; the dev and
test evaluation sets are generated in other seed namespaces.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import memories
from .data import TrainPool, make_world, to_tbatch, stable_seed, n_objects_of, geometry_of

DEFAULTS = dict(steps=20000, batch=256, T=64, lr=3e-4, weight_decay=0.1, clip=1.0, warmup=500,
                dev_every=1000, dev_episodes=512, pool_size=16384, refresh=5000, device="cuda",
                max_seconds=None, overfit=0, curriculum=False, curriculum_gate=None, log_every=50,
                cache=None)

SLOW_BATCH = {"flowm_token": 32}   # memory-heavy structural reference


def _lr(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    frac = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))


@torch.no_grad()
def dev_loss(model, dev, device, chunk=256):
    model.eval()
    tot, cnt = 0.0, 0.0
    n = len(dev["vis"])
    for s in range(0, n, chunk):
        d = {k: (v[s:s + chunk] if isinstance(v, np.ndarray) else v) for k, v in dev.items()}
        b = to_tbatch(d, device)
        out = model.rollout(b)
        m = b.vis * memories.seen_before_mask(b.vis)
        err = memories.obs_diff(b.obs, out["pred"], b.kind).pow(2).sum(-1)
        tot += float((err * m).sum())
        cnt += float(m.sum())
    model.train()
    return tot / max(cnt, 1.0)


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    os.replace(tmp, path)


def _world_with_level(world, level):
    if level is None:
        return world
    sep = "," if ":" in world else ":"
    return f"{world}{sep}hidden_scale={level}"


def train_job(spec: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(spec)
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("status") == "complete":
        return json.loads((out / "status.json").read_text())
    device = torch.device(cfg["device"] if (cfg["device"] != "cuda" or torch.cuda.is_available()) else "cpu")
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cfg["variant"] in SLOW_BATCH:
        cfg["batch"] = min(cfg["batch"], SLOW_BATCH[cfg["variant"]])
    world = cfg["world"]
    probe = make_world(world, 2, cfg["T"], stable_seed("probe", world), cfg["cache"])
    N, A = n_objects_of(probe), probe["app"].shape[-1]
    geom = geometry_of(probe)
    model = memories.build(cfg["variant"], n_objects=N, geom=geom.kind, app_dim=A,
                           grid=int(geom.period) if geom.kind == "torus" else 50).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    levels = [0.25, 0.5, 0.75, 1.0] if cfg["curriculum"] else [None]
    li = 0
    pool = TrainPool(_world_with_level(world, levels[li]), cfg["T"], cfg["pool_size"], cfg["refresh"],
                     seed=stable_seed("pool", world, seed), cache=cfg["cache"])
    dev_world = cfg.get("dev_world") or world          # L1/R select on a held-out split
    dev = make_world(dev_world, cfg["dev_episodes"], cfg["T"], stable_seed("dev_select", dev_world), cfg["cache"])
    fixed = None
    if cfg["overfit"]:
        fixed = make_world(world, int(cfg["overfit"]), cfg["T"], stable_seed("overfit", world), cfg["cache"])
    manifest = {"variant": cfg["variant"], "world": world, "seed": seed, "n_objects": N, "app_dim": A,
                "geometry": geom.to_dict(), "trainable_parameters": n_params,
                "state_scalars": model.state_scalars(N), "config": {k: v for k, v in cfg.items()},
                "device": str(device), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    _write_json(out / "manifest.json", manifest)
    log = open(out / "training.jsonl", "a")
    start = time.time()
    best, best_step, status = math.inf, -1, "complete"
    first_loss = None
    step = 0
    model.train()
    try:
        for step in range(cfg["steps"]):
            if cfg["max_seconds"] and time.time() - start > cfg["max_seconds"]:
                status = "budget_exceeded"
                break
            for g in opt.param_groups:
                g["lr"] = _lr(step, cfg)
            d = fixed if fixed is not None else pool.sample(step, cfg["batch"])
            b = to_tbatch(d, device)
            o = model.rollout(b)
            loss = memories.prequential_loss(o, b, memories.seen_before_mask(b.vis))
            if not torch.isfinite(loss):
                status = "nonfinite_loss"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            lv = float(loss)
            first_loss = lv if first_loss is None else first_loss
            if step % cfg["log_every"] == 0:
                log.write(json.dumps({"step": step, "loss": lv, "grad_norm": float(gn),
                                      "lr": opt.param_groups[0]["lr"], "seconds": time.time() - start,
                                      "level": levels[li]}) + "\n")
                log.flush()
            last = step == cfg["steps"] - 1
            if (step + 1) % cfg["dev_every"] == 0 or last:
                dl = dev_loss(model, dev, device)
                log.write(json.dumps({"step": step, "dev_loss": dl, "seconds": time.time() - start}) + "\n")
                log.flush()
                if dl < best:
                    best, best_step = dl, step
                    torch.save({"state_dict": model.state_dict(), "manifest": manifest, "step": step,
                                "dev_loss": dl}, out / "best.pt")
                gate = cfg["curriculum_gate"]
                if cfg["curriculum"] and li < len(levels) - 1 and gate is not None and dl <= gate:
                    li += 1
                    pool = TrainPool(_world_with_level(world, levels[li]), cfg["T"], cfg["pool_size"],
                                     cfg["refresh"], seed=stable_seed("pool", world, seed, li), cache=cfg["cache"])
    except torch.cuda.OutOfMemoryError:
        status = "oom"
    torch.save({"state_dict": model.state_dict(), "manifest": manifest, "step": step}, out / "final.pt")
    if best_step < 0:
        torch.save({"state_dict": model.state_dict(), "manifest": manifest, "step": step,
                    "dev_loss": None}, out / "best.pt")
    result = {"status": status, "variant": cfg["variant"], "world": world, "seed": seed,
              "steps_done": step + 1, "best_dev_loss": best if best < math.inf else None,
              "best_step": best_step, "first_loss": first_loss, "seconds": time.time() - start,
              "trainable_parameters": n_params, "state_scalars": model.state_scalars(N),
              "peak_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    if cfg["overfit"]:
        result["overfit_final_loss"] = dev_loss(model, fixed, device)
    _write_json(out / "status.json", result)
    log.close()
    return result


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    man = ck["manifest"]
    geom = man["geometry"]
    model = memories.build(man["variant"], n_objects=man["n_objects"], geom=geom["kind"],
                           app_dim=man["app_dim"], grid=int(geom["period"]) if geom["kind"] == "torus" else 50)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), man
