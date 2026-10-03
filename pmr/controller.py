"""Predict per-action final ES gains; select on complete development rollouts."""
from pathlib import Path
from collections import Counter
import math
import json
import numpy as np
import torch
from torch import nn
from .adapter import ACTIONS, FEATURES, FEATURE_SCHEMA, FEATURE_SETS
from .common import seed_all, write_json, clock, sha256
from .report import paired_ci

CONTROLLER_SCHEMA = 3
GAIN_MARGINS = (.01, .05, .1)


class Controller(nn.Module):
    def __init__(self, feature_set="temporal_influence"):
        super().__init__()
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        self.feature_set=feature_set
        self.register_buffer("feature_mask",torch.tensor([float(f in FEATURE_SETS[feature_set]) for f in FEATURES]))
        self.register_buffer("mean", torch.zeros(len(FEATURES)))
        self.register_buffer("std", torch.ones(len(FEATURES)))
        self.register_buffer("gain_margin", torch.tensor(.01))
        self.register_buffer("keep_only", torch.tensor(False))
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        gains = self.net(((x-self.mean)/self.std).clamp(-20, 20)*self.feature_mask)
        return torch.cat((torch.zeros_like(gains[..., :1]), gains), dim=-1)

    @torch.no_grad()
    def actions(self, features, margin=None):
        gains = self(features)
        if not bool(torch.isfinite(gains).all()):
            raise FloatingPointError("Nonfinite controller prediction")
        if bool(self.keep_only):
            return torch.zeros(gains.shape[:-1], dtype=torch.long, device=gains.device)
        candidates = gains[..., 1:].clone()
        candidates[..., 1].masked_fill_(features[..., FEATURES.index("rebuild_available")] < .5, -torch.inf)
        best_gain, best = candidates.max(-1)
        threshold = self.gain_margin if margin is None else margin
        use = (best_gain > threshold) & (features[..., FEATURES.index("has_evidence")] > .5)
        return torch.where(use, best+1, torch.zeros_like(best))

    @torch.no_grad()
    def choose(self, feature):
        return ACTIONS[int(self.actions(feature))]


def training_tensors(rows, device):
    if any(r.get("feature_schema") != FEATURE_SCHEMA or r.get("feature_names") != list(FEATURES) for r in rows):
        raise ValueError("V3 needs fresh branch rows with temporal/probe features; do not reuse V1/V2 labels")
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32, device=device)
    try:
        losses = torch.tensor([[r["branches"][a]["final"]["energy_score"] for a in ACTIONS] for r in rows],
                              dtype=torch.float32, device=device)
    except KeyError as exc:
        raise ValueError("Missing full-horizon final ES in branch labels; regenerate them with V3") from exc
    if not bool(torch.isfinite(x).all() and torch.isfinite(losses).all()):
        raise ValueError("Nonfinite controller training data")
    return x, losses[:, :1]-losses


def select_controller(candidates, model, dev_evaluator, dev_ids, out):
    """The development CI is a tuning gate, not a post-selection significance claim."""
    out=Path(out)
    artifacts=out.parent/(out.name.removesuffix(".selection.json")+".candidates")
    artifacts.mkdir(parents=True,exist_ok=True)
    def save_rows(name, rows):
        path=artifacts/name
        path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        return str(path.relative_to(out.parent))
    def checked(policy):
        rows = dev_evaluator(policy)
        if len(rows) != len(dev_ids) or {r["scene_id"] for r in rows} != dev_ids:
            raise ValueError("Development rollout returned missing, duplicate, or foreign scenes")
        if not all(math.isfinite(r["energy_score"]) for r in rows):
            raise ValueError("Nonfinite development ES")
        return {r["scene_id"]: r for r in rows}

    baseline = checked("keep")
    ids = sorted(dev_ids)
    baseline_file=save_rows("keep.jsonl",[baseline[s] for s in ids])
    ordinary = [baseline[s]["group"] == "ordinary" for s in ids]
    selected = None
    records = []
    for rank, candidate in enumerate(candidates):
        model.load_state_dict(candidate["state_dict"])
        model.gain_margin.fill_(candidate["margin"])
        model.keep_only.fill_(False)
        checkpoint=artifacts/f"candidate{rank:03d}.pt"
        torch.save({"schema":CONTROLLER_SCHEMA,"features":list(FEATURES),"actions":list(ACTIONS),
                    "feature_set":model.feature_set,"training_step":candidate["step"],
                    "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}},checkpoint)
        trial = checked(model.eval())
        if any(trial[s]["group"]!=baseline[s]["group"] for s in ids):
            raise ValueError("Candidate changed reference development groups")
        gains = np.array([baseline[s]["energy_score"]-trial[s]["energy_score"] for s in ids])
        rollout_file=save_rows(f"candidate{rank:03d}.jsonl",[
            dict(trial[s],keep_energy_score=baseline[s]["energy_score"],paired_gain=float(g))
            for s,g in zip(ids,gains)])
        ci = paired_ci(gains)
        ordinary_gain = float(gains[ordinary].mean()) if any(ordinary) else None
        eligible = ci["ci95"][0] > 0 and (ordinary_gain is None or ordinary_gain >= 0)
        item = {"rank": rank, "step": candidate["step"], "gain_margin": candidate["margin"],
                "offline_dev_gain": candidate["offline_dev_gain"],
                "mean_final_ES": float(np.mean([trial[s]["energy_score"] for s in ids])),
                "paired_ES_improvement": ci, "ordinary_mean_gain": ordinary_gain,
                "eligible": eligible,
                "rollout_file":rollout_file,"checkpoint_file":str(checkpoint.relative_to(out.parent)),
                "checkpoint_sha256":sha256(checkpoint),
                "action_counts": dict(Counter(a for s in ids for a in trial[s]["edits"][1:]))}
        records.append(item)
        print(f"dev rollout step={candidate['step']} margin={candidate['margin']}: gain={ci['mean']:.5f}, eligible={eligible}", flush=True)
        if eligible and (selected is None or ci["mean"] > selected["paired_ES_improvement"]["mean"]):
            selected = item
            selected_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if selected is not None:
        model.load_state_dict(selected_weights)
    else:
        model.keep_only.fill_(True)
    selection = {"schema": CONTROLLER_SCHEMA, "split": "dev_from_official_train",
                 "feature_set":model.feature_set,"baseline_rollout_file":baseline_file,
                 "metric": "final_60_frame_energy_score", "n_scenes": len(ids),
                 "keep_ES": float(np.mean([baseline[s]["energy_score"] for s in ids])),
                 "gate": "paired development CI lower bound > 0 and ordinary mean gain >= 0",
                 "mode": "gain_controller" if selected else "keep_fallback",
                 "selected": selected, "candidates": records,
                 "note": "Development tuning only; not evidence of held-out improvement"}
    write_json(out, selection)
    return selection


def train_controller(train_rows, dev_rows, out, device="cpu", steps=500, batch_size=64, seed=0,
                     dev_evaluator=None, max_candidates=3, feature_set="temporal_influence"):
    from .diagnostics import diagnose_rows, regression_metrics, activation_stats, parameter_updates
    seed_all(seed)
    if not train_rows or not dev_rows or steps < 1 or max_candidates < 1:
        raise ValueError("Nonempty scene-disjoint train/dev data and positive budgets required")
    train_ids, dev_ids = ({r["scene_id"] for r in rows} for rows in (train_rows, dev_rows))
    if train_ids & dev_ids:
        raise ValueError("Scene leakage between controller train/dev")
    if dev_evaluator is None:
        raise ValueError("A development-only rollout evaluator is required for V2 model selection")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    controller_start = clock(device)
    diagnostic = diagnose_rows(train_rows, dev_rows, seed, feature_set=feature_set)
    diagnostic_seconds = clock(device)-controller_start
    write_json(out.with_suffix(".diagnostics.json"), diagnostic)
    tiny = diagnostic["tiny_overfit"]
    print(f"tiny-set fit: {tiny['status']}, initial MSE={tiny['initial_MSE']:.6g}, best MSE={tiny['best_MSE']:.6g}", flush=True)
    model = Controller(feature_set).to(device)
    x, gains = training_tensors(train_rows, device)
    xd, gd = training_tensors(dev_rows, device)
    model.mean.copy_(x.mean(0))
    model.std.copy_(x.std(0, unbiased=False).clamp_min(.01))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, pool = [], []
    training_path = out.with_suffix(".training.json")
    initial = {"train_MSE": float(gains[:, 1:].square().mean()),
               "dev_MSE": float(gd[:, 1:].square().mean()),
               "explanation": "Zero-initialized gain head predicts Keep-relative gain 0 for every action"}
    metadata = {"objective": "per_action_final_ES_gain_MSE", "initial": initial,
                "feature_set":feature_set,"active_features":list(FEATURE_SETS[feature_set]),
                "diagnostic_seconds": diagnostic_seconds,
                "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                "candidate_budget": max_candidates, "tiny_fit_status": tiny["status"],
                "dev_baselines": diagnostic["dev_baselines"],
                "parameters": sum(p.numel() for p in model.parameters())}
    write_json(training_path, dict(metadata, history=history, status="fitting"))
    start = clock(device)
    for step in range(1, steps+1):
        log_step = step == 1 or step % 50 == 0 or step == steps
        model.train()
        indices = torch.randint(len(x), (min(batch_size, len(x)),), device=device)
        # All actions are supervised in ES units, including harmful interventions.
        loss = (model(x[indices])[:, 1:]-gains[indices, 1:]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite controller regression loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        before = {k: v.detach().clone() for k, v in model.named_parameters()} if log_step else None
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if log_step:
            model.eval()
            with torch.no_grad():
                train_mse = float((model(x)[:, 1:]-gains[:, 1:]).square().mean())
                dev_metrics = regression_metrics(model(xd), gd, xd)
                dev_mse = dev_metrics["MSE"]
                if not math.isfinite(train_mse + dev_mse):
                    raise FloatingPointError("Nonfinite full-dataset controller MSE")
                weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                for margin in GAIN_MARGINS:
                    actions = model.actions(xd, margin)
                    offline_gain = float(gd.gather(1, actions[:, None]).mean())
                    pool.append({"step": step, "margin": margin, "offline_dev_gain": offline_gain,
                                 "state_dict": weights, "decisions": tuple(actions.cpu().tolist())})
            history.append({"step": step, "train_MSE": train_mse, "batch_MSE_before_update": float(loss.detach()),
                            "dev_MSE": dev_mse, "learning_rate": opt.param_groups[0]["lr"],
                            "gradient_norm_before_clip": float(grad_norm),
                            "gradient_was_clipped": float(grad_norm) > 1.,
                            "parameter_updates": parameter_updates(model, before),
                            "activations": activation_stats(model, x), "dev_at_margin_001": dev_metrics})
            # Durable progress while fitting, so a later failure retains curves.
            write_json(training_path, dict(metadata, history=history, status="fitting", fit_seconds=clock(device)-start))
            print(f"controller step {step}/{steps}: train MSE={train_mse:.5f} dev MSE={dev_mse:.5f} grad={float(grad_norm):.4g}", flush=True)
    fit_seconds = clock(device)-start
    # Counterfactual ranking only makes a bounded shortlist. Final selection
    # uses complete on-policy dev rollouts and always includes Keep fallback.
    candidates, seen = [], set()
    for item in sorted(pool, key=lambda c: (-c["offline_dev_gain"], c["step"], -c["margin"])):
        if item["decisions"] in seen or not any(item["decisions"]):
            continue
        seen.add(item["decisions"])
        candidates.append(item)
        if len(candidates) == max_candidates:
            break
    write_json(training_path, dict(metadata, history=history, status="selecting", fit_seconds=fit_seconds))
    selection = select_controller(candidates, model, dev_evaluator, dev_ids, out.with_suffix(".selection.json"))
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"schema": CONTROLLER_SCHEMA, "state_dict": state, "actions": list(ACTIONS),
                "feature_set":feature_set,
                "features": list(FEATURES), "objective": "per_action_final_ES_gain_MSE", "selection": selection,
                "seed": seed, "steps": steps, "train_scenes": sorted(train_ids), "dev_scenes": sorted(dev_ids)}, out)
    write_json(training_path, dict(metadata, history=history, fit_seconds=fit_seconds, status="complete",
               seconds_including_selection=clock(device)-start,
               seconds_including_diagnostics_and_selection=clock(device)-controller_start,
               selected_mode=selection["mode"]))
    return model.eval()


def load_controller(path, device="cpu"):
    obj = torch.load(path, map_location=device, weights_only=True)
    if obj.get("schema") != CONTROLLER_SCHEMA or obj["features"] != list(FEATURES) or obj["actions"] != list(ACTIONS):
        raise ValueError("Controller schema mismatch: V1/V2 weights cannot be loaded by V3; use a new run directory")
    model = Controller(obj["feature_set"]).to(device)
    model.load_state_dict(obj["state_dict"], strict=True)
    return model.eval()
