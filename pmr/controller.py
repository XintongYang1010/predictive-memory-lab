"""Predict per-action final ES gains; select on complete development rollouts."""
from pathlib import Path
from collections import Counter
import math
import numpy as np
import torch
from torch import nn
from .adapter import ACTIONS, FEATURES, FEATURE_SCHEMA
from .common import seed_all, write_json, clock
from .report import paired_ci

CONTROLLER_SCHEMA = 2
GAIN_MARGINS = (.01, .05, .1)


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(len(FEATURES)))
        self.register_buffer("std", torch.ones(len(FEATURES)))
        self.register_buffer("gain_margin", torch.tensor(.01))
        self.register_buffer("keep_only", torch.tensor(False))
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        gains = self.net(((x-self.mean)/self.std).clamp(-20, 20))
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
        raise ValueError("V2 needs fresh branch rows with stage features and final ES; do not reuse V1 labels")
    x = torch.tensor([r["features"] for r in rows], dtype=torch.float32, device=device)
    try:
        losses = torch.tensor([[r["branches"][a]["final"]["energy_score"] for a in ACTIONS] for r in rows],
                              dtype=torch.float32, device=device)
    except KeyError as exc:
        raise ValueError("Missing full-horizon final ES in branch labels; regenerate them with V2") from exc
    if not bool(torch.isfinite(x).all() and torch.isfinite(losses).all()):
        raise ValueError("Nonfinite controller training data")
    return x, losses[:, :1]-losses


def select_controller(candidates, model, dev_evaluator, dev_ids, out):
    """The development CI is a tuning gate, not a post-selection significance claim."""
    def checked(policy):
        rows = dev_evaluator(policy)
        if len(rows) != len(dev_ids) or {r["scene_id"] for r in rows} != dev_ids:
            raise ValueError("Development rollout returned missing, duplicate, or foreign scenes")
        if not all(math.isfinite(r["energy_score"]) for r in rows):
            raise ValueError("Nonfinite development ES")
        return {r["scene_id"]: r for r in rows}

    baseline = checked("keep")
    ids = sorted(dev_ids)
    ordinary = [baseline[s]["group"] == "ordinary" for s in ids]
    selected = None
    records = []
    for rank, candidate in enumerate(candidates):
        model.load_state_dict(candidate["state_dict"])
        model.gain_margin.fill_(candidate["margin"])
        model.keep_only.fill_(False)
        trial = checked(model.eval())
        gains = np.array([baseline[s]["energy_score"]-trial[s]["energy_score"] for s in ids])
        ci = paired_ci(gains)
        ordinary_gain = float(gains[ordinary].mean()) if any(ordinary) else None
        eligible = ci["ci95"][0] > 0 and (ordinary_gain is None or ordinary_gain >= 0)
        item = {"rank": rank, "step": candidate["step"], "gain_margin": candidate["margin"],
                "offline_dev_gain": candidate["offline_dev_gain"],
                "mean_final_ES": float(np.mean([trial[s]["energy_score"] for s in ids])),
                "paired_ES_improvement": ci, "ordinary_mean_gain": ordinary_gain,
                "eligible": eligible,
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
                 "metric": "final_60_frame_energy_score", "n_scenes": len(ids),
                 "keep_ES": float(np.mean([baseline[s]["energy_score"] for s in ids])),
                 "gate": "paired development CI lower bound > 0 and ordinary mean gain >= 0",
                 "mode": "gain_controller" if selected else "keep_fallback",
                 "selected": selected, "candidates": records,
                 "note": "Development tuning only; not evidence of held-out improvement"}
    write_json(out, selection)
    return selection


def train_controller(train_rows, dev_rows, out, device="cpu", steps=500, batch_size=64, seed=0,
                     dev_evaluator=None, max_candidates=3):
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
    diagnostic = diagnose_rows(train_rows, dev_rows, seed)
    diagnostic_seconds = clock(device)-controller_start
    write_json(out.with_suffix(".diagnostics.json"), diagnostic)
    tiny = diagnostic["tiny_overfit"]
    print(f"tiny-set fit: {tiny['status']}, initial MSE={tiny['initial_MSE']:.6g}, best MSE={tiny['best_MSE']:.6g}", flush=True)
    model = Controller().to(device)
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
        raise ValueError("Controller schema mismatch: V1 weights cannot be loaded by V2; use a new run directory")
    model = Controller().to(device)
    model.load_state_dict(obj["state_dict"], strict=True)
    return model.eval()
