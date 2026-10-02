"""Train/dev-only learning diagnostics; never select a deployment policy here."""
from collections import Counter
import torch
from .adapter import ACTIONS, FEATURES


def available_actions(x):
    allowed = torch.ones((*x.shape[:-1], len(ACTIONS)), dtype=torch.bool, device=x.device)
    allowed[..., 1:] &= x[..., FEATURES.index("has_evidence"), None] > .5
    allowed[..., 2] &= x[..., FEATURES.index("rebuild_available")] > .5
    return allowed


def regression_metrics(predictions, gains, x, margin=.01):
    """Recorded single-edit gains are not closed-loop policy performance."""
    allowed = available_actions(x)
    possible = gains.masked_fill(~allowed, -torch.inf)
    oracle = possible.max(-1).values
    scored = predictions.masked_fill(~allowed, -torch.inf)
    best, actions = scored.max(-1)
    actions = torch.where(best > margin, actions, torch.zeros_like(actions))
    realized = gains.gather(1, actions[:, None]).squeeze(1)
    edits = actions != 0
    return {
        "MSE": float((predictions[:, 1:]-gains[:, 1:]).square().mean()),
        "MSE_by_action": {a: float((predictions[:, i]-gains[:, i]).square().mean())
                          for i, a in enumerate(ACTIONS) if i},
        "offline_mean_gain": float(realized.mean()),
        "offline_oracle_regret": float((oracle-realized).mean()),
        "intervention_fraction": float(edits.float().mean()),
        "harm_fraction_among_interventions": float((realized[edits] < 0).float().mean()) if bool(edits.any()) else None,
        "action_counts": dict(Counter(ACTIONS[i] for i in actions.cpu().tolist())),
    }


def opportunity(x, gains):
    allowed = available_actions(x)
    oracle = gains.masked_fill(~allowed, -torch.inf).max(-1).values
    return {"n_rows": len(x), "oracle_mean_gain": float(oracle.mean()),
            "fraction_with_gain_above_001": float((oracle > .01).float().mean()),
            "per_action": {a: {
                "eligible_rows": int(allowed[:, i].sum()),
                "mean_gain": float(gains[allowed[:, i], i].mean()) if bool(allowed[:, i].any()) else None,
                "positive_fraction": float((gains[allowed[:, i], i] > .01).float().mean()) if bool(allowed[:, i].any()) else None,
            } for i, a in enumerate(ACTIONS) if i}}


def feature_health(x, mean, std):
    z = (x-mean)/std
    return {name: {"mean": float(x[:, i].mean()), "std": float(x[:, i].std(unbiased=False)),
                   "normalized_abs_max": float(z[:, i].abs().max()),
                   "clipped_fraction": float((z[:, i].abs() > 20).float().mean())}
            for i, name in enumerate(FEATURES)}


def tiny_overfit(x, gains, steps=300, seed=0):
    """Fit up to 16 training events with no regularization; preserve caller RNG.

    Failure is diagnostic, not proof of a bug: identical features can have
    different outcomes. Explicitly report that irreducible squared-error floor.
    """
    from .controller import Controller
    if steps < 1:
        raise ValueError("Tiny-fit steps must be positive")
    x, gains = x.detach().cpu(), gains.detach().cpu()
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=gen)[:16]
    xt, yt = x[indices], gains[indices, 1:]
    _, inverse = torch.unique(xt, dim=0, return_inverse=True)
    floor_pred = torch.empty_like(yt)
    for group in inverse.unique():
        mask = inverse == group
        floor_pred[mask] = yt[mask].mean(0)
    floor = float((floor_pred-yt).square().mean())
    with torch.random.fork_rng(devices=[]):
        # CPU-only diagnostic: do not reseed any CUDA generators.
        torch.random.default_generator.manual_seed(seed)
        model = Controller()
        model.mean.copy_(xt.mean(0))
        model.std.copy_(xt.std(0, unbiased=False).clamp_min(.01))
        opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
        initial = float(yt.square().mean())  # zero-gain output initialization
        best, final, history = initial, initial, []
        for step in range(1, steps+1):
            loss = (model(xt)[:, 1:]-yt).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                final = float((model(xt)[:, 1:]-yt).square().mean())
            if not torch.isfinite(torch.tensor(final)):
                raise FloatingPointError("Nonfinite tiny-set fit")
            best = min(best, final)
            if step == 1 or step % 50 == 0 or step == steps:
                history.append({"step": step, "MSE": final})
    ratio = best/initial if initial > 1e-12 else None
    status = ("no_signal" if initial <= 1e-12 else
              "conflicting_targets" if floor > .05*initial else
              "passed" if ratio <= .05 else "needs_investigation")
    return {"status": status, "n_rows": len(xt), "training_row_indices": indices.tolist(),
            "steps": steps, "initial_MSE": initial, "best_MSE": best, "last_MSE": final,
            "best_over_initial_MSE": ratio, "duplicate_feature_MSE_floor": floor,
            "pass_rule": "best MSE <= 5% of zero-gain initial MSE; diagnostic only",
            "history": history}


def diagnose_rows(train_rows, dev_rows, seed=0, tiny_steps=300):
    from .controller import training_tensors
    if not train_rows or not dev_rows:
        raise ValueError("Nonempty train/dev rows required")
    if {r["scene_id"] for r in train_rows} & {r["scene_id"] for r in dev_rows}:
        raise ValueError("Scene leakage between diagnostic train/dev")
    x, g = training_tensors(train_rows, "cpu")
    xd, gd = training_tensors(dev_rows, "cpu")
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(.01)
    train_mean = g.mean(0)
    baselines = {}
    for name, pred in (("zero_gain", torch.zeros_like(gd)),
                       ("train_mean_gain", train_mean.expand_as(gd))):
        baselines[name] = regression_metrics(pred, gd, xd)
    # Stage-only predictor answers whether time alone explains the result.
    stage = FEATURES.index("update_time_seconds")
    pred = train_mean.expand_as(gd).clone()
    for time in x[:, stage].unique():
        pred[xd[:, stage] == time] = g[x[:, stage] == time].mean(0)
    baselines["stage_mean_gain"] = regression_metrics(pred, gd, xd)
    # Fixed ridge penalty, fit on Train only; Dev never chooses its parameters.
    for name, cols in (("residual_ridge", [FEATURES.index("weighted_rmse")]),
                       ("all_feature_ridge", list(range(len(FEATURES))))):
        z = ((x-mean)/std).clamp(-20, 20)[:, cols].double()
        zd = ((xd-mean)/std).clamp(-20, 20)[:, cols].double()
        z = torch.cat((torch.ones(len(z), 1), z), 1)
        zd = torch.cat((torch.ones(len(zd), 1), zd), 1)
        penalty = torch.eye(z.shape[1], dtype=z.dtype)
        penalty[0, 0] = 0
        beta = torch.linalg.solve(z.T@z + penalty, z.T@g.double())
        baselines[name] = regression_metrics((zd@beta).float(), gd, xd)
    data = {}
    for name, xx, gg in (("train", x, g), ("dev", xd, gd)):
        data[name] = {"opportunity": opportunity(xx, gg),
                      "by_update_seconds": {str(float(t)): opportunity(xx[xx[:, stage] == t], gg[xx[:, stage] == t])
                                            for t in xx[:, stage].unique()},
                      "features": feature_health(xx, mean, std)}
    return {"schema": 1, "scope": "train/dev only; offline single-edit diagnostics, not deployed performance",
            "metric": "final_60_frame_energy_score_gain", "seed": seed,
            "data": data, "dev_baselines": baselines,
            "tiny_overfit": tiny_overfit(x, g, tiny_steps, seed)}


@torch.no_grad()
def activation_stats(model, x):
    z = ((x[:256]-model.mean)/model.std).clamp(-20, 20)
    stats = {}
    for i, layer in enumerate(model.net):
        z = layer(z)
        stats[str(i)] = {"mean": float(z.mean()), "std": float(z.std(unbiased=False)),
                         "abs_max": float(z.abs().max())}
    return stats


@torch.no_grad()
def parameter_updates(model, before):
    stats = {}
    for name, p in model.named_parameters():
        old = float(before[name].norm())
        delta = float((p-before[name]).norm())
        stats[name] = {"weight_norm_before": old, "update_norm": delta,
                       "update_weight_ratio": delta/old if old > 1e-12 else None}
    return stats
