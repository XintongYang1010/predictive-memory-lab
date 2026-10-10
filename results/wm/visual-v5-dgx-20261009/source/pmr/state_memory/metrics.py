"""Episode-level metrics for controlled state-memory experiments.

Position RMSE uses Euclidean distance (sum over x/y before averaging).
Forecast MSE is per coordinate. Missing measurements are ``None``, never NaN.
The optional paired rollouts diagnose interference and use of old history;
they are evaluation interventions, not inputs available to the predictor.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


MEAN_METRICS = (
    "future_mse",
    "hidden_position_rmse",
    "changed_post_position_rmse",
    "preserved_post_position_rmse",
    "unsolicited_update_rmse",
    "history_pair_difference_rmse",
    "history_pair_predicted_separation",
    "history_pair_true_separation",
)


def _array(value, shape, name, *, finite=True):
    result = np.asarray(value)
    if result.shape != shape:
        raise ValueError(f"{name} has shape {result.shape}; expected {shape}")
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _position_rmse(delta):
    if delta.size == 0:
        return None
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def evaluate_predictions(
    prediction,
    batch: dict,
    horizons=(0, 1, 4),
    recovery_threshold=.15,
    paired_prediction=None,
    history_prediction=None,
) -> list[dict]:
    """Score state predictions ``[episode, time, object, (x,y,vx,vy)]``.

    Forecasts extrapolate each estimated state by ``xy + horizon * velocity``.
    ``target_mask`` selects object/horizon positions, with two coordinates per
    selected position. Hidden-state and recovery metrics use true current
    states, not extrapolated forecasts. Recovery needs two consecutive frames
    at or after reveal. An unrecovered episode has latency ``None``.

    ``paired_prediction`` is the no-change rollout; its displacement of the
    unaffected object measures unwanted cross-object interference after reveal.
    ``history_prediction`` differs only in early history. Its final difference
    for the preserved object is compared with that world's true difference.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[-1] != 4:
        raise ValueError("prediction must have shape [B,T,N,4]")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction contains nonfinite values")
    b, t, n, _ = prediction.shape
    if not b or not t or not n:
        raise ValueError("prediction dimensions B, T and N must be nonzero")
    horizons = np.asarray(horizons, dtype=np.float64)
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or (horizons < 0).any():
        raise ValueError("horizons must be a nonempty sequence of finite nonnegative values")
    if not math.isfinite(recovery_threshold) or recovery_threshold < 0:
        raise ValueError("recovery_threshold must be finite and nonnegative")
    state = _array(batch["state"], prediction.shape, "state")
    observation = _array(batch["observation"], (b, t, n, 5), "observation")
    h = len(horizons)
    target = _array(batch["targets"], (b, t, n, h, 2), "targets", finite=False)
    mask = _array(batch["target_mask"], (b, t, n, h), "target_mask").astype(bool)
    if not np.isfinite(target[mask]).all():
        raise ValueError("targets contains nonfinite values at scored positions")
    ids = list(batch["episode_id"])
    if len(ids) != b or len(set(ids)) != b:
        raise ValueError("episode_id must contain one unique ID per episode")
    indices = {}
    for key, upper in (("changed_object", n), ("preserved_object", n), ("reveal_step", t)):
        value = _array(batch[key], (b,), key)
        if np.any(value != np.floor(value)) or np.any(value < 0) or np.any(value >= upper):
            raise ValueError(f"{key} must contain valid integer indices")
        indices[key] = value.astype(int)
    if np.any(indices["changed_object"] == indices["preserved_object"]):
        raise ValueError("changed_object and preserved_object must differ")
    if paired_prediction is not None:
        paired_prediction = _array(paired_prediction, prediction.shape, "paired_prediction")
    if history_prediction is not None:
        history_prediction = _array(history_prediction, prediction.shape, "history_prediction")
        history_state = _array(batch["history_pair_state"], prediction.shape, "history_pair_state")
    forecast = prediction[..., None, :2] + prediction[..., None, 2:] * horizons[None, None, None, :, None]
    rows = []
    for i in range(b):
        changed = indices["changed_object"][i]
        preserved = indices["preserved_object"][i]
        reveal = indices["reveal_step"][i]
        error = prediction[i, ..., :2] - state[i, ..., :2]
        forecast_error = forecast[i][mask[i]] - target[i][mask[i]]
        distances = np.linalg.norm(error[reveal:, changed], axis=-1)
        close = distances <= recovery_threshold
        recovered_at = np.flatnonzero(close[:-1] & close[1:])
        row = {
            "episode_id": ids[i],
            "future_mse": float(np.mean(forecast_error ** 2)) if forecast_error.size else None,
            "hidden_position_rmse": _position_rmse(error[observation[i, ..., 4] < .5]),
            "changed_post_position_rmse": _position_rmse(error[reveal:, changed]),
            "preserved_post_position_rmse": _position_rmse(error[reveal:, preserved]),
            "recovered": bool(len(recovered_at)),
            "recovery_latency": int(recovered_at[0]) if len(recovered_at) else None,
            "unsolicited_update_rmse": None,
            "history_pair_difference_rmse": None,
            "history_pair_predicted_separation": None,
            "history_pair_true_separation": None,
        }
        if paired_prediction is not None:
            row["unsolicited_update_rmse"] = _position_rmse(
                prediction[i, reveal:, preserved, :2] - paired_prediction[i, reveal:, preserved, :2]
            )
        if history_prediction is not None:
            predicted_difference = prediction[i, -1, preserved, :2] - history_prediction[i, -1, preserved, :2]
            true_difference = state[i, -1, preserved, :2] - history_state[i, -1, preserved, :2]
            row["history_pair_difference_rmse"] = float(np.linalg.norm(predicted_difference - true_difference))
            row["history_pair_predicted_separation"] = float(np.linalg.norm(predicted_difference))
            row["history_pair_true_separation"] = float(np.linalg.norm(true_difference))
        rows.append(row)
    return rows


def _valid_values(rows, metric):
    values = [row[metric] for row in rows if row.get(metric) is not None]
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{metric} contains nonfinite values")
    return values


def summarize(rows: Sequence[dict]) -> dict:
    """Unweighted mean across episodes; recovery latency is success-only.

    Always report the never-recovered rate beside conditional latency: a
    method that fails often must not appear fast by dropping failed episodes.
    Each metric has its own denominator when episodes lack scored positions.
    """
    rows = list(rows)
    result = {"episodes": len(rows)}
    counts = {}
    for metric in MEAN_METRICS:
        values = _valid_values(rows, metric)
        result[metric] = float(np.mean(values)) if values else None
        counts[metric] = len(values)
    observed = [row for row in rows if row.get("recovered") is not None]
    successes = [row for row in observed if row["recovered"]]
    latencies = _valid_values(successes, "recovery_latency")
    if len(latencies) != len(successes):
        raise ValueError("recovered episodes must have recovery_latency")
    if any(row.get("recovery_latency") is not None for row in observed if not row["recovered"]):
        raise ValueError("unrecovered episodes must have recovery_latency=None")
    result.update({
        "recovered_episodes": len(successes),
        "never_recovered_rate": float(1 - len(successes) / len(observed)) if observed else None,
        "recovery_latency_success_only_mean": float(np.mean(latencies)) if latencies else None,
        "recovery_evaluated_episodes": len(observed),
        "metric_episode_counts": counts,
    })
    return result


def bootstrap_paired(reference_rows, candidate_rows, metric="future_mse", seed=0, draws=1000) -> dict:
    """Bootstrap matched episodes for a *single training seed*.

    Positive gain means a lower candidate error (reference minus candidate).
    The percentile interval resamples paired episodes, never individual times
    or objects. ``seed`` controls bootstrap RNG, not model-training seeds.
    Null metrics are excluded pairwise, and the exclusion count is explicit.
    Multiple model seeds must be reported with separate calls.
    """
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    reference_rows, candidate_rows = list(reference_rows), list(candidate_rows)
    indexed = []
    model_seeds = []
    for rows in (reference_rows, candidate_rows):
        ids = [row["episode_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate episode_id; do not pool training seeds")
        indexed.append(dict(zip(ids, rows)))
        seeds = {row.get("training_seed", row.get("seed")) for row in rows}
        seeds.discard(None)
        if len(seeds) > 1:
            raise ValueError("do not pool multiple training seeds")
        model_seeds.append(seeds)
    if model_seeds[0] and model_seeds[1] and model_seeds[0] != model_seeds[1]:
        raise ValueError("reference and candidate training seeds differ")
    reference, candidate = indexed
    if reference.keys() != candidate.keys():
        raise ValueError("reference and candidate episode_id sets must match exactly")
    gains = []
    for episode_id in sorted(reference, key=str):
        a, b = reference[episode_id].get(metric), candidate[episode_id].get(metric)
        if a is None or b is None:
            continue
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError(f"{metric} contains nonfinite values")
        gains.append(a - b)
    result = {
        "metric": metric, "gain_definition": "reference_minus_candidate",
        "episodes": len(reference), "paired_episodes": len(gains),
        "excluded_episodes": len(reference) - len(gains),
        "mean_gain": None, "ci95": [None, None],
        "bootstrap_draws": int(draws), "bootstrap_seed": seed,
    }
    if gains:
        gains = np.asarray(gains, dtype=np.float64)
        rng = np.random.default_rng(seed)
        # Bound temporary memory for longer suites without changing samples.
        means = np.empty(draws, dtype=np.float64)
        chunk = max(1, 1_000_000 // len(gains))
        for start in range(0, draws, chunk):
            indices = rng.integers(0, len(gains), size=(min(chunk, draws - start), len(gains)))
            means[start:start + len(indices)] = gains[indices].mean(axis=1)
        result["mean_gain"] = float(gains.mean())
        result["ci95"] = np.quantile(means, [.025, .975]).tolist()
    return result
