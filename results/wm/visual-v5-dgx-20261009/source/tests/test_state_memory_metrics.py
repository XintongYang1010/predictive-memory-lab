"""Hand-computed checks for V4 measurements and paired inference."""
import copy
import json

import numpy as np
import pytest

from pmr.state_memory.metrics import evaluate_predictions, summarize, bootstrap_paired


def batch_fixture(batch_size=1, length=5):
    state = np.zeros((batch_size, length, 2, 4))
    observation = np.zeros((batch_size, length, 2, 5))
    observation[..., 4] = 1
    return {
        "state": state,
        "observation": observation,
        "targets": np.zeros((batch_size, length, 2, 3, 2)),
        "target_mask": np.ones((batch_size, length, 2, 3), dtype=bool),
        "changed_object": np.zeros(batch_size, dtype=int),
        "preserved_object": np.ones(batch_size, dtype=int),
        "reveal_step": np.ones(batch_size, dtype=int),
        "episode_id": [f"episode-{i}" for i in range(batch_size)],
    }


def test_forecast_uses_velocity_and_only_masked_coordinates():
    batch = batch_fixture()
    prediction = np.zeros_like(batch["state"])
    prediction[0, 0, 0] = [2, 3, 1, -1]
    batch["targets"].fill(np.nan)
    batch["target_mask"].fill(False)
    # At horizon 4, forecast is [6,-1]. Difference to [3,3] = [3,-4].
    batch["targets"][0, 0, 0, 2] = [3, 3]
    batch["target_mask"][0, 0, 0, 2] = True
    row = evaluate_predictions(prediction, batch)[0]
    assert row["future_mse"] == pytest.approx(25 / 2)
    assert row["hidden_position_rmse"] is None
    json.dumps(row, allow_nan=False)


def test_hidden_and_post_reveal_metrics_use_euclidean_distance():
    batch = batch_fixture(length=4)
    prediction = np.zeros_like(batch["state"])
    # Only this observation is hidden; 3-4-5 Euclidean distance.
    prediction[0, 0, 1, :2] = [3, 4]
    batch["observation"][0, 0, 1, 4] = 0
    prediction[0, 1:, 0, 0] = [1, 2, 2]
    prediction[0, 1:, 1, 1] = [3, 0, 0]
    row = evaluate_predictions(prediction, batch)[0]
    assert row["hidden_position_rmse"] == 5
    assert row["changed_post_position_rmse"] == pytest.approx(np.sqrt(3))
    # Excludes the preserved object's pre-reveal error of 5.
    assert row["preserved_post_position_rmse"] == pytest.approx(np.sqrt(3))


def test_recovery_requires_two_consecutive_frames_and_preserves_failures():
    batch = batch_fixture(batch_size=3, length=6)
    prediction = np.zeros_like(batch["state"])
    prediction[0, 1:, 0, 0] = [.1, .3, .1, .1, .4]
    prediction[1, 1:, 0, 0] = [.3, .1, .3, .1, .3]
    prediction[2, 1:, 0, 0] = [.3, .3, .3, .3, .1]
    rows = evaluate_predictions(prediction, batch)
    assert [r["recovery_latency"] for r in rows] == [2, None, None]
    assert [r["recovered"] for r in rows] == [True, False, False]
    result = summarize(rows)
    assert result["recovery_latency_success_only_mean"] == 2
    assert result["never_recovered_rate"] == pytest.approx(2 / 3)
    assert result["recovered_episodes"] == 1
    never = summarize(rows[1:])
    assert never["recovery_latency_success_only_mean"] is None
    assert never["never_recovered_rate"] == 1
    json.dumps(never, allow_nan=False)


def test_missing_measurements_and_empty_summaries_are_null():
    batch = batch_fixture()
    batch["target_mask"].fill(False)
    rows = evaluate_predictions(batch["state"], batch)
    assert rows[0]["future_mse"] is None
    assert rows[0]["unsolicited_update_rmse"] is None
    summary = summarize(rows)
    assert summary["future_mse"] is None
    assert summary["metric_episode_counts"]["future_mse"] == 0
    assert summarize([])["never_recovered_rate"] is None
    json.dumps([summary, summarize([])], allow_nan=False)


def test_paired_metrics_separate_interference_from_use_of_history():
    batch = batch_fixture(length=4)
    batch["history_pair_state"] = batch["state"].copy()
    batch["history_pair_state"][0, -1, 1, :2] = [-3, -4]
    prediction = batch["state"].copy()
    paired = prediction.copy()
    paired[0, 1:, 1, :2] = [0, 2]
    # Changed-object alterations must not enter preserved-object interference.
    paired[0, :, 0, :2] = 100
    history = prediction.copy()
    history[0, -1, 1, :2] = [-3, 0]
    row = evaluate_predictions(prediction, batch, paired_prediction=paired, history_prediction=history)[0]
    assert row["unsolicited_update_rmse"] == 2
    assert row["history_pair_difference_rmse"] == 4
    assert row["history_pair_predicted_separation"] == 3
    assert row["history_pair_true_separation"] == 5
    # If the predictions are correct in both worlds, difference error is zero.
    perfect = evaluate_predictions(prediction, batch, history_prediction=batch["history_pair_state"])[0]
    assert perfect["history_pair_difference_rmse"] == 0


def test_invalid_values_fail_loudly_instead_of_becoming_nan():
    batch = batch_fixture()
    prediction = batch["state"].copy()
    prediction[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_predictions(prediction, batch)
    prediction.fill(0)
    batch["targets"][0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="scored positions"):
        evaluate_predictions(prediction, batch)
    with pytest.raises(ValueError, match="nonfinite"):
        summarize([{"future_mse": np.nan}])


def test_episode_averages_are_not_frame_pooled():
    rows = [{"future_mse": 1}, {"future_mse": 3}, {"future_mse": None}]
    result = summarize(rows)
    assert result["future_mse"] == 2
    assert result["metric_episode_counts"]["future_mse"] == 2


def test_bootstrap_aligns_scene_ids_and_has_correct_gain_sign():
    reference = [{"episode_id": "a", "future_mse": 2}, {"episode_id": "b", "future_mse": 5}]
    candidate = [{"episode_id": "b", "future_mse": 4}, {"episode_id": "a", "future_mse": 1}]
    result = bootstrap_paired(reference, candidate, draws=100)
    assert result["mean_gain"] == 1
    assert result["ci95"] == [1, 1]
    assert result["paired_episodes"] == 2
    candidate[0]["future_mse"] = None
    result = bootstrap_paired(reference, candidate, draws=100)
    assert result["mean_gain"] == 1
    assert result["paired_episodes"] == result["excluded_episodes"] == 1


def test_bootstrap_rejects_misalignment_duplicates_and_pooled_seeds():
    reference = [{"episode_id": "a", "future_mse": 2}, {"episode_id": "b", "future_mse": 5}]
    candidate = copy.deepcopy(reference)
    candidate[0]["episode_id"] = "c"
    with pytest.raises(ValueError, match="match exactly"):
        bootstrap_paired(reference, candidate)
    with pytest.raises(ValueError, match="duplicate"):
        bootstrap_paired(reference + reference, reference + reference)
    candidate = copy.deepcopy(reference)
    candidate[0]["training_seed"] = 0
    candidate[1]["training_seed"] = 1
    with pytest.raises(ValueError, match="multiple training seeds"):
        bootstrap_paired(reference, candidate)


def test_bootstrap_reproducible_null_case_and_json_safe():
    reference = [{"episode_id": str(i), "future_mse": float(i)} for i in range(10)]
    candidate = [{"episode_id": str(i), "future_mse": 0.} for i in range(10)]
    a = bootstrap_paired(reference, candidate, seed=12)
    b = bootstrap_paired(reference[::-1], candidate[::-1], seed=12)
    assert a == b
    assert a["mean_gain"] == 4.5
    assert a["ci95"][0] < 4.5 < a["ci95"][1]
    empty = bootstrap_paired([], [])
    assert empty["mean_gain"] is None
    assert empty["ci95"] == [None, None]
    json.dumps(empty, allow_nan=False)
