"""Mechanism tests: causality, local interventions and genuine history need."""

from dataclasses import replace
import json

import numpy as np
import pytest

from pmr.state_memory.data import TaskConfig, corrupt_history, history_pairs, make_batch


def test_future_blind_observations_and_velocity_sensor():
    config = TaskConfig(observation_noise=0.0)
    batch = make_batch(config, 12, 7)
    for i, reveal in enumerate(batch["reveal_step"]):
        # A world with a future jump and a world without it have exactly the
        # same observations before the jump, including measured velocities.
        np.testing.assert_array_equal(batch["observation"][i, :reveal], batch["paired_observation"][i, :reveal])
        visible = batch["observation"][i, ..., 4] > 0
        np.testing.assert_array_equal(batch["observation"][i, ..., :4][visible], batch["state"][i][visible])
    # Moving the announcement into the future cannot change its earlier input.
    delayed = make_batch(replace(config, reveal_step=16), 12, 7)
    for i, reveal in enumerate(batch["reveal_step"]):
        np.testing.assert_array_equal(batch["observation"][i, :reveal], delayed["observation"][i, :reveal])


def test_hidden_padding_is_zero_and_roles_randomized():
    batch = make_batch(TaskConfig(), 40, 29)
    for key in ("observation", "paired_observation", "history_pair_observation"):
        obs = batch[key]
        assert np.isfinite(obs).all()
        assert np.all(obs[..., :4][obs[..., 4] == 0] == 0)
    assert len(set(batch["changed_object"])) == 3
    assert len(set(batch["preserved_object"])) == 3
    assert len(set(batch["reveal_step"])) > 1
    for i, preserved in enumerate(batch["preserved_object"]):
        assert preserved != batch["changed_object"][i]
        assert np.all(batch["observation"][i, 2:, preserved] == 0)
        assert np.all(batch["observation"][i, :2, :, 4] == 1)


def test_paired_local_revision_preserves_other_objects_exactly():
    batch = make_batch(TaskConfig(), 15, 19)
    for i, changed in enumerate(batch["changed_object"]):
        others = np.arange(3) != changed
        for key, paired_key in (("observation", "paired_observation"), ("targets", "paired_targets"), ("state", "paired_state")):
            np.testing.assert_array_equal(batch[key][i][:, others], batch[paired_key][i][:, others])
        reveal = batch["reveal_step"][i]
        assert not np.array_equal(batch["observation"][i, reveal, changed], batch["paired_observation"][i, reveal, changed])


def test_split_reproducibility_independence_and_batch_prefix():
    config = TaskConfig()
    a = make_batch(config, 5, 42, "train")
    b = make_batch(config, 5, 42, "train")
    larger = make_batch(config, 8, 42, "train")
    heldout = make_batch(config, 5, 42, "eval")
    np.testing.assert_array_equal(a["observation"], b["observation"])
    np.testing.assert_array_equal(a["observation"], larger["observation"][:5])
    assert a["manifest"] == b["manifest"]
    assert a["episode_id"] == larger["episode_id"][:5]
    assert set(a["episode_id"]).isdisjoint(heldout["episode_id"])
    assert set(a["manifest"]["latent_episode_ids"]).isdisjoint(heldout["manifest"]["latent_episode_ids"])
    assert not np.array_equal(a["observation"], heldout["observation"])
    json.dumps(a["manifest"])


def test_no_unannounced_jump_targets_and_extended_forecast():
    config = TaskConfig(dt=1.0, observation_noise=0.0, horizons=(0, 1, 4, 30))
    batch = make_batch(config, 9, 63)
    assert batch["targets"].shape == (9, 24, 3, 4, 2)
    for i, (changed, reveal) in enumerate(zip(batch["changed_object"], batch["reveal_step"])):
        for t in range(config.length):
            for h_idx, h in enumerate(config.horizons):
                assert batch["target_mask"][i, t, changed, h_idx] == (not (t < reveal <= t + h))
        others = np.arange(config.n_objects) != changed
        assert batch["target_mask"][i][:, others].all()
    assert batch["paired_target_mask"].all()
    # Horizon-zero reconstructs the same state. Final forecasts exceed T using
    # the already-announced dynamics, not truncated or wrapped arrays.
    np.testing.assert_array_equal(batch["targets"][..., 0, :], batch["state"][..., :2])
    for h_idx, horizon in enumerate(config.horizons):
        expected = batch["state"][:, -1, :, :2] + horizon * config.dt * batch["state"][:, -1, :, 2:]
        np.testing.assert_allclose(batch["targets"][:, -1, :, h_idx], expected, rtol=2e-6, atol=2e-6)


def test_current_window_ambiguity_requires_different_histories():
    pairs = history_pairs(TaskConfig(), 10, 23, window=8)
    np.testing.assert_array_equal(pairs["observation"][:, 2:], pairs["paired_observation"][:, 2:])
    for i, preserved in enumerate(pairs["preserved_object"]):
        assert not np.array_equal(pairs["observation"][i, :2, preserved], pairs["paired_observation"][i, :2, preserved])
        assert np.linalg.norm(pairs["targets"][i, -1, preserved] - pairs["paired_targets"][i, -1, preserved]) > 1e-5
        others = np.arange(3) != preserved
        np.testing.assert_array_equal(pairs["targets"][i][:, others], pairs["paired_targets"][i][:, others])
    np.testing.assert_array_equal(pairs["target_mask"], pairs["paired_target_mask"])


def test_history_pair_initial_state_remains_within_training_prior():
    batch = make_batch(TaskConfig(observation_noise=0.0), 40, 91)
    initial = batch["history_pair_state"][:, 0]
    assert np.all(np.abs(initial[..., :2]) <= 1.0)
    assert np.all(np.abs(initial[..., 2:]) <= 0.08)


@pytest.mark.parametrize("mode", ["correct", "delete", "shuffle", "stale"])
def test_history_corruption_is_local_and_preserves_labels(mode):
    batch = make_batch(TaskConfig(), 6, 31)
    original_obs = batch["observation"].copy()
    original_targets = batch["targets"].copy()
    corrupt = corrupt_history(batch["observation"], mode, seed=17)
    np.testing.assert_array_equal(corrupt[:, 2:], original_obs[:, 2:])
    np.testing.assert_array_equal(batch["observation"], original_obs)
    np.testing.assert_array_equal(batch["targets"], original_targets)
    assert not np.shares_memory(corrupt, batch["observation"])
    assert np.all(corrupt[..., :4][corrupt[..., 4] == 0] == 0)
    if mode == "correct":
        np.testing.assert_array_equal(corrupt, original_obs)
    elif mode == "delete":
        assert np.all(corrupt[:, :2] == 0)
    elif mode == "shuffle":
        assert all(not np.array_equal(corrupt[i, :2], original_obs[i, :2]) for i in range(6))
        np.testing.assert_array_equal(corrupt, corrupt_history(original_obs, mode, 17))
    else:
        np.testing.assert_array_equal(corrupt[:, 1], original_obs[:, 0])


def test_single_scene_history_shuffle_and_long_sequence_prefix():
    config = TaskConfig()
    batch = make_batch(config, 1, 10)
    corrupt = corrupt_history(batch["observation"], "shuffle", 41)
    assert not np.array_equal(corrupt[:, :2], batch["observation"][:, :2])
    longer = make_batch(replace(config, length=96), 1, 10)
    np.testing.assert_array_equal(batch["observation"], longer["observation"][:, :24])
    np.testing.assert_array_equal(batch["targets"], longer["targets"][:, :24])


def test_nonunit_dt_rejected_before_creating_inconsistent_targets():
    with pytest.raises(ValueError, match=r"require dt=1\.0"):
        TaskConfig(dt=0.5)


@pytest.mark.parametrize("kwargs", [{"n_objects": 2}, {"horizons": (-1,)}, {"dt": 0}, {"observation_noise": -1}, {"reveal_step": 1}, {"reveal_step": 23}])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        TaskConfig(**kwargs)
