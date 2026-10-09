"""Information-boundary and state-isolation tests for V4 memory models."""

import pytest
import torch

from pmr.state_memory.models import make_model


NAMES = ("window", "retrieval", "gru", "slot", "analytic")


def observations(batch=2, steps=9, objects=3):
    generator = torch.Generator().manual_seed(18)
    obs = torch.randn(batch, steps, objects, 5, generator=generator)
    visible = torch.rand(batch, steps, objects, generator=generator) > 0.4
    obs[..., :4] *= visible[..., None]
    obs[..., 4] = visible
    return obs


@pytest.mark.parametrize("name", NAMES)
def test_causality_shape_and_finite_unobserved_history(name):
    torch.manual_seed(0)
    model = make_model(name, hidden_dim=8, window=3, memory_slots=4).eval()
    obs = observations()
    obs[:, :2] = 0
    full = model(obs)
    assert full.shape == (2, 9, 3, 4)
    assert torch.isfinite(full).all()
    for end in (1, 4, 7):
        torch.testing.assert_close(model(obs[:, :end]), full[:, :end], atol=1e-6, rtol=1e-5)
    changed = obs.clone()
    changed[:, 5:, :, :4] += 100
    torch.testing.assert_close(model(changed)[:, :5], full[:, :5])


@pytest.mark.parametrize("name", NAMES)
def test_scene_reset_batch_independence_and_hidden_values_ignored(name):
    torch.manual_seed(1)
    model = make_model(name, hidden_dim=8, memory_slots=4).eval()
    obs = observations()
    expected = model(obs)
    model(torch.ones_like(obs) * 5)
    torch.testing.assert_close(model(obs), expected)
    separate = torch.cat([model(scene[None]) for scene in obs], dim=0)
    torch.testing.assert_close(separate, expected, atol=1e-6, rtol=1e-5)
    altered = obs.clone()
    hidden = obs[..., 4] == 0
    altered[..., :4][hidden] = 10000
    torch.testing.assert_close(model(altered), expected)


def test_window_cannot_remember_observations_outside_its_window():
    torch.manual_seed(2)
    model = make_model("window", hidden_dim=8, window=3)
    obs = observations()
    changed = obs.clone()
    changed[:, :5] = torch.randn_like(changed[:, :5])
    changed[:, :5, :, 4] = 1
    torch.testing.assert_close(model(obs)[:, 7:], model(changed)[:, 7:])


def test_retrieval_retains_old_observations_but_evicts_at_fixed_budget():
    torch.manual_seed(3)
    model = make_model("retrieval", hidden_dim=8, memory_slots=2)
    obs = torch.zeros(1, 10, 3, 5)
    obs[:, 0, 0] = torch.tensor([1., 2., 0.5, 0., 1.])
    changed = obs.clone()
    changed[:, 0, 0, 0] = 10
    # History survives a gap much longer than the window baseline's W.
    assert not torch.allclose(model(obs)[:, -1], model(changed)[:, -1])
    # Two subsequent visible object events fill the budget and evict that fact.
    obs[:, 4, 1] = torch.tensor([4., 5., 0., 0., 1.])
    obs[:, 6, 2] = torch.tensor([6., 7., 0., 0., 1.])
    changed[:, 4:] = obs[:, 4:]
    torch.testing.assert_close(model(obs)[:, 6:], model(changed)[:, 6:])
    assert model.metadata()["memory_scalars_per_scene"] == 14


def test_slot_new_evidence_does_not_change_other_objects():
    torch.manual_seed(4)
    model = make_model("slot", hidden_dim=8)
    obs = observations()
    changed = obs.clone()
    changed[:, 4:, 1, :4] += 100
    changed[:, 4:, 1, 4] = 1
    original, revised = model(obs), model(changed)
    torch.testing.assert_close(original[:, :, (0, 2)], revised[:, :, (0, 2)])
    assert not torch.allclose(original[:, 4:, 1], revised[:, 4:, 1])


def test_slot_can_advance_latent_state_while_occluded():
    model = make_model("slot", hidden_dim=4)
    with torch.no_grad():
        model.dynamics[-1].bias.fill_(0.25)
        model.decoder.weight.copy_(torch.eye(4))
        model.decoder.bias.zero_()
    obs = torch.zeros(1, 4, 3, 5)
    obs[:, 0, :, 4] = 1
    pred = model(obs)
    torch.testing.assert_close(pred[:, 1:] - pred[:, :-1], torch.full((1, 3, 3, 4), 0.25))


def test_analytic_dead_reckoning_and_local_correction():
    model = make_model("analytic")
    obs = torch.zeros(1, 6, 3, 5)
    obs[0, 1, 0] = torch.tensor([2., 3., 1., -0.5, 1.])
    obs[0, 4, 0] = torch.tensor([20., 30., -2., 4., 1.])
    pred = model(obs)
    expected = torch.tensor([
        [0., 0., 0., 0.], [2., 3., 1., -0.5], [3., 2.5, 1., -0.5],
        [4., 2., 1., -0.5], [20., 30., -2., 4.], [18., 34., -2., 4.],
    ])
    torch.testing.assert_close(pred[0, :, 0], expected)
    assert torch.count_nonzero(pred[0, :, 1:]) == 0
    assert model.metadata()["trainable_parameters"] == 0


@pytest.mark.parametrize("name", NAMES[:-1])
def test_trainable_models_receive_finite_gradients(name):
    torch.manual_seed(5)
    model = make_model(name, hidden_dim=8, memory_slots=4)
    prediction = model(observations())
    horizons = torch.tensor([0., 1., 4.])
    future = prediction[..., None, :2] + prediction[..., None, 2:] * horizons[:, None]
    loss = (future - 1).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert model.metadata()["trainable_parameters"] > 0


def test_metadata_exposes_recurrent_budget_without_claiming_parameter_match():
    gru = make_model("gru", n_objects=3, hidden_dim=8).metadata()
    slot = make_model("slot", n_objects=3, hidden_dim=8).metadata()
    assert gru["memory_scalars_per_scene"] == slot["memory_scalars_per_scene"] == 24
    assert gru["trainable_parameters"] != slot["trainable_parameters"]
    assert slot["object_correspondence_given"]


def test_invalid_contracts_fail_clearly():
    with pytest.raises(ValueError, match="unknown"):
        make_model("future_oracle")
    with pytest.raises(ValueError, match="window"):
        make_model("window", window=0)
    with pytest.raises(ValueError, match="memory_slots"):
        make_model("retrieval", memory_slots=0)
    with pytest.raises(ValueError, match="positive"):
        make_model("gru", n_objects=0)
    with pytest.raises(ValueError, match="expected"):
        make_model("slot")(torch.zeros(2, 5, 4, 5))
    with pytest.raises(ValueError, match="at least one"):
        make_model("analytic")(torch.zeros(2, 0, 3, 5))
    with pytest.raises(TypeError, match="floating"):
        make_model("analytic")(torch.zeros(2, 5, 3, 5, dtype=torch.long))
