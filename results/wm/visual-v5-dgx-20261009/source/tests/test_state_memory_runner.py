import json
import numpy as np
import pytest
import torch

from pmr.state_memory.runner import forecast_loss, parse_args, run, BudgetExceeded


def test_forecast_supervision_does_not_score_unannounced_event():
    prediction = torch.zeros(1, 1, 1, 4, requires_grad=True)
    target = torch.zeros(1, 1, 1, 3, 2)
    target[..., 2, :] = 1e6
    mask = torch.tensor([[[[True, True, False]]]])
    loss = forecast_loss(prediction, target, mask)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize('argument,value', [('--steps', '0'), ('--seeds', '42,42'),
    ('--models', 'typo'), ('--eval-lengths', '3'), ('--max-seconds', 'nan')])
def test_invalid_protocol_rejected(argument, value):
    with pytest.raises(SystemExit):
        parse_args(['--out', '/unused', argument, value])


def test_end_to_end_analytic_and_complete_manifest(tmp_path, monkeypatch):
    import pmr.state_memory.runner as runner
    monkeypatch.setattr(runner, 'save_bad_cases', lambda *a, **k: None)
    out = tmp_path / 'run'
    args = parse_args(['--out', str(out), '--models', 'analytic', '--steps', '1',
                      '--train-episodes', '8', '--dev-episodes', '8', '--eval-episodes', '8',
                      '--seeds', '42', '--eval-lengths', '24', '--smoke'])
    run(args)
    stats = json.loads((out / 'run_stats.json').read_text())
    assert stats['status'] == 'complete' and stats['smoke_only']
    summary = json.loads((out / 'summary.json').read_text())
    by_condition = {row['condition']: row for row in summary['results']}
    assert len(by_condition) == 6
    clean = by_condition['clean']['metrics']
    assert clean['future_mse'] < by_condition['delete']['metrics']['future_mse']
    assert clean['unsolicited_update_rmse'] == 0
    data = np.load(out / 'trajectories' / 'data_T24.npz', allow_pickle=False)
    assert data['observation'].shape == (8, 24, 3, 5)
    manifest = json.loads((out / 'datasets_manifest.json').read_text())
    ids = [i for split in manifest.values() for i in split['episode_ids']]
    assert len(ids) == len(set(ids))
    with pytest.raises(FileExistsError):
        run(args)


def test_budget_exhaustion_is_not_a_completed_experiment(tmp_path, monkeypatch):
    import pmr.state_memory.runner as runner
    def exhausted(_deadline):
        raise BudgetExceeded('unit test budget')
    monkeypatch.setattr(runner, 'check_budget', exhausted)
    args = parse_args(['--out', str(tmp_path / 'partial'), '--models', 'analytic',
                      '--train-episodes', '2', '--dev-episodes', '2', '--eval-episodes', '2',
                      '--seeds', '42', '--eval-lengths', '24'])
    with pytest.raises(BudgetExceeded):
        run(args)
    stats = json.loads((tmp_path / 'partial' / 'run_stats.json').read_text())
    assert stats['status'] == 'budget_exceeded'
    assert not stats['completed_models']
