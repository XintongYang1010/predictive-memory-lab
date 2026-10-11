"""Software-only V5.7 saved-surface numerical and provenance tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import v5_7_cost_diagnosis as diagnostic


def test_synthetic_self_test_without_scientific_inputs():
    assert diagnostic.self_test()['status'] == 'passed'


def test_near_cost_ties_reuse_norm_then_dx_then_dy_on_subset():
    xy = diagnostic.OFFSETS
    cost = np.ones(625)
    def at(point):
        return int(np.flatnonzero(np.all(xy == point, axis=1))[0])
    cost[at([5, 5])] = 0
    cost[at([1, 0])] = 8e-13
    cost[at([-1, 0])] = 8e-13
    cost[at([0, -1])] = 8e-13
    assert diagnostic.choose_minimum(cost) == at([-1, 0])
    row = np.flatnonzero(xy[:, 1] == 0)
    assert diagnostic.choose_minimum(cost, row) == at([-1, 0])
    cost[at([-1, 0])] = 2e-12
    assert diagnostic.choose_minimum(cost) == at([0, -1])
    flat = np.zeros(625)
    rank, ties, gap, margin = diagnostic.rank_record(flat, at([0, 0]), row)
    assert (rank, ties, gap, margin) == (1, 25, 0, 0)


def test_oracle_column_and_row_indexing_donor_preserves_object_and_variant():
    # Distinct row and column minima with an off-row/off-column global optimum.
    truth = np.tile([2., 1., -3., -2.], (8, 1)) / 224
    truth[2:4, 1] = -3. / 224
    truth[2:4, 3] = 4. / 224
    costs = np.full((8, 6, 2, 625), 20.)
    xy = diagnostic.OFFSETS
    def at(x, y):
        return int(np.flatnonzero(np.all(xy == [x, y], axis=1))[0])
    costs[..., at(-5, -5)] = 0
    costs[:, :, 0, at(3, 1)] = 1
    costs[:, :, 0, at(2, 4)] = 2
    costs[:, :, 0, at(-4, -3)] = 3
    costs[:, :, 1, at(-2, -2)] = 1
    costs[:, :, 1, at(-3, 3)] = 2
    costs[:, :, 1, at(5, 4)] = 3
    result = diagnostic.diagnose(costs, truth)
    p = result['predictions'][0, 0] * 224
    np.testing.assert_array_equal(p[1], [3, 1, -2, -2])
    np.testing.assert_array_equal(p[3], [2, 4, -3, 3])
    np.testing.assert_array_equal(p[4], [-4, -3, 5, 4])
    np.testing.assert_array_equal(result['donor_episode_indices'][:2], [2, 3])
    np.testing.assert_array_equal(result['donor_row_matches_gt_row'][0], [False, False])
    assert diagnostic.nearest_axis(-6.8) == 0 and diagnostic.nearest_axis(6.8) == 24


def test_correctness_rescue_harm_separates_zero_and_inclusive_one_pixel():
    truth = np.array([[1, 2, -1, 2], [1, 2, -1, 2], [.5, 2, 0, 2], [2, 2, -2, 2.]]) / 224
    old = truth.copy(); new = truth.copy()
    old[0, 0] *= -1; old[0, 2] = 0
    new[1, 0] *= -1; new[1, 2] = 0
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    report = diagnostic.direction_effect(old, new, truth, draws)
    assert report['active_object_episode_coordinates'] == 6
    assert report['groups']['wrong_to_correct']['count'] == 1
    assert report['groups']['zero_to_correct']['count'] == 1
    assert report['groups']['correct_to_wrong']['count'] == 1
    assert report['groups']['correct_to_zero']['count'] == 1
    assert report['net_correctness_gain']['rate'] == 0
    scoring = diagnostic.score(new, truth, np.zeros(4))
    assert scoring['direction_by_axis']['A_dx']['gt_tiny_0_to_1px_exclusive'] == 1
    assert scoring['direction_by_axis']['B_dx']['gt_zero'] == 1
    assert scoring['direction_by_axis']['B_dx']['non_correct_active'] == 1


def test_subset_effect_resamples_world_error_sums_and_counts_not_equal_world_means():
    truth = np.zeros((4, 4)); old = np.zeros((4, 4)); new = np.zeros((4, 4))
    old[0, 0] = 1; old[2:, [0, 2]] = 2
    mask = np.array([[True, False], [False, False], [True, True], [True, True]])
    draws = np.array([[0, 0], [1, 1], [0, 1]])
    effect = diagnostic.masked_dx_effect(old, new, truth, mask, draws)
    assert effect['mean'] == 17 / 5
    np.testing.assert_allclose(effect['ci95'], np.quantile([1, 4, 17 / 5], [.025, .975]))
    assert effect['object_episode_coordinates'] == 5
    empty = diagnostic.masked_dx_effect(old, new, truth, np.zeros_like(mask), draws)
    assert empty['mean'] is None and empty['ci95'] is None


def test_skill_uses_frozen_mean_and_mean_per_axis_not_pooled_variance():
    truth = np.array([[1., 2., 3., 4.], [3., 4., 5., 6.]])
    mean = np.array([0., 1., 0., 2.]); pred = truth + np.array([1., 2., 1., 2.])
    report = diagnostic.score(pred, truth, mean)
    expected = 1 - np.mean((pred - truth) ** 2, axis=0) / np.mean((truth - mean) ** 2, axis=0)
    np.testing.assert_array_equal(list(report['skill_by_axis'].values()), expected)
    assert report['mean_motion_skill'] == expected.mean()
    assert report['mean_motion_skill'] != 1 - np.mean((pred - truth) ** 2) / np.mean((truth - mean) ** 2)


def test_no_input_load_before_publication_and_no_overwrite(tmp_path, monkeypatch):
    opened = []
    def fail_guard(*args):
        raise ValueError('unpublished')
    monkeypatch.setattr(diagnostic, 'validate_publication', fail_guard)
    monkeypatch.setattr(diagnostic.np, 'load', lambda *a, **k: opened.append(True))
    with pytest.raises(ValueError, match='unpublished'):
        diagnostic.run(tmp_path / 'cache.npz', tmp_path / 'new', tmp_path / 'plan', tmp_path / 'pub')
    assert opened == [] and not (tmp_path / 'new').exists()
    monkeypatch.setattr(diagnostic, 'validate_publication', lambda *a: None)
    out = tmp_path / 'existing'; out.mkdir(); (out / 'keep.txt').write_text('retain')
    with pytest.raises(FileExistsError):
        diagnostic.run(tmp_path / 'input' / 'cache.npz', out, tmp_path / 'plan', tmp_path / 'pub')
    assert (out / 'keep.txt').read_text() == 'retain' and opened == []


def test_publication_requires_all_sources_and_plan_bound_after_design(tmp_path, monkeypatch):
    root = tmp_path / 'repo'; root.mkdir(); (root / 'scripts').mkdir()
    source = root / 'scripts' / 'diagnostic.py'; source.write_text('frozen')
    monkeypatch.setattr(diagnostic, 'ROOT', root)
    monkeypatch.setattr(diagnostic, 'SOURCE_FILES', ('scripts/diagnostic.py',))
    plan = root / 'plan.json'; pub = root / 'publication.json'
    plan.write_text(json.dumps({'design': diagnostic.DESIGN, 'sources': diagnostic.source_identity(),
                               'software_verification': {'status': 'passed'}, 'created_unix': 10.}))
    receipt = {'verified_readback': True, 'verified_unix': 11.,
               'files': {'scripts/diagnostic.py': diagnostic.sha(source), 'plan.json': diagnostic.sha(plan)}}
    pub.write_text(json.dumps(receipt)); diagnostic.validate_publication(pub, plan)
    receipt['verified_unix'] = 9.; pub.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='follow'):
        diagnostic.validate_publication(pub, plan)
    receipt['verified_unix'] = 11.; pub.write_text(json.dumps(receipt)); source.write_text('changed')
    with pytest.raises(ValueError, match='source mismatch'):
        diagnostic.validate_publication(pub, plan)
