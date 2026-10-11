"""Synthetic-only checks; no production images, descriptors or labels opened."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import v5_8_spatial_diagnosis as diagnostic


def test_synthetic_geometry_cosine_and_pool_parity():
    result = diagnostic.self_test()
    assert result['status'] == 'passed'
    assert result['scientific_inputs_opened'] is False
    assert result['old_pool_bank_exact']
    assert result['H_coarse_equals_DR_bitwise']
    assert result['common_support_closed_interval']


def test_global_and_correct_y_row_ranks_are_distinct():
    offsets = diagnostic.old.OFFSETS
    def at(x, y):
        return int(np.flatnonzero(np.all(offsets == [x, y], axis=1))[0])
    cost = np.full(625, 10.)
    cost[at(-5, 5)] = 0
    cost[at(1, -2)] = 1
    cost[at(2, -2)] = 2
    cost[at(-4, 4)] = 2
    row = diagnostic.correct_row_rank(cost, np.array([2.1, -1.9]))
    assert row['true_cell_global_rank'] == 3
    assert row['true_cell_global_ties'] == 2
    assert row['true_cell_global_cost_gap'] == 2
    assert row['true_cell_row_rank'] == 2
    assert row['true_cell_row_ties'] == 1
    assert row['true_cell_row_cost_gap'] == 1
    assert row['correct_row_best_dx'] == 1


def test_frozen_mean_skill_and_unequal_subset_counts():
    # Constant synthetic truth has zero Dev variance: only the fixed FIT mean
    # yields the expected finite skills. All data here are hand-constructed.
    truth = np.tile([1., 2., 3., 4.], (512, 1))/224
    predictions = np.stack((truth, truth+1/224, np.zeros_like(truth), -truth, truth/2), axis=1)
    common = np.ones((512, 2), dtype=bool)
    common[::3, 0] = False
    common[::5, 1] = False
    scores, samples, sums, counts = diagnostic.summarize(predictions, truth, common, np.zeros(8))
    assert samples.shape == (10000, 2, 5, 2)
    assert sums.shape == (2, 256, 5, 2)
    np.testing.assert_array_equal(counts[1], common.reshape(256, 2, 2).sum((1, 2)))
    for subset in scores.values():
        assert len(subset['contrasts']) == 9
        for arm, expected in [('R', 1.), ('DR', 0.), ('DBR', -3.), ('H', .75)]:
            np.testing.assert_allclose(subset['arms'][arm]['per_coordinate_skill'], expected, atol=1e-14, rtol=0)
        assert subset['objects']+subset['objects_excluded'] == 1024
        assert subset['episodes_with_objects']+subset['episodes_excluded'] == 512
        assert subset['worlds_with_objects']+subset['worlds_excluded'] == 256
    included_truth = truth.reshape(512, 2, 2)[common]
    expected_dx = np.mean(included_truth[:, 0]**2)
    np.testing.assert_allclose(scores['common_unclamped']['arms']['DR']['metrics']['dx_mse']['point'], expected_dx, atol=1e-15, rtol=0)


def test_phase_matched_contrasts_keep_registered_arm_order():
    assert diagnostic.METHODS == ('R', 'BR', 'DR', 'DBR', 'H')
    mapping = {name: np.array(weights) for name, weights, _ in diagnostic.CONTRASTS}
    values = np.array([1., 5., 3., 8., 2.])
    assert mapping['matched_dense_phase_filter'] @ values == 3
    assert mapping['coarse_sampling_of_H'] @ values == 1
    assert mapping['phase_matched_interaction'] @ values == 2
    np.testing.assert_array_equal(mapping['phase_matched_interaction'],
        mapping['B_with_coarse_sampling']-mapping['matched_dense_phase_filter'])
