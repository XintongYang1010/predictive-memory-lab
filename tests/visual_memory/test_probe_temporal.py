import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('probe_temporal',ROOT/'scripts/diagnose_visual_probe_temporal_v5.py')
temporal=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(temporal)


class TemporalControls(unittest.TestCase):
    def test_single_frame_and_query_swap_kernels_match_explicit_features(self):
        rng=np.random.default_rng(2209)
        train=rng.normal(size=(8,2,13));dev=rng.normal(size=(5,2,13))
        gram=train.reshape(16,13)@train.reshape(16,13).T
        cross=dev.reshape(10,13)@train.reshape(16,13).T
        views=temporal.temporal_views(gram,cross)
        for index,name in enumerate(['frame6','frame7']):
            np.testing.assert_allclose(views[name]['concat'][0],train[:,index]@train[:,index].T,atol=1e-12)
            np.testing.assert_allclose(views[name]['concat'][1],dev[:,index]@train[:,index].T,atol=1e-12)
        expected=dev[:,::-1].reshape(5,26)@train.reshape(8,26).T
        np.testing.assert_allclose(temporal.swapped_concat_cross(cross),expected,atol=1e-12)

    def test_frame_selection_uses_train_score_and_ignores_dev_fields(self):
        def control(alpha,score,dev):
            return dict(selected_alpha=alpha,trials=[dict(alpha=alpha,mean_score=score)],dev_r2=dev)
        choices={'frame6':control(.1,.7,-1000),'frame7':control(1.,.8,1000)}
        self.assertEqual(temporal.choose_single_frame(choices)['selected_frame'],'frame6')
        choices['frame7']['trials'][0]['mean_score']=.7
        self.assertEqual(temporal.choose_single_frame(choices)['selected_frame'],'frame6')
        choices['frame7']['trials'][0]['mean_score']=.6
        self.assertEqual(temporal.choose_single_frame(choices)['selected_frame'],'frame7')

    def test_reversed_labels_change_endpoint_and_displacement_sign(self):
        p6=np.array([[.1,.2,.3,.4],[.7,.6,.5,.4]])
        p7=np.array([[.2,.1,.5,.6],[.6,.5,.4,.3]])
        targets=np.concatenate([p7,p7-p6],axis=1)
        reverse=temporal.reversed_targets(targets)
        np.testing.assert_allclose(reverse,np.concatenate([p6,p6-p7],axis=1),atol=1e-15)
        np.testing.assert_allclose(temporal.reversed_targets(reverse),targets,atol=1e-15)


if __name__=='__main__':unittest.main()
