"""CPU software fixtures only; no GPU, frozen reader fitting, or Test access."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import extract_v5_2_counterfactual as extract


class FakeEncoder:
    def __init__(self):self.calls=[]
    def __call__(self,images):
        self.calls.append(images.copy())
        values=images.astype(np.float32).mean((1,2,3))
        return torch.from_numpy(np.broadcast_to(values[:,None,None],(len(images),16,2)).copy())


def pil_fixture(center,foreground=100):
    image=Image.new("RGB",(224,224),(30,35,42));draw=ImageDraw.Draw(image)
    for x in range(0,224,14):draw.line((x,0,x,224),fill=(42,47,54))
    draw.ellipse((40,40,65,65),fill=(foreground,60,80))
    draw.rectangle((center-5,224*.48,center+5,224*.52),fill=(240,220,50))
    return np.asarray(image).copy()


def episode_fixture(pair=5,variant=0):
    centers=[int((.5+.4*np.sin(t*2.3+pair))*224) for t in (6.,7.)]
    frames=np.stack([pil_fixture(x,100+variant) for x in centers])
    return SimpleNamespace(rgb=frames,times=np.array([6.,7.]),shutters=np.ones((2,2)),
        probe_state=extract.cue_target_masks(pair,variant)[1],
        metadata=dict(namespace="software_test",variant=variant))


def input_fixture(episodes):
    return dict(cue_rgb=[extract.array_identity(image) for ep in episodes for image in ep.rgb],
                targets=extract.array_identity(np.stack([ep.probe_state for ep in episodes])))


class V52Counterfactual(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)

    def test_original_pillow_rectangle_redraw_is_bitwise_exact(self):
        for center in (22,50,112,200):
            original=pil_fixture(center)
            np.testing.assert_array_equal(extract.draw_nuisance(original,center),original)

    def test_only_old_new_rectangle_union_changes_grid_is_restored(self):
        original=pil_fixture(112)
        modified=extract.intervene(original,112,35)
        allowed=extract.rectangle_mask(112)|extract.rectangle_mask(35)
        np.testing.assert_array_equal(original[~allowed],modified[~allowed])
        np.testing.assert_array_equal(modified,pil_fixture(35))
        # Pixelx112 is a background grid line under the removed yellow rectangle.
        np.testing.assert_array_equal(modified[110,112],np.array([42,47,54],dtype=np.uint8))

    def test_unexpected_pixel_inside_strip_rejects_original_reconstruction(self):
        original=pil_fixture(112);original[110,10]=(8,9,10)
        with self.assertRaisesRegex(ValueError,"Original RGB"):
            extract.intervene(original,112,35)

    def test_four_conditions_reuse_exact_unique_tensors_without_reversing_targets(self):
        episodes=[episode_fixture(5,v) for v in (0,1)]
        targets=np.stack([ep.probe_state for ep in episodes])
        encoder=FakeEncoder()
        features,actual,inputs=extract.encode_pair(episodes,5,encoder,targets,input_fixture(episodes))
        self.assertEqual(features.shape,(2,4,32));self.assertEqual(len(encoder.calls),2)
        self.assertEqual([len(x) for x in encoder.calls],[4,4])
        np.testing.assert_array_equal(actual,targets)
        np.testing.assert_array_equal(extract.CONDITION_INDICES,[[0,3],[0,2],[1,3],[1,2]])
        # The first batch is exactly the archived original bridge batch.
        np.testing.assert_array_equal(encoder.calls[0],np.concatenate([ep.rgb for ep in episodes]))
        for variant,(ep,record) in enumerate(zip(episodes,inputs)):
            images=np.stack((encoder.calls[0][2*variant],encoder.calls[1][2*variant],
                             encoder.calls[1][2*variant+1],encoder.calls[0][2*variant+1]))
            np.testing.assert_array_equal(images[[0,3]],ep.rgb)
            self.assertTrue(record["original_pixel_parity"])
            self.assertTrue(record["outside_rectangle_union_unchanged"])
            self.assertTrue(record["target_masks_disjoint"])
            self.assertTrue(record["target_pixels_unchanged"])
            # Conditions hold original target/time order; only rectanglexchanges.
            for indices in extract.CONDITION_INDICES:
                self.assertEqual(indices[0]//2,0);self.assertEqual(indices[1]//2,1)
            for indices in extract.CONDITION_INDICES[1:3]:
                band=[image[107:117] for image in images[indices]]
                np.testing.assert_array_equal(band[0],band[1])

    def test_changed_original_targets_fail_before_encoder(self):
        episodes=[episode_fixture(5,v) for v in (0,1)];encoder=FakeEncoder()
        targets=np.stack([ep.probe_state for ep in episodes]);targets[0,4]+=.01
        with self.assertRaisesRegex(ValueError,"targets differ"):
            extract.encode_pair(episodes,5,encoder,targets,input_fixture(episodes))
        self.assertEqual(encoder.calls,[])

    def test_archived_rgb_hash_mismatch_fails_before_encoder(self):
        episodes=[episode_fixture(5,v) for v in (0,1)];encoder=FakeEncoder()
        inputs=input_fixture(episodes);inputs["cue_rgb"][0]["sha256"]="0"*64
        targets=np.stack([ep.probe_state for ep in episodes])
        with self.assertRaisesRegex(ValueError,"archived bridge inputs"):
            extract.encode_pair(episodes,5,encoder,targets,inputs)
        self.assertEqual(encoder.calls,[])

    def test_actual_target_mask_overlap_rejects_intervention(self):
        ep=episode_fixture();masks=np.zeros((2,2,224,224),dtype=np.uint8)
        center=int((.5+.4*np.sin(6*2.3+5))*224);masks[0,0,110,center]=1
        with patch.object(extract,"cue_target_masks",return_value=(masks,ep.probe_state)):
            with self.assertRaisesRegex(ValueError,"actual PIL target mask"):
                extract.variant_images(ep,5)

    def test_shard_resume_rejects_hash_or_configuration_changes(self):
        (self.root/"shards").mkdir()
        features=np.zeros((2,4,32),dtype=np.float32);targets=np.zeros((2,8),dtype=np.float32)
        extract.save_shard(self.root,5,"software-config",features,targets,[])
        arrays,_=extract.read_shard(self.root,5,"software-config")
        np.testing.assert_array_equal(arrays["features"],features)
        with self.assertRaises(ValueError):extract.read_shard(self.root,5,"other")
        path,_=extract.shard_paths(self.root,5)
        data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaises(ValueError):extract.read_shard(self.root,5,"software-config")

    def test_prior_scene_and_prior_ledger_cannot_be_output(self):
        prior=self.root/"prior";prior.mkdir()
        args=argparse.Namespace(workers=8,lease_hours=.2,out=prior/"new",prior_scene=prior,budget_root=self.root/"new-ledger")
        with self.assertRaisesRegex(ValueError,"prior scene is read-only"):extract.run(args)
        args.out=self.root/"new";args.budget_root=prior/"ledger"
        with self.assertRaisesRegex(ValueError,"prior ledger is read-only"):extract.run(args)

    def test_ledger_preserves_prior_cost_and_refunds_partial_run(self):
        prior=self.root/"prior";(prior/"ledger").mkdir(parents=True)
        ledger=dict(charged_gpuh=1.7294136471218544,stages={},leases=[])
        (prior/"ledger/budget.json").write_text(json.dumps(ledger))
        new=self.root/"ledger";new.mkdir();(new/"budget.json").write_text(json.dumps(ledger))
        args=argparse.Namespace(workers=8,lease_hours=.2,out=self.root/"out",prior_scene=prior,budget_root=new)
        with patch.object(extract,"run_reserved",side_effect=RuntimeError("software failure")):
            with self.assertRaises(RuntimeError):extract.run(args)
        actual=json.loads((new/"budget.json").read_text())
        self.assertGreaterEqual(actual["charged_gpuh"],ledger["charged_gpuh"])
        self.assertLess(actual["charged_gpuh"],ledger["charged_gpuh"]+.01)
        self.assertEqual(actual["leases"][0]["stage"],"counterfactual_v5_2")
        self.assertEqual(actual["leases"][0]["gpu_count"],8)
        self.assertEqual(json.loads((prior/"ledger/budget.json").read_text()),ledger)


if __name__=="__main__":unittest.main()
