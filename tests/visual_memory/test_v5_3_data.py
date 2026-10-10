"""CPU-only software fixtures: never generate FIT, anchor, or bridge data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
import v5_3_data as data
import extract_v5_3 as extract


class FakeEncoder:
    def __init__(self):self.calls=[]
    def __call__(self,rgb):
        self.calls.append(rgb.copy())
        values=rgb.astype(np.float32).mean((1,2,3))
        return torch.from_numpy(np.broadcast_to(values[:,None,None],(len(rgb),16,2)).copy())


class V53Data(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_standalone_budget_enforces_fixed_round_cap_below_six(self):
        cap=2.2082544136047373+3.75
        self.assertEqual(extract.CUMULATIVE_LIMIT_GPUH,cap)
        extract.check_budget_headroom(cap-.09*8,.09)
        # This would pass the older six-GPUh-only check, but exceeds this round.
        self.assertLess(5.26+.09*8,6.)
        with self.assertRaises(extract.BudgetExceeded):extract.check_budget_headroom(5.26,.09)
        with self.assertRaises(extract.BudgetExceeded):extract.check_budget_headroom(6.,.09)
        for charged,lease in ((float("nan"),.09),(0,float("inf")),(-1,.09),(0,.10)):
            with self.assertRaises(ValueError):extract.check_budget_headroom(charged,lease)

    def test_software_seed_new_namespace_and_bounds(self):
        for block in (0,31,255):
            expected=int.from_bytes(hashlib.sha256(f"v5.3-nuisance-repair-1/software_test/{block}".encode()).digest()[:8],"little")
            self.assertEqual(data.seed_for("software_test",block),expected)
            old=int.from_bytes(hashlib.sha256(f"v5-temporal-confirm-1/software_test/{block}".encode()).digest()[:8],"little")
            self.assertNotEqual(expected,old)
        for role,block in (("test",0),("dev",0),("software_test",-1),("software_test",256),("software_test",True)):
            with self.assertRaises(ValueError):data.seed_for(role,block)

    def test_canonical_target_major_path_minor_indices(self):
        expected=np.array([[0,2],[1,3],[0,3],[1,2], [4,2],[5,3],[4,3],[5,2],
                           [2,4],[3,5],[2,5],[3,4], [2,0],[3,1],[2,1],[3,0]])
        np.testing.assert_array_equal(data.VIEW_INDICES,expected)
        np.testing.assert_array_equal(data.STATIC_VIEW_INDICES,[0,1,4,5,8,9,12,13])
        np.testing.assert_array_equal(data.NEGATIVE_INDICES,[[2,3],[3,2]])

    def test_six_images_reuse_targets_and_exact_reversal(self):
        block=data.make_block("software_test",7)
        repeat=data.make_block("software_test",7)
        self.assertEqual(block.unique_rgb.shape,(6,224,224,3))
        np.testing.assert_array_equal(block.unique_rgb,repeat.unique_rgb)
        self.assertEqual(block.targets.dtype,np.float32)
        target=block.targets.reshape(4,4,8)
        for group in target:
            np.testing.assert_array_equal(group,np.repeat(group[:1],4,axis=0))
        np.testing.assert_array_equal(target[0,0,4:],-target[1,0,4:])
        np.testing.assert_array_equal(target[0,0,4:],target[2,0,4:])
        np.testing.assert_array_equal(target[0,0,4:],-target[3,0,4:])
        for i,(start,end) in enumerate(data.ORDERS):
            np.testing.assert_array_equal(target[i,0,:4],block.positions[end].astype(np.float32))
            np.testing.assert_array_equal(target[i,0,4:],(block.positions[end]-block.positions[start]).astype(np.float32))
        np.testing.assert_array_equal(block.negative_targets[:,4:],np.zeros((2,4)))
        np.testing.assert_array_equal(block.negative_targets[:,:4],np.tile(block.positions[1],(2,1)))
        public=block.actor_inputs()
        self.assertEqual(set(public),{"rgb","times","shutters"})
        np.testing.assert_array_equal(public["rgb"],block.unique_rgb[data.VIEW_INDICES])

    def test_nuisance_independence_preserves_all_non_yellow_band_pixels(self):
        block=data.make_block("software_test",19)
        keep=np.ones((224,224),dtype=bool);keep[107:117]=False
        for pose in range(3):
            np.testing.assert_array_equal(block.unique_rgb[2*pose][keep],block.unique_rgb[2*pose+1][keep])
        centers=block.metadata["nuisance_pixel_centers"]
        for pose in range(3):
            for nuisance in range(2):
                np.testing.assert_array_equal(block.unique_rgb[2*pose+nuisance,110,centers[nuisance]],(240,220,50))

    def test_encoder_receives_six_unique_rgb_once_no_labels(self):
        block=data.make_block("software_test",11);encoder=FakeEncoder()
        x,y,inputs,negative=extract.encode_block(block,encoder)
        self.assertEqual(x.shape,(6,32));self.assertEqual(len(encoder.calls),1)
        np.testing.assert_array_equal(encoder.calls[0],block.unique_rgb)
        np.testing.assert_array_equal(y,block.targets)
        np.testing.assert_array_equal(negative,block.negative_targets)
        self.assertEqual(len(inputs["unique_rgb"]),6)
        self.assertEqual(inputs["metadata"]["role"],"software_test")

    def test_software_shard_roundtrip_and_corruption_rejected(self):
        (self.root/"shards").mkdir()
        values=extract.encode_block(data.make_block("software_test",12),FakeEncoder())
        extract.save_unit(self.root,12,"software_test","fixture",*values)
        arrays,_=extract.read_unit(self.root,12,"fixture","software_test")
        self.assertEqual(set(arrays),{"features","targets","negative_targets"})
        np.testing.assert_array_equal(arrays["features"],values[0])
        with self.assertRaises(ValueError):extract.read_unit(self.root,12,"changed","software_test")
        path,_=extract.unit_paths(self.root,12)
        blob=bytearray(path.read_bytes());blob[-1]^=1;path.write_bytes(blob)
        with self.assertRaises(ValueError):extract.read_unit(self.root,12,"fixture","software_test")

    def release_fixture(self):
        # Metadata-only gate fixtures; no scientific role invokes make_block.
        sources={}
        for name in ("scripts/v5_3_reader.py","scripts/v5_3_data.py","scripts/v5_3_plan.json"):
            path=self.root/name;path.parent.mkdir(exist_ok=True);path.write_text("software fixture")
            sources[name]=extract.digest(path)
        identity=dict(plan_sha256=sources["scripts/v5_3_plan.json"],source_commit="a"*40,
                      source_identity=sources,encoder_identity={"model":"fake encoder"},pooling="4x4_flattened",feature_dtype="float32")
        fresh=self.root/"freshness.json"
        fresh.write_text(json.dumps(dict(schema="v5.3-freshness-audit-1",verified=True,test_opened=False,
             unseen_namespaces=[data.SCHEMA+"/fit",data.SCHEMA+"/anchor"],unseen_original_dev_pair_ids=list(data.BRIDGE_PAIR_IDS),
             source_commit=identity["source_commit"],plan_sha256=identity["plan_sha256"],verified_unix=time.time()-10)))
        files={}
        for name in ("frozen_model.npz","training_config.json"):
            path=self.root/name;path.write_bytes(b"software fixture");files[name]=extract.digest(path)
        freeze=self.root/"freeze_receipt.json"
        freeze.write_text(json.dumps(dict(schema="v5.3-reader-freeze-1",status="frozen_before_evaluation",
            fit_blocks=256,fit_views=4096,views_per_fit_block=16,arm_views={"S":2048,"B":4096},cache_identity=identity,
            freshness_audit_sha256=extract.digest(fresh),sources=sources,files=files,created_unix=time.time()-5)))
        publication=self.root/"freeze_publication.json"
        publication.write_text(json.dumps(dict(verified_readback=True,frozen_reader_sha256=extract.digest(freeze),
            source_commit=identity["source_commit"],plan_sha256=identity["plan_sha256"],verified_unix=time.time()-1)))
        args=argparse.Namespace(role="anchor",freshness_audit=fresh,freeze_receipt=freeze,freeze_publication=publication)
        return args,identity

    def test_gate_requires_freshness_frozen_coefficients_and_github_readback(self):
        args,identity=self.release_fixture()
        with patch.object(extract,"ROOT",self.root):
            receipt=extract.check_release(args,identity)
            self.assertEqual(receipt["freeze_publication_sha256"],extract.digest(args.freeze_publication))
            args.freeze_publication=None
            with self.assertRaisesRegex(ValueError,"GitHub readback"):extract.check_release(args,identity)

    def test_modified_model_or_nonfresh_range_fails_gate(self):
        args,identity=self.release_fixture()
        with patch.object(extract,"ROOT",self.root):
            (self.root/"frozen_model.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"file hash mismatch"):extract.check_release(args,identity)
            fresh=json.loads(args.freshness_audit.read_text());fresh["unseen_original_dev_pair_ids"][0]=128
            args.freshness_audit.write_text(json.dumps(fresh))
            with self.assertRaisesRegex(ValueError,"Freshness audit"):extract.check_release(args,identity)


if __name__=="__main__":unittest.main()
