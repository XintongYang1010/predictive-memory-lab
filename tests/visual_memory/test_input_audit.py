import importlib.util
from pathlib import Path
import unittest

import torch

SPEC=importlib.util.spec_from_file_location("audit_visual_probe_inputs_v5",Path(__file__).resolve().parents[2]/"scripts/audit_visual_probe_inputs_v5.py")
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)


class InputAudit(unittest.TestCase):
    def test_raw_pooling_preserves_spatial_raster_and_storage_dtype(self):
        raw=torch.arange(2*16*3).reshape(2,16,3).bfloat16()
        actual=audit.pool_raw(raw,grid=2,bf16=False)
        grid=raw.float().reshape(2,4,4,3)
        expected=torch.stack([grid[:,y:y+2,x:x+2].mean((1,2)) for y in (0,2) for x in (0,2)],1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(audit.pool_raw(raw,grid=2).dtype,torch.bfloat16)
        with self.assertRaises(ValueError):audit.pool_raw(torch.zeros(2,15,3))

    def test_chunked_gram_matches_concat_and_repeat_frame_equivalence(self):
        torch.manual_seed(42)
        train=[dict(z=torch.randn(2,4,3)) for _ in range(8)]
        dev=[dict(z=torch.randn(2,4,3)) for _ in range(6)]
        components=[audit.gram_components(train,dev,lambda row,i=i:row["z"][i],chunk=5) for i in (0,1)]
        a=torch.stack([r["z"].flatten().double() for r in train]);b=torch.stack([r["z"].flatten().double() for r in dev])
        mean=a.mean(0);a=a-mean;b=b-mean
        torch.testing.assert_close(sum(x[0] for x in components),a@a.T,rtol=1e-14,atol=1e-14)
        torch.testing.assert_close(sum(x[1] for x in components),b@a.T,rtol=1e-14,atol=1e-14)
        targets=torch.randn(8,8,dtype=torch.float64)
        gram,cross=components[0]
        torch.testing.assert_close(audit.ridge_prediction(gram,cross,targets),
                                   audit.ridge_prediction(2*gram,2*cross,targets),rtol=0,atol=0)


if __name__=="__main__":unittest.main()
