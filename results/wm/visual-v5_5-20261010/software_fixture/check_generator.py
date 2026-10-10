#!/usr/bin/env python3
"""Software namespace only: no scientific Dev/FIT, encoder checkpoint, or CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
import v5_5_counterfactual as m
from pmr.visual_memory.data import episode
from pmr.visual_memory.features import Encoder


class SoftwareModel(torch.nn.Module):
    def forward(self,x):
        return F.adaptive_avg_pool2d(x,16).flatten(2).transpose(1,2)


class RecordingEncoder:
    def __init__(self):
        self.encoder=m.LocalizationEncoder.__new__(m.LocalizationEncoder)
        self.encoder.device='cpu';self.encoder.grid=4;self.encoder.model=SoftwareModel()
        self.calls=[]
    def __call__(self,rgb):
        self.calls.append(rgb.copy())
        return self.encoder(rgb)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).with_name('result.json'))
    args=parser.parse_args()
    started=time.time();torch.set_num_threads(4);checked=0
    for pair in range(12):
        for variant in (0,1):
            ep=episode('software_test',pair,variant)
            indices=[int(np.flatnonzero(ep.times==t)[0]) for t in (6.,7.)]
            rgb,target,endpoints,record=m.composite_images('software_test',pair,variant)
            assert np.array_equal(rgb[[0,7]],ep.rgb[indices])
            assert np.array_equal(target,ep.probe_state)
            assert np.array_equal(endpoints[7],target[:4])
            assert record['outside_intervened_object_unions_unchanged'] and record['object_masks_disjoint']
            expected_edges={(s,s|bit) for s in range(8) for bit in (4,2,1) if not s&bit}
            assert len(record['cube_edges'])==12 and record['all_twelve_cube_edges_verified']
            assert {(edge['start_mask'],edge['end_mask']) for edge in record['cube_edges']}==expected_edges
            for edge in record['cube_edges']:
                start,end=edge['start_mask'],edge['end_mask']
                assert (start^end) in (4,2,1) and edge['outside_support_unchanged']
                assert edge['changed_pixels']==int(np.any(rgb[start]!=rgb[end],axis=-1).sum())
            for mask,(a,b,y) in enumerate(m.BITS):
                assert np.array_equal(endpoints[mask,:2],endpoints[7 if a else 0,:2])
                assert np.array_equal(endpoints[mask,2:],endpoints[7 if b else 0,2:])
            checked+=1
    pair=3;episodes=[episode('software_test',pair,v) for v in (0,1)]
    originals=np.concatenate([ep.rgb[[int(np.flatnonzero(ep.times==t)[0]) for t in (6.,7.)]] for ep in episodes])
    targets=np.stack([ep.probe_state for ep in episodes])
    expected=dict(cue_rgb=[m.array_identity(image) for image in originals],targets=m.array_identity(targets))
    encoder=RecordingEncoder()
    arrays,raw,inputs=m.encode_pair(pair,encoder,targets,expected,split='software_test')
    assert [len(x) for x in encoder.calls]==[4,12]
    assert np.array_equal(encoder.calls[0],originals)
    official=Encoder.__call__(encoder.encoder,originals).float().numpy().reshape(2,2,-1)
    assert np.array_equal(arrays['features'][:,[0,7]],official)
    rebuilt=F.adaptive_avg_pool2d(torch.from_numpy(raw.reshape(4,256,3)).transpose(1,2).reshape(4,3,16,16),4).flatten(2).transpose(1,2)
    assert np.array_equal(arrays['pooled_prequant'][:,[0,7]],rebuilt.numpy().reshape(2,2,-1))
    assert np.array_equal(arrays['features'],torch.from_numpy(arrays['pooled_prequant']).to(torch.bfloat16).float().numpy())
    assert raw.dtype==np.float32 and raw.shape==(2,2,256,3)
    bad=dict(expected,cue_rgb=[dict(expected['cue_rgb'][0],sha256='0'*64)]+expected['cue_rgb'][1:])
    try:m.encode_pair(pair,encoder,targets,bad,split='software_test')
    except ValueError:pass
    else:raise AssertionError('Changed archived RGB identity accepted')
    assert len(encoder.calls)==2,'Bad RGB must fail before encoder'
    try:m.cue_geometry('dev',0,0)
    except ValueError:pass
    else:raise AssertionError('Unseen Dev IDs accepted')
    try:m.cue_geometry('test',640,0)
    except ValueError:pass
    else:raise AssertionError('Test accepted')
    files=['scripts/v5_5_counterfactual.py','pmr/visual_memory/data.py','pmr/visual_memory/features.py']
    result=dict(status='software_fixture_passed',software_only=True,scientific_roles_opened=False,
        software_episodes_checked=checked,endpoint_images_checked=8*checked,
        single_object_cube_edges_checked=12*checked,
        checks=['original cue RGB bitwise','original physical labels bitwise','endpoint bit map',
            'outside changed-object mask union unchanged','all12 single-object cube edges per episode',
            'object masks disjoint','exact original4/edited12 batch order',
            'same-call prequant raw pooling','official pooled path bitwise','BF16 roundtrip bitwise',
            'changed archived input rejected before encoding','unseen Dev and Test rejected'],
        source_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        python=sys.version,numpy=np.__version__,torch=torch.__version__,elapsed_seconds=time.time()-started)
    out=args.out
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
