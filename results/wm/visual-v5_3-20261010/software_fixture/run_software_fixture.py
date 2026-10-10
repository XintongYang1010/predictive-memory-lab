#!/usr/bin/env python3
"""CPU scale check only. Does not load encoders or scientific-role data."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--reader-sha256',required=True)
    parser.add_argument('--data-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    sources={'scripts/v5_3_reader.py':args.reader_sha256,'scripts/v5_3_data.py':args.data_sha256}
    if any(sha(args.repo/name)!=expected for name,expected in sources.items()):
        raise ValueError('Software fixture requires the notified stable source hashes')
    if args.out.exists():
        raise FileExistsError('Preserve prior software fixture output; select a fresh output path')
    args.out.mkdir(parents=True)
    def timeout(*_):raise TimeoutError('600-second software-only CPU limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    started=time.time();cpu=time.process_time()
    result=dict(schema='v5.3-software-scale-check-1',software_only=True,status='running',
                scientific_roles_generated=[],test_opened=False,gpu_used=False,source_sha256=sources,
                feature_rng_seed=530103,feature_shape=[256,6,64],target_shape=[256,16,8],
                total_view_rows=4096,arm_view_rows={'S':2048,'B':4096},thread_limit=4,
                environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
                                 executable=sys.executable),start_unix=started)
    try:
        sys.path.insert(0,str(args.repo/'scripts'))
        import v5_3_data as data
        import v5_3_reader as reader
        targets=[];negatives=[];input_hashes=[]
        generation_started=time.perf_counter()
        for block_id in range(256):
            block=data.make_block('software_test',block_id)
            if block.metadata['role']!='software_test':raise AssertionError('Wrong software namespace')
            targets.append(block.targets);negatives.append(block.negative_targets)
            input_hashes.append(hashlib.sha256(block.unique_rgb.tobytes()).hexdigest())
        y=np.stack(targets);negative=np.stack(negatives);ids=np.arange(256,dtype=np.int64)
        x=np.random.default_rng(530103).normal(size=(256,6,64)).astype(np.float32)
        reader.check_targets(y,negative)
        np.savez_compressed(args.out/'software_inputs.npz',features=x,targets=y,negative_targets=negative,
                            block_ids=ids,role=np.array('software_test'))
        result['software_input_sha256']=sha(args.out/'software_inputs.npz')
        result['software_rgb_group_sha256']=input_hashes
        result['software_generation_seconds']=time.perf_counter()-generation_started
        print(json.dumps(dict(stage='six_reader_full_scale_fit',status='starting',view_rows=4096)),flush=True)
        fit_started=time.perf_counter()
        arrays=reader.train_models(x,y,ids)
        result['fit_seconds']=time.perf_counter()-fit_started
        result['coefficient_shapes']={arm+'_'+name:list(arrays[arm+'_'+name+'__coefficients'].shape)
                                      for arm in reader.ARMS for name in reader.INPUTS}
        result['coefficients_finite']=all(np.isfinite(arrays[arm+'_'+name+'__coefficients']).all()
                                          for arm in reader.ARMS for name in reader.INPUTS)
        result['all_saved_arrays_finite']=all(np.isfinite(v).all() for v in arrays.values())
        result['singleframe_fit_parity_max_abs']={name:float(arrays['fit_'+name+'_arm_parity_max_abs'])
                                                for name in ('t6','t7')}
        result['coefficient_sum_max_abs']={arm+'_'+name:float(np.abs(arrays[arm+'_'+name+'__coefficient_sum']).max())
                                          for arm in reader.ARMS for name in reader.INPUTS}
        if not result['coefficients_finite'] or not result['all_saved_arrays_finite']:
            raise AssertionError('Nonfinite software fit')
        if any(v>reader.PARITY_TOLERANCE for v in result['singleframe_fit_parity_max_abs'].values()):
            raise AssertionError('Single-frame S/B software parity failed')
        np.savez_compressed(args.out/'software_models.npz',**arrays)
        result['software_models_sha256']=sha(args.out/'software_models.npz')
        if any(sha(args.repo/name)!=expected for name,expected in sources.items()):
            raise ValueError('Scientific source changed during software CPU check')
        result['status']='passed_software_cpu_scale_only'
        result['interpretation']='Six actual KRR solves at registered world/view counts with random64D features; not encoder throughput, scientific FIT quality, anchor/bridge evidence, or P0 authorization.'
    except BaseException as error:
        result.update(status='failed_software_cpu_scale_check',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        result['elapsed_seconds']=time.time()-started
        result['cpu_seconds']=time.process_time()-cpu
        result['max_rss_platform_units']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['max_rss_units']='bytes on macOS; KiB on Linux'
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('software_rgb_group_sha256','traceback')},indent=2),flush=True)


if __name__=='__main__':main()
