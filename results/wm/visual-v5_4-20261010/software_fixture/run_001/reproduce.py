#!/usr/bin/env python3
"""Software-only full row-count performance/finite fixture; never scientific FIT."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='4'
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
import v5_4_reader as reader
from v5_3_data import make_block as narrow_block
from v5_4_data import make_block as wide_block
OUT=Path(__file__).resolve().parent
source_names=reader.SOURCE_NAMES
before={name:reader.sha256(ROOT/name) for name in source_names}
start=time.monotonic()
rng=np.random.default_rng(54064)
narrow=rng.normal(size=(256,6,64)).astype(np.float32)
wide=(rng.normal(size=(256,6,64))+.2).astype(np.float32)
yn=np.stack([narrow_block('software_test',i).targets for i in range(256)])
yw=np.stack([wide_block('software_test',i).targets for i in range(256)])
fit_start=time.monotonic()
arrays=reader.train_models(narrow,yn,wide,yw,np.arange(256))
fit_seconds=time.monotonic()-fit_start
reader.validate_model_arrays(arrays,blocks=256,width=64)
query=rng.normal(size=(512,64)).astype(np.float32)
pred=reader.predict_models(arrays,query,np.arange(512).reshape(256,2))
assert all(np.isfinite(v).all() for v in arrays.values())
assert all(np.isfinite(v).all() for v in pred.values())
after={name:reader.sha256(ROOT/name) for name in source_names}
assert before==after
result=dict(software_only=True,scientific_data_generated=False,roles_used=['software_test'],
    unique_images=3072,joint_reference_views=8192,arm_supervised_views={'N':4096,'W':4096},
    feature_width=64,production_feature_width_not_exercised=20480,
    regularization=reader.training_config()['regularization'],fit_seconds=fit_seconds,
    total_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    all_models_and_predictions_finite=True,source_unchanged=True,source_sha256=after,
    coefficient_sum_max_abs={a:float(np.abs(arrays[a+'_concat__coefficient_sum']).max()) for a in reader.ARMS},
    python=sys.version,numpy=np.__version__,threads=4)
with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(result,indent=2))
