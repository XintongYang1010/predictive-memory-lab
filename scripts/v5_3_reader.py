#!/usr/bin/env python3
"""FIT-only matched static/balanced KRR, followed by frozen CPU evaluation."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_key, '4')
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
import numpy as np

from v5_3_data import (SCHEMA, VIEW_INDICES, STATIC_VIEW_INDICES, NEGATIVE_INDICES,
                       UNIQUE_ORDER, PATH_NAMES)

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'scripts/v5_3_plan.json'
INPUTS = ('concat', 't6', 't7')
ARMS = ('S', 'B')
ALPHA = 7.8125e-7
BOOTSTRAPS = 10000
BOOT_SEED = 7811
PARITY_TOLERANCE = 1e-7
FREEZE_SCHEMA = 'v5.3-reader-freeze-1'
IDENTITY_KEYS = ('plan_sha256', 'source_commit', 'encoder_identity', 'source_identity', 'pooling', 'feature_dtype')
SOURCE_NAMES = ('scripts/v5_3_reader.py', 'scripts/v5_3_data.py', 'scripts/v5_3_plan.json')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def entry_identity(array):
    a = np.asarray(array)
    return dict(sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                dtype=str(a.dtype), shape=list(a.shape))


def sources():
    return {name: sha256(ROOT/name) for name in SOURCE_NAMES}


def validate_identity(receipt):
    if any(k not in receipt for k in IDENTITY_KEYS):
        raise ValueError('Missing cache identity')
    identity = {k: receipt[k] for k in IDENTITY_KEYS}
    if re.fullmatch('[0-9a-f]{40}', identity['source_commit']) is None:
        raise ValueError('A frozen source commit is required')
    if identity['plan_sha256'] != sha256(PLAN):
        raise ValueError('Plan SHA mismatch')
    plan = read_json(PLAN)
    expected_encoder = {k: plan['encoder'][k] for k in ('source_commit', 'checkpoint_sha256', 'model')}
    if identity['encoder_identity'] != expected_encoder:
        raise ValueError('Encoder identity differs from registered encoder')
    if identity['pooling'] != '4x4_flattened' or identity['feature_dtype'] != 'float32':
        raise ValueError('Pooling/precision storage mismatch')
    required = set(SOURCE_NAMES) | {'scripts/extract_v5_3.py', 'scripts/launch_v5_3.py',
        'scripts/v5_temporal_confirmation.py', 'docs/V5_3_EXECUTION_PROTOCOL.md'}
    required |= {p.relative_to(ROOT).as_posix() for p in (ROOT/'pmr/visual_memory').glob('*.py')}
    mapping = identity['source_identity']
    if not isinstance(mapping, dict) or not required.issubset(mapping):
        raise ValueError('Required source identity entries missing')
    for name, digest in mapping.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha256(ROOT/path) != digest:
            raise ValueError('Source changed: '+name)
    return identity


def validate_arrays(data, role):
    """Strict production role, shape, target semantics and whole-world counts."""
    if role not in ('fit', 'anchor', 'bridge'):
        raise ValueError('Only fit, anchor and bridge are scientific roles')
    keys = {'features', 'targets', 'role', 'pair_ids'} if role == 'bridge' else {
        'features', 'targets', 'negative_targets', 'block_ids', 'role'}
    if set(data) != keys or data['role'].shape != () or str(data['role'].item()) != role:
        raise ValueError('Cache entry or role mismatch')
    x, y = data['features'], data['targets']
    shape = (256, 2, 2, 20480) if role == 'bridge' else (256, 6, 20480)
    target_shape = (256, 2, 8) if role == 'bridge' else (256, 16, 8)
    if x.shape != shape or y.shape != target_shape or x.dtype != np.float32 or y.dtype != np.float32:
        raise ValueError('Exactly 256 worlds, official feature width and float32 caches required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite cache')
    ids = data['pair_ids' if role == 'bridge' else 'block_ids']
    expected = np.arange(384, 640) if role == 'bridge' else np.arange(256)
    if not np.issubdtype(ids.dtype, np.integer) or not np.array_equal(ids, expected):
        raise ValueError('Exact ordered unique role IDs required')
    if role != 'bridge':
        check_targets(y, data['negative_targets'])


def check_targets(targets, negatives=None):
    y = np.asarray(targets).reshape(-1, 4, 4, 8)
    if not np.array_equal(y, np.repeat(y[:, :, :1], 4, axis=2)):
        raise ValueError('Nuisance schedules must have identical target labels')
    t = y[:, :, 0]
    if not (np.array_equal(t[:, 0, 4:], -t[:, 1, 4:]) and
            np.array_equal(t[:, 0, 4:], t[:, 2, 4:]) and
            np.array_equal(t[:, 0, 4:], -t[:, 3, 4:]) and
            np.array_equal(t[:, 0, :4], t[:, 1, :4]) and
            np.allclose(t[:, 2, :4]-t[:, 0, :4], t[:, 0, 4:], rtol=0, atol=1e-7) and
            np.allclose(t[:, 0, :4]-t[:, 3, :4], t[:, 0, 4:], rtol=0, atol=1e-7)):
        raise ValueError('Target view order or displacement semantics mismatch')
    if negatives is not None:
        n = np.asarray(negatives)
        if (n.shape != (len(y), 2, 8) or n.dtype != np.float32 or not np.isfinite(n).all()
                or np.any(n[:, :, 4:] != 0)
                or not np.array_equal(n[:, :, :4], np.repeat(t[:, :1, :4], 2, axis=1))):
            raise ValueError('Center-pose negative labels must have exactly zero motion')


def validate_publication(path, frozen):
    publication = read_json(path)
    if (publication.get('verified_readback') is not True or
            publication.get('frozen_reader_sha256') != frozen['receipt_sha256'] or
            publication.get('source_commit') != frozen['cache_identity']['source_commit'] or
            publication.get('plan_sha256') != frozen['cache_identity']['plan_sha256'] or
            not isinstance(publication.get('verified_unix'), (int, float)) or
            publication['verified_unix'] < frozen['created_unix']):
        raise ValueError('Evaluation requires verified publication after this complete freeze')
    return sha256(path)


def load_cache(cache, receipt_path, role, frozen=None, publication=None):
    receipt = read_json(receipt_path)
    if receipt.get('schema') != SCHEMA or receipt.get('role') != role or receipt.get('test_opened') is not False:
        raise ValueError('Cache schema, role or Test declaration mismatch')
    if receipt.get('cache_sha256') != sha256(cache):
        raise ValueError('Cache hash mismatch')
    identity = validate_identity(receipt)
    if re.fullmatch('[0-9a-f]{64}', receipt.get('freshness_audit_sha256', '')) is None:
        raise ValueError('Freshness audit identity required')
    if re.fullmatch('[0-9a-f]{64}', receipt.get('input_records_sha256', '')) is None:
        raise ValueError('Rendered input record identity required')
    for key, expected in (('view_indices', VIEW_INDICES.tolist()), ('negative_indices', NEGATIVE_INDICES.tolist()),
                          ('static_view_indices', STATIC_VIEW_INDICES.tolist()), ('unique_order', list(UNIQUE_ORDER))):
        if receipt.get(key) != expected:
            raise ValueError('View mapping mismatch: '+key)
    if role == 'fit':
        if frozen is not None or publication is not None:
            raise ValueError('FIT must not consume evaluation release metadata')
    else:
        if frozen is None or publication is None:
            raise ValueError('Evaluation requires a complete freeze and publication')
        publication_sha = validate_publication(publication, frozen)
        if (identity != frozen['cache_identity'] or
                receipt.get('freshness_audit_sha256') != frozen['freshness_audit_sha256'] or
                receipt.get('frozen_reader_sha256') != frozen['receipt_sha256'] or
                receipt.get('freeze_publication_sha256') != publication_sha or
                receipt.get('frozen_before_generation') is not True or receipt.get('unseen_before_release') is not True):
            raise ValueError('Evaluation cache was not released against this published freeze')
    with np.load(cache, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    validate_arrays(data, role)
    if receipt.get('entry_hashes') != {k: entry_identity(v) for k, v in data.items()}:
        raise ValueError('Cache entry hash/dtype/shape mismatch')
    return data, receipt


def block_indices(blocks, local=VIEW_INDICES):
    return (np.arange(blocks)[:, None, None]*6+np.asarray(local)[None]).reshape(-1, 2)


def dot_gram(a, b, chunk=2048):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError('Feature matrix mismatch')
    g = np.zeros((len(a), len(b)), dtype=np.float64)
    for lo in range(0, a.shape[1], chunk):
        g += a[:, lo:lo+chunk].astype(np.float64) @ b[:, lo:lo+chunk].astype(np.float64).T
    if not np.isfinite(g).all():
        raise ValueError('Nonfinite Gram')
    return g


def view_gram(unique, query, reference, input_name):
    if input_name not in INPUTS:
        raise ValueError('Unknown input')
    columns = (0, 1) if input_name == 'concat' else ((0,) if input_name == 't6' else (1,))
    return sum(unique[np.ix_(query[:, c], reference[:, c])] for c in columns)


def shared_kernel(gram):
    """Fit both centering/trace stages on complete B reference, never S subset."""
    g = np.asarray(gram, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise ValueError('Complete finite FIT Gram required')
    state = {}
    for level in ('linear', 'quadratic'):
        columns, grand = g.mean(0), float(g.mean())
        centered = g-g.mean(1, keepdims=True)-columns+grand
        scale = max(float(np.diag(centered).mean()), 1e-12)
        state[level+'_columns'], state[level+'_grand'], state[level+'_scale'] = columns, np.array(grand), np.array(scale)
        g = centered/scale
        if level == 'linear':
            g = (1+g)**2
    return g, state


def apply_shared_kernel(cross, state):
    q = np.asarray(cross, dtype=np.float64)
    for level in ('linear', 'quadratic'):
        if q.shape[1] != len(state[level+'_columns']):
            raise ValueError('Query moments require all B-reference columns')
        q = (q-q.mean(1, keepdims=True)-state[level+'_columns']+state[level+'_grand'])/state[level+'_scale']
        if level == 'linear':
            q = (1+q)**2
    return q


def ridge_fit(shared_gram, y, rows, alpha=ALPHA):
    rows = np.asarray(rows, dtype=np.int64)
    if not np.isfinite(alpha) or alpha <= 0 or len(rows) == 0:
        raise ValueError('Nonempty arm and positive fixed regularization required')
    g = np.asarray(shared_gram)[np.ix_(rows, rows)].copy()
    g.flat[::len(rows)+1] += len(rows)*alpha
    # Nine RHS share one solve of the SPD system. The Schur complement enforces
    # sum(coefficients)=0 and fits an unpenalized intercept for every output.
    solution = np.linalg.solve(g, np.column_stack((np.asarray(y)[rows],np.ones(len(rows)))))
    u,v = solution[:,:-1],solution[:,-1]
    denominator = float(v.sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError('Invalid intercept Schur complement')
    intercept = u.sum(0)/denominator
    coefficients = u-v[:,None]*intercept
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise ValueError('Nonfinite ridge fit')
    return coefficients,intercept


def train_models(features, targets, block_ids):
    """Small synthetic worlds allowed here; production load_cache enforces 256."""
    x, y, ids = np.asarray(features), np.asarray(targets), np.asarray(block_ids)
    if x.ndim != 3 or x.shape[1] != 6 or y.shape != (len(x), 16, 8) or ids.shape != (len(x),):
        raise ValueError('Complete six-image, sixteen-view worlds required')
    if len(np.unique(ids)) != len(ids) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite unique FIT worlds required')
    check_targets(y)
    index = block_indices(len(x))
    arm_rows = {'B': np.arange(len(x)*16),
                'S': (np.arange(len(x))[:, None]*16+STATIC_VIEW_INDICES[None]).ravel()}
    flat = y.reshape(-1, 8).astype(np.float64)
    mean = flat.mean(0)
    if not np.array_equal(mean, flat[arm_rows['S']].mean(0)):
        raise ValueError('S/B target means must be exactly identical')
    arrays = dict(fit_features=x.astype(np.float32), fit_block_ids=ids.astype(np.int64),
                  fit_view_indices=index, shared_target_mean=mean,
                  fit_motion_rms=np.sqrt(np.mean(flat[:,4:]**2,axis=0)),
                  S_rows=arm_rows['S'], B_rows=arm_rows['B'], mean_loss_alpha=np.array(ALPHA))
    raw = dot_gram(x.reshape(-1, x.shape[-1]), x.reshape(-1, x.shape[-1]))
    for input_name in INPUTS:
        g, state = shared_kernel(view_gram(raw, index, index, input_name))
        arrays.update({input_name+'__'+k: v for k, v in state.items()})
        fitted = {}
        for arm in ARMS:
            coef,intercept = ridge_fit(g,flat,arm_rows[arm])
            arrays[arm+'_'+input_name+'__coefficients'] = coef
            arrays[arm+'_'+input_name+'__intercept'] = intercept
            arrays[arm+'_'+input_name+'__target_mean'] = mean.copy()
            arrays[arm+'_'+input_name+'__coefficient_sum'] = coef.sum(0)
            if input_name in ('t6','t7'):
                fitted[arm] = g[:,arm_rows[arm]]@coef+intercept
        if input_name in ('t6','t7'):
            difference = float(np.max(np.abs(fitted['S']-fitted['B'])))
            arrays['fit_'+input_name+'_arm_parity_max_abs'] = np.array(difference)
            if difference > PARITY_TOLERANCE:
                raise ValueError('FIT single-frame S/B numerical equivalence failed')
    return arrays


def predict_models(arrays, query_unique, query_index, chunk=256):
    """Feature-only frozen prediction; all query means use complete B reference."""
    fit = arrays['fit_features']
    raw = dot_gram(query_unique, fit.reshape(-1, fit.shape[-1]))
    reference = arrays['fit_view_indices']
    output = {arm+'_'+name: np.empty((len(query_index), 8)) for name in INPUTS for arm in ARMS}
    for name in INPUTS:
        state = {k.split('__', 1)[1]: v for k, v in arrays.items() if k.startswith(name+'__')}
        for lo in range(0, len(query_index), chunk):
            q = apply_shared_kernel(view_gram(raw, query_index[lo:lo+chunk], reference, name), state)
            for arm in ARMS:
                key = arm+'_'+name
                output[key][lo:lo+chunk] = q[:, arrays[arm+'_rows']] @ arrays[key+'__coefficients']+arrays[key+'__intercept']
    if any(not np.isfinite(v).all() for v in output.values()):
        raise ValueError('Nonfinite frozen prediction')
    return output


def training_config():
    return dict(kernel='quadratic', mean_loss_alpha=ALPHA, inputs=list(INPUTS), arms=list(ARMS),
        normalization='shared_full_balanced_FIT_linear_then_quadratic_center_and_mean_diagonal',
        intercept='arm_specific_unpenalized_intercept_via_SPD_solve_and_Schur_complement',
        baseline='shared_FIT_target_mean_for_scoring_only',loss='mean_within_world_then_equal_mean_over_worlds',
        cv=False, selection=False, view_indices=VIEW_INDICES.tolist(), static_view_indices=STATIC_VIEW_INDICES.tolist(),
        regularization={'S': 2048*ALPHA, 'B': 4096*ALPHA})


def fit(cache, receipt, out):
    started = time.time()
    data, cache_receipt = load_cache(cache, receipt, 'fit')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arrays = train_models(data['features'], data['targets'], data['block_ids'])
    np.savez(out/'frozen_model.npz', **arrays)
    write_json(out/'training_config.json', training_config())
    frozen = dict(schema=FREEZE_SCHEMA, status='frozen_before_evaluation', created_unix=time.time(),
        fit_blocks=256, views_per_fit_block=16, fit_views=4096, arm_views={'S':2048, 'B':4096},
        cache_identity={k:cache_receipt[k] for k in IDENTITY_KEYS},
        freshness_audit_sha256=cache_receipt['freshness_audit_sha256'], sources=sources(),
        fit_cache_sha256=sha256(cache), fit_cache_receipt_sha256=sha256(receipt),
        files={name:sha256(out/name) for name in ('frozen_model.npz','training_config.json')},
        model_entry_hashes={k:entry_identity(v) for k,v in arrays.items()},
        fit_singleframe_arm_parity={name:float(arrays['fit_'+name+'_arm_parity_max_abs']) for name in ('t6','t7')},
        coefficient_sum_max_abs={arm+'_'+name:float(np.max(np.abs(arrays[arm+'_'+name+'__coefficient_sum'])))
            for arm in ARMS for name in INPUTS},
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),
            threads={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        prediction_inputs='features_only', fit_elapsed_seconds=time.time()-started,
        test_opened=False, p0_authorized=False, original_v5_gate='failed_unchanged', v5_1_bridge='failed_unchanged')
    write_json(out/'freeze_receipt.json', frozen)
    return frozen


def load_frozen(folder):
    folder = Path(folder)
    receipt = read_json(folder/'freeze_receipt.json')
    if (receipt.get('schema') != FREEZE_SCHEMA or receipt.get('status') != 'frozen_before_evaluation' or
            receipt.get('fit_blocks') != 256 or receipt.get('fit_views') != 4096 or
            receipt.get('views_per_fit_block') != 16 or receipt.get('arm_views') != {'S':2048,'B':4096} or
            receipt.get('sources') != sources() or receipt.get('test_opened') is not False or
            not isinstance(receipt.get('created_unix'), (int,float))):
        raise ValueError('Incomplete or changed reader freeze')
    validate_identity(receipt['cache_identity'])
    for name in ('frozen_model.npz', 'training_config.json'):
        if receipt.get('files', {}).get(name) != sha256(folder/name):
            raise ValueError('Frozen artifact changed: '+name)
    if read_json(folder/'training_config.json') != training_config():
        raise ValueError('Frozen training definition changed')
    with np.load(folder/'frozen_model.npz', allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    if receipt.get('model_entry_hashes') != {k:entry_identity(v) for k,v in arrays.items()}:
        raise ValueError('Frozen array identity mismatch')
    if (arrays['fit_features'].shape != (256,6,20480) or arrays['fit_features'].dtype != np.float32 or
            not np.array_equal(arrays['fit_block_ids'], np.arange(256)) or
            not np.array_equal(arrays['fit_view_indices'], block_indices(256)) or
            not np.array_equal(arrays['S_rows'], (np.arange(256)[:,None]*16+STATIC_VIEW_INDICES).ravel()) or
            not np.array_equal(arrays['B_rows'], np.arange(4096)) or float(arrays['mean_loss_alpha']) != ALPHA):
        raise ValueError('Frozen model dimensions, rows or regularization mismatch')
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite frozen arrays')
    for name in INPUTS:
        for arm,count in (('S',2048),('B',4096)):
            if (arrays[arm+'_'+name+'__coefficients'].shape != (count,8) or
                    arrays[arm+'_'+name+'__intercept'].shape != (8,) or
                    not np.array_equal(arrays[arm+'_'+name+'__target_mean'], arrays['shared_target_mean'])):
                raise ValueError('Frozen arm coefficients or baseline mean mismatch')
    if any(float(arrays['fit_'+name+'_arm_parity_max_abs']) > PARITY_TOLERANCE for name in ('t6','t7')):
        raise ValueError('Frozen FIT single-frame equivalence failed')
    receipt['receipt_sha256'] = sha256(folder/'freeze_receipt.json')
    return arrays, receipt


def score_outputs(prediction, target, mean):
    p, y = np.asarray(prediction).reshape(-1,8), np.asarray(target,dtype=np.float64).reshape(-1,8)
    mse = np.mean((p-y)**2,axis=0)
    den = np.mean((y-mean)**2,axis=0)
    if np.any(den <= 0):
        raise ValueError('Degenerate FIT-mean score denominator')
    skill = 1-mse/den
    return dict(per_dimension_mse=mse.tolist(),per_dimension_skill=skill.tolist(),
        position_skill=float(skill[:4].mean()),motion_skill=float(skill[4:].mean()),motion_nmse=float((mse[4:]/den[4:]).mean()),
        motion_direction_error_rate=np.mean(np.sign(p[:,4:])!=np.sign(y[:,4:]),axis=0).tolist(),
        motion_target_zero_count=np.sum(y[:,4:]==0,axis=0).tolist(),
        motion_prediction_zero_count=np.sum(p[:,4:]==0,axis=0).tolist(),
        motion_sign_tie_count=np.sum(p[:,4:]*y[:,4:]==0,axis=0).tolist(),
        motion_signed_bias=np.mean(p[:,4:]-y[:,4:],axis=0).tolist(),
        motion_prediction_target_covariance=np.mean((p[:,4:]-p[:,4:].mean(0))*(y[:,4:]-y[:,4:].mean(0)),axis=0).tolist(),
        motion_absolute_amplitude_ratio=(np.mean(np.abs(p[:,4:]),axis=0)/np.mean(np.abs(y[:,4:]),axis=0)).tolist(),
        motion_rms_ratio=np.sqrt(np.mean(p[:,4:]**2,axis=0)/np.mean(y[:,4:]**2,axis=0)).tolist(),
        motion_std_ratio=(np.std(p[:,4:],axis=0)/np.std(y[:,4:],axis=0)).tolist())


def score_bridge(predictions, target, mean, replicates=BOOTSTRAPS):
    y = np.asarray(target,dtype=np.float64)
    if y.ndim != 3 or y.shape[1:] != (2,8):
        raise ValueError('Bridge must retain both variants of each world')
    scores = {name:score_outputs(p,y,mean) for name,p in predictions.items()}
    errors = {name:((p-y)**2).mean(1) for name,p in predictions.items()}
    den = ((y-mean)**2).mean(1)
    differences = {name:errors['S_'+name]-errors['B_'+name] for name in INPUTS}
    boots = {name:[] for name in INPUTS}
    rng = np.random.default_rng(BOOT_SEED)
    for begin in range(0,replicates,250):
        indices = rng.integers(0,len(y),(min(250,replicates-begin),len(y)))
        base = den[indices,4:].mean(1)
        for name,diff in differences.items():
            d = diff[indices].mean(1)
            boots[name].extend(np.stack((d[:,[4,6]].mean(1),(d[:,4:]/base).mean(1)),axis=1))
    comparisons = {}
    for name,diff in differences.items():
        ci = np.quantile(np.asarray(boots[name]),[.025,.975],axis=0)
        comparisons[name] = dict(S_minus_B_dx_mse=float(diff[:,[4,6]].mean()),
            dx_mse_paired_95ci=ci[:,0].tolist(),
            S_minus_B_motion_nmse=float((diff[:,4:].mean(0)/den[:,4:].mean(0)).mean()),
            motion_nmse_descriptive_paired_95ci=ci[:,1].tolist())
    return dict(primary_reader='concat',primary=comparisons['concat'],comparisons=comparisons,readers=scores,
        primary_improvement_interval_above_zero=bool(comparisons['concat']['dx_mse_paired_95ci'][0]>0),
        bootstrap=dict(seed=BOOT_SEED,replicates=replicates,unit='whole_world_both_variants',primary_endpoints=1),
        p0_authorized=False,interpretation='One fresh bridge comparison of predeclared arms; support metrics are not a new P0 gate')


def bootstrap_weights(blocks, replicates):
    rng = np.random.default_rng(BOOT_SEED)
    weights = np.empty((replicates,blocks),dtype=np.float64)
    for b in range(replicates):
        weights[b] = np.bincount(rng.integers(0,blocks,blocks),minlength=blocks)/blocks
    return weights


def bootstrap_mean(weights, values):
    values = np.asarray(values,dtype=np.float64)
    return (weights@values.reshape(len(values),-1)).reshape((len(weights),)+values.shape[1:])


def score_anchor(predictions, target, negative_predictions, negative_target, mean, fit_motion_rms,
                 replicates=BOOTSTRAPS):
    y = np.asarray(target,dtype=np.float64).reshape(-1,4,4,8)
    fit_motion_rms = np.asarray(fit_motion_rms,dtype=np.float64)
    if fit_motion_rms.shape != (4,) or np.any(fit_motion_rms<=0):
        raise ValueError('Positive nonzero FIT motion RMS required')
    weights = bootstrap_weights(len(y),replicates)
    boot_denom = bootstrap_mean(weights,((y-mean)**2).mean(1))
    boot_delta_denom = bootstrap_mean(weights,y[:,0,:,4:]**2)
    fit_overall_rms = float(np.sqrt(np.mean(fit_motion_rms**2)))
    result,boot_contrasts,boot_motion,boot_negative = {},{},{},{}
    for key,prediction in predictions.items():
        p = prediction.reshape(-1,4,4,8)
        contrasts = np.stack(((p[:,0,:,4:]-p[:,1,:,4:])/2,
                              (p[:,2,:,4:]-p[:,3,:,4:])/2),axis=2)
        delta = y[:,0,:,4:]
        contrast_error = (contrasts-delta[:,:,None,:])**2
        boot_contrasts[key] = 1-bootstrap_mean(weights,contrast_error)/boot_delta_denom[:,:,None,:]
        boot_motion[key] = 1-(bootstrap_mean(weights,((p-y)**2).mean(1))[:,:,4:]/boot_denom[:,:,4:]).mean(2)
        paths = {}
        for path,name in enumerate(PATH_NAMES):
            mse = contrast_error[:,path].mean(0)
            denominator = (delta[:,path]**2).mean(0)
            paths[name] = dict(end_start_per_coordinate_skill=(1-mse/denominator).tolist(),
                end_start_per_coordinate_skill_descriptive_95ci=np.quantile(boot_contrasts[key][:,path],[.025,.975],axis=0).tolist(),
                end_start_motion_mse=mse.tolist(),
                end_start_prediction_rms=np.sqrt((contrasts[:,path]**2).mean(0)).tolist(),
                target_displacement_rms=np.sqrt(denominator).tolist(),
                absolute_outputs=score_outputs(p[:,:,path,:],y[:,:,path,:],mean),
                mean_motion_skill_descriptive_95ci=np.quantile(boot_motion[key][:,path],[.025,.975]).tolist())
        groups = {}
        for name,indices in (('static',(0,1)),('moving',(2,3))):
            groups[name] = score_outputs(p[:,:,indices,:],y[:,:,indices,:],mean)
            groups[name]['mean_motion_skill_descriptive_95ci']=np.quantile(boot_motion[key][:,indices].mean(1),[.025,.975]).tolist()
        negative = negative_predictions[key]
        # Square each direction before aggregation: ab/ba sign reversal must not
        # cancel a nuisance response in this zero-target diagnostic.
        sq = negative[:,:,4:]**2
        rms = np.sqrt(sq.mean((0,1)))
        boot_by_direction = np.sqrt(bootstrap_mean(weights,sq))
        boot_rms = np.sqrt(np.mean(boot_by_direction**2,axis=1))
        boot_overall = np.sqrt(np.mean(boot_rms**2,axis=1))
        boot_negative[key] = dict(axis=boot_rms,direction=boot_by_direction,overall=boot_overall)
        result[key] = dict(all_views=score_outputs(prediction,target,mean),paths=paths,schedule_groups=groups,
            negative_control_motion_rms=rms.tolist(),
            negative_control_motion_rms_by_direction=np.sqrt(sq.mean(0)).tolist(),
            negative_control_motion_rms_by_direction_over_FIT=(np.sqrt(sq.mean(0))/fit_motion_rms).tolist(),
            negative_control_motion_rms_over_FIT_motion_rms=(rms/fit_motion_rms).tolist(),
            negative_control_overall_motion_rms=float(np.sqrt(sq.mean())),
            negative_control_overall_motion_rms_over_FIT=float(np.sqrt(sq.mean())/fit_overall_rms),
            negative_control_motion_rms_descriptive_95ci=np.quantile(boot_rms,[.025,.975],axis=0).tolist(),
            negative_control_motion_rms_by_direction_descriptive_95ci=np.quantile(boot_by_direction,[.025,.975],axis=0).tolist(),
            negative_control_motion_rms_by_direction_over_FIT_descriptive_95ci=(np.quantile(boot_by_direction,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            negative_control_motion_rms_over_FIT_descriptive_95ci=(np.quantile(boot_rms,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            negative_control_overall_motion_rms_descriptive_95ci=np.quantile(boot_overall,[.025,.975]).tolist(),
            negative_control_overall_motion_rms_over_FIT_descriptive_95ci=(np.quantile(boot_overall,[.025,.975])/fit_overall_rms).tolist(),
            negative_control_motion_mean=negative[:,:,4:].mean((0,1)).tolist(),
            negative_control_position_mse=np.mean((negative[:,:,:4]-negative_target[:,:,:4])**2,axis=(0,1)).tolist())
        if key.endswith('t6'):
            result[key]['shared_start_singleframe_null_max_abs']=np.max(np.abs(p[:,2]-p[:,3]),axis=(0,2)).tolist()
        if key.endswith('t7'):
            result[key]['shared_end_singleframe_null_max_abs']=np.max(np.abs(p[:,0]-p[:,1]),axis=(0,2)).tolist()
    comparisons={}
    for name in INPUTS:
        s,b='S_'+name,'B_'+name
        axis_delta=boot_negative[s]['axis']-boot_negative[b]['axis']
        direction_delta=boot_negative[s]['direction']-boot_negative[b]['direction']
        comparisons[name]=dict(
            B_minus_S_per_path_mean_motion_skill=[result[b]['paths'][path]['absolute_outputs']['motion_skill']-
                result[s]['paths'][path]['absolute_outputs']['motion_skill'] for path in PATH_NAMES],
            B_minus_S_per_path_end_start_coordinate_skill=[(np.array(result[b]['paths'][path]['end_start_per_coordinate_skill'])-
                np.array(result[s]['paths'][path]['end_start_per_coordinate_skill'])).tolist() for path in PATH_NAMES],
            B_minus_S_per_path_mean_motion_skill_descriptive_95ci=np.quantile(boot_motion[b]-boot_motion[s],[.025,.975],axis=0).tolist(),
            B_minus_S_per_path_end_start_coordinate_skill_descriptive_95ci=np.quantile(boot_contrasts[b]-boot_contrasts[s],[.025,.975],axis=0).tolist(),
            S_minus_B_negative_motion_rms=(np.array(result[s]['negative_control_motion_rms'])-np.array(result[b]['negative_control_motion_rms'])).tolist(),
            S_minus_B_negative_motion_rms_descriptive_95ci=np.quantile(axis_delta,[.025,.975],axis=0).tolist(),
            S_minus_B_negative_motion_rms_over_FIT=((np.array(result[s]['negative_control_motion_rms'])-
                np.array(result[b]['negative_control_motion_rms']))/fit_motion_rms).tolist(),
            S_minus_B_negative_motion_rms_over_FIT_descriptive_95ci=(np.quantile(axis_delta,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            S_minus_B_negative_motion_rms_by_direction_descriptive_95ci=np.quantile(direction_delta,[.025,.975],axis=0).tolist(),
            S_minus_B_negative_motion_rms_by_direction_over_FIT_descriptive_95ci=(np.quantile(direction_delta,[.025,.975],axis=0)/fit_motion_rms).tolist(),
            S_minus_B_negative_overall_motion_rms=result[s]['negative_control_overall_motion_rms']-result[b]['negative_control_overall_motion_rms'],
            S_minus_B_negative_overall_motion_rms_descriptive_95ci=np.quantile(boot_negative[s]['overall']-boot_negative[b]['overall'],[.025,.975]).tolist())
    return dict(readers=result,comparisons=comparisons,fit_motion_rms=fit_motion_rms.tolist(),
        bootstrap=dict(seed=BOOT_SEED,replicates=replicates,unit='whole_block_all_views_directions_arms',intervals='descriptive_unadjusted'),
        support_only=True,p0_authorized=False)


def evaluate(role, cache, receipt, model, publication, out):
    if role not in ('anchor','bridge'):
        raise ValueError('Evaluation cannot fit or consume FIT/Test roles')
    arrays,frozen = load_frozen(model)
    data,cache_receipt = load_cache(cache,receipt,role,frozen,publication)
    x,y = data['features'],data['targets']
    index = (np.arange(512*2).reshape(512,2) if role == 'bridge' else block_indices(256))
    prediction = predict_models(arrays,x.reshape(-1,x.shape[-1]),index)
    prediction = {k:v.reshape(y.shape) for k,v in prediction.items()}
    negative = {}
    if role == 'anchor':
        negative = {k:v.reshape(256,2,8) for k,v in predict_models(
            arrays,x.reshape(-1,x.shape[-1]),block_indices(256,NEGATIVE_INDICES)).items()}
        score = score_anchor(prediction,y,negative,data['negative_targets'],arrays['shared_target_mean'],arrays['fit_motion_rms'])
    else:
        score = score_bridge(prediction,y,arrays['shared_target_mean'])
    single_parity = {name:float(np.max(np.abs(prediction['S_'+name]-prediction['B_'+name]))) for name in ('t6','t7')}
    if negative:
        single_parity.update({'negative_'+name:float(np.max(np.abs(negative['S_'+name]-negative['B_'+name]))) for name in ('t6','t7')})
    numerical_passed = all(v<=PARITY_TOLERANCE for v in single_parity.values())
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    saved = dict(prediction,targets=y,role=np.array(role))
    saved['pair_ids' if role == 'bridge' else 'block_ids'] = data['pair_ids' if role == 'bridge' else 'block_ids']
    if negative:
        saved.update({'negative_'+k:v for k,v in negative.items()})
        saved['negative_targets'] = data['negative_targets']
    np.savez_compressed(out/'predictions.npz',**saved)
    report = dict(schema='v5.3-reader-evaluation-1',role=role,
        status='evaluation_complete' if numerical_passed else 'numerical_equivalence_failed',scores=score,
        singleframe_S_B_max_abs=single_parity,groups=256,examples=512 if role=='bridge' else 4096,
        singleframe_equivalence_passed=numerical_passed,singleframe_equivalence_tolerance=PARITY_TOLERANCE,
        coincident_nuisance_pixel_blocks=cache_receipt.get('coincident_nuisance_pixel_blocks'),
        cache_sha256=sha256(cache),cache_receipt_sha256=sha256(receipt),
        frozen_reader_sha256=frozen['receipt_sha256'],freeze_publication_sha256=sha256(publication),
        predictions_sha256=sha256(out/'predictions.npz'),test_opened=False,p0_authorized=False,
        original_v5_gate='failed_unchanged',v5_1_bridge='failed_unchanged')
    write_json(out/'cache_receipt.json',cache_receipt)
    write_json(out/'report.json',report)
    if not numerical_passed:
        raise RuntimeError('Single-frame S/B equivalence failed; preserve outputs and stop')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','anchor','bridge'))
    for name in ('cache','receipt','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--publication',type=Path)
    args = parser.parse_args()
    if args.phase == 'fit':
        if args.model is not None or args.publication is not None:
            parser.error('FIT accepts no prior model or evaluation publication')
        result = fit(args.cache,args.receipt,args.out)
    else:
        if args.model is None or args.publication is None:
            parser.error('Evaluation requires --model and --publication')
        result = evaluate(args.phase,args.cache,args.receipt,args.model,args.publication,args.out)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
