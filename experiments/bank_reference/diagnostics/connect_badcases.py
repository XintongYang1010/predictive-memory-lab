#!/usr/bin/env python3
"""Frozen-parent signal/noise accounting; no fitting or new online policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
D = 34
R25_SHA = "5d0cf16b8625b9a545bc5bc7ccf6fa5a0181fd98acdebea199c244568d01767e"
R50_SHA = "c256d8ee6b80e422c5ddc6d6e120b3eb3bfb8272a9083f3b64ca5683db9700b3"
TOL = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(arr):
    return {"shape": list(arr.shape), "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()}


def components(r25, r50):
    r25, r50 = r25.astype(np.float64), r50.astype(np.float64)
    c = (r25 - np.sqrt(.5) * r50) / (np.sqrt(.75) - .5)
    e = np.sqrt(2.) * r50 - c
    return np.sqrt(.5) * c, np.sqrt(.5) * e


def reader_weights(observed, keys, query_key):
    eligible = observed.astype(np.float64) * (keys == query_key)[:, None]
    den = eligible.sum(axis=0)
    weights = np.zeros_like(eligible)
    np.divide(eligible, den, out=weights, where=den > 0)
    return weights


def loss_parts(pu, pn, u, n, mask):
    den = float(mask.sum())
    eu, en = pu-u, pn-n
    return {"signal": float((eu*eu*mask).sum()/den),
            "noise": float((en*en*mask).sum()/den),
            "cross": float((2*eu*en*mask).sum()/den),
            "total": float(((eu+en)**2*mask).sum()/den)}


def prefix_hash(r, o, key, valid_times, decision_time):
    h = hashlib.sha256(np.asarray([decision_time], np.int64).tobytes())
    h.update(np.ascontiguousarray(r[valid_times]*o[valid_times]).tobytes())
    h.update(np.ascontiguousarray(o[valid_times]).tobytes())
    h.update(np.ascontiguousarray(key[valid_times]).tobytes())
    return h.hexdigest()


def bounded_hash(r, o, keys, chosen_times, candidate_ord, recent_ord, decision_time):
    h = hashlib.sha256(np.asarray([decision_time, len(candidate_ord), len(recent_ord)], np.int64).tobytes())
    h.update(np.asarray(candidate_ord+recent_ord, np.int64).tobytes())
    h.update(np.ascontiguousarray(r[chosen_times]*o[chosen_times]).tobytes())
    h.update(np.ascontiguousarray(o[chosen_times]).tobytes())
    h.update(np.ascontiguousarray(keys[chosen_times]).tobytes())
    return h.hexdigest()


def analyze_parent(parent, arrays, un):
    cond, i = parent['condition'], parent['source_index']
    r, o, w, keys = (arrays[f'{cond}__{x}'][i] for x in ('R','O','W','key'))
    u, n = un[cond][0][i], un[cond][1][i]
    valid = np.flatnonzero(w)
    co, ro = parent['candidate_times'], parent['recent_times']
    ct, rt = valid[co], valid[ro]
    future = np.asarray(parent['future_original_times'])
    all_t = np.concatenate((ct, rt))
    assert keys[ct].tolist() == parent['candidate_keys']
    assert keys[future].tolist() == parent['future_keys']
    assert np.all(all_t <= parent['original_t']) and np.all(future > parent['original_t'])
    assert len(ct) == int(parent['method'].split('/')[0][1:]) + 1 and len(rt) == 4
    mask = o[future].astype(np.float64)
    zero = loss_parts(np.zeros_like(u[future]), np.zeros_like(n[future]), u[future], n[future], mask)
    losses, coverage, teacher_rows, query_detail = [], [], [], []
    for t in future:
        weights = reader_weights(o[all_t], keys[all_t], keys[t])
        att = weights[:len(ct)].mean(axis=1)
        if att.sum() > 0:
            teacher_rows.append(att/att.sum())
    for drop in range(len(ct)):
        retained = np.delete(all_t, drop)
        pus, pns, cov, queries = [], [], [], []
        for t in future:
            weights = reader_weights(o[retained], keys[retained], keys[t])
            pu, pn = (weights*u[retained]).sum(0), (weights*n[retained]).sum(0)
            pus.append(pu); pns.append(pn)
            covered = (weights.sum(0)>0) * o[t]
            cov.append(float(covered.sum()))
            if drop in {parent['chosen_drop'], parent['oracle_drop']}:
                queries.append({'original_t': int(t), 'query_key': int(keys[t]),
                                'covered_scored_dimensions': int(covered.sum()),
                                'scored_dimensions': int(o[t].sum()),
                                'parts': loss_parts(pu, pn, u[t], n[t], o[t])})
        parts = loss_parts(np.stack(pus), np.stack(pns), u[future], n[future], mask)
        losses.append(parts); coverage.append(sum(cov)/float(mask.sum()))
        if queries:
            query_detail.append({'drop': drop, 'queries': queries})
    original = np.asarray(parent['candidate_losses'])
    reconstructed = np.array([x['total'] for x in losses])
    max_error = float(np.max(np.abs(original-reconstructed)))
    assert max_error < 1e-11, (parent['method'],max_error)
    for part in losses:
        assert abs(part['signal']+part['noise']+part['cross']-part['total']) < 1e-11
    best = np.flatnonzero(reconstructed <= reconstructed.min()+TOL).tolist()
    signal = np.array([x['signal'] for x in losses])
    signal_best = np.flatnonzero(signal <= signal.min()+TOL).tolist()
    chosen, oracle = parent['chosen_drop'], parent['oracle_drop']
    signal_delta = losses[chosen]['signal']-losses[oracle]['signal']
    teacher = None
    if teacher_rows:
        salience = np.max(teacher_rows, axis=0)
        probability = salience/salience.sum()
        teacher_drop = int(np.argmin(probability))
        teacher = {'nonempty_queries':len(teacher_rows),'probability':probability.tolist(),
                   'chosen_drop':teacher_drop,'loss':float(reconstructed[teacher_drop]),
                   'regret':float(reconstructed[teacher_drop]-reconstructed.min()),
                   'probability_at_oracle_drop':float(probability[oracle]),
                   'max_attention_in_task_optimal_drop_set': bool(set(np.flatnonzero(probability >= probability.max()-TOL)) & set(best)),
                   'all_max_attention_items_are_task_optimal_drops': bool(set(np.flatnonzero(probability >= probability.max()-TOL)) <= set(best))}
    return {**{k:parent[k] for k in ('condition','source_index','method','original_t','decision_time','chosen_drop','oracle_drop','future_den','candidate_times','candidate_keys','recent_times')},
            'candidate_original_times':ct.tolist(), 'candidate_parts':losses,
            'candidate_scored_coverage':coverage,'zero_memory_parts':zero,
            'chosen_minus_oracle':{k:losses[chosen][k]-losses[oracle][k] for k in losses[chosen]},
            'chosen_minus_zero':{k:losses[chosen][k]-zero[k] for k in zero},
            'task_optimal_drop_set':best,'signal_optimal_drop_set':signal_best,
            'disjoint_task_signal_optimal_sets':not bool(set(best)&set(signal_best)),
            'task_oracle_signal_regret':float(signal[oracle]-signal.min()),
            'positive_task_regret':bool(parent['regret']>TOL),
            'oracle_improves_signal':bool(signal_delta>TOL),
            'oracle_worsens_signal':bool(signal_delta < -TOL),
            'oracle_drop_pre_splice':bool(ct[oracle]<400),
            'oracle_drop_age_valid_steps':int(parent['decision_time']-co[oracle]),
            'teacher':teacher, 'query_detail':query_detail,
            'bounded_past_sha256':bounded_hash(r,o,keys,all_t,co,ro,parent['decision_time']),
            'full_available_past_sha256':prefix_hash(r,o,keys,valid[valid<=parent['original_t']],parent['decision_time']),
            'reconstruction_max_abs_error':max_error}


def aggregate(rows):
    out={}
    for method in sorted({r['method'] for r in rows}):
        for condition in ('EXACT','NEWPATH','CHANGE','ALL'):
            rr=[r for r in rows if r['method']==method and (condition=='ALL' or r['condition']==condition)]
            den=sum(r['future_den'] for r in rr)
            def avg(get): return sum(get(r)*r['future_den'] for r in rr)/den
            pos=[r for r in rr if r['positive_task_regret']]
            tr=[r for r in rr if r['teacher'] is not None]
            out[f'{method}/{condition}']={'parents':len(rr),'positive_task_regret':len(pos),
                'chosen_loss':avg(lambda r:r['candidate_parts'][r['chosen_drop']]['total']),
                'oracle_loss':avg(lambda r:r['candidate_parts'][r['oracle_drop']]['total']),
                'zero_memory_loss':avg(lambda r:r['zero_memory_parts']['total']),
                'chosen_minus_oracle':{k:avg(lambda r:r['chosen_minus_oracle'][k]) for k in ('signal','noise','cross','total')},
                'chosen_minus_zero':{k:avg(lambda r:r['chosen_minus_zero'][k]) for k in ('signal','noise','cross','total')},
                'positive_regret_oracle_improves_signal':sum(r['oracle_improves_signal'] for r in pos),
                'positive_regret_oracle_worsens_signal':sum(r['oracle_worsens_signal'] for r in pos),
                'disjoint_task_signal_optimal_sets':sum(r['disjoint_task_signal_optimal_sets'] for r in rr),
                'positive_regret_oracle_drop_pre_splice':sum(r['oracle_drop_pre_splice'] for r in pos),
                'teacher_nonempty_parents':len(tr),'teacher_positive_regret':sum(r['teacher']['regret']>TOL for r in tr),
                'teacher_weighted_regret':sum(r['teacher']['regret']*r['future_den'] for r in tr)/sum(r['future_den'] for r in tr) if tr else None,
                'positive_regret_teacher_highest_attention_also_optimal_drop':sum(r['positive_task_regret'] and r['teacher']['all_max_attention_items_are_task_optimal_drops'] for r in tr)}
    return out


def collision_audit(rows):
    out={}
    for field in ('bounded_past_sha256','full_available_past_sha256'):
        groups={}
        for r in rows:
            # Full-prefix equality does not imply identical candidate sets.
            key=(r[field],tuple(r['candidate_times']),tuple(r['recent_times']))
            groups.setdefault(key,[]).append(r)
        duplicated=[g for g in groups.values() if len(g)>1]
        conflicts=[]
        for g in duplicated:
            common=set(g[0]['task_optimal_drop_set'])
            for r in g[1:]: common &= set(r['task_optimal_drop_set'])
            if not common:
                conflicts.append([{'method':r['method'],'condition':r['condition'],'source_index':r['source_index'],'original_t':r['original_t'],'optimal':r['task_optimal_drop_set']} for r in g])
        out[field]={'unique_states':len(groups),'duplicate_groups':len(duplicated),
                    'duplicate_rows':sum(map(len,duplicated)),'conflicting_optimal_set_groups':len(conflicts),'conflicts':conflicts,
                    'limit':'Only exact archived input collisions; zero conflicts does not establish learnability or exclude unobserved future branches.'}
    return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();assert not args.out.exists(),'Preserve previous output';args.out.mkdir(parents=True)
    old=args.source_root/'deployment/v6-h800-20261015'
    paths={'r25':old/'s2-global-decay-diagnostic/output/original_a8g_trace.npz',
           'm25':old/'s2-global-decay-diagnostic/output/manifest.json',
           'r50':old/'a2-gate-attribution-prepared/frozen/a2_frozen_inputs.npz',
           'm50':old/'a2-gate-attribution-prepared/frozen/manifest.json',
           'parents':HERE.parent/'results/br-0.1-dev-seed17/parents.json',
           'old_a2_conclusion':args.source_root/'results/v6_1/v6_1-20261019-a2-object-cost/conclusion.json',
           'old_a2_components':args.source_root/'results/v6_1/v6_1-20261018-a2-beta-context/signal-noise.json',
           'protocol':HERE/'PROTOCOL.md','runner':Path(__file__)}
    hashes={k:sha(p) for k,p in paths.items()}
    write(args.out/'frozen-protocol.json',{'protocol':'BR-0.2','created_utc':datetime.now(timezone.utc).isoformat(),
          'before_new_accounting':True,'source_sha256':hashes,'parents':864,'new_training_steps':0,'gpu_seconds':0,
          'same_budget_comparison':'Original chosen K subset versus same-parent oracle K subset; reader/input/horizon/loss unchanged',
          'dev_only':True,'new_online_policy':False})
    start=time.perf_counter()
    assert hashes['r25']==R25_SHA and hashes['r50']==R50_SHA
    m25,m50=(json.loads(paths[x].read_text()) for x in ('m25','m50'))
    assert m25['noise_seed']==m50['noise_seed']==0
    assert m25['sbar_inj']==.25 and m50['track']['sbar_inj']==.5
    assert m25['inputs']['slow']['sha256']==m50['source_hashes']['slow_checkpoint']
    assert m25['source_sha256']['pipeline.py']==m50['source_hashes']['pipeline.py']
    assert m50['ordered_a_ids']==m25['selected_ids'][:32]
    with np.load(paths['r25'],allow_pickle=False) as z25,np.load(paths['r50'],allow_pickle=False) as z50:
        arrays={k:z50[k] for k in z50.files};un={};audit={}
        for cond in ('EXACT','NEWPATH','CHANGE'):
            for a,b in (('Z','Z'),('EPS','EPS'),('O','O'),('W','W'),('KEY','key')):
                assert m25['input_arrays'][cond][a]==m50['array_identities'][f'{cond}__{b}']
            for x in ('R','O','W','key'):
                assert identity(z25[f'{cond}__{x}'])==m25['trace']['arrays'][f'{cond}__{x}']
                assert identity(z50[f'{cond}__{x}'])==m50['array_identities'][f'{cond}__{x}']
            for x in ('O','W','key'): assert np.array_equal(z25[f'{cond}__{x}'],z50[f'{cond}__{x}'])
            u,n=components(z25[f'{cond}__R'],z50[f'{cond}__R']);un[cond]=(u,n)
            err=float(np.max(np.abs(u+n-z50[f'{cond}__R'])))
            assert err<1e-12;audit[cond]={'reconstruction_error':err,'same_Z_EPS_O_W_key_identity':True}
    parents=json.loads(paths['parents'].read_text());assert len(parents)==864
    assert {p['source_index'] for p in parents}==set(range(24,32))
    rows=[analyze_parent(p,arrays,un) for p in parents]
    groups=aggregate(rows);collisions=collision_audit(rows)
    top=next(r for r in rows if (r['method'],r['condition'],r['source_index'],r['original_t'])==('K16/PaFu','CHANGE',31,464))
    summary={'protocol':'BR-0.2','scope':'Connect existing cases via exact frozen-parent component accounting; no new online policy',
        'parents':len(rows),'losses_verified':sum(len(r['candidate_parts']) for r in rows),
        'max_loss_reconstruction_error':max(r['reconstruction_max_abs_error'] for r in rows),
        'input_identity_audit':audit,'groups':groups,'collision_audit':collisions,
        'runtime_seconds':time.perf_counter()-start,'training_steps':0,'gpu_seconds':0,'new_scientific_online_arms':0,
        'positive_regret_parents':sum(r['positive_task_regret'] for r in rows),
        'positive_regret_signal_improves':sum(r['positive_task_regret'] and r['oracle_improves_signal'] for r in rows),
        'positive_regret_signal_worsens':sum(r['positive_task_regret'] and r['oracle_worsens_signal'] for r in rows),
        'disjoint_task_signal_optimal_sets':sum(r['disjoint_task_signal_optimal_sets'] for r in rows),
        'limitations':['U is a standardized pre-injection slow-model residual, not physical ground truth.',
                      'Saved parents reuse eight A clusters and adjacent CHANGE B sources; 864 rows are not independent.',
                      'Frozen-path realized decomposition does not identify causal effects of noise on policy selection.',
                      'Raw Z/actions/continuous pose/RGB absent; no visual task metric is reconstructed.',
                      'Exact collision audit is not a trained causal-predictability test.']}
    for k,p in paths.items():assert sha(p)==hashes[k],f'Source changed: {k}'
    write(args.out/'parents.json',rows);write(args.out/'summary.json',summary);write(args.out/'previous_top_case.json',top)
    write(args.out/'source_manifest.json',{'source_sha256':hashes,'sources_unchanged':True,
         'published_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in args.out.iterdir() if p.is_file()}})
    print(json.dumps({k:summary[k] for k in ('parents','losses_verified','max_loss_reconstruction_error','runtime_seconds','positive_regret_parents','positive_regret_signal_improves','positive_regret_signal_worsens','disjoint_task_signal_optimal_sets')},indent=2))


if __name__=='__main__': main()
