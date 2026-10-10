#!/usr/bin/env python3
"""Fixed raw/pool correspondence and equal-byte token-bank diagnostics (CPU only).

Scientific scoring requires a published, read-back source/design receipt.
Source RGB colors supply a privileged ROI; no target RGB/labels select motion.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='4'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from v5_5_counterfactual import cue_geometry,render_endpoint
from v5_4_reader import entry_identity,score_outputs
METHODS=('raw_native','pool_native','uniform_budget','roi_adaptive_budget','random_budget','change_priority_budget')
BUDGET_METHODS=METHODS[2:]
DESIGN=dict(schema='v5.6-correspondence-fixed-1',posthoc=True,population=dict(split='dev',pair_ids=[640,895],episodes=512),
    immutable_inputs={'archive_manifest.json':'b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663',
      'counterfactual_cache/receipt.json':'000ec48d9a2a7ba7ea07025f87e246ae87045d3a122ca478f6d7b10c4e5d4f46',
      'report/predictions.npz':'f3f66de4fb643ccbf4bb011761111d24aa49170e539c186a41d93068026eb294'},
    raw='V5.5 original000/111 same-call stored FP32 spatial256x1280 tokens; BF16 encoder forward',
    roi='A RGB R235 B60;B R60 B235;source-frame color bbox25/50/75percent xy Cartesian3x3. No target RGB ROI or labels.',
    grid=dict(minimum_pixels=-6.,maximum_pixels=6.,step_pixels=.5,count=625,order='dy major,dx minor'),
    interpolation='Bilinear,physical pixel centers(native grid centers=(i+.5)*224/side-.5),border replication;record clamp',
    descriptor='Center each interpolated descriptor over channels;cosine similarity;mean cosine distance across9 ROI points;norm squared <=1e-24 gives similarity0 and marked undefined',
    tie=dict(absolute_cost_tolerance=1e-12,selection='smallest squared displacement then dx then dy',near_optimal_tolerance=1e-6),
    native_comparison='Full raw16x16 versus same-raw meanpool4x4;FP64 accumulation then FP32 storage,no final BF16 quantization;same displacement candidates,ROI,cost and interpolation;not bitwise GPU pooling',
    budget=dict(tokens_per_frame=16,channels=1280,feature_dtype='float32',index_dtype='uint16',bytes_per_frame=81952,bytes_two_frames=163904,
      common_header='Method identifier and common known-grid/precision constants excluded equally from every method;no extra per-object token-bank copy',
      uniform='4x4 spatial means;16 tokens',
      others='2x2 global spatial means4 tokens plus12 raw tokens;identical selected indices used in bothframes',
      roi_adaptive='6 perobject;source RGB color coverage perrawcell descending,then distance tosource color centroid,then rasterindex;withoutreplacement',
      random='12 unique rawcells;default_rng(5617+2*pair_id+variant);no target/labels',
      change_priority='12 largest raw feature-change squared L2;ties rasterindex;two-frame lookahead,not causal memory selection',
      common_reading='All4 budget banks:inverse-distance-squared interpolation from selected centers to16x16 centers(exact atstoredcenter),then same bilinear+cosine reader',
      codes='raw0..255;2x2means256..259;4x4means260..275'),
    metrics='All512 predictions/costs/ties/margins/border flags;continuous archive displacement with original frozenFIT mean;no method selection/no refit',
    limits='Privileged local-correspondence and two-frame storage prototypes;not supervised decoder capacity bounds,encoder failure proof,long-term memory evidence or fresh confirmation',
    no_training=True,test_opened=False,p0_authorized=False)
SOURCE_FILES=('scripts/v5_6_correspondence.py','scripts/v5_5_counterfactual.py','scripts/v5_4_reader.py',
              'scripts/v5_3_reader.py','pmr/visual_memory/data.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def source_identity():return {name:sha(ROOT/name) for name in SOURCE_FILES}
def offsets():
    axis=np.arange(-6,6.01,.5);dx,dy=np.meshgrid(axis,axis)
    return np.stack((dx.ravel(),dy.ravel()),axis=1)
OFFSETS=offsets()


def centers(side):
    a=(np.arange(side)+.5)*224/side-.5;x,y=np.meshgrid(a,a)
    return np.stack((x.ravel(),y.ravel()),axis=1)


def pool(raw,side):
    x=np.asarray(raw);channels=x.shape[-1];factor=16//side
    if x.shape[-2]!=256 or 16%side:raise ValueError('Fixed16x16 spatial grid required')
    return x.reshape(*x.shape[:-2],side,factor,side,factor,channels).mean(axis=(-4,-2),dtype=np.float64).astype(np.float32).reshape(*x.shape[:-2],side*side,channels)


def bilinear_weights(points,side):
    points=np.asarray(points,dtype=np.float64);u=(points+.5)/(224/side)-.5
    clamp=np.any((u<0)|(u>side-1),axis=-1);u=np.clip(u,0,side-1)
    left=np.floor(u).astype(int);right=np.minimum(left+1,side-1);frac=u-left
    weights=np.zeros((points.reshape(-1,2).shape[0],side*side));row=np.arange(len(weights));f=frac.reshape(-1,2);l=left.reshape(-1,2);r=right.reshape(-1,2)
    for xx,yy,w in ((l[:,0],l[:,1],(1-f[:,0])*(1-f[:,1])),(r[:,0],l[:,1],f[:,0]*(1-f[:,1])),
                    (l[:,0],r[:,1],(1-f[:,0])*f[:,1]),(r[:,0],r[:,1],f[:,0]*f[:,1])):
        np.add.at(weights,(row,yy*side+xx),w)
    return weights.reshape(*points.shape[:-1],side*side),clamp


def idw_reconstruction(points):
    distance=np.sum((centers(16)[:,None,:]-np.asarray(points)[None,:,:])**2,axis=-1)
    w=1/np.maximum(distance,1e-24)
    exact=distance==0
    for row in np.flatnonzero(exact.any(1)):w[row]=exact[row].astype(float)
    return w/w.sum(1,keepdims=True)


def source_roi(rgb):
    result=[];masks=[];centroids=[]
    for red,blue in ((235,60),(60,235)):
        mask=(rgb[:,:,0]==red)&(rgb[:,:,2]==blue);yy,xx=np.nonzero(mask)
        if not len(xx):raise ValueError('Source color ROI absent')
        xs=xx.min()+(xx.max()-xx.min())*np.array([.25,.5,.75]);ys=yy.min()+(yy.max()-yy.min())*np.array([.25,.5,.75])
        x,y=np.meshgrid(xs,ys);result.append(np.stack((x.ravel(),y.ravel()),1));masks.append(mask);centroids.append([xx.mean(),yy.mean()])
    return np.asarray(result),np.asarray(masks),np.asarray(centroids)


def select_banks(raw,rgb,pair,variant):
    if raw.shape[:2]!=(2,256) or not np.isfinite(raw).all():raise ValueError('Two finite raw frames required')
    roi,masks,centroids=source_roi(rgb);cellcenters=centers(16);chosen=[]
    for obj in (0,1):
        coverage=masks[obj].reshape(16,14,16,14).sum((1,3)).ravel()
        dist=((cellcenters-centroids[obj])**2).sum(1)
        order=np.lexsort((np.arange(256),dist,-coverage))
        selected=[int(i) for i in order if int(i) not in chosen][:6];chosen.extend(selected)
    random=np.random.default_rng(5617+2*int(pair)+int(variant)).choice(256,12,replace=False)
    energy=np.sum((raw[1].astype(np.float64)-raw[0].astype(np.float64))**2,axis=1)
    change=np.lexsort((np.arange(256),-energy))[:12]
    p4=pool(raw,4);p2=pool(raw,2)
    banks={'uniform_budget':(p4,np.arange(260,276,dtype=np.uint16),centers(4))}
    for name,idx in (('roi_adaptive_budget',chosen),('random_budget',random),('change_priority_budget',change)):
        idx=np.asarray(idx,dtype=int)
        banks[name]=(np.concatenate((p2,raw[:,idx]),axis=1),np.r_[np.arange(256,260),idx].astype(np.uint16),np.concatenate((centers(2),cellcenters[idx])))
    for bank,codes,_ in banks.values():
        if bank.dtype!=np.float32 or bank.shape!=(2,16,raw.shape[-1]) or codes.shape!=(16,) or len(np.unique(codes))!=16:raise ValueError('Equal-byte bank identity failed')
    return banks,roi


def costs(bank,source_weights,target_weights):
    """Exact centered cosine via small Gram matrices; no high-dimensional query tensor."""
    s=np.asarray(source_weights);t=np.asarray(target_weights)
    active=np.flatnonzero((np.abs(s).sum(0)+np.abs(t).sum((0,1)))>0)
    s=s[:,active];t=t[:,:,active]
    x=bank[0,active].astype(np.float64);y=bank[1,active].astype(np.float64)
    x-=x.mean(1,keepdims=True);y-=y.mean(1,keepdims=True)
    source=s@x;source_norm=np.sum(source*source,1)
    source_target=source@y.T;target_gram=y@y.T
    numerator=np.einsum('pn,kpn->kp',source_target,t)
    norm=np.einsum('kpn,nm,kpm->kp',t,target_gram,t)
    if np.min(norm)<-1e-8:raise ValueError('Negative descriptor norm beyond roundoff')
    denominator=np.sqrt(source_norm[None,:]*np.maximum(norm,0))
    valid=(source_norm[None,:]>1e-24)&(norm>1e-24)
    cosine=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=valid)
    cost=np.mean(1-np.clip(cosine,-1,1),1)
    if not np.isfinite(cost).all():raise ValueError('Nonfinite candidate costs')
    return cost,dict(undefined_comparisons=int((~valid).sum()),total_comparisons=int(valid.size),source_zero_norm_count=int((source_norm<=1e-24).sum()))


def match(bank,roi,side=None,stored_centers=None):
    reconstruction=None if stored_centers is None else idw_reconstruction(stored_centers)
    if (side is None)==(reconstruction is None):raise ValueError('Choose native or reconstructed field exactly once')
    side=16 if reconstruction is not None else side
    source,source_clamp=bilinear_weights(roi,side);target,target_clamp=bilinear_weights(roi[None]+OFFSETS[:,None],side)
    if reconstruction is not None:source=source@reconstruction;target=target@reconstruction
    cost,diagnostics=costs(bank,source,target)
    tied=np.flatnonzero(cost<=cost.min()+1e-12)
    order=np.lexsort((OFFSETS[tied,1],OFFSETS[tied,0],np.sum(OFFSETS[tied]**2,1)));best=int(tied[order[0]])
    sorted_cost=np.sort(cost)
    diagnostics.update(best_index=best,best_cost=float(cost[best]),second_minus_best=float(sorted_cost[1]-sorted_cost[0]),
        tie_count=int(len(tied)),near_best_count=int(np.sum(cost<=cost.min()+1e-6)),boundary_optimum=bool(np.any(np.abs(OFFSETS[best])==6)),
        source_clamped_points=int(source_clamp.sum()),best_target_clamped_points=int(target_clamp[best].sum()),all_target_clamped_points=int(target_clamp.sum()))
    return OFFSETS[best].copy(),cost,diagnostics


def self_test():
    rng=np.random.default_rng(56061);locations=centers(16);motion=np.array([2.5,-3.])
    linear=rng.normal(size=(2,7));bias=rng.normal(size=7)*50
    raw=np.stack((locations@linear+bias,(locations-motion)@linear+bias)).astype(np.float32)
    roi=np.array([[x,y] for y in (85,91,97) for x in (85,91,97)],dtype=float)
    recovered={}
    for name,bank,side in (('raw',raw,16),('pool',pool(raw,4),4)):
        displacement,cost,d=match(bank,roi,side=side)
        np.testing.assert_array_equal(displacement,motion);recovered[name]=displacement.tolist()
        # Independent explicitly interpolated descriptors verify Gram shortcut.
        sw,_=bilinear_weights(roi,side);tw,_=bilinear_weights(roi[None]+OFFSETS[:,None],side)
        source=sw@bank[0].astype(float);target=tw@bank[1].astype(float)
        source-=source.mean(-1,keepdims=True);target-=target.mean(-1,keepdims=True)
        expected=1-np.einsum('pc,kpc->kp',source,target)/(np.linalg.norm(source,axis=-1)[None]*np.linalg.norm(target,axis=-1))
        np.testing.assert_allclose(cost,expected.mean(1),atol=1e-11,rtol=0)
    # Software RGB; no scientific generator, labels or images accessed.
    rgb=np.zeros((224,224,3),dtype=np.uint8);rgb[35:55,70:90]=[235,77,60];rgb[145:165,130:150]=[60,90,235]
    banks,regions=select_banks(raw,rgb,0,0)
    for name,(bank,codes,xy) in banks.items():
        if bank[0].nbytes+codes.nbytes!=16*7*4+16*2:raise AssertionError('Bytes differ')
        pred,cost,d=match(np.stack((bank[0],bank[0])),regions[0],stored_centers=xy)
        np.testing.assert_array_equal(pred,[0,0])
    flat=np.ones((2,256,7),dtype=np.float32);p,c,d=match(flat,roi,side=16)
    np.testing.assert_array_equal(p,[0,0]);assert d['tie_count']==625 and d['source_zero_norm_count']==9
    # Index order is spatial raster, not channel or object order.
    sentinel=np.arange(256).reshape(1,256,1).repeat(2,0).astype(np.float32)
    expected=np.array([[sentinel[0].reshape(16,16)[y:y+4,x:x+4].mean() for x in range(0,16,4)] for y in range(0,16,4)])
    np.testing.assert_array_equal(pool(sentinel,4)[0,:,0],expected.ravel())
    return dict(status='passed',software_only=True,known_translation=recovered,gram_matches_explicit_descriptors=True,
        equal_payload_bytes_verified=True,constant_ambiguity_preserved=True,stationary_match_zero=True)


def validate_publication(publication,plan):
    p=json.loads(Path(publication).read_text());design=json.loads(Path(plan).read_text())
    if p.get('verified_readback') is not True or design['design']!=DESIGN or design['sources']!=source_identity():raise ValueError('Frozen design/readback mismatch')
    if p.get('verified_unix',0)<design['created_unix']:raise ValueError('Publication must follow design freeze')
    files=p.get('files',{})
    for path in (ROOT/'scripts/v5_6_correspondence.py',Path(plan).resolve()):
        rel=path.relative_to(ROOT).as_posix();record=files.get(rel);h=record.get('sha256') if isinstance(record,dict) else record
        if h!=sha(path):raise ValueError('Published source/plan hash missing or different: '+rel)
    return p,design


def run(scene,out,publication,plan):
    started=time.time();scene=Path(scene).resolve();out=Path(out).resolve();pub,design=validate_publication(publication,plan)
    initial_sources=source_identity();initial_plan=sha(plan);initial_publication=sha(publication)
    if out==scene or scene in out.parents:raise ValueError('Output must be outside immutable input scene')
    if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh result directory required; preserve previous results')
    out.mkdir(parents=True,exist_ok=False)
    for relative,digest in DESIGN['immutable_inputs'].items():
        if sha(scene/relative)!=digest:raise ValueError('Pinned original input changed: '+relative)
    manifest=json.loads((scene/'archive_manifest.json').read_text());inputs={}
    def checked(relative):
        path=scene/relative;record=dict(bytes=path.stat().st_size,sha256=sha(path))
        if manifest['files'][relative]!=record:raise ValueError('Sealed scene input changed: '+relative)
        inputs[relative]=record;return path
    receipt=json.loads(checked('counterfactual_cache/receipt.json').read_text())
    if receipt['groups']!=256 or receipt['test_opened'] is not False:raise ValueError('Original cache population changed')
    with np.load(checked('report/predictions.npz'),allow_pickle=False) as z:
        truth=z['targets'][...,7,:].reshape(512,8).astype(np.float64)
    with np.load(checked('lane_reader/frozen_model.npz'),allow_pickle=False) as z:mean=z['shared_target_mean'].copy()
    predictions=np.empty((512,len(METHODS),4),dtype=np.float64)
    all_costs=np.empty((512,len(METHODS),2,625),dtype=np.float64);rows=[];banks_log=[]
    for pair in range(640,896):
        raw_record=receipt['raw_records'][str(pair)];rawpath=checked('counterfactual_cache/'+raw_record['path'])
        if sha(rawpath)!=raw_record['sha256']:raise ValueError('Raw receipt hash differs')
        with np.load(rawpath,allow_pickle=False) as z:raw=z['raw']
        if raw.shape!=(2,2,256,1280) or raw.dtype!=np.float32 or {'raw':entry_identity(raw)}!=raw_record['entry_hashes']:raise ValueError('Original raw identity mismatch')
        shard=json.loads(checked(f'counterfactual_cache/shards/{pair:06d}.json').read_text())
        for variant in (0,1):
            idx=2*(pair-640)+variant;pos,colors,yellow=cue_geometry('dev',pair,variant)
            # Only original source RGB is used by ROI selection/prediction.
            rgb,_=render_endpoint(pos[0],colors,yellow[0]);original=shard['inputs']['variants'][variant]
            if entry_identity(rgb)!=original['unique_rgb'][0]:raise ValueError('Source RGB differs from authenticated original')
            labels=np.concatenate((pos[1].ravel(),(pos[1]-pos[0]).ravel())).astype(np.float32)
            np.testing.assert_array_equal(labels.astype(np.float64),truth[idx])
            banks,regions=select_banks(raw[variant],rgb,pair,variant)
            for name,(bank,codes,xy) in banks.items():
                if bank[0].nbytes+codes.nbytes!=DESIGN['budget']['bytes_per_frame']:raise ValueError('Production payload differs')
                banks_log.append(dict(pair_id=pair,variant=variant,method=name,codes=codes.tolist(),
                    descriptor_hashes=[entry_identity(x) for x in bank],metadata_hash=entry_identity(codes),
                    bytes_per_frame=bank[0].nbytes+codes.nbytes,source_rgb_sha256=original['unique_rgb'][0]['sha256']))
            for method,name in enumerate(METHODS):
                if name=='raw_native':bank=raw[variant];args=dict(side=16)
                elif name=='pool_native':bank=pool(raw[variant],4);args=dict(side=4)
                else:bank,_,xy=banks[name];args=dict(stored_centers=xy)
                for obj in (0,1):
                    pred,cost,d=match(bank,regions[obj],**args);predictions[idx,method,obj*2:obj*2+2]=pred/224;all_costs[idx,method,obj]=cost
                    rows.append(dict(pair_id=pair,variant=variant,object=('A','B')[obj],method=name,
                        prediction_dx_pixels=float(pred[0]),prediction_dy_pixels=float(pred[1]),
                        target_dx_pixels=float(truth[idx,4+obj*2]*224),target_dy_pixels=float(truth[idx,5+obj*2]*224),**d))
        if (pair-639)%32==0:print(json.dumps(dict(progress_worlds=pair-639,total_worlds=256)),flush=True)
    scores={};y=truth[:,4:]
    for i,name in enumerate(METHODS):
        p=predictions[:,i];e=p-y;den=np.mean((y-mean[4:])**2,axis=0)
        scores[name]=dict(per_axis_mse=np.mean(e*e,0).tolist(),per_axis_skill=(1-np.mean(e*e,0)/den).tolist(),
            motion_skill=float(np.mean(1-np.mean(e*e,0)/den)),dx_mse=float(np.mean(e[:,[0,2]]**2)),
            direction_errors=(np.sign(p)!=np.sign(y)).sum(0).tolist(),predicted_zero_count=(p==0).sum(0).tolist(),
            mean_abs_amplitude_ratio=(np.mean(np.abs(p),0)/np.mean(np.abs(y),0)).tolist())
    np.savez_compressed(out/'costs_and_predictions.npz',costs=all_costs,predictions=predictions,targets=y,offsets_pixels=OFFSETS,
                        methods=np.array(METHODS),pair_ids=np.repeat(np.arange(640,896),2),variants=np.tile([0,1],256),frozen_fit_mean=mean)
    with (out/'per_episode_object.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'selected_banks.json',banks_log)
    for relative,record in inputs.items():
        if sha(scene/relative)!=record['sha256']:raise ValueError('Archived input changed during run')
    if source_identity()!=initial_sources or sha(plan)!=initial_plan or sha(publication)!=initial_publication:raise ValueError('Source/design/publication changed during run')
    save(out/'report.json',dict(schema=DESIGN['schema'],status='complete',design=DESIGN,scores=scores,
        provenance=dict(source_identity=source_identity(),publication_sha256=sha(publication),plan_sha256=sha(plan),
            old_scene_manifest_sha256=sha(scene/'archive_manifest.json'),inputs=inputs,command=sys.argv),
        ambiguity={name:dict(tie_gt1=sum(row['tie_count']>1 for row in rows if row['method']==name),
            boundary_optima=sum(row['boundary_optimum'] for row in rows if row['method']==name)) for name in METHODS},
        runtime=dict(python=sys.version,numpy=np.__version__,threads=4,elapsed_seconds=time.time()-started),
        test_opened=False,no_training=True,no_long_term_memory_claim=True))
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='phase',required=True)
    sub.add_parser('self-test');p=sub.add_parser('freeze-design');p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--publication',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    if args.phase=='self-test':answer=self_test()
    elif args.phase=='freeze-design':
        answer=dict(design=DESIGN,sources=source_identity(),software_verification=self_test(),created_unix=time.time());save(args.out,answer)
    else:
        result_path=args.out.resolve();scene_path=args.scene.resolve()
        failure_output_allowed=not result_path.exists() and result_path!=scene_path and scene_path not in result_path.parents
        try:answer=run(args.scene,args.out,args.publication,args.plan)
        except Exception as exc:
            if failure_output_allowed:
                args.out.mkdir(parents=True,exist_ok=True)
                path=args.out/'failure.json'
                if not path.exists():save(path,dict(status='failed',error=str(exc),traceback=traceback.format_exc(),command=sys.argv))
            raise
    print(json.dumps(answer,indent=2,allow_nan=False))

if __name__=='__main__':main()
