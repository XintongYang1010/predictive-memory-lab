#!/usr/bin/env python3
"""Known-cue RGB observability audit; fixed color centroid primary, bbox auxiliary."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='4'
import argparse,csv,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
import PIL
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
from v5_5_counterfactual import cue_geometry,render_endpoint
from v5_4_reader import entry_identity,score_outputs
from pmr.visual_memory.data import seed_for
AXES=('A_dx','A_dy','B_dx','B_dy')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')

def rgb_centers(rgb):
 """No latent position, object masks, per-scene sampled color or labels enter."""
 centers=[];bbox=[];masks=[]
 for red,blue in ((235,60),(60,235)):
  mask=(rgb[:,:,0]==red)&(rgb[:,:,2]==blue);yy,xx=np.nonzero(mask)
  if len(xx)==0:raise ValueError('Fixed color detector found no pixels')
  centers.append([xx.mean(),yy.mean()]);bbox.append([(xx.min()+xx.max())/2,(yy.min()+yy.max())/2]);masks.append(mask)
 return np.array(centers),np.array(bbox),masks

def run(scene,v55,out):
 start=time.time();out.mkdir(parents=True,exist_ok=True)
 if (out/'report.json').exists():raise FileExistsError('Prior result preserved')
 manifest=json.loads((scene/'archive_manifest.json').read_text());inputs={}
 def checked(path,root=scene,records=None):
  rec=dict(bytes=path.stat().st_size,sha256=sha(path));relative=path.relative_to(root).as_posix()
  if rec!=(manifest['files'] if records is None else records)[relative]:raise ValueError('Archive mismatch: '+relative)
  inputs[str(path.resolve())]=rec
  return path
 predpath=checked(scene/'bridge_report/predictions.npz')
 with np.load(predpath,allow_pickle=False) as z:truth=z['targets'].reshape(512,8).astype(np.float64)
 modelpath=checked(scene/'reader/frozen_model.npz')
 with np.load(modelpath,allow_pickle=False) as z:mean=z['shared_target_mean'].copy()
 reportdir=v55/'report-snapshot/report';rm=json.loads((reportdir/'manifest.json').read_text())
 p=checked(reportdir/'predictions.npz',reportdir,rm)
 with np.load(p,allow_pickle=False) as z:
  np.testing.assert_array_equal(z['targets'][...,7,:].reshape(512,8),truth)
  old_predictions={name:z[name][...,7,:].reshape(512,8) for name in ('W_concat','lane_W')}
 rp=checked(reportdir/'report.json',reportdir,rm);oldreport=json.loads(rp.read_text())
 allrows=[];centroid=[];bbox=[];frame_records=[];pixelcounts=[];same_frames=0;same_objectmasks=0
 image_seen={};input_seen={};image_collisions=[];input_collisions=[];reflection_counts=np.zeros(4,dtype=int)
 sources={name:sha(ROOT/name) for name in ('pmr/visual_memory/data.py','scripts/v5_5_counterfactual.py','scripts/v5_4_reader.py','scripts/v5_3_reader.py')}
 for pair in range(640,896):
  path=checked(scene/'bridge_cache/shards'/f'{pair:06d}.json');record=json.loads(path.read_text())
  if record['unit']!=pair or record['role']!='bridge':raise ValueError('Episode ordering mismatch')
  for variant in (0,1):
   index=2*(pair-640)+variant;pos,colors,yellow=cue_geometry('dev',pair,variant)
   labels=np.concatenate((pos[1].ravel(),(pos[1]-pos[0]).ravel())).astype(np.float32)
   np.testing.assert_array_equal(labels.astype(np.float64),truth[index])
   rng=np.random.default_rng(seed_for('dev',pair));initial=rng.uniform([.2,.23],[.8,.32],(2,2));initial[1,1]+=.42
   velocity=rng.uniform(.008,.018,(2,2))*rng.choice([-1,1],(2,2))*(1 if variant==0 else -1)
   lo=np.array([[.1,.15],[.1,.6]]);hi=np.array([[.9,.4],[.9,.85]])
   reflection=(np.floor((initial+velocity*6-lo)/(hi-lo))!=np.floor((initial+velocity*7-lo)/(hi-lo))).ravel();reflection_counts+=reflection
   cc=[];bb=[];hashes=[];masks=[]
   for frame in (0,1):
    rgb,object_masks=render_endpoint(pos[frame],colors,yellow[frame]);identity=entry_identity(rgb)
    if identity!=record['inputs']['cue_rgb'][variant*2+frame]:raise ValueError('Original cue RGB hash mismatch')
    c,b,m=rgb_centers(rgb);cc.append(c);bb.append(b);masks.append(m);hashes.append(identity['sha256'])
    pixelcounts.extend(int(mm.sum()) for mm in m)
    if np.any(object_masks.sum(0)>1):raise ValueError('Original objects overlap')
    if identity['sha256'] in image_seen and not np.array_equal(image_seen[identity['sha256']],pos[frame]):image_collisions.append([pair,variant,frame+6])
    image_seen[identity['sha256']]=pos[frame].copy()
    frame_records.append(dict(pair_id=pair,variant=variant,time=frame+6,rgb=identity,hash_matches_original=True,
      centroid_xy_pixels=c.tolist(),auxiliary_bbox_xy_pixels=b.tolist(),color_pixel_counts=[int(mm.sum()) for mm in m]))
   if hashes[0]==hashes[1]:same_frames+=1
   same_objectmasks+=sum(np.array_equal(masks[0][obj],masks[1][obj]) for obj in (0,1))
   key=tuple(hashes)
   if key in input_seen and not np.array_equal(input_seen[key],labels):input_collisions.append([pair,variant])
   input_seen[key]=labels.copy()
   cc=np.asarray(cc);bb=np.asarray(bb)
   pc=np.concatenate((cc[1].ravel(),(cc[1]-cc[0]).ravel()))/224
   pb=np.concatenate((bb[1].ravel(),(bb[1]-bb[0]).ravel()))/224
   centroid.append(pc);bbox.append(pb)
   for d,axis in enumerate(AXES):
    ypx=truth[index,d+4]*224;classify='zero' if ypx==0 else ('tiny_nonzero_lt_1px' if abs(ypx)<1 else 'at_least_1px')
    row=dict(pair_id=pair,variant=variant,axis=axis,label_dx_dy_pixels=ypx,centroid_pixels=pc[d+4]*224,
      auxiliary_bbox_pixels=pb[d+4]*224,global_pixels=old_predictions['W_concat'][index,d+4]*224,
      lane_pixels=old_predictions['lane_W'][index,d+4]*224,motion_bin=classify,cue_interval_reflection=bool(reflection[d]))
    allrows.append(row)
 predictions={'rgb_centroid_primary':np.asarray(centroid),'rgb_bbox_auxiliary':np.asarray(bbox),**old_predictions}
 scores={};t=truth[:,4:]*224
 bins={'zero':t==0,'tiny_nonzero_lt_1px':(np.abs(t)>0)&(np.abs(t)<1),'at_least_1px':np.abs(t)>=1}
 for name,pred in predictions.items():
  metrics=score_outputs(pred,truth,mean);p=pred[:,4:]*224;e=p-t
  metrics.update(dx_mse_normalized=float(np.mean((pred[:,[4,6]]-truth[:,[4,6]])**2)),dx_rmse_pixels=float(np.sqrt(np.mean(e[:,[0,2]]**2))),
    per_axis_motion_rmse_pixels=np.sqrt(np.mean(e*e,0)).tolist(),per_axis_max_abs_error_pixels=np.max(np.abs(e),0).tolist(),direction_by_bin={})
  for key,mask in bins.items():
   metrics['direction_by_bin'][key]=dict(count_per_axis=mask.sum(0).tolist(),
    strict_opposite_per_axis=((p*t<0)&mask).sum(0).tolist(),prediction_zero_per_axis=((p==0)&mask).sum(0).tolist(),
    sign_mismatch_including_zero_per_axis=((np.sign(p)!=np.sign(t))&mask).sum(0).tolist())
  scores[name]=metrics
 for name in old_predictions:
  np.testing.assert_allclose(scores[name]['per_dimension_skill'],oldreport['readers'][name]['original']['per_dimension_skill'],rtol=0,atol=1e-12)
 badcases=[row for row in allrows if row['pair_id'] in (761,844,891) and row['variant']==1]
 np.savez_compressed(out/'predictions.npz',targets=truth,**predictions,pair_ids=np.repeat(np.arange(640,896),2),variants=np.tile([0,1],256),frozen_fit_mean=mean)
 with (out/'per_episode_coordinate.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 save(out/'frame_hashes_and_centers.json',frame_records)
 result=dict(schema='v5.5-known-rgb-observability-1',posthoc=True,privileged_renderer_color_knowledge=True,primary='rgb_centroid_primary',auxiliary_not_selected='rgb_bbox_auxiliary',
  population=dict(split='dev',pair_ids=[640,895],episodes=512,frames=1024),axes=list(AXES),
  prediction_rule='A pixels R=235,B=60;B pixels R=60,B=235. Arithmetic centroid of qualifying RGB pixels, or fixed auxiliary min/max midpoint. No latent position/masks/per-case sampled green value/labels enter prediction.',
  target_definition='Continuous reflected endpoint position p7 and p7-p6, cast FP32 in original data.py; not raster center displacement or unreflected latent velocity.',
  denominator=dict(rule='Exact same original V5.4 frozen joint-FIT mean as V5.5 score_outputs; no Dev centering',frozen_fit_mean=mean.tolist(),
    per_dimension=np.mean((truth-mean)**2,0).tolist(),old_model_sha256=sha(modelpath)),
  tiny_definition='zero iff float32 archive displacement==0; tiny iff0<abs(displacement*224)<1pixel; at_least_1px iff abs>=1pixel. No case excluded.',
  scores=scores,badcases=badcases,all1024_rgb_hashes_match=True,all512_labels_match=True,
  visibility=dict(open_shutters_at_cues=True,all2048_target_masks_disjoint_and_detected=True,color_pixel_count_minmax=[min(pixelcounts),max(pixelcounts)],
    no_canvas_clipping='centers x[.1,.9],A_y[.15,.4],B_y[.6,.85],radius.06; both outlines within canvas; yellow_y[.48,.52] lies between target extents'),
  reflection_per_axis=reflection_counts.tolist(),identical_two_cue_rgb_episode_count=same_frames,
  identical_two_cue_target_color_masks_count=same_objectmasks,same_rgb_different_latent_position_observed=image_collisions,same_two_rgb_different_labels_observed=input_collisions,
  limits='A privileged handcrafted RGB diagnostic, not an encoder result, a fair learned-reader comparison or a Bayes-optimal error floor. No collisions observed in these finite samples does not make continuous latent labels globally identifiable from rasterized RGB.',
  source_identity=sources,inputs=inputs,script_sha256=sha(__file__),command=sys.argv,
  runtime=dict(python=sys.version,numpy=np.__version__,pillow=PIL.__version__,elapsed_seconds=time.time()-start),
  no_training=True,no_new_population=True,test_opened=False)
 save(out/'report.json',result)
 save(out/'input_inventory.json',dict(archive_manifest_sha256=sha(scene/'archive_manifest.json'),report_manifest_sha256=sha(reportdir/'manifest.json'),inputs=inputs,sources=sources))
 print(json.dumps(dict(status='complete',primary=scores['rgb_centroid_primary'],auxiliary=scores['rgb_bbox_auxiliary'],reflections=reflection_counts.tolist()),indent=2,allow_nan=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--v55',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 try:run(args.scene,args.v55,args.out)
 except Exception as exc:
  args.out.mkdir(parents=True,exist_ok=True)
  save(args.out/'failure.json',dict(status='failed',error=str(exc),traceback=traceback.format_exc(),command=sys.argv))
  raise
