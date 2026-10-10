#!/usr/bin/env python3
"""Posthoc top-three V5.4 episode errors; CPU-only rerender of authenticated t6/t7 cues."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import PIL
from PIL import Image,ImageDraw

GENERATOR_SHA='ce5082f3be9006dcceb46fbaffadc27349f977a77abcf8992eda3e80f8775368'
RULE='Rank the 512 already evaluated Dev640..895 episodes by W concat mean squared error over A_dx and B_dx, descending; ties: smaller pair ID, then variant. Select the first three episodes; no distinct-pair constraint.'
ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def identity(a):
    a=np.asarray(a)
    return dict(sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),dtype=str(a.dtype),shape=list(a.shape))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def cue_world(pair,variant):
    """Exact original pre-update seed/draw/reflect math, evaluated only at t6/t7."""
    if not 640<=pair<=895 or variant not in (0,1):raise ValueError('Only already evaluated V5.4 bridge IDs allowed')
    seed=int.from_bytes(hashlib.sha256(f'v5-rgb-pairs-1/dev/{pair}'.encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed)
    p=rng.uniform([.2,.23],[.8,.32],(2,2));p[1,1]+=.42
    v=rng.uniform(.008,.018,(2,2))*rng.choice([-1,1],(2,2));v*=1 if variant==0 else -1
    rng.uniform(.008,.018,2)*rng.choice([-1,1],2)*(1 if variant==0 else -1)
    colors=[tuple(int(x) for x in c) for c in rng.integers(60,240,(2,3))]
    colors[0]=(235,colors[0][1]//2,60);colors[1]=(60,colors[1][1]//2,235)
    low=np.array([[.10,.15],[.10,.60]]);high=np.array([[.90,.40],[.90,.85]]);span=high-low
    raw=np.stack((p+6*v,p+7*v));pos=low+span-np.abs(np.mod(raw-low,2*span)-span)
    target=np.r_[pos[1].ravel(),(pos[1]-pos[0]).ravel()].astype(np.float32)
    crossing=np.floor((raw[0]-low)/span)!=np.floor((raw[1]-low)/span)
    ties=np.isclose(np.mod(raw-low,span),0,rtol=0,atol=1e-14)
    return dict(pos=pos,target=target,colors=colors,crossing=crossing,boundary_ties=ties,
                yellow_x=[int((.5+.4*np.sin(t*2.3+pair))*224) for t in (6,7)])


def render_cue(world,t):
    """Pinned generator's unmodified open-shutter cue RGB recipe."""
    size=224;im=Image.new('RGB',(size,size),(30,35,42));draw=ImageDraw.Draw(im)
    for xx in range(0,size,max(1,size//16)):draw.line((xx,0,xx,size),fill=(42,47,54))
    radius=size*.06
    for obj in (0,1):
        x,y=world['pos'][t,obj]*size
        box=(int(x-radius),int(y-radius),int(x+radius),int(y+radius))
        draw.ellipse(box,fill=world['colors'][obj],outline=(250,250,240),width=max(1,size//100))
        draw.line((x-radius/2,y,x+radius/2,y),fill=(255,255,255),width=max(1,size//60))
        if obj==1:draw.line((x,y-radius/2,x,y+radius/2),fill=(255,255,255),width=max(1,size//60))
    x=world['yellow_x'][t];draw.rectangle((x-5,size*.48,x+5,size*.52),fill=(240,220,50))
    return np.asarray(im).copy()


def direction(v):return 'R' if v>0 else ('L' if v<0 else '0')
def reflection_text(crossing):
    return ', '.join(name+':'+('/'.join(axis for j,axis in enumerate(('x','y')) if crossing[i,j]) or 'none') for i,name in enumerate(('A','B')))


def build(scene,verification,generator,out):
    scene=Path(scene).resolve();out=Path(out).resolve()
    if out==scene or scene in out.parents:raise ValueError('Cannot write into immutable input snapshot')
    inventory=json.loads(Path(verification).read_text())
    if inventory.get('status')!='all_snapshot_files_verified':raise ValueError('Verified report snapshot required')
    used={}
    def checked(relative):
        path=scene/relative;expected=inventory['files'][relative]
        actual=dict(bytes=path.stat().st_size,sha256=sha(path))
        if actual!=expected:raise ValueError('Snapshot artifact changed: '+relative)
        used[relative]=actual
        return path
    report=json.loads(checked('bridge_report/report.json').read_text())
    receipt=json.loads(checked('bridge_cache/receipt.json').read_text())
    if (report.get('schema')!='v5.4-reader-evaluation-1' or report.get('status')!='evaluation_complete' or
        report.get('role')!='bridge' or report.get('groups')!=256 or report.get('examples')!=512 or
        report.get('test_opened') is not False or report.get('p0_authorized') is not False):
        raise ValueError('Completed bridge report required before posthoc reconstruction')
    path=checked('bridge_report/predictions.npz')
    if sha(path)!=report['predictions_sha256']:raise ValueError('Prediction identity mismatch')
    with np.load(path,allow_pickle=False) as z:a={key:z[key] for key in z.files}
    if not np.array_equal(a['pair_ids'],np.arange(640,896)) or str(a['role'].item())!='bridge':raise ValueError('Unexpected evaluation IDs')
    for key in ('N_concat','W_concat','targets'):
        if a[key].shape!=(256,2,8) or not np.isfinite(a[key]).all():raise ValueError('Invalid '+key)
    if sha(generator)!=GENERATOR_SHA or receipt['source_identity']['pmr/visual_memory/data.py']!=GENERATOR_SHA:
        raise ValueError('Original generator source changed')
    if sha(scene/'bridge_cache/receipt.json')!=report['cache_receipt_sha256']:raise ValueError('Cache receipt mismatch')
    error=np.mean((a['W_concat'][:,:, [4,6]]-a['targets'][:,:, [4,6]].astype(float))**2,axis=2)
    order=sorted(((float(error[i,v]),int(pair),v,i) for i,pair in enumerate(a['pair_ids']) for v in (0,1)),key=lambda x:(-x[0],x[1],x[2]))[:3]
    out.mkdir(parents=True,exist_ok=False)
    rows=[];images=[];checks=[];worlds=[]
    # All six raw-array hashes must match before any contact sheet is written.
    for rank,(mse,pair,variant,i) in enumerate(order,1):
        shard_path=checked(f'bridge_cache/shards/{pair:06d}.json')
        shard=json.loads(shard_path.read_text());inputs=shard['inputs']
        if (shard['role']!='bridge' or shard['unit']!=pair or inputs['original_generator_sha256']!=GENERATOR_SHA or
            inputs['cue_times']!=[6,7] or inputs['images_encoded']!=4):raise ValueError('Shard input identity mismatch')
        if inputs['targets']!=identity(a['targets'][i]) or shard['entry_hashes']['targets']!=inputs['targets']:
            raise ValueError('Archived pair labels differ from prediction targets')
        meta=inputs['metadata'][variant]
        if meta['episode_id']!=f'dev-{pair:06d}-{variant}' or meta['split']!='dev':raise ValueError('Episode metadata mismatch')
        world=cue_world(pair,variant)
        if not np.array_equal(world['target'],a['targets'][i,variant]):raise ValueError('Reconstructed labels differ from evaluated labels')
        rgb=np.stack([render_cue(world,t) for t in (0,1)])
        for t in (0,1):
            expected=inputs['cue_rgb'][2*variant+t];actual=identity(rgb[t])
            checks.append(dict(rank=rank,pair_id=pair,variant=variant,time=6+t,expected=expected,actual=actual,exact=actual==expected))
        row=dict(rank=rank,pair_id=pair,variant=variant,episode_id=meta['episode_id'],W_dx_mse=mse,
                 N_dx_mse=float(np.mean((a['N_concat'][i,variant,[4,6]]-a['targets'][i,variant,[4,6]])**2)),
                 cue_reflection=bool(world['crossing'].any()),reflection_axes=reflection_text(world['crossing']),
                 boundary_tie_count=int(world['boundary_ties'].sum()),yellow_pixel_dx=int(world['yellow_x'][1]-world['yellow_x'][0]))
        for obj,d in (('A',4),('B',6)):
            for name,key in (('true','targets'),('N','N_concat'),('W','W_concat')):
                value=float(a[key][i,variant,d]);row[f'{obj}_{name}_dx']=value;row[f'{obj}_{name}_direction']=direction(value)
        rows.append(row);images.append(rgb);worlds.append(world)
    verified=all(row['exact'] for row in checks)
    verification_result=dict(status='verified' if verified else 'RGB_HASH_MISMATCH',posthoc=True,
        selection_rule=RULE,selected=rows,rgb_checks=checks,labels_exact=True,
        original_generator=dict(path=str(Path(generator).resolve()),sha256=GENERATOR_SHA),
        helper_sha256=sha(__file__),source_commit=inventory['source_commit'],
        runtime=dict(python=sys.version,numpy=np.__version__,pillow=PIL.__version__,platform=platform.platform()),
        snapshot_verification=dict(path=str(Path(verification).resolve()),sha256=sha(verification)),input_files=used,
        scientific_source_changed=False,test_opened=False,new_scientific_samples=False)
    dump(out/'render_verification.json',verification_result)
    if not verified:raise ValueError('RGB_HASH_MISMATCH: retained verification; no contact sheet produced')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(10,14))
    fig.subplots_adjust(top=.92,bottom=.10,left=.07,right=.97,hspace=.62,wspace=.12)
    for r,(row,rgb) in enumerate(zip(rows,images)):
        for t in (0,1):
            axes[r,t].imshow(rgb[t],interpolation='nearest');axes[r,t].axis('off')
            axes[r,t].set_title((f'#{r+1} {row["episode_id"]} | t6' if t==0 else f't7 | W dx MSE={row["W_dx_mse"]:.6f}'),fontsize=11)
        lines=[]
        for obj in ('A','B'):
            lines.append(obj+' dx  '+ ' | '.join(f'{name}: {row[f"{obj}_{name}_dx"]:+.5f} {row[f"{obj}_{name}_direction"]}' for name in ('true','N','W')))
        lines.append('Cue reflection '+row['reflection_axes']+f' | yellow pixel dx={row["yellow_pixel_dx"]:+d}')
        axes[r,0].text(0,-.10,'\n'.join(lines),transform=axes[r,0].transAxes,ha='left',va='top',fontsize=10,family='monospace')
    fig.suptitle('V5.4 posthoc: top 3 W episode horizontal errors\nN=narrow / W=wide; exact archived t6/t7 RGB; R=right, L=left',fontsize=14)
    fig.text(.5,.025,'Outcome-selected examples, not representative or causal evidence.\nTrue and predicted dx use normalized coordinates; all six RGB hashes and labels verified.',ha='center',fontsize=10)
    fig.savefig(out/'top3_bad_cases.png',dpi=180);plt.close(fig)
    with (out/'selected_bad_cases.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# V5.4 已评估坏例：posthoc 选择','',
        '按已评估 512 个 episode 的 W concat A_dx/B_dx 平均平方误差降序，取前三个；并列时按 pair ID、variant 升序。'
        '选择单位是 episode，不要求不同 pair。没有重选模型、计算新 CI、生成新 split/ID 或启动 GPU。','',
        '| Rank | Episode | W dx MSE | N dx MSE | t6→t7 反射轴 |','|---|---|---:|---:|---|']
    for row in rows:lines.append(f'| {row["rank"]} | {row["episode_id"]} | {row["W_dx_mse"]:.8g} | {row["N_dx_mse"]:.8g} | {row["reflection_axes"]} |')
    lines+=['','![Top three evaluated bad cases](top3_bad_cases.png)','',
        'A 是上方红色目标，B 是下方蓝色目标。图中 true/N/W 为 t6→t7 的归一化水平位移及左右方向。'
        'RGB 没有美化或改动；contact sheet 仅按最近邻显示原像素，标注放在图像外。',
        '','三个 episode 的 6 张 cue RGB 数组 SHA256 均逐项等于原 encoded-input shard 收据；重建标签逐 bit 等于已评估 predictions targets。'
        f'原生成器 SHA256：`{GENERATOR_SHA}`；本地 Pillow `{PIL.__version__}`。完整源、输入、环境及逐 RGB 哈希见 `render_verification.json`。','',
        '反射标志来自原生成器的镜像位置公式：t6/t7 的未反射位置落入不同镜像区间即标记该轴。'
        '这不是根据图像猜测遮挡，也不把反射、黄色位移或误差共现解释成因果。'
        '这些样本按误差结果选出，不代表总体分布；总体结论以已冻结主端点及完整评估报告为准。','',
        '```sh','python3 execution/V54/posthoc_bad_cases.py \\',
        '  --scene execution/visual-v5_4-20261010/report-snapshot \\',
        '  --verification execution/visual-v5_4-20261010/report-snapshot-verification.json \\',
        '  --generator pmr/visual_memory/data.py --out execution/V54/bad_cases','```','']
    (out/'README.md').write_text('\n'.join(lines))
    verification_result['contact_sheet_sha256']=sha(out/'top3_bad_cases.png')
    dump(out/'render_verification.json',verification_result)
    for name,record in used.items():
        path=scene/name
        if path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:raise ValueError('Input changed during rendering')
    if sha(generator)!=GENERATOR_SHA:raise ValueError('Scientific generator changed during rendering')
    files=sorted(p for p in out.iterdir() if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    return dict(status='all_6_RGB_hashes_and_labels_verified',selected=[r['episode_id'] for r in rows],
                output=str(out),contact_sheet_sha256=sha(out/'top3_bad_cases.png'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','verification','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--generator',type=Path,default=ROOT/'pmr/visual_memory/data.py')
    a=parser.parse_args();print(json.dumps(build(a.scene,a.verification,a.generator,a.out),indent=2))


if __name__=='__main__':main()
