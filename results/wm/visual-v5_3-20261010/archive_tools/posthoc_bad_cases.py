#!/usr/bin/env python3
"""Posthoc diagnostics of already evaluated V5.3 Dev384..639 only; no fitting/GPU.

Nothing is read/generated at import. Run only after the complete bridge report is
released for analysis. Three examples follow explicit deterministic posthoc rules.
RGB must match archived actual-input hashes; a CPU-rendered cue archive is accepted
when local Pillow differs. This tool never calls SSH or changes scientific files.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np

GENERATOR_SHA = 'ce5082f3be9006dcceb46fbaffadc27349f977a77abcf8992eda3e80f8775368'
IDS = np.arange(384, 640)
AXES = ('A_dx', 'A_dy', 'B_dx', 'B_dy')
SELECTION_RULES = [
    'Largest B concat dx MSE (both variants and both targets); tie: smaller pair ID.',
    'Remaining pair with most B horizontal sign errors; tie: largest B dx MSE, then smaller ID.',
    'Remaining pair without a cue-interval reflection with largest B dx MSE; fallback: largest remaining B dx MSE.',
]
# Rounded coordinatewise outer envelope implied by the frozen make_block
# schema, not a fit to observed FIT or bridge coordinates. Both center and d
# are rounded to 2^-20 before the three positions C-d,C,C+d are rendered.
QUANTUM = 2**20
FIT_CENTER_LOW = np.round(np.array([.20,.20,.20,.65])*QUANTUM)/QUANTUM
FIT_CENTER_HIGH = np.round(np.array([.80,.35,.80,.80])*QUANTUM)/QUANTUM
FIT_MAX_D = np.round(.018*QUANTUM)/QUANTUM
FIT_POSITION_LOW = FIT_CENTER_LOW-FIT_MAX_D
FIT_POSITION_HIGH = FIT_CENTER_HIGH+FIT_MAX_D


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_identity(a):
    a = np.asarray(a)
    return dict(sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(), dtype=str(a.dtype), shape=list(a.shape))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_module(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def load_inputs(scene, generator, fit_generator):
    # Reuse the already written report verifier without reading anchor results.
    helper = load_module(Path(__file__).with_name('build_results.py'), 'v53_posthoc_report_verifier')
    evidence = helper.Evidence(scene)
    bridge = helper.load_role(evidence, 'bridge')
    r, arrays = bridge['report'], bridge['arrays']
    if r['status'] != 'evaluation_complete' or not r['singleframe_equivalence_passed']:
        raise ValueError('Completed and numerically valid bridge evaluation required')
    for key in helper.KEYS:
        actual=((arrays[key]-arrays['targets'])**2).mean((0,1))
        helper.equal(actual,r['scores']['readers'][key]['per_dimension_mse'],key+'/MSE')
    delta=(((arrays['S_concat']-arrays['targets'])**2)-((arrays['B_concat']-arrays['targets'])**2))[:,:,[4,6]].mean()
    helper.equal(delta,r['scores']['primary']['S_minus_B_dx_mse'],'registered bridge primary')
    if digest(generator) != GENERATOR_SHA:
        raise ValueError('Original generator source identity changed')
    receipts, records = {}, {}
    for i, pair in enumerate(IDS):
        receipt = evidence.read(f'bridge_cache/shards/{pair:06d}.json')
        x = receipt['inputs']
        if receipt['unit'] != int(pair) or receipt['role'] != 'bridge':
            raise ValueError('Shard ID/role mismatch')
        if (x['original_generator_sha256'] != GENERATOR_SHA or x['cue_times'] != [6, 7]
                or x['images_encoded'] != 4 or len(x['cue_rgb']) != 4):
            raise ValueError('Original cue identity differs')
        target = arrays['targets'][i].astype(np.float32)
        if x['targets'] != array_identity(target) or receipt['entry_hashes']['targets'] != x['targets']:
            raise ValueError('Shard/report target mismatch')
        if not np.array_equal(target.astype(np.float64), arrays['targets'][i]):
            raise ValueError('Report targets are not exactly the archived float32 values')
        for variant, meta in enumerate(x['metadata']):
            if (meta['episode_id'] != f'dev-{pair:06d}-{variant}' or meta['pair_id'] != f'dev-{pair:06d}'
                    or meta['split'] != 'dev' or meta['variant'] != variant):
                raise ValueError('Metadata mapping differs')
        receipts[int(pair)] = receipt
        records[str(pair)] = x
    cache = evidence.read('bridge_cache/receipt.json')
    if cache['source_identity']['scripts/v5_3_data.py'] != digest(fit_generator):
        raise ValueError('V5.3 generator source differs from frozen extraction source')
    if cache['input_records_sha256'] != json_sha(records):
        raise ValueError('Aggregate receipt does not bind the supplied shard input records')
    return arrays, receipts, evidence.inventory, r


def analytic_world(pair, variant):
    # Exact original seed and first RNG draws; no new IDs or RGB are generated.
    seed = int.from_bytes(hashlib.sha256(f'v5-rgb-pairs-1/dev/{pair}'.encode()).digest()[:4], 'little')
    rng = np.random.default_rng(seed)
    p = rng.uniform([.2, .23], [.8, .32], (2, 2)); p[1, 1] += .42
    v = rng.uniform(.008, .018, (2, 2))*rng.choice([-1, 1], (2, 2))
    v *= 1 if variant == 0 else -1
    # Consume the exact revised-velocity draw before colors, even though update
    # time is strictly after the two cue frames.
    rng.uniform(.008, .018, 2)*rng.choice([-1, 1], 2)*(1 if variant == 0 else -1)
    colors = [tuple(int(x) for x in c) for c in rng.integers(60, 240, (2, 3))]
    colors[0] = (235, colors[0][1]//2, 60); colors[1] = (60, colors[1][1]//2, 235)
    low = np.array([[.10, .15], [.10, .60]])
    high = np.array([[.90, .40], [.90, .85]])
    span = high-low
    raw = np.stack((p+6*v, p+7*v))
    pos = low+span-np.abs(np.mod(raw-low, 2*span)-span)
    tile = np.floor((raw-low)/span)
    crossing = tile[0] != tile[1]
    reflected_by7 = (raw[1] < low) | (raw[1] > high)
    boundary_ties = np.isclose(np.mod(raw-low, span), 0, rtol=0, atol=1e-14)
    y = np.concatenate((pos[1].reshape(-1), (pos[1]-pos[0]).reshape(-1))).astype(np.float32)
    yellow_x = np.array([int((.5+.4*np.sin(t*2.3+pair))*224) for t in (6, 7)])
    # Inclusive Pillow bounding rectangles: disjoint y extents imply no occlusion.
    boxes = np.stack((np.trunc(pos*224-224*.06), np.trunc(pos*224+224*.06)), axis=-2).astype(int)
    yellow_y = [int(224*.48), int(224*.52)]
    box_overlap = (boxes[:,:,0,1] <= yellow_y[1]) & (boxes[:,:,1,1] >= yellow_y[0])
    return dict(pos=pos, target=y, crossing=crossing, reflected_by7=reflected_by7,
                boundary_ties=boundary_ties, yellow_x=yellow_x, yellow_dx=float(np.diff(yellow_x)[0]/224),
                colors=colors, yellow_target_ybox_overlap=box_overlap)


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(mask, y, predictions):
    count = int(np.sum(mask))
    if count == 0:
        return dict(paired_worlds=0)
    yy = y[mask]
    result = dict(paired_worlds=count)
    for arm in ('S', 'B'):
        pp = predictions[arm][mask]
        mse = ((pp-yy)**2).mean((0, 1))
        zero = (yy**2).mean((0, 1))
        result[arm] = dict(dx_mse=float(mse[[0, 2]].mean()), motion_mse=float(mse.mean()),
                          per_axis_mse=mse.tolist(), zero_motion_skill=(1-mse/zero).tolist(),
                          horizontal_sign_error_rate=float((np.sign(pp[:,:,[0,2]]) != np.sign(yy[:,:,[0,2]])).mean()))
    return result


def diagnostics(arrays, receipts):
    y = arrays['targets'][:,:,4:].astype(np.float64)
    pp = {arm:arrays[arm+'_concat'][:,:,4:].astype(np.float64) for arm in ('S', 'B')}
    worlds, rows = {}, []
    crossing, reflected = np.zeros(256, bool), np.zeros(256, bool)
    horizontal_crossing, horizontal_reflected = np.zeros(256, bool), np.zeros(256, bool)
    nuisance = np.empty(256)
    ties = overlap = 0
    object_rows = []
    outside = np.zeros((256,2,4),bool)
    for i, pair_np in enumerate(IDS):
        pair = int(pair_np)
        variants = [analytic_world(pair, variant) for variant in (0, 1)]
        worlds[pair] = variants
        for variant, world in enumerate(variants):
            if not np.array_equal(world['target'], arrays['targets'][i, variant]):
                raise ValueError(f'Analytic original labels do not match evaluated target {pair}/{variant}')
            ties += int(world['boundary_ties'].sum())
            overlap += int(world['yellow_target_ybox_overlap'].sum())
            coord=world['pos'].reshape(2,4)
            outside[i,variant]=((coord<FIT_POSITION_LOW)|(coord>FIT_POSITION_HIGH)).any(0)
            for obj in (0, 1):
                true_dx = y[i, variant, obj*2]
                relation = 'same_direction' if true_dx*world['yellow_dx'] > 0 else ('opposite_direction' if true_dx*world['yellow_dx'] < 0 else 'zero_product')
                object_rows.append(dict(pair_id=pair, variant=variant, target='AB'[obj],
                    target_dx=float(true_dx), yellow_dx=world['yellow_dx'], relative_dx=float(true_dx-world['yellow_dx']),
                    relation=relation, cue_reflection=bool(world['crossing'][obj].any()),
                    cue_horizontal_reflection=bool(world['crossing'][obj,0]),
                    cue_vertical_reflection=bool(world['crossing'][obj,1]),
                    reflected_by_t7=bool(world['reflected_by7'][obj].any()),
                    S_dx_error=float(pp['S'][i,variant,obj*2]-true_dx),
                    B_dx_error=float(pp['B'][i,variant,obj*2]-true_dx)))
        crossing[i] = any(v['crossing'].any() for v in variants)
        reflected[i] = any(v['reflected_by7'].any() for v in variants)
        horizontal_crossing[i] = any(v['crossing'][:,0].any() for v in variants)
        horizontal_reflected[i] = any(v['reflected_by7'][:,0].any() for v in variants)
        nuisance[i] = variants[0]['yellow_dx']
        row = dict(pair_id=pair, cue_interval_reflection=bool(crossing[i]), reflected_by_t7=bool(reflected[i]),
                   cue_horizontal_reflection=bool(horizontal_crossing[i]), horizontal_reflected_by_t7=bool(horizontal_reflected[i]),
                   any_cue_outside_FIT_coordinate_envelope=bool(outside[i].any()),
                   yellow_dx=float(nuisance[i]), family=receipts[pair]['inputs']['metadata'][0]['family'])
        for arm in ('S', 'B'):
            err = (pp[arm][i]-y[i])**2
            row[arm+'_dx_mse'] = float(err[:,[0,2]].mean())
            row[arm+'_motion_mse'] = float(err.mean())
            row[arm+'_horizontal_sign_errors'] = int((np.sign(pp[arm][i][:,[0,2]]) != np.sign(y[i][:,[0,2]])).sum())
        rows.append(row)
    summary = dict(scope='Posthoc descriptive analysis of already evaluated Dev384..639; no new confirmation inference.',
        counts=dict(paired_worlds=256, episodes=512, target_observations=1024),
        reflection_definitions=dict(cue_interval='Any object/axis/variant crosses a mirror tile boundary between t6 and t7.',
            reflected_by_t7='Any unreflected t7 coordinate lies outside its lane; initial positions lie inside and pre-update velocity is constant.',
            boundary_ties_at_1e_14=ties),
        visibility=dict(cue_shutters='Both targets open at t6 and t7 by the fixed original generator.',
            yellow_target_y_bounding_box_intersections=overlap,
            implication='Zero intersections proves no yellow-on-target occlusion in these cue images; it does not imply absent encoder distraction.'),
        groups={label:summarize(mask, y, pp) for label, mask in {
            'all':np.ones(256, bool), 'cue_reflection':crossing, 'no_cue_reflection':~crossing,
            'reflected_by_t7':reflected, 'no_reflection_by_t7':~reflected,
            'cue_horizontal_reflection':horizontal_crossing, 'no_cue_horizontal_reflection':~horizontal_crossing,
            'horizontal_reflected_by_t7':horizontal_reflected, 'no_horizontal_reflection_by_t7':~horizontal_reflected,
            'outside_FIT_coordinate_envelope':outside.any((1,2)),
            'inside_FIT_coordinate_envelope':~outside.any((1,2))}.items()},
        nuisance_pair_common_residual={})
    summary['spatial_support_proxy'] = dict(
        definition='Outside if any cue t6/t7, variant, target coordinate exceeds the coordinatewise rounded schema outer envelope. This is not a full joint-support or empirical FIT coverage test.',
        coordinate_order=['A_x','A_y','B_x','B_y'],
        nominal_unrounded_low=[.182,.182,.182,.632], nominal_unrounded_high=[.818,.368,.818,.818],
        actual_quantized_low=FIT_POSITION_LOW.tolist(),actual_quantized_high=FIT_POSITION_HIGH.tolist(),
        rounding_quantum=1/QUANTUM,comparison_tolerance=0,source='scripts/v5_3_data.py: make_block center and displacement rounding',
        posthoc=True,confidence_intervals=None,coordinate_groups={})
    for j,name in enumerate(AXES):
        result={}
        for label,mask in [('outside',outside[:,:,j]),('inside',~outside[:,:,j])]:
            result[label]=dict(target_coordinate_observations=int(mask.sum()),
                unique_worlds=int(mask.any(1).sum()),
                S_mse=float(((pp['S'][:,:,j]-y[:,:,j])[mask]**2).mean()) if mask.any() else None,
                B_mse=float(((pp['B'][:,:,j]-y[:,:,j])[mask]**2).mean()) if mask.any() else None)
        summary['spatial_support_proxy']['coordinate_groups'][name]=result
    for arm in ('S', 'B'):
        residual = (pp[arm]-y).mean(1)[:,[0,2]]
        summary['nuisance_pair_common_residual'][arm] = {
            target:dict(pearson_r=correlation(nuisance,residual[:,obj]),
                linear_slope=float(np.dot(nuisance-nuisance.mean(), residual[:,obj]-residual[:,obj].mean())/np.sum((nuisance-nuisance.mean())**2)))
            for obj,target in enumerate(('A_dx','B_dx'))}
    summary['relative_motion_groups'] = {}
    for relation in ('same_direction', 'opposite_direction', 'zero_product'):
        rr = [r for r in object_rows if r['relation']==relation]
        summary['relative_motion_groups'][relation] = dict(target_observations=len(rr),
            unique_worlds=len({r['pair_id'] for r in rr}),
            S_dx_mse=float(np.mean([r['S_dx_error']**2 for r in rr])) if rr else None,
            B_dx_mse=float(np.mean([r['B_dx_error']**2 for r in rr])) if rr else None)
    summary['relative_motion_group_caveat'] = 'Correlated target/variant observations; no independent-observation confidence interval or causal group effect is claimed.'
    selected = [sorted(rows,key=lambda r:(-r['B_dx_mse'],r['pair_id']))[0]]
    remaining = [r for r in rows if r not in selected]
    selected.append(sorted(remaining,key=lambda r:(-r['B_horizontal_sign_errors'],-r['B_dx_mse'],r['pair_id']))[0])
    remaining = [r for r in rows if r not in selected]
    eligible = [r for r in remaining if not r['cue_interval_reflection']] or remaining
    selected.append(sorted(eligible,key=lambda r:(-r['B_dx_mse'],r['pair_id']))[0])
    selection = []
    for n, row in enumerate(selected):
        i = int(row['pair_id']-384)
        dx = ((pp['B'][i]-y[i])**2)[:,[0,2]].mean(1)
        selection.append(dict(rule=SELECTION_RULES[n], **row, shown_variant=int(np.argmax(dx))))
    summary['selection'] = selection
    return summary, rows, object_rows, worlds


def render_cue(world, time_index):
    # This is the exact cue-only RGB recipe from the pinned original generator.
    # Actual-input hash checking below is mandatory, including Pillow rasterization.
    from PIL import Image, ImageDraw
    size = 224
    im = Image.new('RGB',(size,size),(30,35,42)); draw = ImageDraw.Draw(im)
    for xx in range(0,size,max(1,size//16)):
        draw.line((xx,0,xx,size),fill=(42,47,54))
    radius = size*.06
    for obj in (0,1):
        x,y = world['pos'][time_index,obj]*size
        box = (int(x-radius),int(y-radius),int(x+radius),int(y+radius))
        draw.ellipse(box,fill=world['colors'][obj],outline=(250,250,240),width=max(1,size//100))
        draw.line((x-radius/2,y,x+radius/2,y),fill=(255,255,255),width=max(1,size//60))
        if obj==1:
            draw.line((x,y-radius/2,x,y+radius/2),fill=(255,255,255),width=max(1,size//60))
    x = world['yellow_x'][time_index]
    draw.rectangle((x-5,size*.48,x+5,size*.52),fill=(240,220,50))
    return np.asarray(im).copy()


def verified_cues(summary, worlds, receipts, out, cue_archive):
    import PIL
    external = None
    if cue_archive:
        with np.load(cue_archive,allow_pickle=False) as z:
            external = {key:z[key] for key in z.files}
    images, checks = {}, []
    for row in summary['selection']:
        pair, variant = row['pair_id'],row['shown_variant']
        key = f'pair_{pair:06d}_variant_{variant}'
        rgb = external[key] if external is not None else np.stack([render_cue(worlds[pair][variant],t) for t in (0,1)])
        if rgb.shape != (2,224,224,3) or rgb.dtype != np.uint8:
            raise ValueError('Invalid cue RGB dimensions/dtype')
        images[pair] = rgb
        for t in (0,1):
            expected = receipts[pair]['inputs']['cue_rgb'][variant*2+t]
            actual = array_identity(rgb[t])
            checks.append(dict(pair_id=pair,variant=variant,time=6+t,expected=expected,actual=actual,exact=expected==actual))
    verification = dict(status='verified' if all(c['exact'] for c in checks) else 'RGB_HASH_MISMATCH',
        source='external cue archive' if external is not None else 'local pinned cue-only re-render',
        pillow=PIL.__version__,numpy=np.__version__,checks=checks,
        external_archive=None if cue_archive is None else dict(path=str(cue_archive),sha256=digest(cue_archive)))
    dump(out/'render_verification.json',verification)
    if verification['status'] != 'verified':
        raise ValueError('RGB differs from actual encoded inputs; retained evidence. CPU-render selected cues in the original environment and pass --cue-archive. No approximate example figure was written.')
    return images


def plots(out, summary, rows, arrays, worlds, images):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(11,4.4))
    for reflect,color,label in ((False,'#008b8b','No cue-interval reflection'),(True,'#d07120','Cue-interval reflection')):
        group=[r for r in rows if r['cue_interval_reflection']==reflect]
        axes[0].scatter([r['S_dx_mse'] for r in group],[r['B_dx_mse'] for r in group],s=20,alpha=.65,c=color,label=label)
    upper=max(max(r['S_dx_mse'],r['B_dx_mse']) for r in rows)*1.03
    axes[0].plot([0,upper],[0,upper],c='gray',ls='--',lw=1)
    axes[0].set(xlabel='S dx MSE',ylabel='B dx MSE',title='All 256 paired worlds; one point per world');axes[0].legend(fontsize=8)
    eta=np.array([r['yellow_dx'] for r in rows]); y=arrays['targets']
    for obj,color in ((0,'#c33c32'),(1,'#3169b0')):
        error=(arrays['B_concat']-y).mean(1)[:,4+obj*2]
        axes[1].scatter(eta,error,s=14,alpha=.55,color=color,label='AB'[obj]+' dx residual')
    axes[1].axhline(0,c='gray',lw=.8);axes[1].axvline(0,c='gray',lw=.8)
    axes[1].set(xlabel='Yellow pixel dx / 224',ylabel='B pair-common dx residual',title='Residual association (posthoc, not intervention)');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'paired_error_diagnostics.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    for ax,arm in zip(axes,('S','B')):
        for j,(label,color) in enumerate((('inside','#008b8b'),('outside','#d07120'))):
            data=summary['spatial_support_proxy']['coordinate_groups']
            values=[data[name][label][arm+'_mse'] for name in AXES]
            ax.bar(np.arange(4)+(j-.5)*.32,values,width=.32,color=color,label=label)
        ax.set(xticks=np.arange(4),xticklabels=AXES,title=arm+' motion MSE by coordinate envelope',ylabel='Per-coordinate MSE')
        ax.legend()
    fig.suptitle('Posthoc coordinate envelope proxy; correlated observations; no confidence intervals',fontsize=10)
    fig.tight_layout();fig.savefig(out/'spatial_support_proxy.png');plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,11),gridspec_kw={'width_ratios':[1,1,1.4]})
    styles=[('GT','#222222',1),('S','#d07120',.8),('B','#008b8b',.8)]
    for rowidx,row in enumerate(summary['selection']):
        pair,variant=row['pair_id'],row['shown_variant'];i=pair-384;world=worlds[pair][variant]
        for t in (0,1):
            axes[rowidx,t].imshow(images[pair][t]);axes[rowidx,t].axis('off')
            axes[rowidx,t].set_title(f'Dev {pair}, variant {variant}, t={6+t}')
        ax=axes[rowidx,2]
        ax.set(xlim=(0,1),ylim=(1,0),aspect='equal',xlabel='Normalized x',ylabel='Normalized y')
        ax.set_title(f'B dx MSE {row["B_dx_mse"]:.5g}; sign errors {row["B_horizontal_sign_errors"]}/4\nAll arrows 4x; origins are true t6 centers',fontsize=9)
        for obj in (0,1):
            origin=world['pos'][0,obj]
            ax.scatter(*origin,c=['#c33c32','#3169b0'][obj],s=32,zorder=5)
            ax.text(origin[0],origin[1]-.035,'AB'[obj])
            for model,color,alpha in styles:
                value=(arrays['targets'] if model=='GT' else arrays[model+'_concat'])[i,variant,4+obj*2:6+obj*2]
                ax.quiver(*origin,*(4*value),angles='xy',scale_units='xy',scale=1,color=color,alpha=alpha,width=.009,
                          label=model if obj==0 else None,zorder={'GT':4,'S':2,'B':3}[model])
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3);ax.grid(alpha=.15)
    fig.suptitle('Three deterministic posthoc examples; exact hashes match actual encoded RGB inputs',fontsize=11)
    fig.tight_layout();fig.savefig(out/'three_verified_examples.png');plt.close(fig)


def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def readme(summary, out):
    g=summary['groups'];v=summary['visibility'];sels=summary['selection']
    lines=['# V5.3 已评估 bridge 的后验 bad-case 诊断','',
        '只分析本轮已经评估的 Dev384–639（256 个配对 world、512 个 episode）。未增加新样本、未训练或选择读出器、未使用 Test 或 GPU。这里的分组、相关性和三例选择均是后验探索；不构成新确认实验或 P0 许可。','',
        '三例按公开的固定后验规则选择：① B 的 pair dx MSE 最大；② 剩余 pair 中水平符号错误数最多；③ 剩余且 t6→t7 无反射的 pair 中 B dx MSE 最大。并列依次按误差及较小 ID；每个 pair 展示 B dx MSE 最大的 variant（并列选 0）。完整排序可由 `paired_worlds.csv` 复核。','',
        '| 分组 | world 数 | S dx MSE | B dx MSE | B 水平方向错误率 |','|---|---:|---:|---:|---:|']
    for label,name in [('all','全部'),('cue_reflection','t6→t7 发生反射（任意轴）'),('no_cue_reflection','t6→t7 无反射（任意轴）'),('cue_horizontal_reflection','t6→t7 发生水平反射'),('no_cue_horizontal_reflection','t6→t7 无水平反射'),('reflected_by_t7','截至 t7 曾反射（任意轴）'),('no_reflection_by_t7','截至 t7 未反射（任意轴）'),('outside_FIT_coordinate_envelope','至少一个 cue 坐标超出 FIT 外包络'),('inside_FIT_coordinate_envelope','全部 cue 坐标处于 FIT 外包络')]:
        x=g[label]
        lines.append(f'| {name} | {x["paired_worlds"]} | '+(f'{x["S"]["dx_mse"]:.8g} | {x["B"]["dx_mse"]:.8g} | {x["B"]["horizontal_sign_error_rate"]:.2%} |' if x['paired_worlds'] else '— | — | — |'))
    lines += ['', '反射由原始固定 seed 的初始位置、速度及镜面公式解析判定，并与归档的全部目标标签逐值一致；没有从预测方向推断反射。“任意轴”包含垂直反射，因此同时列出与 dx 直接对应的水平反射组。两个时间定义不同，且这些分组不互相独立。分组样本量与误差为描述统计，未做多重比较校正，也未把目标或 variant 当作独立 world。', '',
        '位置外包络按冻结生成器 C±d 推导：名义 x∈[.182,.818]，A_y∈[.182,.368]，B_y∈[.632,.818]；实际比较使用中心和位移按 2⁻²⁰ 舍入后的精确边界（见 `summary.json`），不从本轮得分选择阈值。这只是逐坐标外包络代理，处于其中不代表完整联合支持或经验 FIT 覆盖；分组没有 CI 或因果解释。完整 motion MSE 和逐坐标 in/out MSE 也保存在 `summary.json`。', '',
        f'两提示帧的目标快门均打开。黄色矩形与目标包围盒的 y 区间交集计数为 {v["yellow_target_y_bounding_box_intersections"]}，因而这些提示帧不存在黄色遮挡目标的情况；残差关联不能解释为遮挡，也不能单独证明编码器内的因果机制。', '',
        '同向/反向相对运动见 `object_observations.csv` 与 `summary.json`。归一化 motion skill 使用零运动基线（文件中标为 `zero_motion_skill`），不与正式报告的冻结 FIT 均值基线混写。全部相关性是在同一已看 bridge 上的后验残差关联；修复的确认结论仍应以正式 bridge 主比较及 anchor/负控共同评估。', '',
        '| pair-common dx 残差与黄色 pixel dx 的相关系数 | S | B |','|---|---:|---:|']
    for target in ('A_dx','B_dx'):
        corr=summary['nuisance_pair_common_residual']
        lines.append(f'| {target} | {corr["S"][target]["pearson_r"]:.5f} | {corr["B"][target]["pearson_r"]:.5f} |')
    lines += ['', '配对均值先减各自真实 motion 再平均两个 variant；这张表表明原有干扰残差关联减弱，不能证明所有残差均由黄色运动造成。B 仍有水平符号错误，故不能把改善写为完全解决。','',
        '| 坐标 | 外/内 episode 数 | S 外/内 MSE | B 外/内 MSE |','|---|---:|---:|---:|']
    for name,group in summary['spatial_support_proxy']['coordinate_groups'].items():
        o,i=group['outside'],group['inside']
        lines.append(f'| {name} | {o["target_coordinate_observations"]}/{i["target_coordinate_observations"]} | {o["S_mse"]:.6g}/{i["S_mse"]:.6g} | {o["B_mse"]:.6g}/{i["B_mse"]:.6g} |')
    lines += ['', '这些“外/内”组仅按相应坐标的两提示位置划分，每个坐标合计 512 个 episode。不同坐标的组成员不同，同一 world 的两个 variant 有相关性。外包络组误差更高是进一步研究空间覆盖的线索，不能据此断言扩大 FIT 支持就必然修复，亦不能将全部残差归于反射或特征细度。','',
        '| 实例 | pair / variant | B pair dx MSE | 水平符号错数（两 variant × 两目标） |','|---|---:|---:|---:|']
    for idx,x in enumerate(sels):
        lines.append(f'| {idx+1} | {x["pair_id"]} / {x["shown_variant"]} | {x["B_dx_mse"]:.8g} | {x["B_horizontal_sign_errors"]}/4 |')
    lines += ['', '![配对误差与残差关联](paired_error_diagnostics.png)', '', '![位置外包络分组](spatial_support_proxy.png)', '', '![三例输入与预测](three_verified_examples.png)', '',
        '例图的两帧 RGB 已逐个对照实际编码输入的 SHA256、shape、dtype；详见 `render_verification.json`。箭头均为 4 倍位移、起点为真实 t6 位置，便于比较方向与幅度，不能把箭头终点当作实际 t7 预测位置。','',
        '所有旧失败 gate 保持不变，后验诊断不授权 P0。原 scene、冻结模型及 cache 均只读。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--generator',type=Path,default=Path(__file__).resolve().parents[2]/'pmr/visual_memory/data.py')
    parser.add_argument('--fit-generator',type=Path,default=Path(__file__).resolve().parents[2]/'scripts/v5_3_data.py')
    parser.add_argument('--cue-archive',type=Path,help='Optional CPU-rendered NPZ; pair_000384_variant_0-style keys, exact uint8 actual-input RGB')
    args=parser.parse_args();scene=args.scene.resolve();out=args.out.resolve()
    if out==scene or scene in out.parents:
        raise ValueError('Output must be outside the immutable input scene')
    if out.exists():
        raise FileExistsError('Preserve prior output; choose a fresh output directory')
    arrays,receipts,inventory,report=load_inputs(scene,args.generator,args.fit_generator)
    out.mkdir(parents=True)
    summary,rows,object_rows,worlds=diagnostics(arrays,receipts)
    summary['primary_report'] = report['scores']['primary']
    summary['singleframe_equivalence_passed']=report['singleframe_equivalence_passed']
    dump(out/'summary.json',summary);write_csv(out/'paired_worlds.csv',rows);write_csv(out/'object_observations.csv',object_rows)
    dump(out/'input_inventory.json',dict(scene=str(scene),files=inventory,
        generator_sha256=digest(args.generator),fit_generator_sha256=digest(args.fit_generator),tool_sha256=digest(__file__),
        verifier_sha256=digest(Path(__file__).with_name('build_results.py')),selection_rules=SELECTION_RULES))
    images=verified_cues(summary,worlds,receipts,out,args.cue_archive)
    plots(out,summary,rows,arrays,worlds,images);readme(summary,out)
    (out/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps(dict(status='posthoc_complete',out=str(out),selected=summary['selection']),indent=2))


if __name__=='__main__':
    main()
