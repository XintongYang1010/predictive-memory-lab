#!/usr/bin/env python3
"""Read-only V5.3 posthoc tables/plots. Never generate data or fit/select models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ARMS = ('S', 'B')
INPUTS = ('concat', 't6', 't7')
KEYS = tuple(a+'_'+n for a in ARMS for n in INPUTS)
PATHS = ('aa', 'bb', 'ab', 'ba')
DIMS = ('A_dx', 'A_dy', 'B_dx', 'B_dy')
COLORS = ('#657786', '#008b8b')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def equal(actual, expected, name):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise ValueError('Prediction/report mismatch: '+name)


def check_interval(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
        raise ValueError('Invalid confidence interval: '+name)


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inventory = {}

    def record(self, relative):
        path = self.root/relative
        self.inventory[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return path

    def read(self, relative, optional=False):
        if optional and not (self.root/relative).exists():
            return None
        return json.loads(self.record(relative).read_text())


def load_role(evidence, role):
    report = evidence.read(role+'_report/report.json')
    if (report.get('schema') != 'v5.3-reader-evaluation-1' or report.get('role') != role or
            report.get('groups') != 256 or report.get('examples') != (512 if role == 'bridge' else 4096) or
            report.get('test_opened') is not False or report.get('p0_authorized') is not False or
            report.get('original_v5_gate') != 'failed_unchanged' or report.get('v5_1_bridge') != 'failed_unchanged'):
        raise ValueError('Role/schema/count/gate declarations differ: '+role)
    path = evidence.record(role+'_report/predictions.npz')
    if sha(path) != report['predictions_sha256']:
        raise ValueError('Prediction SHA256 mismatch: '+role)
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    shape = (256, 2, 8) if role == 'bridge' else (256, 16, 8)
    id_key = 'pair_ids' if role == 'bridge' else 'block_ids'
    if (str(arrays['role'].item()) != role or
            not np.array_equal(arrays[id_key], np.arange(384, 640) if role == 'bridge' else np.arange(256))):
        raise ValueError('Role/ID mapping changed: '+role)
    for key in (*KEYS, 'targets'):
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError('Prediction/target shape or finite check failed: '+role+'/'+key)
    copied = evidence.read(role+'_report/cache_receipt.json')
    # The reader serializes its report copy with sorted keys. Compare original
    # receipt bytes when available, plus semantic equality of the report copy.
    original = evidence.read(role+'_cache/receipt.json', optional=True)
    if original is not None:
        if sha(evidence.root/(role+'_cache/receipt.json')) != report['cache_receipt_sha256'] or original != copied:
            raise ValueError('Original receipt SHA or serialized-copy identity mismatch: '+role)
        receipt_verification = 'original receipt byte SHA plus report-copy JSON equality'
    elif sha(evidence.root/(role+'_report/cache_receipt.json')) == report['cache_receipt_sha256']:
        receipt_verification = 'report copy is byte-identical to original receipt hash'
    else:
        raise ValueError('Original '+role+'_cache/receipt.json required: report copy was reserialized')
    if (copied.get('cache_sha256') != report['cache_sha256'] or
            copied.get('frozen_reader_sha256') != report['frozen_reader_sha256']):
        raise ValueError('Report/cache receipt references disagree')
    if report['singleframe_equivalence_tolerance'] != 1e-7:
        raise ValueError('Unexpected numerical tolerance')
    parity = {name:float(np.max(np.abs(arrays['S_'+name]-arrays['B_'+name]))) for name in ('t6','t7')}
    if role == 'anchor':
        if arrays['negative_targets'].shape != (256,2,8) or np.any(arrays['negative_targets'][:,:,4:] != 0):
            raise ValueError('Negative control must have zero target motion')
        for key in KEYS:
            if arrays['negative_'+key].shape != (256,2,8) or not np.isfinite(arrays['negative_'+key]).all():
                raise ValueError('Invalid negative-control prediction')
        parity.update({'negative_'+name:float(np.max(np.abs(arrays['negative_S_'+name]-arrays['negative_B_'+name])))
                       for name in ('t6','t7')})
    for key, value in parity.items():
        equal(value, report['singleframe_S_B_max_abs'][key], role+'/parity/'+key)
    passed = all(v <= 1e-7 for v in parity.values())
    if report['singleframe_equivalence_passed'] != passed or (report['status']=='evaluation_complete') != passed:
        raise ValueError('Numerical status inconsistent with actual predictions')
    boot = report['scores']['bootstrap']
    if boot.get('seed') != 7811 or boot.get('replicates') != 10000:
        raise ValueError('Frozen bootstrap contract changed')
    return dict(report=report, arrays=arrays, receipt_verification=receipt_verification)


def verify_metrics(anchor, bridge):
    for role, data in (('anchor',anchor), ('bridge',bridge)):
        r, a = data['report'], data['arrays']
        y = a['targets'].astype(np.float64)
        for key in KEYS:
            p = a[key]
            actual = ((p-y)**2).reshape(-1,8).mean(0)
            row = r['scores']['readers'][key]
            score = row['all_views'] if role == 'anchor' else row
            equal(actual, score['per_dimension_mse'], role+'/'+key+'/MSE')
            skills = np.asarray(score['per_dimension_skill'])
            equal(skills[4:].mean(), score['motion_skill'], role+'/'+key+'/motion')
            equal(skills[:4].mean(), score['position_skill'], role+'/'+key+'/position')
            if role == 'anchor':
                yy, pp = y.reshape(256,4,4,8), p.reshape(256,4,4,8)
                for i, path in enumerate(PATHS):
                    contrast = np.stack(((pp[:,0,i,4:]-pp[:,1,i,4:])/2,
                                         (pp[:,2,i,4:]-pp[:,3,i,4:])/2), axis=1)
                    d = yy[:,0,i,4:]
                    skill = 1-((contrast-d[:,None])**2).mean(0)/(d**2).mean(0)
                    equal(skill, row['paths'][path]['end_start_per_coordinate_skill'], key+'/'+path+'/contrast')
                    equal(((pp[:,:,i]-yy[:,:,i])**2).mean((0,1)),
                          row['paths'][path]['absolute_outputs']['per_dimension_mse'], key+'/'+path+'/MSE')
                rms = np.sqrt((a['negative_'+key][:,:,4:]**2).mean((0,1)))
                equal(rms, row['negative_control_motion_rms'], key+'/negative RMS')
                equal(rms/np.asarray(r['scores']['fit_motion_rms']),
                      row['negative_control_motion_rms_over_FIT_motion_rms'], key+'/normalized negative RMS')
    br, ba = bridge['report']['scores'], bridge['arrays']
    for name in INPUTS:
        err = {arm:((ba[arm+'_'+name]-ba['targets'])**2).mean(1) for arm in ARMS}
        delta = (err['S'][:,[4,6]]-err['B'][:,[4,6]]).mean()
        equal(delta, br['comparisons'][name]['S_minus_B_dx_mse'], name+'/primary MSE difference')
        check_interval(br['comparisons'][name]['dx_mse_paired_95ci'], name+'/primary CI')
    if br['primary'] != br['comparisons']['concat']:
        raise ValueError('Primary must be the registered concat contrast')
    if br['primary_improvement_interval_above_zero'] != (br['primary']['dx_mse_paired_95ci'][0] > 0):
        raise ValueError('Primary interval flag inconsistent')


def plots(out, anchor, bridge):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    ar, br = anchor['report']['scores'], bridge['report']['scores']
    fig, axes = plt.subplots(1,2,figsize=(11,4.5))
    for i, arm in enumerate(ARMS):
        row = br['readers'][arm+'_concat']
        axes[0].bar(np.arange(4)+(i-.5)*.32, row['per_dimension_skill'][4:], width=.32, color=COLORS[i], label=arm)
        axes[1].bar(np.arange(4)+(i-.5)*.32, row['motion_absolute_amplitude_ratio'], width=.32, color=COLORS[i], label=arm)
    axes[0].axhline(0,c='black',lw=.8); axes[1].axhline(1,c='black',lw=.8,ls='--')
    for ax in axes:
        ax.set_xticks(np.arange(4), DIMS); ax.legend()
    axes[0].set(title='Fresh original bridge: motion skill',ylabel='1 - MSE / frozen FIT-mean baseline')
    axes[1].set(title='Motion amplitude diagnostic',ylabel='Mean |prediction| / mean |target|')
    fig.suptitle('Fixed concat readers; 256 paired worlds; S static supervision, B balanced supervision',fontsize=10)
    fig.tight_layout(); fig.savefig(out/'bridge_motion_skills.png'); plt.close(fig)

    fig, axes = plt.subplots(2,1,figsize=(11,9),gridspec_kw={'height_ratios':[1,2.1]})
    for i, arm in enumerate(ARMS):
        row = ar['readers'][arm+'_concat']
        vals = [row['paths'][path]['absolute_outputs']['motion_skill'] for path in PATHS]
        axes[0].bar(np.arange(4)+(i-.5)*.32,vals,width=.32,label=arm,color=COLORS[i])
    axes[0].set(xticks=np.arange(4),xticklabels=['aa: static','bb: static','ab: moving','ba: moving'],
                ylabel='Mean absolute motion skill',title='Fresh anchor views by nuisance path')
    axes[0].axhline(0,c='black',lw=.8);axes[0].legend()
    heat = np.array([ar['readers'][arm+'_concat']['paths'][path]['end_start_per_coordinate_skill']
                     for arm in ARMS for path in PATHS]).reshape(8,8)
    bound=max(1.,float(np.abs(heat).max()))
    im=axes[1].imshow(heat,cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
    axes[1].set(yticks=np.arange(8),yticklabels=[a+'/'+p for a in ARMS for p in PATHS],
                xticks=np.arange(8),xticklabels=['end '+d for d in DIMS]+['start '+d for d in DIMS],
                title='All coordinate contrast skills; nonzero target d, shared image anchors')
    axes[1].tick_params(axis='x',rotation=30)
    for row in range(8):
        for col in range(8):
            axes[1].text(col,row,f'{heat[row,col]:.2f}',ha='center',va='center',fontsize=8,
                         color='white' if abs(heat[row,col])>bound*.6 else 'black')
    fig.colorbar(im,ax=axes[1],label='Contrast skill',shrink=.8)
    fig.tight_layout(); fig.savefig(out/'anchor_path_diagnostics.png'); plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(11,4.5))
    fit=np.asarray(ar['fit_motion_rms']); overall=float(np.sqrt(np.mean(fit**2)))
    for i,arm in enumerate(ARMS):
        row=ar['readers'][arm+'_concat']
        axes[0].bar(np.arange(4)+(i-.5)*.32,row['negative_control_motion_rms_over_FIT_motion_rms'],
                    width=.32,label=arm,color=COLORS[i])
        direction=np.sqrt(np.mean(np.asarray(row['negative_control_motion_rms_by_direction'])**2,axis=1))/overall
        axes[1].bar(np.arange(2)+(i-.5)*.32,direction,width=.32,label=arm,color=COLORS[i])
    for ax in axes:
        ax.set_ylabel('Predicted motion RMS / nonzero FIT motion RMS');ax.axhline(1,c='black',lw=.8,ls='--');ax.legend()
    axes[0].set(xticks=np.arange(4),xticklabels=DIMS,title='Static-target / moving-yellow negative controls')
    axes[1].set(xticks=[0,1],xticklabels=['a → b','b → a'],title='Both directions retained before squaring')
    fig.suptitle('True target motion = 0; lower nuisance response is useful only with preserved nonzero-target performance',fontsize=10)
    fig.tight_layout();fig.savefig(out/'negative_control_rms.png');plt.close(fig)


def fmt(value):
    return '未提供' if value is None else f'{float(value):.7g}'


def ci(value):
    return '['+', '.join(fmt(v) for v in value)+']'


def write_readme(out, anchor, bridge, meta):
    ar,br=anchor['report']['scores'],bridge['report']['scores']
    primary=br['primary']; lower,upper=primary['dx_mse_paired_95ci']
    b=br['readers']['B_concat']; s=br['readers']['S_concat']
    valid=all(x['report']['singleframe_equivalence_passed'] for x in (anchor,bridge))
    if not valid:
        conclusion='数值等价检查失败：结果仅保留排错，不能解释成科学改善。'
    elif lower>0:
        conclusion='B 在新原分布 bridge 上降低了水平位移误差，主配对区间整体为正。'
        if b['motion_skill']<=0:
            conclusion+='但 B 的总体 motion skill 仍非正，只能称相对误差改善，不能称读出器修复。'
        elif min(b['per_dimension_skill'][4:])<=0:
            conclusion+='B 的总体 motion skill 为正，但仍有运动坐标未超过基线，不能称全部目标运动信息恢复。'
        else:
            conclusion+='B 的四个运动坐标点估计均超过基线；是否保留目标运动还须结合逐路径 anchor 和静止目标负控，不能据单端点宣布全面修复。'
    elif upper<0:
        conclusion='B 在主端点上比 S 更差：S−B 的配对区间整体为负，本轮不支持平衡监督带来相对改善。'
    else:
        conclusion='S−B 的主配对区间包含零；本轮不能明确确认平衡监督降低 fresh bridge 水平位移误差。'
    lines=['# V5.3：匹配 S/B 读出器修复诊断','',conclusion,'',
        'S 使用静止干扰监督，B 使用静止与双向移动干扰的平衡监督。两组共享同一 FIT 图像、'
        '核标准化、quadratic 核和 alpha；每个世界总损失权重相同，两组均拟合不受惩罚的截距。'
        'S 也使用共享的无标签 FIT 标准化，不能把它描述为完全没见过移动视图特征分布。','',
        f'- 唯一主端点：fresh bridge Dev 384–639 的 pair 内两 variants × A_dx/B_dx 原始 MSE；'
        f'**S−B = {fmt(primary["S_minus_B_dx_mse"])}**，95% CI **{ci(primary["dx_mse_paired_95ci"])}**。',
        f'- 绝对表现：S motion skill **{fmt(s["motion_skill"])}**，B **{fmt(b["motion_skill"])}**；'
        f'B 最低单坐标 motion skill **{fmt(min(b["per_dimension_skill"][4:]))}**。',
        '- Bootstrap：10,000 次完整 pair 重采样，seed 7811，双侧 percentile 95%。'
        '区间以本轮已拟合、冻结的模型为条件，不包含 FIT 抽样或重新训练的不确定性。',
        '- 主端点为正只代表相对水平误差改善；绝对分数、非零目标 anchor 和零目标负控共同约束解释，未新增 P0 gate。','',
        '## Fresh bridge：全部固定读出器','',
        '| 读出器 | motion skill | A_dx | A_dy | B_dx | B_dy | position skill |','|---|---:|---:|---:|---:|---:|---:|']
    for key in KEYS:
        row=br['readers'][key]
        lines.append('| '+key+' | '+' | '.join(fmt(v) for v in [row['motion_skill'],*row['per_dimension_skill'][4:],row['position_skill']])+' |')
    lines+=['','| 输入 | S−B dx MSE | 配对 95% CI |','|---|---:|---:|']
    for name in INPUTS:
        row=br['comparisons'][name]
        lines.append(f'| {name} | {fmt(row["S_minus_B_dx_mse"])} | {ci(row["dx_mse_paired_95ci"])} |')
    lines+=['','concat 是固定主读出器，t6/t7 是固定对照，不根据评估分数挑选。单帧 S/B 的加权输入—标签分布相同，'
            '它们的等价是实现检查，不是独立科学重复。','', '![Fresh bridge motion](bridge_motion_skills.png)','',
            '## Fresh anchor：逐路径而非只看总平均','',
            '| Arm/path | absolute motion skill | 8 项 contrast 最低点估计 | mean-motion 描述性 95% CI |',
            '|---|---:|---:|---:|']
    for arm in ARMS:
        for path in PATHS:
            row=ar['readers'][arm+'_concat']['paths'][path]
            lines.append(f'| {arm}/{path} | {fmt(row["absolute_outputs"]["motion_skill"])} | '
                         f'{fmt(np.min(row["end_start_per_coordinate_skill"]))} | {ci(row["mean_motion_skill_descriptive_95ci"])} |')
    lines+=['','aa/bb 为静止干扰，ab/ba 为两种移动方向。共享 end/start anchor 对照以非零目标位移 d 评分。'
            '各路径先计算 contrast 与误差，再平均；图中保留全部八个坐标对照。'
            '辅助区间为未调整的描述性区间，不把多个相关正分当独立复制。','',
            '![Anchor path diagnostics](anchor_path_diagnostics.png)','',
            '## 静止目标、移动干扰负控','',
            '| 读出器 | A_dx RMS/FIT | A_dy RMS/FIT | B_dx RMS/FIT | B_dy RMS/FIT | overall RMS/FIT |',
            '|---|---:|---:|---:|---:|---:|']
    for key in KEYS:
        row=ar['readers'][key]
        lines.append('| '+key+' | '+' | '.join(fmt(v) for v in [*row['negative_control_motion_rms_over_FIT_motion_rms'],row['negative_control_overall_motion_rms_over_FIT']])+' |')
    lines+=['','两方向分别平方后再平均计算 RMS，未通过先平均正负响应使误差相消。归一化分母是固定 FIT '
            '非零位移 RMS。负控越低表示对黄色干扰的虚假运动响应越小，但若非零目标 motion 同时塌缩，不能称修复。',
            f'Anchor 中 a/b 落在同一像素的世界数：{anchor["report"].get("coincident_nuisance_pixel_blocks", "未提供")}；按注册保留。','',
            '![Normalized negative-control RMS](negative_control_rms.png)','',
            '## 数值与科学边界','',
            '| 评估 | t6 S/B 最大差 | t7 S/B 最大差 | 数值等价 |','|---|---:|---:|---|']
    for role,data in (('anchor',anchor),('bridge',bridge)):
        r=data['report'];dif=r['singleframe_S_B_max_abs']
        lines.append(f'| {role} | {fmt(dif["t6"])} | {fmt(dif["t7"])} | {r["singleframe_equivalence_passed"]} |')
    lines+=['','固定数值阈值为 1e-7；anchor 的 negative 单帧等价也包含在检查中。'
            '图表脚本独立核对预测 SHA、实际 MSE、S−B 主效应、每路径 contrast 和负控 RMS；'
            '置信区间直接使用冻结评分器输出，未重选或改变统计方案。','',
            '原 V5 gate 和 V5.1 bridge 始终 failed_unchanged。V5.3 没有自动启动 P0，Test 未打开；'
            '结果只针对冻结视觉表示与读出器，不是 world-model 训练或记忆有效性的结论。'
            'BF16 推理后转 FP32 存储的 cache 不等于全精度视觉提取。','',
            '## 执行与归档','']
    for name in ('fit_controller.json','evaluation_controller.json'):
        c=meta.get(name)
        if not c:continue
        lines.append(f'- `{name}`：状态 `{c.get("status")}`，源码 `{c.get("source_commit")}`。')
        for job in c.get('jobs',[]):
            ids=sorted({m.group(1) for path in job.get('logs',[]) for m in [re.search(r'_(\d+\.\d+)\.log$',path)] if m})
            lines.append(f'  - `{job.get("stage")}`：`{job.get("status")}`；step `{", ".join(ids) or "未提供"}`；选卡 {job.get("selected_gpus","未提供")}。')
    budget=meta.get('ledger/budget.json')
    if budget:lines.append(f'- 下载 ledger 累计：{budget.get("charged_gpuh")} / 60 GPUh；最终以 Slurm 对账记录为准。')
    accounting=meta.get('final_accounting.json')
    if accounting:
        lines.append('- 最终对账原始字段见 `input_inventory.json` 所列 `final_accounting.json`；本报告不推定其未提供的清理状态。')
        for key in ('cumulative_charged_gpuh','all_new_steps_complete','active_new_leases','cleanup'):
            if key in accounting:lines.append(f'- `{key}`：`{accounting[key]}`。')
    lines+=['- 所有输出、原始预测、模型和旧失败档案应保留；归档校验通过不改变失败现场的保留要求。',
            '- `input_inventory.json` 记录本报告实际使用的文件；`SHA256SUMS` 覆盖新生成的图表、表格和 README。','',
            '```sh','python3 build_results.py --scene <downloaded-v53-scene> --out <new-analysis-directory>','```','']
    (out/'README.md').write_text('\n'.join(lines))


def build(scene,out):
    evidence=Evidence(scene)
    anchor,bridge=(load_role(evidence,role) for role in ('anchor','bridge'))
    if anchor['report']['frozen_reader_sha256']!=bridge['report']['frozen_reader_sha256']:
        raise ValueError('Anchor/bridge use different frozen readers')
    verify_metrics(anchor,bridge)
    meta={name:evidence.read(name,optional=True) for name in ('fit_controller.json','evaluation_controller.json',
        'registration_publication.json','freeze_publication.json','reader/freeze_receipt.json',
        'ledger/budget.json','final_accounting.json')}
    frozen=meta['reader/freeze_receipt.json']
    if frozen and sha(evidence.root/'reader/freeze_receipt.json')!=anchor['report']['frozen_reader_sha256']:
        raise ValueError('Frozen reader receipt hash mismatch')
    out=Path(out).resolve()
    if out==evidence.root or evidence.root in out.parents:
        raise ValueError('Derived reports must be outside the immutable input scene')
    out.mkdir(parents=True,exist_ok=False)
    plots(out,anchor,bridge)
    with (out/'bridge_axis_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['reader','axis','mse','skill','direction_error','abs_amplitude_ratio','std_ratio'])
        for key in KEYS:
            row=bridge['report']['scores']['readers'][key]
            for d,name in enumerate(DIMS):
                writer.writerow([key,name,row['per_dimension_mse'][4+d],row['per_dimension_skill'][4+d],
                    row['motion_direction_error_rate'][d],row['motion_absolute_amplitude_ratio'][d],row['motion_std_ratio'][d]])
    with (out/'anchor_contrast_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['reader','path','anchor','axis','skill','lower95','upper95'])
        for key in KEYS:
            for path in PATHS:
                row=anchor['report']['scores']['readers'][key]['paths'][path]
                for side,side_name in enumerate(('end','start')):
                    for d,name in enumerate(DIMS):
                        writer.writerow([key,path,side_name,name,row['end_start_per_coordinate_skill'][side][d],
                            row['end_start_per_coordinate_skill_descriptive_95ci'][0][side][d],
                            row['end_start_per_coordinate_skill_descriptive_95ci'][1][side][d]])
    write_readme(out,anchor,bridge,meta)
    dump(out/'input_inventory.json',dict(schema='v5.3.posthoc-report-inputs.1',inputs=evidence.inventory,
        renderer_sha256=sha(__file__),frozen_reader_sha256=anchor['report']['frozen_reader_sha256'],
        receipt_verification={role:data['receipt_verification'] for role,data in (('anchor',anchor),('bridge',bridge))},
        checks=['prediction SHA256','exact registered counts/IDs','per-coordinate MSE recomputation',
                'paired S-B primary recomputation','per-path contrast recomputation','negative RMS recomputation','single-frame arm equivalence'],
        confidence_intervals='copied from frozen scorer; conditional on frozen models; no new bootstrap or selection'))
    paths=sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in paths))
    for line in (out/'SHA256SUMS').read_text().splitlines():
        digest,relative=line.split('  ',1)
        if sha(out/relative)!=digest:raise ValueError('Output integrity failure')
    return dict(output=str(out),files=len(paths)+1,verified_input_files=len(evidence.inventory),
                scientific_source_modified=False,test_opened=False,p0_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(build(args.scene,args.out),indent=2))


if __name__=='__main__':main()
