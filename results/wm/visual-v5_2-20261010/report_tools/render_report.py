#!/usr/bin/env python3
"""Render verified V5.2 result files; never extract, fit, select, or change gates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

CONDITIONS = ('original', 'static_t6', 'static_t7', 'swap_nuisance')
READERS = ('concat', 't6', 't7')
AXES = ('A_x', 'A_y', 'B_x', 'B_y', 'A_dx', 'A_dy', 'B_dx', 'B_dy')
COLORS = ('#455a64', '#007f86', '#3aa6a5', '#d67530')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def same(actual, expected, name):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise ValueError('Prediction/report mismatch: '+name)


def interval(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or a[0] > a[1]:
        raise ValueError('Invalid interval: '+name)


def read_inputs(scene):
    scene = Path(scene).resolve()
    inventory = {}

    def register(path):
        inventory[str(path.relative_to(scene))] = dict(bytes=path.stat().st_size, sha256=sha(path))

    def read(relative, optional=False):
        path = scene/relative
        if optional and not path.exists():
            return None
        register(path)
        return json.loads(path.read_text())

    report = read('report/report.json')
    if (report.get('schema') != 'v5.2.nuisance-counterfactual.1' or
            report.get('primary_reader') != 'concat' or
            report.get('conditions') != list(CONDITIONS) or
            report.get('test_access') is not False or report.get('p0_authorized') is not False or
            report.get('original_v5_gate') != 'failed_unchanged' or
            report.get('v5_1_bridge') != 'failed_unchanged'):
        raise ValueError('Unexpected schema, fixed conditions, or gate declarations')
    path = scene/'report/predictions.npz'
    register(path)
    if sha(path) != report.get('predictions_sha256'):
        raise ValueError('Predictions SHA256 mismatch')
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if (not np.array_equal(data.get('pair_ids'), np.arange(128, 384)) or
            tuple(data.get('conditions', ())) != CONDITIONS or
            data['targets'].shape != (256, 2, 8) or not np.isfinite(data['targets']).all()):
        raise ValueError('Unexpected pair IDs, condition order, or target shape')
    losses = {}
    for reader in READERS:
        p, y = data[reader], data['targets']
        if p.shape != (256, 2, 4, 8) or not np.isfinite(p).all():
            raise ValueError('Invalid prediction shape/values: '+reader)
        error = ((p-y[:, :, None, :])**2).mean(1)
        dx = error[:, :, [4, 6]].mean(2)
        losses[reader] = dict(error=error, dx=dx)
        score = report['scores'][reader]
        for i, name in enumerate(CONDITIONS):
            c = score['conditions'][name]
            same(error[:, i].mean(0), c['per_dimension_mse'], reader+'/'+name+'/mse')
            same(dx[:, i].mean(), c['dx_mse'], reader+'/'+name+'/dx_mse')
            skills = np.asarray(c['per_dimension_skill'], dtype=float)
            if skills.shape != (8,) or not np.isfinite(skills).all():
                raise ValueError('Invalid skill vector')
            same(skills[:4].mean(), c['position_skill'], reader+'/'+name+'/position_skill')
            same(skills[4:].mean(), c['motion_skill'], reader+'/'+name+'/motion_skill')
            same(1-skills[4:].mean(), c['motion_nmse'], reader+'/'+name+'/motion_nmse')
        same((dx[:, 0]-(dx[:, 1]+dx[:, 2])/2).mean(), score['primary_dx_mse_reduction'], reader+'/primary')
        same(((dx[:, 0]+dx[:, 3]-dx[:, 1]-dx[:, 2])/2).mean(),
             score['secondary_balanced_dx_mse_reduction'], reader+'/balanced')
        for key in ('primary_dx_paired_95ci', 'secondary_balanced_dx_paired_95ci', 'secondary_motion_paired_95ci'):
            interval(score[key], reader+'/'+key)
        for i, name in enumerate(CONDITIONS[1:], 1):
            c = score['individual_comparisons'][name]
            same((dx[:, 0]-dx[:, i]).mean(), c['dx_mse_reduction'], reader+'/'+name+'/reduction')
            interval(c['dx_mse_reduction_descriptive_95ci'], reader+'/'+name+'/ci')
        boot = score['bootstrap']
        if boot.get('seed') != 7810 or boot.get('replicates') != 10000 or boot.get('primary_endpoints') != 1:
            raise ValueError('Bootstrap settings differ from frozen protocol')
    # Validate reused-input checks directly, independently of their JSON flags.
    reuse = dict(t6_original_static6=float(np.max(np.abs(data['t6'][:, :, 0]-data['t6'][:, :, 1]))),
                 t6_static7_swap=float(np.max(np.abs(data['t6'][:, :, 2]-data['t6'][:, :, 3]))),
                 t7_original_static7=float(np.max(np.abs(data['t7'][:, :, 0]-data['t7'][:, :, 2]))),
                 t7_static6_swap=float(np.max(np.abs(data['t7'][:, :, 1]-data['t7'][:, :, 3]))))
    for key, value in reuse.items():
        same(value, report['reused_single_frame_input_max_differences'][key], key)
    if report['reused_single_frame_inputs_passed'] != all(x <= 1e-10 for x in reuse.values()):
        raise ValueError('Reused-input status inconsistent')
    parity = report['original_prediction_parity']
    for reader in READERS:
        row = parity[reader]
        checks = row['checks']
        if row['passed'] != all(checks.values()):
            raise ValueError('Parity checks inconsistent: '+reader)
    valid = all(parity[k]['passed'] for k in READERS) and report['reused_single_frame_inputs_passed']
    if (report['status'] == 'diagnostic_complete') != valid:
        raise ValueError('Report status disagrees with numerical checks')
    meta = {name: read(name, optional=True) for name in
            ('diagnostic_controller.json', 'registration_publication.json', 'source_manifest.json',
             'ledger/budget.json', 'counterfactual_cache/receipt.json', 'creation.json', 'final_accounting.json')}
    receipt = meta['counterfactual_cache/receipt.json']
    if receipt is not None and (receipt.get('cache_sha256') != report['cache_sha256'] or
                                receipt.get('frozen_reader_sha256') != report['frozen_reader_sha256']):
        raise ValueError('Cache receipt and scientific report disagree')
    accounting, budget = meta['final_accounting.json'], meta['ledger/budget.json']
    if accounting is not None and budget is not None:
        same(accounting['cumulative_charged_gpuh'], budget['charged_gpuh'], 'final accounting/ledger')
    return scene, report, data, losses, meta, inventory


def nuisance_dx(pairs):
    return np.array([(int((.5+.4*np.sin(7*2.3+int(p)))*224)-
                      int((.5+.4*np.sin(6*2.3+int(p)))*224))/224 for p in pairs])


def plot_all(out, report, data, losses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 180})
    subtitle = ('SOFTWARE FIXTURE: NO EXPERIMENTAL EVIDENCE' if report.get('software_only') else
                'Known bridge: 256 paired worlds; fixed V5.1 readers; posthoc diagnostic')
    names = ('Original', 'Static at t6', 'Static at t7', 'Nuisance swap')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.7))
    x, width = np.arange(4), .24
    for j, reader in enumerate(READERS):
        vals = [report['scores'][reader]['conditions'][c]['motion_skill'] for c in CONDITIONS]
        axs[0].bar(x+(j-1)*width, vals, width=width, label=reader, color=('#007f86','#818a95','#d67530')[j])
    axs[0].axhline(0, color='black', lw=.8)
    axs[0].set(xticks=x, xticklabels=names, ylabel='Mean motion skill (4 coordinates)', title='Frozen reader comparison')
    axs[0].tick_params(axis='x', rotation=15)
    axs[0].legend(ncols=3, fontsize=9)
    for ci, cond in enumerate(CONDITIONS):
        vals = np.asarray(report['scores']['concat']['conditions'][cond]['per_dimension_mse'])[[4,6]]
        axs[1].bar(np.arange(2)+(ci-1.5)*.2, vals, width=.2, label=names[ci], color=COLORS[ci])
    axs[1].set(xticks=[0,1], xticklabels=['A_dx','B_dx'], ylabel='MSE (normalized coordinate squared)',
               title='Primary reader: horizontal error')
    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs[1].legend(fontsize=8)
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout(); fig.savefig(out/'condition_comparison.png'); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
    dx = losses['concat']['dx']
    variants = (dx[:,1], dx[:,2], (dx[:,1]+dx[:,2])/2)
    for ax, vals, title in zip(axs, variants, ('Static at t6', 'Static at t7', 'Mean of two static losses')):
        limit = max(float(dx[:,0].max()), float(vals.max()))*1.04
        ax.scatter(dx[:,0], vals, s=13, alpha=.65, color='#007f86', edgecolors='none')
        ax.plot([0,limit], [0,limit], color='#6f6f6f', ls='--', lw=1)
        ax.set(xlim=(0,limit), ylim=(0,limit), xlabel='Original pair dx MSE', ylabel='Static pair dx MSE', title=title)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        ax.text(.04,.95,'Below diagonal = lower static error', transform=ax.transAxes, va='top', fontsize=8)
    fig.suptitle('Each dot retains both variants and both dx coordinates', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'paired_dx_error.png'); plt.close(fig)

    eta = nuisance_dx(data['pair_ids'])
    prediction, target = data['concat'], data['targets']
    response = (prediction[:,:,0,4:]-prediction[:,:,3,4:]).mean(1)/2
    orig = (prediction[:,:,0,4:]-target[:,:,4:]).mean(1)
    swap = (prediction[:,:,3,4:]-target[:,:,4:]).mean(1)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.7))
    for col, (d, name) in enumerate(((0,'A_dx'),(2,'B_dx'))):
        slope = report['nuisance_swap_response']['concat'][name]['through_origin_slope']
        same(eta@response[:,d]/(eta@eta), slope, 'swap_slope/'+name)
        axs[0,col].scatter(eta, response[:,d], s=15, alpha=.65, color='#007f86')
        span = np.array([eta.min(),eta.max()])
        axs[0,col].plot(span, slope*span, color='#d67530', label=f'Descriptive slope = {slope:.4f}')
        axs[0,col].set(xlabel='Original yellow dx (integer pixels / 224)',
                       ylabel='Mean(original - swap prediction) / 2', title=name+' nuisance response')
        axs[0,col].legend(fontsize=8)
        axs[1,col].scatter(eta, orig[:,d], s=12, alpha=.55, color=COLORS[0], label='Original residual')
        axs[1,col].scatter(eta, swap[:,d], s=12, alpha=.55, color=COLORS[3], label='Nuisance-swap residual')
        axs[1,col].set(xlabel='Original yellow dx (integer pixels / 224)',
                       ylabel='Pair-common signed residual', title=name+' residual direction')
        axs[1,col].legend(fontsize=8)
    for ax in axs.flat:
        ax.axhline(0, color='black', lw=.6); ax.axvline(0, color='black', lw=.6)
    fig.suptitle('Target trajectory and target labels remain unchanged', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'nuisance_swap_response.png'); plt.close(fig)


def f(value):
    return '未提供' if value is None else f'{float(value):.7g}'


def ci(value):
    return '['+f(value[0])+', '+f(value[1])+']'


def make_readme(out, report, meta, inventory):
    main = report['scores']['concat']
    valid = report['status'] == 'diagnostic_complete'
    primary, bounds = main['primary_dx_mse_reduction'], main['primary_dx_paired_95ci']
    if report.get('software_only'):
        conclusion = '**软件样例：这些数字不能作为实验结论。**'
    elif not valid:
        conclusion = '**数值一致性检查未通过，本次诊断不能作科学解释；以下数字仅保留排错现场。**'
    elif bounds[0] > 0:
        conclusion = '在这批已看过的 bridge 场景上，固定两个静止干扰条件的平均水平位移误差低于原条件；配对区间整体为正。'
    elif bounds[1] < 0:
        conclusion = '在这批已看过的 bridge 场景上，固定两个静止干扰条件的平均水平位移误差高于原条件；配对区间整体为负。'
    else:
        conclusion = '主效应的配对区间跨过零；本次诊断尚不能明确区分平均静止干扰的改善、恶化或无变化。'
    lines = ['# V5.2：冻结读出器的黄色干扰物反事实诊断', '', conclusion, '',
        '这是已检查过的 Dev pairs 128–383（256 对世界、512 episodes）的探索性干预。'
        '目标轨迹、时间顺序、标签与背景保持不变，只改变黄色干扰物的位置组合；没有训练或选择新的读出器。', '',
        f'- 报告状态：`{report["status"]}`。',
        '- 主端点：两 variants × A_dx/B_dx 的 pair 内原始 MSE；单位是归一化图像坐标的平方。',
        f'- 主效应 Δorig = 原条件损失 − 两个静止条件损失均值：**{f(primary)}**，95% 配对区间 **{ci(bounds)}**。',
        f'- 相对本次 original 的 dx MSE，平均静止条件减少 **{100*primary/main["conditions"]["original"]["dx_mse"]:.2f}%**；这是误差减少比例，不是解释方差比例。',
        f'- 辅助平衡效应 Δbalanced = 两个移动条件损失均值 − 两个静止条件损失均值：**{f(main["secondary_balanced_dx_mse_reduction"])}**，95% 区间 **{ci(main["secondary_balanced_dx_paired_95ci"])}**。',
        f'- 四维 motion NMSE 平均减少：{f(main["secondary_static_average_motion_nmse_reduction"])}，95% 区间 {ci(main["secondary_motion_paired_95ci"])}。',
        '- 10,000 次完整 pair bootstrap，seed 7810。唯一主端点是 Δorig；其他区间为未调整的描述性区间。'
        '静止条件平均的是损失，不是预测值，未构造预测集成。', '',
        '## 每个条件都报告', '',
        '| 条件 | dx MSE | A_dx MSE | B_dx MSE | 四维 motion skill | position skill |',
        '|---|---:|---:|---:|---:|---:|']
    for name in CONDITIONS:
        c = main['conditions'][name]
        lines.append(f'| {name} | {f(c["dx_mse"])} | {f(c["per_dimension_mse"][4])} | {f(c["per_dimension_mse"][6])} | {f(c["motion_skill"])} | {f(c["position_skill"])} |')
    lines += ['', '| 与 original 比较 | dx MSE 减少 | 描述性 95% 区间 |', '|---|---:|---:|']
    for name in CONDITIONS[1:]:
        c = main['individual_comparisons'][name]
        lines.append(f'| {name} | {f(c["dx_mse_reduction"])} | {ci(c["dx_mse_reduction_descriptive_95ci"])} |')
    lines += ['', '正减少表示误差变小。两个静止锚点和 swap 都完整列出，不选最好的一项。'
              'skill 的参考是冻结 FIT 目标均值，不能把本表的正分解释成新的 gate 通过。', '',
              '![条件与读出器比较](condition_comparison.png)', '',
              '![完整 pair 的原条件与静止条件误差](paired_dx_error.png)', '',
              '## 单帧对照', '', '| 冻结读出器 | Δorig | 主式的描述性 95% 区间 | Δbalanced |', '|---|---:|---:|---:|']
    for name in READERS:
        c = report['scores'][name]
        lines.append(f'| {name} | {f(c["primary_dx_mse_reduction"])} | {ci(c["primary_dx_paired_95ci"])} | {f(c["secondary_balanced_dx_mse_reduction"])} |')
    lines += ['', 'concat 是预先固定的主读出器；t6/t7 只是已冻结的对照。单帧成绩为正或变化明显，'
              '都不能证明利用了目标的时间信息。单帧 balanced 对照接近零是匹配时间槽位置边际带来的代数恒等，'
              '不能计为独立实验支持。输入复用的四项最大预测差：', '']
    for key, val in report['reused_single_frame_input_max_differences'].items():
        lines.append(f'- `{key}`：{f(val)}。')
    lines += ['', '## Nuisance swap 响应', '',
              'η 是原黄色干扰物实际整数像素位移 / 224；R 是 pair 内两 variants 的 '
              '(original − swap) 预测均值 / 2。只交换干扰物，不交换目标帧或标签。', '',
              '| 主读出器轴 | R 对 η 的过原点描述性斜率 | 同向比例（排除零值） | 同向/反向 | residual 反号对数 |',
              '|---|---:|---:|---:|---:|']
    for name in AXES[4:]:
        row = report['nuisance_swap_response']['concat'][name]
        lines.append(f'| {name} | {f(row["through_origin_slope"])} | {f(row["signed_alignment_fraction"])} | '
                     f'{row["aligned_count"]}/{row["opposed_count"]} | {row["residual_sign_reversal_count"]}/{row["total_pairs"]} |')
    lines += ['', '零 η、零 R 和 residual 平局计数保存在原始 report.json；任何零值都没有从主 MSE 分析删除。'
              '该斜率只是描述预测对干扰物交换的响应，没有用于训练、校正或选择模型。', '',
              '![黄色干扰物交换响应](nuisance_swap_response.png)', '', '## Original parity', '',
              '| 读出器 | parity | 最大 motion RMS drift / target RMS | 最大 motion max drift / target RMS | 最大 motion skill drift |',
              '|---|---|---:|---:|---:|']
    for name in READERS:
        row = report['original_prediction_parity'][name]
        lines.append(f'| {name} | {"通过" if row["passed"] else "失败"} | '
                     f'{f(max(row["motion_rms_drift_fraction"]))} | {f(max(row["motion_max_drift_fraction"]))} | '
                     f'{f(max(row["all_dimension_skill_abs_drift"][4:]))} |')
    lines += ['', '固定阈值依次为 0.01、0.10、0.02，分别检查四个 motion 维度和三个读出器。'
              '比较使用本次重算的 matched original；旧 original 仅是数值 parity 基准。'
              '若结果与数值漂移相当，应保留方向不确定性。', '', '## 解释边界', '',
              '通过像素与数值一致性检查后，配对干预支持的是“黄色渲染分配对该冻结 encoder＋reader '
              '在这些已知场景上的误差贡献”。balanced 对照进一步匹配了每个时间槽的黄色位置边际。'
              '这不定位到 encoder 内部的某个机制，不证明目标运动信息缺失，也不证明目标时间信息或记忆被利用。', '',
              '旧 V5 gate 与 V5.1 bridge 仍为 failed_unchanged；P0、适配、replay 均不因本诊断启动。'
              'Test 未访问；没有新鲜确认集。即使静止条件恢复，也没有恢复原移动干扰物任务。'
              '多个相关的次要指标不构成独立重复证据。', '',
              '编码沿用 BF16 autocast，FP32 pooling 后再量化为 BF16，最终转 FP32 存储；'
              '“FP32 cache”不表示全精度提取。', '', '## 执行证据与复现', '']
    controller = meta['diagnostic_controller.json'] or {}
    publication = meta['registration_publication.json'] or {}
    source = controller.get('source_commit') or publication.get('source_commit')
    lines.append(f'- 冻结执行源码：`{source}`。' if source else '- 冻结源码 commit：scene 中未提供。')
    lines.append(f'- 冻结 reader receipt SHA256：`{report["frozen_reader_sha256"]}`。')
    jobs = controller.get('jobs', [])
    if jobs:
        for job in jobs:
            ids = sorted({m.group(1) for s in job.get('logs', []) for m in [re.search(r'_(\d+\.\d+)\.log$', s)] if m})
            lines.append(f'- `{job.get("stage")}`：状态 `{job.get("status")}`，实际日志 step ID '
                         f'`{", ".join(ids) if ids else "未提供"}`，选卡数 {job.get("selected_gpus", "未提供")}。')
    else:
        lines.append('- scene 中未提供 controller job 记录，不能从报告推定 job ID。')
    budget = meta['ledger/budget.json']
    if budget:
        lines.append(f'- 下载 ledger 的累计选卡记账：{f(budget.get("charged_gpuh"))} / 60 GPUh；以归档中的最终 Slurm 对账为准。')
    accounting = meta['final_accounting.json']
    if accounting:
        lines.append(f'- 最终对账累计：**{accounting["cumulative_charged_gpuh"]:.16g} / 60 GPUh**；'
                     f'本轮科学计算 {f(accounting.get("new_science_gpuh"))} GPUh，管理开销 {f(accounting.get("management_gpuh"))} GPUh。')
        lines.append(f'- 新 step 全部完成：`{accounting.get("all_new_steps_complete")}`；'
                     f'仍有新 lease：`{accounting.get("active_new_leases")}`。')
        for key, name in (('indexmem_step','既有 IndexMem'), ('parent_allocation','共享 parent allocation')):
            entry = accounting.get(key)
            if entry and len(entry) >= 3:
                lines.append(f'- 最终对账时 {name} `{entry[0]}` 状态 `{entry[2]}`；保留原工作负载。')
        if accounting.get('cleanup') == 'retained scientific failure investigation scene; no deletion':
            lines.append('- 清理状态：**远端诊断现场保留，未删除**。本次技术诊断完成，但原科学门槛仍失败；'
                         '归档成功不改变用户要求保留失败现场的边界。')
        elif accounting.get('cleanup'):
            lines.append('- 清理记录：`'+str(accounting['cleanup'])+'`。')
    lines += ['- 本生成器只读 JSON/NPZ，并独立重算 MSE、主效应、balanced、锚点差和输入复用差。'
              '置信区间沿用已冻结 scorer 的输出，未在绘图时改变端点或重选模型。',
              '- `input_inventory.json` 记录使用的文件字节数与 SHA256；`SHA256SUMS` 覆盖本次生成的 README、图和表。', '',
              '```sh', 'python3 render_report.py --scene <downloaded-scene> --out <new-report-directory>', '```', '']
    (out/'README.md').write_text('\n'.join(lines))


def render(scene, out):
    scene, report, data, losses, meta, inventory = read_inputs(scene)
    out = Path(out).resolve()
    if out == scene or scene in out.parents:
        raise ValueError('Write derived reports outside the immutable input scene')
    out.mkdir(parents=True, exist_ok=False)
    plot_all(out, report, data, losses)
    with (out/'condition_metrics.csv').open('w', newline='') as fobj:
        columns = ['reader', 'condition', 'dx_mse', 'motion_skill', 'position_skill']+[a+'_mse' for a in AXES]
        writer = csv.DictWriter(fobj, fieldnames=columns); writer.writeheader()
        for reader in READERS:
            for condition in CONDITIONS:
                c = report['scores'][reader]['conditions'][condition]
                row = dict(reader=reader, condition=condition, **{key:c[key] for key in columns[2:5]})
                row.update({a+'_mse':v for a,v in zip(AXES,c['per_dimension_mse'])}); writer.writerow(row)
    dx = losses['concat']['dx']
    with (out/'paired_dx_losses.csv').open('w', newline='') as fobj:
        writer = csv.writer(fobj); writer.writerow(['pair', *CONDITIONS, 'static_mean_loss', 'primary_reduction', 'balanced_reduction'])
        for i, pair in enumerate(data['pair_ids']):
            stat = (dx[i,1]+dx[i,2])/2
            writer.writerow([int(pair), *dx[i], stat, dx[i,0]-stat, (dx[i,0]+dx[i,3])/2-stat])
    make_readme(out, report, meta, inventory)
    dump(out/'input_inventory.json', dict(schema='v5.2.posthoc-render-inputs.1', inputs=inventory,
        renderer_sha256=sha(__file__), report_status=report['status'], software_only=bool(report.get('software_only')),
        verified=dict(predictions_digest=True, fixed_counts=True, mse_and_primary_effect_recomputed=True,
                      reused_input_differences_recomputed=True, confidence_intervals='copied from frozen scorer; not reselected')))
    files = sorted(p for p in out.rglob('*') if p.is_file())
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    for line in (out/'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        if sha(out/relative) != expected:
            raise ValueError('Rendered file verification failed')
    return dict(output=str(out), report_status=report['status'], files=len(files)+1,
                software_only=bool(report.get('software_only')), input_files=len(inventory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.scene, args.out), indent=2))


if __name__ == '__main__':
    main()
