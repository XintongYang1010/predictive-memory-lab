#!/usr/bin/env python3
"""Posthoc same-output spatial controls using saved predictions/features only."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REPORT = HERE.parent / 'report'
V55 = REPO / 'execution/visual-v5_5-20261010/diagnostic-20261010-v55'
SEED = 7816
REPLICATES = 10000


def identity(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for value in iter(lambda: f.read(2**20), b''):
            h.update(value)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def summarize(effect, indices):
    # All 2 variants x 4 A/B configurations remain together within each world.
    value = effect.reshape(256, 2, 4) * 224
    world_mse = np.mean(value**2, axis=(1, 2))
    world_mean = np.mean(value, axis=(1, 2))
    boot_rms = np.sqrt(world_mse[indices].mean(1))
    return dict(rms_px=float(np.sqrt(world_mse.mean())),
                rms_px_paired_world_95ci=interval(boot_rms),
                signed_mean_px=float(world_mean.mean()),
                signed_mean_px_paired_world_95ci=interval(world_mean[indices].mean(1)),
                max_abs_px=float(np.max(np.abs(value)))), boot_rms


def main():
    inputs = {str(path.relative_to(REPO)): identity(path) for path in
              (REPORT / 'feature_paths.npz', REPORT / 'transitions.npz', REPORT / 'report.json',
               V55 / 'counterfactual_cache/cache.npz', V55 / 'archive_manifest.json',
               V55 / 'batch_match_run/comparison/report.json', V55 / 'batch_match_run/cache/receipt.json',
               REPO / 'scripts/v5_6_replay.py')}
    old_report = json.loads((REPORT / 'report.json').read_text())
    assert old_report['status'] == 'complete' and not old_report['test_opened']
    assert inputs[str((V55 / 'counterfactual_cache/cache.npz').relative_to(REPO))] == old_report['cache']
    assert identity(V55 / 'archive_manifest.json')['sha256'] == 'b6e0ca1e893a595edb3a8c60507f72de647f5affcb20daf8223bf3af20e72663'
    seal = json.loads((V55 / 'archive_manifest.json').read_text())['files']
    for name in ('counterfactual_cache/cache.npz', 'batch_match_run/comparison/report.json', 'batch_match_run/cache/receipt.json'):
        assert identity(V55 / name) == seal[name]
    assert identity(REPO / 'scripts/v5_6_replay.py')['sha256'] == old_report['source_identity']['scripts/v5_6_replay.py']
    paths = load(REPORT / 'feature_paths.npz')
    transitions = load(REPORT / 'transitions.npz')
    assert set(paths) == {f'{m}_object{j}' for m in ('global_W', 'lane_W') for j in (0, 1)}
    indices = np.random.default_rng(SEED).integers(0, 256, size=(REPLICATES, 256))
    summary = {}; difference = {}; consistency = {}; effects = {}; boot = {}
    for model in ('global_W', 'lane_W'):
        summary[model] = {}; difference[model] = {}
        for region, suffix in (('top_outer_row', 0), ('bottom_outer_row', 1)):
            values = paths[f'{model}_object{suffix}']
            assert values.shape == (512, 4, 4, 8) and np.isfinite(values).all()
            lo = transitions[model][:, 0, [0, 2, 4, 6]]
            hi = transitions[model][:, 0, [1, 3, 5, 7]]
            consistency[f'{model}/{region}'] = dict(
                baseline_query_max_abs=float(np.max(np.abs(values[:, :, 0] - lo))),
                all_high_query_max_abs=float(np.max(np.abs(values[:, :, 3] - hi))))
            assert max(consistency[f'{model}/{region}'].values()) < 1e-7
            # v00/v10/v01/v11: outside row / selected row independently uses
            # low-Y or high-Y real-frame features. Reverse means removal from v11.
            tensors = dict(row_forward=values[:, :, 1] - values[:, :, 0],
                           row_reverse=values[:, :, 2] - values[:, :, 3],
                           complement_forward=values[:, :, 2] - values[:, :, 0],
                           complement_reverse=values[:, :, 1] - values[:, :, 3],
                           interaction=values[:, :, 3] - values[:, :, 2] - values[:, :, 1] + values[:, :, 0],
                           full_yellow=values[:, :, 3] - values[:, :, 0])
            for output, col in (('A_dx', 4), ('B_dx', 6)):
                summary[model].setdefault(output, {})[region] = {}
                for name, tensor in tensors.items():
                    key = f'{model}/{output}/{region}/{name}'
                    effects[key] = tensor[..., col].reshape(256, 2, 4)
                    stats, samples = summarize(tensor[..., col], indices)
                    summary[model][output][region][name] = stats
                    boot[key] = samples
        for output in ('A_dx', 'B_dx'):
            difference[model][output] = {}
            for name in ('row_forward', 'row_reverse', 'interaction'):
                top = summary[model][output]['top_outer_row'][name]['rms_px']
                bottom = summary[model][output]['bottom_outer_row'][name]['rms_px']
                d = boot[f'{model}/{output}/top_outer_row/{name}'] - boot[f'{model}/{output}/bottom_outer_row/{name}']
                difference[model][output][name] = dict(top_minus_bottom_rms_px=top-bottom,
                    paired_world_95ci=interval(d), same_output=True)

    with np.load(V55 / 'counterfactual_cache/cache.npz', allow_pickle=False) as cached:
        x = cached['features'].reshape(256, 2, 8, 16, 1280)
        assert np.array_equal(cached['pair_ids'], np.arange(640, 896))
    rgb_changed = np.zeros((256, 2, 4), dtype=bool)
    shard_evidence = {}; unchanged = []
    for i, pair in enumerate(range(640, 896)):
        name = f'counterfactual_cache/shards/{pair:06d}.json'
        path = V55 / name
        record_identity = identity(path)
        assert record_identity == seal[name]
        shard_evidence[name] = record_identity
        record = json.loads(path.read_text())
        assert record['pair_id'] == pair
        for variant, values in enumerate(record['inputs']['variants']):
            assert values['variant'] == variant
            for j, mask in enumerate((0, 2, 4, 6)):
                changed = values['unique_rgb'][mask]['sha256'] != values['unique_rgb'][mask+1]['sha256']
                rgb_changed[i, variant, j] = changed
                if not changed:
                    unchanged.append(dict(pair=pair,variant=variant,low_mask=mask,nuisance_centers=values['nuisance_centers']))
    real_change = x[:, :, 1::2].astype(np.float64) - x[:, :, ::2].astype(np.float64)
    response = {}
    for region, tokens in (('top_outer_row', list(range(4))), ('bottom_outer_row', list(range(12, 16)))):
        change = real_change[..., tokens, :]
        norms = np.sqrt(np.sum(change**2, axis=(-1, -2)))
        assert np.array_equal(norms > 0, rgb_changed)
        world_energy = (norms**2).mean((1, 2))
        response[region] = dict(token_indices=tokens,
            contexts=int(norms.size), contexts_with_any_nonzero_change=int(np.count_nonzero(norms)),
            actual_rgb_changed_contexts=int(rgb_changed.sum()),
            all_actual_changed_rgb_contexts_have_nonzero_feature_change=True,
            unchanged_rgb_contexts_have_zero_feature_change=True,
            feature_coordinates=int(change.size), nonzero_feature_coordinates=int(np.count_nonzero(change)),
            max_abs_feature_change=float(np.max(np.abs(change))),
            mean_feature_l2=float(norms.mean()), rms_feature_l2=float(np.sqrt(world_energy.mean())),
            rms_feature_l2_paired_world_95ci=interval(np.sqrt(world_energy[indices].mean(1))),
            feature_l2_quantiles={str(q):float(np.quantile(norms,q)) for q in (0,.25,.5,.75,1)})

    caveats = [
        'Posthoc fixed spatial comparison on already opened Dev640..895; no new fitting, images, encoder calls or tuning.',
        'Names object0/object1 in the historical NPZ index top/bottom path construction; all eight saved outputs are used here. They do not certify object coverage.',
        'Top tokens0..3 and bottom tokens12..15 are outer pooled rows. Objects move and need not lie inside them. These are not target tokens.',
        'Both top and bottom effects are compared on the identical A_dx or identical B_dx output, using identical paired worlds and A/B configurations.',
        'Activation replacements mix parts of two real encodings; mixed states can be off manifold. Effects are reader responses under this intervention, not unique encoder responsibility percentages.',
        'Lane A ignores bottom outer tokens and lane B ignores top outer tokens by construction. Their near-zero opposite-row effect is an expected access-control check, not a discovered encoder property.',
        'The 10000 paired-world percentile intervals are conditional exploratory intervals, with no multiplicity correction or new pass gate.',
        'Nonzero outer-row changes under real yellow-only pixel changes establish contextual feature sensitivity outside the directly changed yellow pooling area, not an encoder defect or pure object identity.',
        'The archived full batch4 supplement established exact equality of every canonical feature; the former batch4/12 concern is excluded for these inputs.'
    ]
    report = dict(schema='v5.6-posthoc-spatial-control-1', status='complete',
        method='No production functions called: subtract saved four-condition predictions and bootstrap per-world second moments; directly subtract cached real yellow endpoint features.',
        inputs=inputs, population=dict(worlds=256, variants_per_world=2, ab_contexts_per_variant=4, endpoints_per_context=2),
        bootstrap=dict(seed=SEED,replicates=REPLICATES,unit='whole paired world',paired_top_bottom=True,posthoc=True),
        coordinate_unit='pixels for horizontal prediction effects (normalized output multiplied by224)',
        conditions=['00 both low-yellow','10 selected row high-yellow, complement low','01 selected row low-yellow, complement high','11 both high-yellow'],
        effects=dict(row_forward='10-00',row_reverse='01-11',complement_forward='01-00',complement_reverse='10-11',interaction='11-01-10+00',full_yellow='11-00'),
        consistency=consistency,same_output_spatial_comparison=summary,top_minus_bottom=difference,
        real_yellow_outer_feature_changes=response,
        rgb_identity_audit=dict(shard_files=shard_evidence,actual_rgb_changed_contexts=int(rgb_changed.sum()),
                                unchanged_contexts=unchanged),
        caveats=caveats,test_opened=False,new_fit=False)
    (HERE / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(HERE / 'per_world_effects.npz', **effects)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey=False)
    for i, model in enumerate(('global_W', 'lane_W')):
        for j, output in enumerate(('A_dx', 'B_dx')):
            ax=axes[i,j]
            for offset, region, color in ((-.16,'top_outer_row','#31708e'),(.16,'bottom_outer_row','#d39045')):
                records=[summary[model][output][region][name] for name in ('row_forward','row_reverse','interaction')]
                vals=np.array([v['rms_px'] for v in records]);ci=np.array([v['rms_px_paired_world_95ci'] for v in records])
                ax.bar(np.arange(3)+offset,vals,width=.30,color=color,label=region.replace('_',' '))
                ax.errorbar(np.arange(3)+offset,vals,yerr=np.maximum(0,np.vstack((vals-ci[:,0],ci[:,1]-vals))),fmt='none',ecolor='#333333',capsize=3)
            ax.set_title(f'{model}: same {output} output');ax.set_xticks(range(3),['Row insert','Row removal','Interaction']);ax.set_ylabel('Prediction effect RMS (px)')
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Posthoc outer-row control: compare the same output across spatial regions',fontsize=13)
    fig.text(.5,.012,'256 paired worlds; 95% conditional bootstrap intervals. Outer rows are fixed spatial locations, not target-token labels.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(HERE/'same_output_spatial_controls.png',dpi=160);plt.close(fig)
    lines=['# V5.6 事后空间控制：同一输出上的上、下外侧行', '',
      '本报告只重算已保存的路径预测和特征。没有训练模型、调用编码器、生成新样本或调整阈值。', '',
      '上外侧行是 4×4 池化后的 tokens 0–3，下外侧行是 tokens 12–15；它们是固定空间位置，不能称为“目标 tokens”，也不保证总覆盖目标本体。'
      '这里分别固定 A_dx、固定 B_dx，在同一输出上比较两行；不再用“上行→A”与“下行→B”充当空间对照。', '',
      '| reader / 同一输出 | 上行插入 RMS px [95% CI] | 下行插入 RMS px [95% CI] | 上−下 RMS px [配对 CI] |',
      '|---|---|---|---|']
    fmt=lambda p,ci:f'{p:.6f} [{ci[0]:.6f}, {ci[1]:.6f}]'
    for model in ('global_W','lane_W'):
        for output in ('A_dx','B_dx'):
            top=summary[model][output]['top_outer_row']['row_forward'];bottom=summary[model][output]['bottom_outer_row']['row_forward'];diff=difference[model][output]['row_forward']
            lines.append(f"| {model} / {output} | {fmt(top['rms_px'],top['rms_px_paired_world_95ci'])} | {fmt(bottom['rms_px'],bottom['rms_px_paired_world_95ci'])} | {fmt(diff['top_minus_bottom_rms_px'],diff['paired_world_95ci'])} |")
    lines += ['', '![同一输出空间控制](same_output_spatial_controls.png)', '',
      '“插入”以整幅 low-yellow 编码为基底，只换选定外侧行；“移除”以整幅 high-yellow 编码为基底，换回 low-yellow 的该行。'
      '移除与插入的背景特征不同，因此二次核下不必严格等幅。interaction 是四条件差分 11−01−10+00。'
      '两个方向、互补区域替换、交互与完整 yellow 变化的 RMS、带符号均值及配对世界区间均在 `report.json`；原逐世界效应在 `per_world_effects.npz`。', '',
      '## 真实黄色变化是否改变外侧行特征', '',
      '| 区域 | 非零变化上下文 /2048 | 非零坐标 /总坐标 | 特征差 L2 RMS [95% CI] |',
      '|---|---:|---:|---|']
    for region,v in response.items():
        lines.append(f"| {region} | {v['contexts_with_any_nonzero_change']} | {v['nonzero_feature_coordinates']} / {v['feature_coordinates']} | {fmt(v['rms_feature_l2'],v['rms_feature_l2_paired_world_95ci'])} |")
    lines += ['', '这些差异来自实际 yellow-only 两张图的编码，并非人造特征替换。黄色矩形位于图像中间，而外侧池化行也会改变：'
      '编码器的上下文敏感性可把变化带到外侧区域。再结合路径替换结果，可说明冻结读出器能使用这类变化；'
      '不能据此判为 encoder 实现错误，也不能把各效应 RMS 相除解释成模块责任百分比。'
      '逐一回读并验证 256 份原 RGB 哈希记录后，2040 个实际 RGB 变化上下文全部在两侧外行产生非零特征差；'
      '剩余 8 个来自 world816 的两 variants ×4 布局，黄色像素中心均为 [148,148]，RGB 本就相同且特征差为零。', '',
      '## 范围和限制', ''] + ['- '+item for item in caveats] + ['',
      '读取范围是已看 Dev640..895 的 256 个世界，每世界 2 variants ×4 个 A/B 布局；完整世界一起重采样 10000 次，seed7816。'
      '历史全 batch4 补验已证明全部 canonical 特征逐值相等，此处不存在此前提出的 batch4/12 输入差异。', '']
    (HERE/'README.md').write_text('\n'.join(lines))
    files=['analyze.py','report.json','per_world_effects.npz','same_output_spatial_controls.png','README.md']
    manifest={name:identity(HERE/name) for name in files}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='complete',top_minus_bottom=difference,real_feature_changes=response,manifest=identity(HERE/'manifest.json')),indent=2))


if __name__=='__main__':
    main()
