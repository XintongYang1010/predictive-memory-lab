#!/usr/bin/env python3
"""Posthoc report-only audit; no fit/render/encoder, initial mixed-batch results."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='4'
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;FOLDER=BASE/'report-snapshot/report'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rms(x):return np.sqrt(np.mean(x*x,axis=(0,1)))
manifest=json.loads((FOLDER/'manifest.json').read_text())
for name,record in manifest.items():
 p=FOLDER/name
 if dict(bytes=p.stat().st_size,sha256=sha(p))!=record:raise ValueError('Report archive mismatch '+name)
r=json.loads((FOLDER/'report.json').read_text())
with np.load(FOLDER/'predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
with np.load(BASE/'lane_original_eval/predictions.npz',allow_pickle=False) as z:local={k:z[k] for k in z.files}
y=a['targets'][...,7,4:];case_ids=(844,761,891);result={};case_table=[]
for name in ('W_concat','lane_W'):
 p=a[name][...,:,4:];phi=a[name+'_shapley'][...,:,4:]
 own=np.stack((phi[...,0,0],phi[...,0,1],phi[...,1,2],phi[...,1,3]),-1)
 other=np.stack((phi[...,1,0],phi[...,1,1],phi[...,0,2],phi[...,0,3]),-1);yellow=phi[...,2,:];base=p[...,0,:]
 error=own-y;components=np.stack((base,error,other,yellow),-1)
 closure=float(np.max(np.abs(components.sum(-1)-(p[...,7,:]-y))))
 if closure>1e-12:raise ValueError('Error decomposition mismatch')
 moment=np.einsum('wvdc,wvde->dce',components,components)/512
 np.testing.assert_allclose(moment.sum((1,2)),np.mean((p[...,7,:]-y)**2,axis=(0,1)),rtol=1e-12,atol=1e-15)
 onlyown=np.stack((p[...,4,0],p[...,4,1],p[...,2,2],p[...,2,3]),-1)
 onlyother=np.stack((p[...,2,0],p[...,2,1],p[...,4,2],p[...,4,3]),-1)
 summary=dict(error_components=['repeated_image','own_shapley_minus_truth','other_target_shapley','yellow_shapley'],
  error_recomposition_max_abs=closure,error_component_second_moment_by_axis=moment.tolist(),
  error_component_second_moment_dx_average=moment[[0,2]].mean(0).tolist(),
  note='Diagonal entries are component mean squares; off-diagonal entries are signed products counted twice in MSE. Not independent causal responsibility shares.',
  own_gain_through_origin_descriptive=(np.mean(own*y,(0,1))/np.mean(y*y,(0,1))).tolist(),
  own_direction_errors=(np.sign(own)!=np.sign(y)).sum((0,1)).tolist(),
  own_error_rms=rms(error).tolist(),other_target_rms=rms(other).tolist(),yellow_rms=rms(yellow).tolist(),
  own_only_error_rms=rms(onlyown-y).tolist(),own_only_direction_errors=(np.sign(onlyown)!=np.sign(y)).sum((0,1)).tolist(),
  other_only_false_motion_rms=rms(onlyother).tolist(),yellow_only_false_motion_rms=rms(p[...,1,:]).tolist(),
  both_targets_yellow_static_dx_mse=float(np.mean((p[...,6,[0,2]]-y[...,[0,2]])**2)),
  original_dx_mse=float(np.mean((p[...,7,[0,2]]-y[...,[0,2]])**2)),
  masks={mask:{k:v[k] for k in ('dx_mse','motion_mse','motion_direction_error_rate_nonzero','motion_target_zero_count','motion_response_rms')}
         for mask,v in r['readers'][name]['masks'].items()},
  original_scores=r['readers'][name]['original'],original_direction=r['readers'][name]['original_direction'])
 for pair in case_ids:
  i=pair-640;v=1
  case_table.append(dict(pair_id=pair,variant=v,reader=name,true_dx_pixels=(y[i,v,[0,2]]*224).tolist(),
   predicted_dx_pixels=(p[i,v,7,[0,2]]*224).tolist(),own_shapley_dx_pixels=(own[i,v,[0,2]]*224).tolist(),
   other_shapley_dx_pixels=(other[i,v,[0,2]]*224).tolist(),yellow_shapley_dx_pixels=(yellow[i,v,[0,2]]*224).tolist(),
   original_dx_mse=float(np.mean((p[i,v,7,[0,2]]-y[i,v,[0,2]])**2))))
 result[name]=summary
original_local_parity={name:float(np.max(np.abs(a[name][...,7,:]-local[key]))) for name,key in (('W_concat','global_W'),('lane_W','lane_W'))}
if max(original_local_parity.values())>1e-9:raise ValueError('Published reader original evaluation differs')
output=dict(status='provisional_factorial_mechanism_pending_uniform_batch4_check',
 original_reader_comparison_status='stable_on_archived_original_cues_and_exact_GPU_original_endpoints',
 population=dict(worlds=256,episodes=512,already_opened_dev=True),axes=['A_dx','A_dy','B_dx','B_dy'],
 provenance=dict(report_sha256=sha(FOLDER/'report.json'),predictions_sha256=sha(FOLDER/'predictions.npz'),
  report_manifest_sha256=sha(FOLDER/'manifest.json'),snapshot_verification_sha256=sha(BASE/'report-snapshot-verification.json'),
  analysis_script_sha256=sha(__file__),source_commit=r['provenance']['source_commit']),
 original_local_parity=original_local_parity,readers=result,global_vs_lane=r['global_vs_lane'],cases=case_table,
 pooling=r['pooling'],bf16_rounding=r['final_bf16_rounding'],
 conclusions=dict(stable='With identical encoder features, the prespecified lane-restricted readout lowers original dx MSE by29.78%, all four motion-coordinate skills improve. This changes access and kernel geometry, not the encoder.',
  provisional='Under current Shapley convention, lane improves own-target response and reduces other-target response; yellow response does not consistently improve. Own-response residual has larger RMS than either nuisance component but covariances prevent treating norms as MSE percentages.',
  unresolved='Batch4 original versus batch12 intermediates can confound factorial mechanism attribution; uniform batch4 reevaluation is pending. Both models retain shrinkage, wrong directions, yellow response and substantial own-target errors.',
  pooling='About9.98% of raw feature-change energy survives spatial mean projection; this is not a semantic-motion-information fraction or proof pooling causes errors.',
  bf16='Final pooled BF16 rounding has very small effect on original scores; this does not test full FP32 encoder inference.'),
 no_refit=True,no_new_images=True,test_opened=False,p0_authorized=False)
(BASE/'independent_interpretation.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
lines=['# V5.5 独立解释（原始比较稳定；干预机制暂定）','',
'本分析只读取已经校验的正式报告及预测。原始 512 个 Dev 片段已经查看过，区间是给定固定模型的描述性配对区间，不是独立确认。中间六种干预图使用 batch12，而原始端点使用 batch4；全量统一 batch4 复算正在进行，因此下述对象机制解释均标为 provisional。','',
'## 已稳定复现的读出比较','',
'同一编码器特征下，固定上下半区读出将原始水平 MSE 从 0.0001184554 降至 0.0000831771（下降 29.78%）；global−lane 差值 0.0000352782，整世界配对 95% CI [0.0000283909, 0.0000423095]。运动 skill 0.4340→0.5799；至少一个水平方向错误 196→139/512。四个运动坐标均改善，预测幅度仍偏小。新 GPU 的原始端点与先前 CPU 独立预测相符（最大绝对差 '+str(max(original_local_parity.values()))+'）。','',
'## 暂定模式：自身响应改善与目标间串扰减少并存','',
'下表是 Shapley 约定下的响应幅度；数字不是误差责任占比。像素单位是归一化坐标乘224，不是离散光栅中心的整数位移。','',
'| 指标（A_dx / B_dx） | global W | lane W |','|---|---:|---:|']
for label,key in [('自身响应对真值的过原点描述性增益','own_gain_through_origin_descriptive'),('自身响应误差 RMS','own_error_rms'),('另一目标响应 RMS','other_target_rms'),('黄色响应 RMS','yellow_rms')]:
 vals=[]
 for name in ('W_concat','lane_W'):
  v=result[name][key];vals.append(f'{v[0]:.6f} / {v[2]:.6f}')
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','自身响应误差的 RMS 大于另外两个响应项；自身响应增益从约0.42上升到0.58–0.61，但仍远低于1。这支持“仍存在显著自身运动估计不足”，不能解释成纯粹串扰问题。另一方面，另一目标响应明显减小；黄色响应没有一致减少，因此不能称为黄色移除。自身响应中的方向错误 A_dx/B_dx 从54/64降到29/46，也仍未消失。','',
'为了避免把分摊幅度当误差占比，JSON 保存逐轴完整二阶矩阵：e = repeated-image + (own−truth) + other + yellow；MSE 等于所有对角项加两倍交叉项。交叉项可正可负，Shapley 本身还分摊交互项；这里不赋予神经模块因果责任。','',
'## 全八种条件','',
'| ABY | global dx MSE | lane dx MSE | global−lane 95% CI |','|---|---:|---:|---|']
for mask in [format(i,'03b') for i in range(8)]:
 g=result['W_concat']['masks'][mask]['dx_mse'];l=result['lane_W']['masks'][mask]['dx_mse'];ci=r['global_vs_lane'][mask]['paired_95ci']
 lines.append(f'| {mask} | {g:.8f} | {l:.8f} | [{ci[0]:.8f}, {ci[1]:.8f}] |')
lines+=['','000 的零响应是数值实现检查。001（只有黄色变化）没有改善，区间跨零。100/010 的自身目标变化条件改善；110（两个目标动、黄色静止）水平误差下降。不能由此忽略 batch 形状控制尚未完成的限制，也不能对所有条件逐个做新的通过门槛。','',
'## 原先三个坏例仍揭示剩余错误','',
'| pair/variant | 真值 A/B dx（像素） | global A/B | lane A/B |','|---|---|---|---|']
for pair in case_ids:
 g=next(c for c in case_table if c['pair_id']==pair and c['reader']=='W_concat');l=next(c for c in case_table if c['pair_id']==pair and c['reader']=='lane_W')
 fmt=lambda x:' / '.join(f'{v:+.3f}' for v in x)
 lines.append(f"| {pair}/1 | {fmt(g['true_dx_pixels'])} | {fmt(g['predicted_dx_pixels'])} | {fmt(l['predicted_dx_pixels'])} |")
lines+=['','844/1：两个方向变正确，但预测接近零，幅度仍严重不足。761/1：两个方向仍错，A 还稍变差；此分摊下自身正向响应太弱，其他目标/黄色负响应可压过它。891/1：A 接近真值，B 仍方向错；自身 B 响应很弱，其他目标/黄色负响应依然存在。这是预先保留的坏例说明，不代表总体因果比例。','',
'## 池化与精度边界','',
'同一 raw 张量独立平均池化和保存的 prequant 最大差2.414e-6、RMSE5.258e-8；BF16 转存逐位一致。4×4空间平均投影保留原始两帧差分能量的均值9.9758%（范围5.5755%–15.1208%），不能称为“只剩10%运动信息”。是否被移除的特征能量包含可用任务信号，仍需匹配解码实验。','',
'仅去掉最终 pooled BF16 转存，原 global 运动 skill 为0.4339805，对比canonical的0.4339992，差约−1.87e-5；运动预测变化RMS约6.6e-5–7.0e-5（约0.015像素）。这排除不了 BF16 编码器前向误差，只说明最后一次 pooled 存储舍入不是当前大误差的主要数值来源。','',
'没有新训练、图像、GPU、Test、P0 或调参。后续统一 batch4 校验应单独归档；本文件保留初始批次条件与 provisional 标记。']
(BASE/'independent_interpretation.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(original_local_parity=original_local_parity,outputs=['independent_interpretation.json','independent_interpretation.md']),indent=2))
