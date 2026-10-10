#!/usr/bin/env python3
"""Read-only compact batch-match validation and fixed-case interpretation."""
from pathlib import Path
import hashlib,json
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
monitor=BASE/'latest-batch-monitor.txt';raw=json.loads(monitor.read_text());c=raw['comparison/report.json']
r=json.loads((BASE/'report-snapshot/report/report.json').read_text())
prior=json.loads((BASE/'independent_interpretation.json').read_text())
receipt=json.loads((BASE/'report-snapshot/counterfactual_cache/receipt.json').read_text())
if (c['schema']!='v5.5-batch-match-1' or c['status']!='complete_cpu_frozen_comparison' or
    c['groups']!=256 or c['episodes']!=512 or c['masks']!=8 or
    c['new_fit'] is not False or c['test_opened'] is not False or c['no_case_selection'] is not True):raise ValueError('Population/status mismatch')
masks={format(m,'03b') for m in range(8)}
def zero(summary,count):
 if summary['count']!=count or summary['exact_zero_count']!=count:raise ValueError('Non-exact coordinate count')
 if any(summary[k]!=0 for k in ('mean','rms','max_abs')) or any(v!=0 for v in summary['absolute_quantiles'].values()):raise ValueError('Nonzero difference')
if set(c['feature_differences'])!=masks or set(c['prediction_differences'])!={'N_concat','W_concat','lane_W'}:raise ValueError('Missing mask or reader')
for m in masks:zero(c['feature_differences'][m],512*20480)
for name,rows in c['prediction_differences'].items():
 if set(rows)!=masks:raise ValueError('Missing prediction mask')
 for m in masks:
  zero(rows[m]['all_outputs'],512*8)
  if len(rows[m]['outputs'])!=8:raise ValueError('Missing output axis')
  for x in rows[m]['outputs']:zero(x,512)
if c['readers']['first']!=c['readers']['batch4']:raise ValueError('Derived metrics changed')
pro=c['provenance']
if pro['first_cache_sha256']!=receipt['cache_sha256'] or pro['source_sha256']!=sha(ROOT/'scripts/v5_5_batch_match.py'):raise ValueError('Source/cache identity mismatch')
if pro['lane_reader_sha256']!=r['provenance']['lane_model_receipt']['receipt_sha256']:raise ValueError('Lane identity mismatch')
if pro['frozen_reader_sha256']!=r['provenance']['lane_model_receipt']['original_frozen_reader_sha256']:raise ValueError('Global identity mismatch')
with np.load(BASE/'report-snapshot/report/predictions.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
if sha(BASE/'report-snapshot/report/predictions.npz')!=prior['provenance']['predictions_sha256']:raise ValueError('Original predictions changed')
examples=[]
for pair in (891,844):
 i=pair-640;v=1
 examples.append(dict(pair_id=pair,variant=v,units='normalized coordinate times224; not integer raster displacement',
  masks={format(m,'03b'):dict(true_dx_pixels=(a['targets'][i,v,m,[4,6]]*224).tolist(),
      global_dx_pixels=(a['W_concat'][i,v,m,[4,6]]*224).tolist(),lane_dx_pixels=(a['lane_W'][i,v,m,[4,6]]*224).tolist()) for m in range(8)}))
paragraph=('这轮可以定位到一个可改进的下游读出环节：编码器及其输出保持不变，仅采用预先固定的上下半区特征访问，'
 '原始512个片段的水平MSE下降29.78%，运动skill从0.4340升至0.5799，至少一个水平方向错误从196降至139。'
 '这是一项同时改变特征访问、维度、核几何及对应正则化几何的组合干预，不能把收益单独归因于目标分离，也不能据此宣判编码器正常或故障。'
 '全量统一batch4重编码后，8种条件的canonical特征和三个固定读出器的预测均逐项相同，因此本次观察到的对象干预响应不由原batch4/batch12差异造成。'
 '在固定Shapley分摊下，局部读出增强自身响应、减小另一目标响应，但黄色响应没有一致下降；自身运动响应不足和方向错误仍与串扰并存。'
 '这些是真实有限图像干预对整个编码→池化→读出链的响应，Shapley响应幅度不是模块责任或MSE占比；结论仍局限于已查看Dev、一个冻结FIT和当前渲染干预，旧gate及P0状态不变。')
result=dict(status='final_for_this_registered_diagnostic_after_zero_batch_shape_difference',
 evidence_scope='Independent validation of complete compact comparison report and original archived prediction arrays; new raw feature files were not independently downloaded by this analysis',
 batch_audit=dict(feature_coordinates=8*512*20480,prediction_outputs=3*8*512*8,all_exactly_zero=True,
   derived_metrics_identical=True,workers_complete=sorted(w['worker'] for w in raw['workers'] if w['status']=='complete' and w['completed_pairs']==32),
   actual_gpu_step='415453.42',actual_cpu_step='415453.43',charged_gpuh=raw['ledger/budget.json']['charged_gpuh'],provenance=pro),
 conclusion_zh=paragraph,examples=examples,original_comparison=r['global_vs_lane']['111'],
 prior_mechanism_summary={k:{name:prior['readers'][k][name] for name in ('own_gain_through_origin_descriptive','own_error_rms','other_target_rms','yellow_rms')} for k in ('W_concat','lane_W')},
 input_sha256={str(p.relative_to(BASE)):sha(p) for p in (monitor,BASE/'independent_interpretation.json',BASE/'report-snapshot/report/predictions.npz')},
 analysis_script_sha256=sha(__file__),test_opened=False,p0_authorized=False,no_new_fit=True,no_new_images=True)
for step in ('415453.42','415453.43'):
 if not any(line.startswith(step+'|') and '|COMPLETED|0:0|' in line for line in raw['accounting']):raise ValueError('Completed step evidence missing')
if result['batch_audit']['workers_complete']!=list(range(8)):raise ValueError('Incomplete workers')
(BASE/'final_interpretation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
lines=['# V5.5 最终机制解释','',paragraph,'',
'## 全量 batch4 核对','',
'逐条件检查8×512×20480＝83,886,080个canonical特征坐标；逐模型、条件、输出检查3×8×512×8＝98,304个预测值：zero count等于总数，mean/RMS/max/所有分位数均为0。first和batch4派生统计字典完全一致。八个worker各完成32个world；GPU415453.42、CPU415453.43均COMPLETED/0:0。','',
'本次独立核验读取完整的紧凑比较报告和原始归档预测，并核对来源哈希；它没有另行下载补充实验的全部特征文件。该范围与全量归档回读验证相区分。','',
'## 两个具体反事实例子','',
'下表ABY表示第二帧中哪些对象移至原t7位置，其余留在t6；第一帧固定。只改变A时，B真位移为0；只改变B时，A真位移为0。单位为连续坐标×224像素。','',
'| pair/variant | 条件 | 真值A/B dx | global预测A/B | lane预测A/B |','|---|---|---|---|---|']
for case in examples:
 for mask in ('100','010','001','111'):
  item=case['masks'][mask];fmt=lambda x:' / '.join(f'{v:+.3f}' for v in x)
  lines.append(f"| {case['pair_id']}/1 | {mask} | {fmt(item['true_dx_pixels'])} | {fmt(item['global_dx_pixels'])} | {fmt(item['lane_dx_pixels'])} |")
lines+=['',
'891/1：B应向右3.348像素。只让B动时，global已向左预测0.990，lane改为向右0.303，方向改善但远低于真值。只让A动时，静止B仍被预测向左1.926/global、0.832/lane；只动黄色也诱发B向左1.087/global、0.920/lane。全部移动后B仍向左：global−4.009、lane−1.669。这同时展示自身响应不足和非自身对象驱动，而非单一“读错方向”或单一黄色原因。','',
'844/1：只让A向左3.278像素，global却预测A向右1.837，并让实际静止的B向右4.166；lane分别为+0.111/+1.056，串扰和错误自身响应都减弱但未消失。全部移动时，global两个方向都错（+1.707/+4.439）；lane方向变正确（−0.229/−0.444），但对真值−3.278/−3.690仍严重低估。不能把方向修正等同于完整修复。','',
'## 仍然不能推出的结论','',
'两个单对象效应及其组合可能非线性，不能把表中条件直接相加代替完整预测；Shapley只是按固定规则分摊交互。特征上下半区仍经过全局注意力，而且黄色跨越边界。固定alpha不意味着屏蔽前后有相同的有效正则化几何。因而本轮支持“下游读出有可改善瓶颈”，不支持“编码器没问题”“池化是唯一原因”或“串扰占误差多少百分比”。','',
'最终pooled BF16存储舍入对原始skill影响约−1.87e-5；空间均值投影保留约9.98%的raw差分能量。前者不是全FP32编码器比较，后者不是运动信息保留率。旧失败gate保持，未开Test/P0；所有文件保留已知Dev诊断属性。']
(BASE/'final_interpretation.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status=result['status'],feature_coordinates=result['batch_audit']['feature_coordinates'],prediction_outputs=result['batch_audit']['prediction_outputs'],files=['final_interpretation.md','final_interpretation.json']),ensure_ascii=False))
