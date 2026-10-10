"""Curate already measured V5.4 evidence without changing scientific files."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
B=Path(__file__).resolve().parent;root=B.parents[1];R=root/'results/wm/visual-v5_4-20261010';R.mkdir(parents=True,exist_ok=True)
S=B/('diagnostic-20261010-v54' if a.final else 'report-snapshot')
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
for name in ['fit_controller.json','evaluation_controller.json','registration_publication.json','freeze_publication.json','ledger/budget.json','reader/freeze_receipt.json','reader/training_config.json','fit_cache/receipt.json','anchor_cache/receipt.json','bridge_cache/receipt.json','fit_extraction.json','eval_extraction.json','source_manifest.json','freshness_audit.json','creation.json']:
 copy(S/name,R/name)
for folder in ['anchor_report','bridge_report','fit_workers','eval_workers','logs']:
 for path in (S/folder).rglob('*'):
  if path.is_file():copy(path,R/folder/path.relative_to(S/folder))
for path in (B/'posthoc_report').rglob('*'):
 if path.is_file():copy(path,R/'analysis'/path.relative_to(B/'posthoc_report'))
for path in (B/'bad_cases').rglob('*'):
 if path.is_file():copy(path,R/'bad_cases'/path.relative_to(B/'bad_cases'))
for path in (B/'training-generalization').rglob('*'):
 if path.is_file():copy(path,R/'training_generalization'/path.relative_to(B/'training-generalization'))
for name in ['training-generalization-execution.json','training-generalization-numeric-verification.json','training-generalization.stdout.log','training-generalization.stderr.log']:
 if (B/name).exists():copy(B/name,R/'training_generalization'/name)
text=(B/'posthoc_report/README.md').read_text()
for path in (B/'posthoc_report').glob('*.png'):text=text.replace(']('+path.name+')','](analysis/'+path.name+')')
text+='\n逐坐标预注册分组显示覆盖取舍：A_dx域外MSE由2.0862e-4降至1.7522e-4（70世界），域内由9.6844e-5升至1.1627e-4（186世界）；B_dx域外1.4026e-4降至1.1570e-4（65世界），域内8.3189e-5升至1.0072e-4（191世界）。这些是描述性分组点估计，无分组显著性结论。本次固定256世界的扩域未确认总体水平收益，不能据此否定所有覆盖方案。网页端复核见 [完整讨论](web_result_review.md)。\n'
text+='\n按W水平误差排序的三个事后坏例与实际输入RGB哈希复核见 [逐例图像](bad_cases/README.md)；这是选择的坏例，不代表总体或因果结论。\n'
text+='\n完整冻结的两组 reader、共同特征基底和原始 FIT 收据见 [V5.4 FIT Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_4-fit-20261010)。\n'
text+='\n只读CPU复核显示N/W自身FIT运动分数为0.999493/0.999508，训练方向错误均为0，而fresh bridge仅0.353835/0.433999。全部65536个训练预测另经KRR正规方程核验，最大偏差1.73e-12；详见 [训练与新样本的差距](training_generalization/README.md)。这是事后诊断，不能单独定位编码器、池化、正则化或数据密度的原因。\n'
text+='\n硬件是既有 Slurm 分配415453中的8张 H800，与 IndexMem 共用。GPUh按选卡数量乘实际wall time计账，不换算为A100等效吞吐。原V5/V5.1门槛未更改，Test未打开，P0与memory训练未启动。\n'
if a.final:
 for name in ['final_accounting.json','slurm_accounting.txt','dependencies.json','archive_manifest.json']:
  copy(S/name,R/('remote_scene_manifest.json' if name=='archive_manifest.json' else name))
 copy(B/'local_snapshot_receipt.json',R/'release_asset.json')
 text+='\n完整现场、所有新缓存、模型、日志与复现资料见 [V5.4完整归档](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_4-20261010)。独立GitHub回读凭证位于 results/wm/visual-v5_4-20261010-verification.json。科学问题尚未全部解决，现场保留，未取消IndexMem或删除既有内容。\n'
(R/'README.md').write_text(text.rstrip()+'\n')
for name in ['web_design_review.md','web_result_review.md','profile_reuse.json','precommit-tests.log']:
 if (B/name).exists():copy(B/name,R/name)
for path in B.glob('*.py'):copy(path,R/'archive_tools'/path.name)
for path in (B/'software_fixture').rglob('*'):
 if path.is_file():copy(path,R/'software_fixture'/path.relative_to(B/'software_fixture'))
(R/'software_fixture/FINAL_SOURCE_NOTE.md').write_text('The full8192-view software fixture ran before final receipt provenance additions and protocol grouping wording. Fit/predict mathematics are unchanged, but its recorded source hashes differ from the final source. Final complete visual-memory suite:167passed. Random64D software features do not measure full20480D or GPU throughput. RuntimeWarnings are preserved; all finite checks passed.\n')
print(R)
