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
text=(B/'posthoc_report/README.md').read_text()
for path in (B/'posthoc_report').glob('*.png'):text=text.replace(']('+path.name+')','](analysis/'+path.name+')')
text+='\n完整冻结的两组 reader、共同特征基底和原始 FIT 收据见 [V5.4 FIT Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_4-fit-20261010)。\n'
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
