"""Copy measured reports and provenance for review; no model or data mutation."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args()
B=Path('execution/visual-v5_3-20261010');R=Path('results/wm/visual-v5_3-20261010');R.mkdir(parents=True,exist_ok=True)
scene=B/('diagnostic-20261010-v53' if a.final else 'report-snapshot')
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
for name in ['fit_controller.json','evaluation_controller.json','registration_publication.json','freeze_publication.json','ledger/budget.json','reader/freeze_receipt.json','reader/training_config.json','fit_cache/receipt.json','anchor_cache/receipt.json','bridge_cache/receipt.json']:
 copy(scene/name,R/name)
for folder in ['anchor_report','bridge_report']:
 for path in (scene/folder).glob('*'):
  if path.is_file():copy(path,R/folder/path.name)
for path in (B/'posthoc_report').rglob('*'):
 if path.is_file():copy(path,R/'analysis'/path.relative_to(B/'posthoc_report'))
copy(B/'posthoc_report/README.md',R/'README.md')
text=(R/'README.md').read_text().replace('](bridge_motion_skills.png)','](analysis/bridge_motion_skills.png)').replace('](anchor_path_diagnostics.png)','](analysis/anchor_path_diagnostics.png)').replace('](negative_control_rms.png)','](analysis/negative_control_rms.png)')
text+='\n完整模型见 [FIT reader Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_3-fit-20261010)。硬件为既有分配中的8张H800，与IndexMem共用；GPUh按选卡数量乘实际wall time保守计账，不声称A100等效速度。\n'
(R/'README.md').write_text(text)
for name in ['build_results.py','download_reports.py','publish_frozen.py','register_and_deploy.py','remote_control.py','seal_and_download.py','verify_github.py','curate_results.py']:
 if (B/name).is_file():copy(B/name,R/'archive_tools'/name)
for name in ['README.md','run_software_fixture.py','invocation.json','run_001.log','run_001/result.json']:
 copy(B/'software_fixture'/name,R/'software_fixture'/name)
(R/'software_fixture/FINAL_SOURCE_NOTE.md').write_text('Full-size fixture exercised the same fitting mathematics before final receipt/report-only additions. Its recorded reader SHA differs from the frozen final source; it is not a byte-for-byte execution of that final file. The final nine reader pytest tests passed independently. Software fixture artifacts contain only software_test worlds and random 64D features.\n')
if a.final:
 for name in ['final_accounting.json','slurm_accounting.txt','dependencies.json','creation.json','freshness_audit.json','source_manifest.json']:
  copy(scene/name,R/name)
 copy(scene/'archive_manifest.json',R/'remote_scene_manifest.json')
 meta=json.loads((B/'local_snapshot_receipt.json').read_text());(R/'release_asset.json').write_text(json.dumps(meta,indent=2)+'\n')
 for path in (scene/'logs').glob('*'):
  if path.is_file():copy(path,R/'logs'/path.name)
 for folder in ['analysis','bad_cases']:
  if (scene/folder).exists():
   for path in (scene/folder).rglob('*'):
    if path.is_file():copy(path,R/folder/path.relative_to(scene/folder))
 for n in ['web_review.md']:
  if (scene/n).exists():copy(scene/n,R/n)
 text=(R/'README.md').read_text()+'\n完整现场、缓存、全部检查点和复现资料见 [V5.3 Release](https://github.com/XintongYang1010/predictive-memory-lab/releases/tag/visual-v5_3-20261010)。归档逐文件回读凭证会单独存入 results/wm/visual-v5_3-20261010-verification.json。未修复的科学问题及历史失败现场均保留，未删除WM已有目录或停止IndexMem。\n'
 (R/'README.md').write_text(text)
print(R)
