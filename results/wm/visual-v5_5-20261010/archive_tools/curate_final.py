"""Curate reviewable results; large complete caches/models stay in Release."""
import json,shutil
from pathlib import Path
import register_deploy as h
dest=h.ROOT/'results/wm/visual-v5_5-20261010';dest.mkdir(parents=True,exist_ok=True)
for name in ['figures','lane_original_eval']:
 shutil.copytree(h.HERE/name,dest/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
for folder in ['report-snapshot','batch-report-snapshot']:
 shutil.copytree(h.HERE/folder,dest,dirs_exist_ok=True)
names=['REPRODUCE.md','independent_interpretation.md','independent_interpretation.json','final_interpretation.md','final_interpretation.json','web_review.md','figure_visual_review.json','prior-tests-summary.json','batch-report-snapshot-verification.json','report-snapshot-verification.json','batch_registration_publication.json','batch_deployment.json','batch_source_manifest.json','registration_publication.json']
for name in names:shutil.copy2(h.HERE/name,dest/name)
shutil.copy2(h.HERE/'FINAL_README.md',dest/'README.md')
tools=dest/'archive_tools';tools.mkdir(exist_ok=True)
for name in ['register_deploy.py','control.py','control_batch.py','register_batch.py','download_report.py','download_batch_report.py','make_figures.py','recompute_interpretation.py','finalize_interpretation.py','test_controller_lifecycle.py','seal_download.py','release_and_verify.py','cleanup_verified.py','upload_derived.py','curate_final.py']:
 shutil.copy2(h.HERE/name,tools/name)
if (h.HERE/'remote_seal_receipt.json').exists():
 scene=h.HERE/'diagnostic-20261010-v55'
 for src,target in [('archive_manifest.json','remote_scene_manifest.json'),('final_accounting.json','final_accounting.json')]:shutil.copy2(scene/src,dest/target)
 for name in ['derived_upload_receipt.json','remote_export.json','local_archive_verification.json']:shutil.copy2(h.HERE/name,dest/name)
print(json.dumps(dict(files=sum(p.is_file() for p in dest.rglob('*')),out=str(dest))))
