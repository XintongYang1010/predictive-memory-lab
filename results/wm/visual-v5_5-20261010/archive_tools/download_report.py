import json
from pathlib import Path
import register_deploy as h
h.helper.REMOTE=h.REMOTE
paths=['report','counterfactual_cache/receipt.json','counterfactual_cache/manifest.json','extraction_controller.json','owned_step.json','ledger/budget.json','source_manifest.json','registration_publication.json','gpu_selection','logs','score_process.json']
records=h.helper.download_snapshot(paths,h.HERE/'report-snapshot.tar.gz',h.HERE/'report-snapshot')
h.save(h.HERE/'report-snapshot-verification.json',dict(files=records,all_downloaded_files_verified=True))
print(json.dumps(dict(files=len(records),directory=str(h.HERE/'report-snapshot'))))
