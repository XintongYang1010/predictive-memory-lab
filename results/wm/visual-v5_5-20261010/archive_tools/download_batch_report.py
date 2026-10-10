import json
import register_deploy as h
h.helper.REMOTE=h.REMOTE
paths=['batch_match_run/comparison','batch_match_run/cache/receipt.json','batch_match_run/cache/manifest.json','batch_match_run/extraction_controller.json','batch_match_run/owned_step.json','batch_match_run/ledger/budget.json','batch_match_run/gpu_selection','batch_match_run/logs','batch_match_run/creation.json']
records=h.helper.download_snapshot(paths,h.HERE/'batch-report-snapshot.tar.gz',h.HERE/'batch-report-snapshot')
h.save(h.HERE/'batch-report-snapshot-verification.json',dict(files=records,all_downloaded_files_verified=True))
print(json.dumps(dict(files=len(records))))
