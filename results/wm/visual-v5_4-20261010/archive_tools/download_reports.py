#!/usr/bin/env python3
"""Download a checksummed V5.4 report snapshot after all five steps complete."""
import argparse
from helper_common import BASE, completed_gate, download_snapshot, frozen_source, save_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    paths = ['anchor_report','bridge_report','reader/freeze_receipt.json','reader/training_config.json',
        'fit_controller.json','evaluation_controller.json','registration_publication.json',
        'freeze_publication.json','freshness_audit.json','source_manifest.json','creation.json',
        'ledger/budget.json','ledger/inherited_budget.json','fit_extraction.json','eval_extraction.json',
        'fit_workers','eval_workers','logs']
    for role, ids in (('fit',range(256)),('anchor',range(256)),('bridge',range(640,896))):
        paths += [role+'_cache/receipt.json',role+'_cache/manifest.json']
        paths += [role+'_cache/shards/'+f'{i:06d}.json' for i in ids]
    if not a.execute:
        print('Preview: read-only download of two reports, three cache receipts/input records, shared '
              'fit_workers/eval_workers, controllers, ledger and logs after all five science steps pass. '
              'Feature NPZs and model basis remain in their separate full archives.')
        return
    identity = frozen_source()
    gate = completed_gate(['fit','evaluation'])
    if gate['source_commit'] != identity['source_commit']:
        raise ValueError('Completed controller/source freeze mismatch')
    records = download_snapshot(paths, BASE/'report-snapshot.tar.gz', BASE/'report-snapshot')
    save_new(BASE/'report-snapshot-verification.json', dict(status='all_snapshot_files_verified',
        source_commit=identity['source_commit'],steps=gate['steps'],files=records,
        note='Report/input-record snapshot only; the full sealed archive must retain all feature caches and frozen model.'))
    print(str(BASE/'report-snapshot'))


if __name__ == '__main__':
    main()
