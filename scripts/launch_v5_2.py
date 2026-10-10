#!/usr/bin/env python3
"""One bounded diagnostic in the existing allocation; never starts P0."""
import argparse
import json
from pathlib import Path
import time
from launch_v5_1 import Controller, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'python', 'assets', 'prior-scene'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('source-commit', 'allocation-job', 'account', 'partition', 'qos'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    a.action = 'diagnostic'
    c = Controller(a)
    try:
        publication=json.loads((a.root/'registration_publication.json').read_text())
        protocol=c.code/'docs/V5_2_CONTROLLED_DIAGNOSTIC_DRAFT.md'
        if (publication.get('verified_readback') is not True or
                publication.get('source_commit')!=a.source_commit or
                publication.get('protocol_sha256')!=sha(protocol)):
            raise ValueError('Verified GitHub protocol/code publication required before extraction')
        command = [str(a.python), 'scripts/extract_v5_2_counterfactual.py',
            '--out', str(a.root/'counterfactual_cache'), '--prior-scene', str(a.prior_scene),
            '--ijepa', str(a.assets/'ijepa'),
            '--checkpoint', str(a.assets/'IN1K-vit.h.14-300e.pth.tar'),
            '--source-commit', a.source_commit, '--budget-root', str(a.root/'ledger'),
            '--allocation-job', a.allocation_job, '--account', a.account,
            '--partition', a.partition, '--qos', a.qos,
            '--workers', '8', '--lease-hours', '.20', '--allow-gpu-sharing']
        c.execute('counterfactual_v5_2', command, gpu_count=8, minutes=12)
        c.execute('counterfactual_score_v5_2', [str(a.python), 'scripts/score_v5_2_counterfactual.py',
            '--cache', str(a.root/'counterfactual_cache/cache.npz'),
            '--receipt', str(a.root/'counterfactual_cache/receipt.json'),
            '--model', str(a.prior_scene/'reader'),
            '--prior-predictions', str(a.prior_scene/'bridge_report/predictions.npz'),
            '--out', str(a.root/'report')], gpu_count=0)
        c.state.update(status='diagnostic_complete', p0_authorized=False, end_unix=time.time())
    except BaseException as exc:
        c.state.update(status='execution_failed', reason=str(exc), end_unix=time.time())
        save(c.status_path, c.state)
        raise
    save(c.status_path, c.state)


if __name__ == '__main__':
    main()
