import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from launch_v5_4 import check_budget, VERSION_BASELINE


class ControllerBudgetTests(unittest.TestCase):
    def test_two_worst_case_envelopes_preserve_pilot(self):
        ledger=dict(charged_gpuh=VERSION_BASELINE,stages={},leases=[],controller_envelopes=[])
        for stage in ('v54_fit','v54_eval'):
            check_budget(ledger,stage)
            ledger['charged_gpuh']+=1.2;ledger['stages'][stage]=1.2
        self.assertLess(ledger['charged_gpuh'],6)
        with self.assertRaises(RuntimeError):check_budget(ledger,'v54_eval')

    def test_rejects_unreconciled_repeated_and_insufficient_budget(self):
        base=dict(charged_gpuh=VERSION_BASELINE,stages={},leases=[],controller_envelopes=[])
        with self.assertRaises(ValueError):check_budget(base,'v54_anchor')
        for patch in [dict(charged_gpuh=VERSION_BASELINE+1.200001),
                      dict(charged_gpuh=4.78),
                      dict(leases=[dict(status='reserved_or_interrupted')]),
                      dict(controller_envelopes=[dict(status='active')]),
                      dict(controller_envelopes=[dict(status='unreconciled_timeout')]),
                      dict(stages={'v54_fit':.01})]:
            value=copy.deepcopy(base);value.update(patch)
            with self.assertRaises(RuntimeError):check_budget(value,'v54_fit')


if __name__=='__main__':unittest.main()
