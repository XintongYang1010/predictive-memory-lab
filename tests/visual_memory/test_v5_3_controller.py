import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from launch_v5_3 import check_budget


class ControllerBudgetTests(unittest.TestCase):
    def test_three_envelopes_fit_before_reserved_pilot(self):
        ledger=dict(charged_gpuh=2.2105,stages={},leases=[],controller_envelopes=[])
        for stage in ['v53_fit','v53_anchor','v53_bridge']:
            check_budget(ledger,stage)
            ledger['charged_gpuh']+=1.2;ledger['stages'][stage]=1.2
        self.assertLess(ledger['charged_gpuh'],6)
        with self.assertRaises(RuntimeError):check_budget(ledger,'v53_bridge')

    def test_rejects_unknown_over_budget_and_unreconciled(self):
        base=dict(charged_gpuh=2.21,stages={},leases=[],controller_envelopes=[])
        with self.assertRaises(ValueError):check_budget(base,'other')
        for patch in [dict(charged_gpuh=4.8001),dict(charged_gpuh=4.76),
                      dict(leases=[dict(status='reserved_or_interrupted')]),
                      dict(controller_envelopes=[dict(status='active')]),
                      dict(controller_envelopes=[dict(status='unreconciled_timeout')]),
                      dict(stages={'v53_fit':.1})]:
            value=copy.deepcopy(base);value.update(patch)
            with self.assertRaises(RuntimeError):check_budget(value,'v53_fit')


if __name__=='__main__':unittest.main()
