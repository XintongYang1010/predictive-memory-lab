import importlib.util,json,os,pathlib,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('controller',ROOT/'scripts/launch_v5_5.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Clock:
 def __init__(self):self.t=0.
 def monotonic(self):return self.t
 def time(self):return 1000+self.t
 def sleep(self,t):self.t+=t
class Process:
 def __init__(self,clock,exit_at=10,rc=0):self.c=clock;self.exit_at=exit_at;self.rc=rc;self.returncode=None
 def poll(self):
  if self.c.t>=self.exit_at:self.returncode=self.rc
  return self.returncode
 def terminate(self):self.exit_at=self.c.t;self.rc=-15;self.poll()
 def kill(self):self.exit_at=self.c.t;self.rc=-9;self.poll()
 def wait(self,timeout):return self.poll()
class Tests(unittest.TestCase):
 def fixture(self,kind):
  clock=Clock();proc=Process(clock,10 if kind=='normal' else 539 if kind=='overrun' else 10000)
  query_count=0;cancelled=[]
  def query(*_):
   nonlocal query_count
   query_count+=1
   if kind=='overrun':clock.t+=10
   terminal=kind in ('normal','overrun') or bool(cancelled)
   if terminal and kind=='timeout':proc.exit_at=clock.t;proc.rc=-15;proc.poll()
   return dict(known={'415453.40'},live={} if terminal else {'415453.40':'RUNNING'},
     terminal={'415453.40':dict(state='COMPLETED' if kind!='timeout' else 'CANCELLED',exit_code='0:0')} if terminal else {},live_query_ok=True,evidence={})
  def cancel(sid,allocation):cancelled.append(sid);return dict(step_id=sid)
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);(root/'ledger').mkdir();(root/'logs').mkdir()
   lease={};env={};entry=dict(stage='test');state=dict(jobs=[entry]);budget=dict(stages={})
   with patch.object(m,'time',clock),patch.object(m.subprocess,'Popen',return_value=proc),patch.object(m,'receipt_step',return_value='415453.40'),patch.object(m,'query_steps',side_effect=query),patch.object(m,'cancel_owned',side_effect=cancel):
    ok=m.monitor(['not-executed'],root,state,entry,budget,lease,env,'415453','nonce','unique')
   return ok,entry,lease,budget,cancelled
 def test_success_requires_terminal_and_accounts_query(self):
  ok,e,l,b,c=self.fixture('normal');self.assertTrue(ok);self.assertEqual(e['step_ids'],['415453.40']);self.assertTrue(l['exit_confirmed']);self.assertEqual(c,[])
 def test_timeout_cancels_only_owned_and_bills_until_confirmed(self):
  ok,e,l,b,c=self.fixture('timeout');self.assertFalse(ok);self.assertEqual(c,['415453.40']);self.assertGreater(e['charged_gpuh'],1.2);self.assertTrue(e['exit_confirmed']);self.assertEqual(l['status'],'closed')
 def test_zero_exit_over_envelope_is_failed(self):
  ok,e,l,b,c=self.fixture('overrun');self.assertFalse(ok);self.assertTrue(e['envelope_exceeded']);self.assertGreater(e['charged_gpuh'],1.2)
 def test_parent_and_foreign_cancel_refused(self):
  for sid in ['415453','415454.2','415453.batch','415453.2,415453.3','415453.*']:
   with self.assertRaises(ValueError):m.exact_step(sid,'415453')
 def test_unconfirmed_reservation_not_undercharged(self):
  self.assertEqual(m.charge_for(2,False),1.2);self.assertGreater(m.charge_for(600,False),1.2)
 def test_symlink_root_rejected_before_resolve(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d)/'real';root.mkdir();root=root.resolve();s=root.stat();(root/'creation.json').write_text(json.dumps(dict(root=str(root),inode=s.st_ino,device=s.st_dev)))
   link=pathlib.Path(d)/'link';link.symlink_to(root,target_is_directory=True)
   self.assertEqual(m.checked_root(root),root)
   with self.assertRaises(ValueError):m.checked_root(link)
 def test_exact_lane_inventory_guard(self):
  with tempfile.TemporaryDirectory() as d:
   root=pathlib.Path(d);f=root/'lane_reader';f.mkdir()
   for n in ('freeze_receipt.json','frozen_model.npz'):(f/n).write_text('fixture')
   pub=dict(lane_reader=dict(all_files_readback_verified=True,files={p.name:dict(bytes=p.stat().st_size,sha256=m.sha(p)) for p in f.iterdir()}))
   m.validate_lane_publication(root,pub);(f/'extra').write_text('bad')
   with self.assertRaises(ValueError):m.validate_lane_publication(root,pub)
if __name__=='__main__':unittest.main(verbosity=2)
