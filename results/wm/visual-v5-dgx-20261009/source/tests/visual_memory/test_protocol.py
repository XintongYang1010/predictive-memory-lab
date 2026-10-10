import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from pmr.visual_memory.data import episode,specs,pooled_masks
from pmr.visual_memory.memory import Writer,Reader,MemoryBank,persistent_cost,matched_fifo_frames,VARIANTS
from pmr.visual_memory.predictor import Predictor,Candidate,batch
from pmr.visual_memory.common import Budget,BudgetExceeded
from pmr.visual_memory.train import optimize,evaluate,search_and_train

torch.set_num_threads(2)


def fake_records(count=8,split="smoke"):
    # Tiny RGB-derived deterministic features for SOFTWARE tests only.
    # They never stand in for pretrained features in the production CLI.
    rows=[]
    for sp,p,v in specs(split,count):
        e=episode(sp,p,v,size=32,gap=8,writes=0)
        def encode(rgb):
            x=torch.from_numpy(rgb.copy()).permute(0,3,1,2).float()/255
            z=torch.nn.functional.adaptive_avg_pool2d(x,2).flatten(2).transpose(1,2)
            return torch.cat([z,z.square()],-1)
        rows.append(dict(features=encode(e.rgb),targets=encode(e.target_rgb),times=torch.from_numpy(e.times),
                         shutters=torch.from_numpy(e.shutters),target_masks=torch.from_numpy(pooled_masks(e.target_masks,2)).float(),
                         metadata=e.metadata))
    return rows


class Unlimited:
    def check(self):pass


class Protocol(unittest.TestCase):
    def test_pairs_and_splits(self):
        for family in ("hold","interference","update"):
            a,b=[episode("train",0,v,family=family) for v in (0,1)]
            np.testing.assert_array_equal(a.rgb[-4:],b.rgb[-4:])
            np.testing.assert_array_equal(a.times,b.times)
            np.testing.assert_array_equal(a.shutters,b.shutters)
            self.assertGreater(np.abs(a.target_rgb.astype(float)-b.target_rgb).sum(),0)
        self.assertNotEqual(episode("train",0,0).metadata["pair_id"],episode("dev",0,0).metadata["pair_id"])
        with self.assertRaises(ValueError):specs("train",3)

    def test_gap_and_write_axes(self):
        a=episode("train",0,0,gap=8,writes=8)
        b=episode("train",0,0,gap=32,writes=8)
        self.assertEqual(len(a.times),len(b.times))
        c=episode("train",0,0,gap=8,writes=32)
        self.assertEqual(a.times[-1],c.times[-1])
        self.assertEqual(len(c.times)-len(a.times),24)

    def test_delta_revises_one_orthogonal_address(self):
        w=Writer("delta",1,2)
        with torch.no_grad():w.gate.bias.fill_(100)
        m=w.reset(1,"cpu")
        k1=torch.tensor([[[1.,0.]]]); k2=torch.tensor([[[0.,1.]]])
        m=w.write(m,k1,torch.tensor([[[2.,3.]]]))
        m=w.write(m,k2,torch.tensor([[[4.,5.]]]))
        m=w.write(m,k1,torch.tensor([[[6.,7.]]]))
        torch.testing.assert_close(m,torch.tensor([[[[6.,7.],[4.,5.]]]]))

    def test_delta_formula_nontrivial_decay(self):
        torch.manual_seed(4);w=Writer("delta",2,3)
        m=torch.randn(2,2,3,3);k=torch.randn(2,2,3);v=torch.randn(2,2,3)
        kn=torch.nn.functional.normalize(k,dim=-1)
        expected=.975*m+.5*kn.unsqueeze(-1)*(v-torch.einsum("bhd,bhde->bhe",kn,.975*m)).unsqueeze(-2)
        torch.testing.assert_close(w.write(m,k,v),expected)

    def test_skip_reset_chronological_and_size(self):
        k=torch.randn(2,2,11,3);v=torch.randn_like(k);valid=torch.ones(2,11,dtype=torch.bool)
        for rule in ("add","gated_add","delta"):
            w=Writer(rule,2,3)
            empty=w.reset(2,"cpu");m=w.replay(k,v,valid)
            state=empty
            for t in range(11):state=w.write(state,k[:,:,t],v[:,:,t])
            torch.testing.assert_close(m,state)
            self.assertEqual(m.numel(),empty.numel())
            torch.testing.assert_close(w.replay(k,v,torch.zeros_like(valid)),empty)
            torch.testing.assert_close(w.reset(2,"cpu"),empty)
        w=Writer("delta",2,3)
        self.assertFalse(torch.allclose(w.replay(k,v,valid),w.replay(k.flip(2),v.flip(2),valid)))

    def test_reader_empty_and_identity(self):
        q=torch.randn(2,4,9,8);m=torch.randn(2,4,8,8)
        a,b=Reader(8,False),Reader(8,True)
        torch.testing.assert_close(a.read(q,m),b.read(q,m))
        self.assertTrue(torch.isfinite(b.read(q,torch.zeros_like(m))).all())

    def test_actor_does_not_receive_evaluation_fields(self):
        rows=fake_records(4);actor,target,_=batch(rows,"cpu")
        model=Predictor(6,4,16,2,1).eval()
        first=model(actor)
        changed=copy.deepcopy(rows)
        for row in changed:
            row["targets"]*=99;row["metadata"]["variant"]=999;row["target_masks"]*=0
        actor2,_,_=batch(changed,"cpu")
        torch.testing.assert_close(first,model(actor2))

    def test_all_candidates_gradients_and_frozen_writer(self):
        rows=fake_records(4)
        # Ensure some real observations fall out of E+R, not only padded values.
        for row in rows:
            for key in ("features","times","shutters"):
                row[key]=torch.cat([row[key][:8],row[key]],0)
        a,y,_=batch(rows,"cpu")
        base=Predictor(6,4,16,2,1)
        for variant in VARIANTS:
            torch.manual_seed(42);m=Candidate(copy.deepcopy(base),variant)
            pred,state=m(a,return_state=True)
            self.assertTrue(torch.isfinite(pred).all())
            self.assertEqual(state.shape,(4,2,8,8))
            loss=(pred-y).square().mean();loss.backward()
            self.assertIsNotNone(m.memory_scale.grad)
            self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))
            if hasattr(m.writer,"gate"):self.assertGreater(float(m.writer.gate.weight.grad.abs().sum()),0)
            if m.reader.calibrated:self.assertGreater(float(m.reader.up.weight.grad.abs().sum()),0)
            with torch.no_grad():
                torch.testing.assert_close(m(a,fixed_state=state),m(a))
        c=Candidate(copy.deepcopy(base),"C1");c.writer.requires_grad_(False)
        before={k:v.clone() for k,v in c.writer.state_dict().items()}
        state_before=c(a,return_state=True)[1].detach().clone()
        opt=torch.optim.Adam([p for p in c.parameters() if p.requires_grad],lr=.01)
        opt.zero_grad();c(a).square().mean().backward();opt.step()
        for k,v in before.items():torch.testing.assert_close(v,c.writer.state_dict()[k])
        torch.testing.assert_close(state_before,c(a,return_state=True)[1])

    def test_equal_cache_budget(self):
        self.assertEqual(matched_fifo_frames(),10)
        self.assertEqual(persistent_cost()["total_bytes"],persistent_cost(exact=10,memory=False)["total_bytes"])

    def test_streaming_fifo_writes_once_and_matches_replay(self):
        torch.manual_seed(55);writer=Writer("delta",2,4)
        bank=MemoryBank(writer,1,"cpu")
        frames=torch.randn(35,1,2,8)
        def project(frame):
            x=frame.reshape(1,2,2,4).transpose(1,2)
            return x,x*.3
        costs=[]
        with torch.no_grad():
            for frame in frames:
                bank.append(frame,project);costs.append(bank.bytes())
            self.assertEqual(len(set(costs[11:])),1)
            self.assertEqual(int(bank.control[1]),23)
            kv=[project(x) for x in frames[:23]]
            k=torch.cat([x[0] for x in kv],2).to(torch.bfloat16).float()
            v=torch.cat([x[1] for x in kv],2).to(torch.bfloat16).float()
            torch.testing.assert_close(bank.state,writer.replay(k,v,torch.ones(1,46,dtype=torch.bool)))
            expected=persistent_cost(heads=2,width=4,tokens=2)["total_bytes"]
            self.assertEqual(bank.bytes(),expected)
            bank.reset();self.assertEqual(len(bank.exact),0);self.assertEqual(int(bank.control[1]),0)

    def test_budget_resume_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Budget(tmp,"phase",1,.1)
            b.close("partial")
            data=json.loads((Path(tmp)/"budget.json").read_text())
            self.assertLess(data["charged_gpuh"],.1)
            with self.assertRaises(BudgetExceeded):Budget(tmp,"phase",1,1.1)

    def test_checkpoint_resume_and_tiny_training(self):
        rows=fake_records(4)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(9);m=Predictor(6,4,16,2,1)
            score=optimize(m,rows,rows,tmp,4,.001,"cpu",Unlimited(),4,eval_every=2,p0=True)
            expected={k:v.clone() for k,v in m.state_dict().items()}
            restored=Predictor(6,4,16,2,1)
            score2=optimize(restored,rows,rows,tmp,4,.001,"cpu",Unlimited(),4,eval_every=2,p0=True)
            self.assertEqual(score,score2)
            for k,v in expected.items():torch.testing.assert_close(v,restored.state_dict()[k])
            self.assertEqual(len(evaluate(restored,rows,"cpu",interventions=True)),4)

    def test_interrupted_optimizer_resume_matches_uninterrupted(self):
        class StopAfterTwo:
            count=0
            def check(self):
                self.count+=1
                if self.count>2:raise BudgetExceeded("simulated scheduler warning")
        rows=fake_records(4)
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(17);initial=Predictor(6,4,16,2,1)
            direct=copy.deepcopy(initial);partial=copy.deepcopy(initial)
            optimize(direct,rows,rows,Path(tmp)/"direct",4,.001,"cpu",Unlimited(),4,eval_every=2,p0=True)
            with self.assertRaises(BudgetExceeded):
                optimize(partial,rows,rows,Path(tmp)/"resume",4,.001,"cpu",StopAfterTwo(),4,eval_every=2,p0=True)
            ck=torch.load(Path(tmp)/"resume"/"last.pt",weights_only=True)
            self.assertEqual(ck["status"],"partial");self.assertEqual(ck["step"],2)
            restored=copy.deepcopy(initial)
            optimize(restored,rows,rows,Path(tmp)/"resume",4,.001,"cpu",Unlimited(),4,eval_every=2,p0=True)
            for k,v in direct.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)


if __name__=="__main__":unittest.main()
