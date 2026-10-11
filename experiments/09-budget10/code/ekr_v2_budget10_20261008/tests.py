import unittest
import copy
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from core import IncrementRouter, AnchorModel


def bank():
    g = torch.Generator().manual_seed(77)
    units = F.normalize(torch.randn(6, 4, 300, generator=g), dim=-1)
    names = units[:, 0].clone()
    shuffled = units.clone()
    shuffled[:, 1:] = units[torch.tensor([1, 2, 3, 4, 5, 0]), 1:]
    mask = torch.ones(6, 4, dtype=torch.bool)
    return dict(a=dict(unit=names,units=units,mask=mask,shuffle_units=shuffled,shuffle_mask=mask.clone()),
                o=dict(unit=F.normalize(torch.randn(3, 300, generator=g),dim=-1)))


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.bank = bank()
        self.x = torch.randn(8, 300)
        self.ai = torch.tensor([0, 0, 1, 2, 3, 4, 5])
        self.oi = torch.tensor([0, 2, 1, 2, 0, 1, 2])

    def test_equal_capacity_and_initial_state(self):
        states = []
        for v in ('V_name_align','V_clean_align','V_shuffle_align','V_clean_fixed'):
            torch.manual_seed(31)
            m = IncrementRouter(self.bank,v)
            expected = 28834 if v.endswith('_fixed') else 48034
            self.assertEqual(sum(p.numel() for p in m.parameters()),48034)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),expected)
            states.append(m.state_dict())
        for state in states[1:]:
            self.assertTrue(all(torch.equal(state[k],states[0][k]) for k in state))

    def test_zero_gain_exact_off_and_identity_value(self):
        for v in ('V_name_align','V_clean_align','V_shuffle_align','V_clean_fixed'):
            m = IncrementRouter(self.bank,v)
            self.assertTrue(torch.equal(m(self.x,self.ai,self.oi),torch.zeros(8,7)))
            self.assertTrue(torch.allclose(m.projected_values(),m.units/2,atol=1e-7))
            with torch.no_grad():m.raw_gain.fill_(0.7)
            self.assertTrue(torch.equal(m(self.x,self.ai,self.oi,force_off=True),torch.zeros(8,7)))

    def test_name_control_nonzero_and_query_gradient(self):
        m = IncrementRouter(self.bank,'V_name_align')
        with torch.no_grad():m.raw_gain.fill_(0.5)
        y = m(self.x,self.ai,self.oi)
        self.assertGreater(float(y.abs().max()),0)
        y.square().sum().backward()
        for p in (m.qx.weight,m.qo.weight,m.pk.weight,m.null_key,m.value_up.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.abs().sum()),0)

    def test_candidate_order_and_chunk_invariance(self):
        m = IncrementRouter(self.bank,'V_clean_align',object_chunk=1)
        with torch.no_grad():m.raw_gain.fill_(0.6)
        expected = m(self.x,self.ai,self.oi)
        order = torch.tensor([6,3,0,5,2,1,4])
        self.assertTrue(torch.allclose(m(self.x,self.ai[order],self.oi[order]),expected[:,order],atol=1e-7))
        m.object_chunk=16
        self.assertTrue(torch.allclose(m(self.x,self.ai,self.oi),expected,atol=1e-7))

    def test_raw_name_slot_unchanged_by_description_change(self):
        other = bank()
        other['a']['units'][:,1:] = other['a']['units'].roll(1,0)[:,1:]
        torch.manual_seed(3); first = IncrementRouter(self.bank,'V_clean_align')
        torch.manual_seed(3); second = IncrementRouter(other,'V_clean_align')
        self.assertTrue(torch.equal(first.units[:,0],second.units[:,0]))
        self.assertTrue(torch.equal(first.projected_values()[:,0],second.projected_values()[:,0]))
        with torch.no_grad():first.raw_gain.fill_(.5);second.raw_gain.fill_(.5)
        self.assertFalse(torch.allclose(first(self.x,self.ai,self.oi),second(self.x,self.ai,self.oi)))

    def test_bank_and_name_control_isolation(self):
        before=self.bank['a']['units'].clone()
        m=IncrementRouter(self.bank,'V_clean_align')
        with torch.no_grad():m.units.add_(.1)
        self.assertTrue(torch.equal(self.bank['a']['units'],before))
        other=bank();other['a']['units'][:,1:]=0
        torch.manual_seed(5);a=IncrementRouter(self.bank,'V_name_align')
        torch.manual_seed(5);b=IncrementRouter(other,'V_name_align')
        with torch.no_grad():a.raw_gain.fill_(.5);b.raw_gain.fill_(.5)
        self.assertTrue(torch.equal(a(self.x,self.ai,self.oi),b(self.x,self.ai,self.oi)))

    def test_zero_init_gradient_stages(self):
        m=IncrementRouter(self.bank,'V_clean_align')
        opt=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=0)
        target=torch.tensor([0,1,2,3,4,5,6,0])
        base=torch.randn(8,7)*.1
        opt.zero_grad();F.cross_entropy((base+m(self.x,self.ai,self.oi))/.05,target).backward()
        self.assertGreater(float(m.raw_gain.grad.abs()),0)
        self.assertEqual(float(m.qx.weight.grad.abs().sum()),0)
        opt.step()
        opt.zero_grad();F.cross_entropy((base+m(self.x,self.ai,self.oi))/.05,target).backward()
        self.assertGreater(float(m.qx.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.value_up.weight.grad.abs().sum()),0)
        self.assertEqual(float(m.value_down.weight.grad.abs().sum()),0)
        opt.step()
        opt.zero_grad();F.cross_entropy((base+m(self.x,self.ai,self.oi))/.05,target).backward()
        self.assertGreater(float(m.value_down.weight.grad.abs().sum()),0)

    def test_value_bound(self):
        m=IncrementRouter(self.bank,'V_clean_align')
        with torch.no_grad():m.value_up.weight.normal_();m.raw_gain.fill_(3)
        v=m.projected_values()
        self.assertTrue(torch.allclose(v.norm(dim=-1),torch.full((6,4),.5),atol=1e-6))
        self.assertLessEqual(float(m(self.x,self.ai,self.oi).abs().max()),.05+1e-6)

    def test_anchor_is_frozen_and_off_preserves_name_score(self):
        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__();self.feature=torch.nn.Linear(300,300,bias=False)
            def raw_forward(self,batch):
                feature=self.feature(batch['x'])
                return dict(attr_feature=feature,pred=feature[:,:7])
        holder=torch.nn.Module();holder.base=Base()
        holder.router=IncrementRouter(self.bank,'V_name_align');holder.bank=self.bank
        with torch.no_grad():holder.router.raw_gain.fill_(.4)
        increment=IncrementRouter(self.bank,'V_clean_align')
        model=AnchorModel(holder,increment);model.train(True)
        self.assertFalse(model.base.training);self.assertFalse(model.name_anchor.training)
        batch=dict(x=self.x)
        anchor,feature=model.components(batch,self.ai,self.oi)
        self.assertTrue(torch.equal(model.score(batch,self.ai,self.oi),anchor))
        with torch.no_grad():increment.raw_gain.fill_(.5)
        self.assertTrue(torch.equal(model.score(batch,self.ai,self.oi,force_off=True),anchor))
        model.score(batch,self.ai,self.oi).square().sum().backward()
        self.assertTrue(all(p.grad is None for module in (model.base,model.name_anchor) for p in module.parameters()))

    def test_gate_needs_name_and_shuffle_comparisons(self):
        from pilot import decision
        def record(auc):
            return dict(cw=dict(metrics_raw=dict(AUC=auc,best_hm=.55)),
                        ow=dict(metrics_raw=dict(AUC=.30,best_unseen=.56)))
        baseline=record(.43)
        records=dict(V_clean_align=record(.435),V_name_align=record(.431),
                     V_shuffle_align=record(.432),V_clean_fixed=record(.434))
        result=decision(records,baseline)
        self.assertTrue(result['proceed']);self.assertFalse(result['old_pilot_proceed'])
        records['V_shuffle_align']=record(.434)
        self.assertFalse(decision(records,baseline)['proceed'])
        records['V_shuffle_align']=record(.432);records['V_name_align']=record(.434)
        self.assertFalse(decision(records,baseline)['proceed'])

    def test_guard_blocks_despite_auc_gain(self):
        from pilot import decision
        def record(auc):
            return dict(cw=dict(metrics_raw=dict(AUC=auc,best_hm=.55)),
                        ow=dict(metrics_raw=dict(AUC=.30,best_unseen=.56)))
        baseline=record(.43)
        records={v:record(.43) for v in ('V_name_align','V_clean_align','V_shuffle_align','V_clean_fixed')}
        records['V_clean_align']=record(.435)
        records['V_clean_align']['ow']['metrics_raw']['AUC']=.29
        self.assertFalse(decision(records,baseline)['proceed'])

    def test_atomic_checkpoint_and_rng_resume(self):
        from pilot import rng_state,restore_rng,save_checkpoint
        m=IncrementRouter(self.bank,'V_clean_align')
        opt=torch.optim.AdamW(m.parameters(),lr=.005)
        generator=torch.Generator().manual_seed(99)
        def step():
            x=torch.randn(8,300)
            dummy=float(np.random.random())+random.random()+float(torch.rand((),generator=generator))
            target=torch.arange(8)%7
            opt.zero_grad();loss=F.cross_entropy(m(x,self.ai,self.oi)/.05,target)
            loss.backward();opt.step()
            return float(loss),dummy
        step()
        snapshot=dict(state=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(generator))
        expected=step();state=copy.deepcopy(m.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pth'
            save_checkpoint(path,snapshot)
            self.assertFalse(path.with_name(path.name+'.tmp').exists())
            restored=torch.load(str(path),map_location='cpu')
        m.load_state_dict(restored['state']);opt.load_state_dict(restored['optimizer'])
        restore_rng(restored['rng'],generator)
        self.assertEqual(step(),expected)
        self.assertTrue(all(torch.equal(m.state_dict()[k],v) for k,v in state.items()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
