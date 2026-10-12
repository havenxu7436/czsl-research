"""CPU unit tests for support-mask construction and the fixed-value router."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from core import AnchorModel, IncrementRouter
from pilot import build_support_mask, decision


def bank_fixture():
    names=torch.eye(300)[:16]
    units=names[:,None].repeat(1,4,1)
    units[:,2]=-names
    shuffled=units.roll(1,0);shuffled[:,0]=units[:,0]
    return {'a':dict(unit=names,units=units,mask=torch.ones(16,4,dtype=torch.bool),
        shuffle_units=shuffled,shuffle_mask=torch.ones(16,4,dtype=torch.bool)),
        'o':dict(unit=torch.eye(300)[:12])}


def support_mask():
    value=torch.zeros(16,4,dtype=torch.bool);value[:,0]=True;value[:,1]=True;value[:,3]=True
    return value


class RouterTests(unittest.TestCase):
    def test_equal_capacity_and_initialization(self):
        states=[]
        for variant in ('G_name','G_support','G_shuffle'):
            torch.manual_seed(0);router=IncrementRouter(bank_fixture(),variant,support_mask())
            self.assertEqual(sum(p.numel() for p in router.parameters() if p.requires_grad),28834)
            states.append({k:v.clone() for k,v in router.state_dict().items()})
        for key in states[0]:
            self.assertTrue(torch.equal(states[0][key],states[1][key]))
            self.assertTrue(torch.equal(states[0][key],states[2][key]))

    def test_masks_and_fixed_values(self):
        b=bank_fixture();m=support_mask()
        name=IncrementRouter(b,'G_name',m);support=IncrementRouter(b,'G_support',m)
        shuffle=IncrementRouter(b,'G_shuffle',m)
        self.assertTrue(torch.equal(name.slot_mask.sum(1),torch.ones(16,dtype=torch.long)))
        self.assertTrue(torch.equal(support.slot_mask,m) and torch.equal(shuffle.slot_mask,m))
        self.assertFalse(any('value' in n for n,p in support.named_parameters()))
        self.assertTrue(torch.equal(support.units[:,0],shuffle.units[:,0]))

    def test_zero_increment_and_gradient_stages(self):
        torch.manual_seed(0);r=IncrementRouter(bank_fixture(),'G_support',support_mask())
        feature=torch.randn(5,300);ai=torch.tensor([0,1,2,3]);oi=torch.tensor([0,1,2,3])
        y=r(feature,ai,oi);self.assertEqual(float(y.abs().max()),0.0)
        y.sum().backward()
        active={n for n,p in r.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0}
        self.assertEqual(active,{'raw_gain'})
        with torch.no_grad():r.raw_gain.fill_(.2)
        r.zero_grad();r(feature,ai,oi).sum().backward()
        active={n for n,p in r.named_parameters() if p.grad is not None and float(p.grad.abs().sum())>0}
        self.assertIn('qx.weight',active);self.assertIn('pk.weight',active);self.assertIn('qo.weight',active)

    def test_residual_bound_and_candidate_permutation(self):
        torch.manual_seed(1);r=IncrementRouter(bank_fixture(),'G_support',support_mask())
        with torch.no_grad():r.raw_gain.fill_(10)
        f=torch.randn(7,300);ai=torch.tensor([0,1,2,3,4]);oi=torch.tensor([0,1,2,3,4])
        y=r(f,ai,oi);self.assertLessEqual(float(y.abs().max()),.050001)
        order=torch.tensor([4,2,0,3,1]);yp=r(f,ai[order],oi[order])
        self.assertTrue(torch.allclose(y[:,order],yp,atol=1e-6,rtol=1e-6))
        self.assertEqual(float(r(f,ai,oi,force_off=True).abs().max()),0.0)

    def test_bad_support_or_shuffle_is_rejected(self):
        bad=support_mask();bad[:,0]=False
        with self.assertRaises(ValueError):IncrementRouter(bank_fixture(),'G_support',bad)
        b=bank_fixture();b['a']['shuffle_units'][:,0]=0
        with self.assertRaises(ValueError):IncrementRouter(b,'G_shuffle',support_mask())


class SupportTests(unittest.TestCase):
    def test_object_conditioned_two_fold_mask(self):
        attrs=['a%d'%i for i in range(16)];objs=['o%d'%i for i in range(12)]
        pairs=[(attrs[a],objs[o]) for o in range(12) for a in range(16)][:83]
        a=[];o=[];p=[];ids=[];features=[]
        for pi,(an,on) in enumerate(pairs):
            ai=attrs.index(an);oi=objs.index(on)
            for j in range(6):
                a.append(ai);o.append(oi);p.append(pi);ids.append('%03d_%d'%(pi,j))
                features.append(torch.eye(300)[ai].numpy())
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);np.save(str(root/'attr_features.npy'),np.asarray(features,dtype=np.float32))
            np.savez(str(root/'labels.npz'),attr=np.asarray(a),obj=np.asarray(o),pair=np.asarray(p))
            (root/'image_ids.json').write_text(json.dumps(ids),encoding='utf-8')
            (root/'candidates.json').write_text(json.dumps(pairs),encoding='utf-8')
            mask,record=build_support_mask({'support_cache':str(root)},SimpleNamespace(attrs=attrs,objs=objs),bank_fixture())
        self.assertTrue(torch.as_tensor(mask[:,0]).all())
        self.assertTrue(torch.as_tensor(mask[:,1]).all())
        self.assertFalse(torch.as_tensor(mask[:,2]).any())
        self.assertTrue(torch.as_tensor(mask[:,3]).all())
        self.assertEqual(record['selected_ae_slots'],32)
        self.assertEqual(record['train_images'],498)


def metrics(auc=.4,hm=.5,unseen=.6):
    return {'metrics_raw':{'AUC':auc,'best_hm':hm,'best_unseen':unseen}}


class GateTests(unittest.TestCase):
    def test_gate_requires_baseline_name_and_shuffle(self):
        baseline={'cw':metrics(.4),'ow':metrics(.3,.4,.5)}
        records={v:{'cw':metrics(.405),'ow':metrics(.302,.4,.5)} for v in ('G_name','G_support','G_shuffle')}
        records['G_name']['cw']=metrics(.403);records['G_shuffle']['cw']=metrics(.402)
        self.assertTrue(decision(records,baseline)['proceed'])
        records['G_shuffle']['cw']=metrics(.4045)
        self.assertFalse(decision(records,baseline)['proceed'])

    def test_guards_block_apparent_auc_gain(self):
        baseline={'cw':metrics(.4,.5,.6),'ow':metrics(.3,.4,.6)}
        records={v:{'cw':metrics(.405,.5,.6),'ow':metrics(.302,.4,.6)} for v in ('G_name','G_support','G_shuffle')}
        records['G_name']['cw']=metrics(.4,.5,.6);records['G_shuffle']['cw']=metrics(.4,.5,.6)
        records['G_support']['ow']=metrics(.302,.4,.57)
        self.assertFalse(decision(records,baseline)['proceed'])


if __name__=='__main__':unittest.main(verbosity=2)
