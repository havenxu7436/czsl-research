import copy
import unittest
import torch
import torch.nn.functional as F
from transition.name_router import NameRouter


def bank():
    g=torch.Generator().manual_seed(1729)
    result={}
    for axis,n in [('a',16),('o',12)]:
        units=F.normalize(torch.randn(n,4,300,generator=g),dim=-1)
        mask=torch.ones(n,4,dtype=torch.bool)
        perm=torch.roll(torch.arange(n),1)
        shuffled=units.clone(); shuffled[:,1:]=units[perm,1:]
        result[axis]={'units':units,'shuffle_units':shuffled,'mask':mask,
            'shuffle_mask':mask.clone(),'correct':F.normalize(units.mean(1),dim=-1),
            'unit':units[:,0].clone()}
    return result


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.bank=bank()
        self.feature=torch.randn(5,300)
        self.ai=torch.arange(16).repeat_interleave(12)
        self.oi=torch.arange(12).repeat(16)

    def active(self,mode):
        torch.manual_seed(7)
        m=NameRouter(self.bank,mode)
        with torch.no_grad():m.raw_gain.fill_(0.5)
        return m

    def test_equal_capacity_and_zero_initialization(self):
        counts=[]
        for mode in ['name_only','ekr','shuffle','no_object','no_null']:
            m=NameRouter(self.bank,mode)
            counts.append(sum(p.numel() for p in m.parameters()))
            self.assertTrue(torch.equal(m(self.feature,self.ai,self.oi),torch.zeros(5,192)))
        self.assertEqual(set(counts),{28834})

    def test_name_control_independent_of_both_description_banks(self):
        altered=copy.deepcopy(self.bank)
        for axis in ('a','o'):
            altered[axis]['units'][:,1:]=torch.randn_like(altered[axis]['units'][:,1:])*20
            altered[axis]['shuffle_units'][:,1:]=torch.randn_like(altered[axis]['shuffle_units'][:,1:])*20
            altered[axis]['correct']=torch.randn_like(altered[axis]['correct'])*20
        m=self.active('name_only')
        other=NameRouter(altered,'name_only');other.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(m(self.feature,self.ai,self.oi),other(self.feature,self.ai,self.oi)))

    def test_object_query_has_no_description_leak_for_any_variant(self):
        altered=copy.deepcopy(self.bank)
        altered['o']['correct']=torch.randn_like(altered['o']['correct'])*100
        altered['o']['units'][:,1:]=torch.randn_like(altered['o']['units'][:,1:])*100
        for mode in ['name_only','ekr','shuffle','no_object','no_null']:
            m=self.active(mode)
            other=NameRouter(altered,mode);other.load_state_dict(m.state_dict())
            self.assertTrue(torch.equal(m(self.feature,self.ai,self.oi),other(self.feature,self.ai,self.oi)))

    def test_description_changes_can_change_active_clean_router(self):
        altered=copy.deepcopy(self.bank)
        altered['a']['units'][:,1:]=F.normalize(torch.randn_like(altered['a']['units'][:,1:]),dim=-1)
        m=self.active('ekr'); other=NameRouter(altered,'ekr');other.load_state_dict(m.state_dict())
        self.assertFalse(torch.allclose(m(self.feature,self.ai,self.oi),other(self.feature,self.ai,self.oi)))

    def test_bound_off_order_and_chunk_invariance(self):
        for mode in ['name_only','ekr','shuffle','no_object','no_null']:
            m=self.active(mode)
            output=m(self.feature,self.ai,self.oi)
            self.assertLessEqual(float(output.abs().max()),0.100001)
            self.assertTrue(torch.equal(m(self.feature,self.ai,self.oi,force_off=True),torch.zeros_like(output)))
            perm=torch.randperm(192)
            self.assertTrue(torch.allclose(output[:,perm],m(self.feature,self.ai[perm],self.oi[perm]),atol=1e-7))
            m.object_chunk=3
            self.assertTrue(torch.allclose(output,m(self.feature,self.ai,self.oi),atol=1e-7))

    def test_learns_after_zero_gain_update(self):
        for mode in ['name_only','ekr','shuffle']:
            m=NameRouter(self.bank,mode)
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
            counts=[]
            for step in range(3):
                scores=torch.randn(5,192,generator=torch.Generator().manual_seed(80))+m(self.feature,self.ai,self.oi)
                loss=F.cross_entropy(scores/0.05,torch.arange(5))
                opt.zero_grad();loss.backward()
                grads=[p.grad for p in m.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                counts.append(sum(int(g.abs().sum()>0) for g in grads));opt.step()
            self.assertGreaterEqual(max(counts[1:]),2)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
