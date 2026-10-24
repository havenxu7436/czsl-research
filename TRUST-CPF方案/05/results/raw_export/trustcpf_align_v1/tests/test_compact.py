import copy,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from trustcpf import compact
from trustcpf.common import read_json,write_json,save_torch,load_torch

class CompactTests(unittest.TestCase):
    def test_episode_mean_se_and_noise_not_double_counted(self):
        raw=dict(y=torch.tensor([1.,3.,2.,6.]),edge=torch.tensor([0,0,1,1]),noise=torch.tensor([.2,.4,.8,1.2]))
        s=compact.aggregate_measurements(raw,2)
        np.testing.assert_allclose(s['target'],[2,4]);np.testing.assert_allclose(s['se2'],[1,4])
        np.testing.assert_allclose(s['background_se2'],[.15,.5])
    def test_ridge_normal_equation_and_permutation(self):
        rng=np.random.RandomState(42);types=np.repeat(np.arange(3),6)
        xs=[rng.randn(18,8) for _ in range(3)];ys=[rng.randn(18)*.01 for _ in range(3)]
        m=compact.fit_ridge(xs,ys,types,10)
        t=np.tile(types,3);z=compact.design(np.concatenate(xs),t,m['center'],m['spread'])
        np.testing.assert_allclose((z.T@z+10*np.eye(27))@m['beta'],z.T@(np.concatenate(ys)/m['scales'][t]),atol=1e-12)
        s={k:np.arange(18)+offset for k,offset in [('target',0),('se2',100),('count',200),('background_se2',300)]}
        a,p=compact.shuffled_stats(s,types,7,1);b,q=compact.shuffled_stats(s,types,7,1)
        np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(types[p],types)
        np.testing.assert_array_equal(a['se2']-a['target'],np.full(18,100))
        self.assertFalse(np.array_equal(p,np.arange(18)))
    def test_regret_endpoints_signed_dual_and_type_constant(self):
        ty=np.repeat(np.arange(3),3);cfg=dict(tau_absolute=.01,fixed_type_budgets={'0':0.,'1':1.5,'2':3.})
        y=np.linspace(-.05,.01,9)
        result=compact.regret(y,y,ty,cfg);self.assertAlmostEqual(result['regret'],0.,places=12)
        p,_=compact.policy(np.repeat([-.4,.2,100],3),ty,cfg)
        np.testing.assert_allclose(p.numpy(),np.repeat([0.,.5,1.],3),atol=1e-12)
        self.assertGreaterEqual(compact.regret(-y,y,ty,cfg)['regret'],0)
        self.assertTrue(np.isfinite(compact.objective(p,y,.01)))
    def test_q5_cannot_select_alpha_and_shuffle_shares_alpha_and_scaling(self):
        rng=np.random.RandomState(4);ty=np.repeat(np.arange(3),4)
        with tempfile.TemporaryDirectory() as td:
            work=Path(td);stats=[]
            for j in range(1,6):
                stats.append(dict(target=rng.randn(12)*.01,se2=np.ones(12)*1e-5,count=np.full(12,10),background_se2=np.ones(12)*1e-6))
                save_torch(work/'responses'/('fold%d.pt'%j),dict(features=torch.from_numpy(rng.randn(12,8))))
            cfg=dict(ridge_alphas=[.1,1,10,100],label_shuffle_seed=8,seed=3351,tau_absolute=.01,fixed_type_budgets={str(t):1. for t in range(3)})
            write_json(work/'recipe.json',cfg);save_torch(work/'inputs/edge_statistics.pt',dict(types=ty,folds=stats))
            a=compact.fit(work);selection=read_json(work/'utility/selection.json')
            stats[4]['target'][:]=9999;save_torch(work/'inputs/edge_statistics.pt',dict(types=ty,folds=stats))
            b=compact.fit(work);self.assertEqual(selection,read_json(work/'utility/selection.json'))
            for k in ('center','spread','scales'):
                np.testing.assert_array_equal(a['compact'][k],b['shuffled_utility'][k])
            self.assertEqual(a['compact']['alpha'],a['shuffled_utility']['alpha'])

if __name__=='__main__':unittest.main()
