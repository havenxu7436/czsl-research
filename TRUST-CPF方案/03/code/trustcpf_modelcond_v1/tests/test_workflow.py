import contextlib,copy,io,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from test_formulas import synthetic_cache
from trustcpf.common import ROOT,read_json,write_json,load_torch,sha
from trustcpf import engine,stages,workflow

class WorkflowTests(unittest.TestCase):
    def test_complete_three_seed_workflow_reuse_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);cache=root/'cache';cache.mkdir();synthetic_cache(cache)
            source=root/'strong';cfg=dict(read_json(ROOT/'configs/default.json'),synthetic=True,utility_input='full',
                variant='robust',backbone_epochs=1,teacher_epochs=1,adapt_epochs=1,utility_epochs=1,
                fit_episodes=2,eval_episodes=2,utility_models=2,summaries=2,graphs=2,backgrounds=2,
                audit_backgrounds=2,support_images=4,batch_size=16,edges_per_type=2,rank=4,message_rank=2,
                backbone_rank=4,backbone_adapter_rank=2,eval_batch_size=64)
            write_json(root/'old.json',cfg)
            for seed in workflow.SEEDS:
                old=source/'work'/('seed%d'%seed)
                a=SimpleNamespace(cache=str(cache),work=str(old),config=str(root/'old.json'),seed=seed,
                    variant='robust',quick=False,fold='all',fit_all=False,device='cpu')
                engine.teachers(a);stages.measure(a);stages.fit_utility(a);stages.utility_audit(a)
                a.fit_all=True;stages.fit_utility(a);engine.reference(a)
                write_json(old/'tuning.json',dict(selected=dict(lambda_var=.5,lambda_ens=.25,budget=.25,tau_scale=.5,tau_absolute=.001)))
                stages.structure(a)
                for v in ['drop','robust']:
                    a.variant=v;stages.adapt(a)
            old_hashes={str(p):sha(p) for p in source.rglob('*') if p.is_file()}
            config=read_json(ROOT/'configs/default.json');config['response_backgrounds']=2
            write_json(root/'new.json',config)
            for seed in workflow.SEEDS:
                a=SimpleNamespace(source=str(source),cache=str(cache),work=str(root/'new'),config=str(root/'new.json'),seed=seed,device='cpu')
                workflow.run_seed(a)
                dest=root/'new'/('seed%d'%seed)
                original=load_torch(source/'work'/('seed%d'%seed)/'reference/last.pt')['model']
                final=load_torch(dest/'runs/conditioned/best.pt')
                for k,v in original.items():
                    if k.startswith('backbone.'):self.assertTrue(torch.equal(v,final['model'][k]))
                self.assertEqual(final['cfg']['utility_input'],'model_response')
                measured=load_torch(dest/'utility/audit_predictions.pt')
                q5=load_torch(source/'work'/('seed%d'%seed)/'measurements/fold5.pt')
                self.assertTrue(torch.equal(measured['target'],q5['y']))
                for j in range(1,6):
                    r=read_json(dest/'responses'/('fold%d.json'%j));self.assertEqual(r['query_overlap'],0)
                bank=load_torch(dest/'runs/conditioned/bank.pt')
                for t,b in final['cfg']['fixed_type_budgets'].items():
                    self.assertAlmostEqual(float(final['pi'][bank['types']==int(t)].sum()),b,places=5)
                utility=load_torch(dest/'utility/model0.pt')
                self.assertEqual(utility['dimension'],q5['x'].shape[1]+8)
            status=workflow.report(a);self.assertTrue(status['complete'],status)
            before={str(p):p.stat().st_mtime_ns for p in (root/'new').rglob('*.pt')}
            workflow.run_seed(a)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in (root/'new').rglob('*.pt')})
            self.assertEqual(old_hashes,{str(p):sha(p) for p in source.rglob('*') if p.is_file()})
            write_json(root/'synthetic_workflow.json',dict(status=status,source_unchanged=True,
                restart_preserves_checkpoints=True,query_labels_unchanged=True,backbone_frozen=True,synthetic_only=True))

if __name__=='__main__':unittest.main()
