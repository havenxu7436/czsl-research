import contextlib,copy,io,tempfile,unittest,shutil
from pathlib import Path
from types import SimpleNamespace
import torch
from test_formulas import synthetic_cache
from trustcpf.common import ROOT,read_json,write_json,load_torch,sha
from trustcpf import engine,stages,workflow,response

class WorkflowTests(unittest.TestCase):
    def test_complete_three_seed_workflow_reuse_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);cache=root/'cache';cache.mkdir();synthetic_cache(cache)
            source=root/'strong';response_source=root/'modelcond'
            cfg=dict(read_json(ROOT/'configs/default.json'),synthetic=True,utility_input='full',
                variant='robust',backbone_epochs=1,teacher_epochs=1,adapt_epochs=1,utility_epochs=1,
                fit_episodes=2,eval_episodes=2,utility_models=2,summaries=2,graphs=2,backgrounds=2,
                audit_backgrounds=2,support_images=4,batch_size=16,edges_per_type=8,rank=4,message_rank=2,
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
                resp=response_source/'work'/('seed%d'%seed);(resp/'reference').mkdir(parents=True)
                shutil.copy2(old/'reference/last.pt',resp/'reference/last.pt')
                write_json(resp/'reuse.json',dict(source_seed=str(old)))
                bank=load_torch(old/'runs/robust/bank.pt');pi=load_torch(old/'runs/robust/best.pt')['pi']
                recipe=dict(cfg,seed=seed,response_backgrounds=2,tau_absolute=.001,
                    fixed_type_budgets={str(t):float(pi[bank['types']==t].double().sum()) for t in range(3)})
                write_json(resp/'recipe.json',recipe)
                write_json(resp/'runs/conditioned/metrics.json',read_json(old/'runs/robust/metrics.json'))
                a.work=str(resp);a.config=str(resp/'recipe.json');response.build_responses(a)
            old_hashes={str(p):sha(p) for base in (source,response_source) for p in base.rglob('*') if p.is_file()}
            for seed in workflow.SEEDS:
                a=SimpleNamespace(source=str(source),response_source=str(response_source),cache=str(cache),
                    work=str(root/'new'),config=str(ROOT/'configs/default.json'),seed=seed,device='cpu')
                workflow.run_seed(a);dest=root/'new'/('seed%d'%seed)
                original=load_torch(source/'work'/('seed%d'%seed)/'reference/last.pt')['model']
                for variant in ('compact','shuffled_utility'):
                    final=load_torch(dest/'runs'/variant/'best.pt')
                    for k,v in original.items():
                        if k.startswith('backbone.'):self.assertTrue(torch.equal(v,final['model'][k]))
                    self.assertEqual(final['cfg']['utility_input'],'compact_response')
                    bank=load_torch(dest/'runs'/variant/'bank.pt')
                    for t,b in final['cfg']['fixed_type_budgets'].items():
                        self.assertAlmostEqual(float(final['pi'][bank['types']==int(t)].double().sum()),b,places=5)
                self.assertEqual(sha(dest/'runs/drop/metrics.json'),sha(source/'work'/('seed%d'%seed)/'runs/drop/metrics.json'))
            status=workflow.report(a);self.assertTrue(status['complete'],status)
            before={str(p):p.stat().st_mtime_ns for p in (root/'new').rglob('*.pt')}
            workflow.run_seed(a)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for p in (root/'new').rglob('*.pt')})
            self.assertEqual(old_hashes,{str(p):sha(p) for base in (source,response_source) for p in base.rglob('*') if p.is_file()})
            model=root/'new/seed3353/utility/compact/model.pt'
            with model.open('ab') as f:f.write(b'changed')
            self.assertFalse(workflow.report(a)['complete'])
            with self.assertRaisesRegex(ValueError,'output changed'):workflow.run_seed(a)

if __name__=='__main__':unittest.main()
