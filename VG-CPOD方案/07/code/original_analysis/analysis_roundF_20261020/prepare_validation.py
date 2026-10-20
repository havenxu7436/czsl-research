"""Adapt the existing CLI integration check for F1 and its E1 beta baseline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'analysis_roundE_20261020/validate_release.py').read_text(encoding='utf-8')
    for left,right in [('three E seeds','three F seeds'),('vgcpod_v1_roundD','vgcpod_v1_roundE'),
                       ('vgcpod_v1_roundE\'\n','vgcpod_v1_roundF\'\n'),
                       ('vgcpod_e_cli_','vgcpod_f_cli_'),('E_source','F_source'),('D_source','E_source'),
                       ('configs/round_e/E1.json','configs/round_f/F1.json'),('test_e1.json','test_f1.json'),
                       ('round_e.py','round_f.py'),('E_out','F_out'),('round_e_','round_f_'),
                       ('ROUND_E_COMPLETE','ROUND_F_COMPLETE'),('/E1/','/F1/'),('/\'E1\'/','/\'F1\'/'),
                       ('_E1_seed','_F1_seed'),('Legacy rho=.15 run','Legacy E trainer at beta=.05'),
                       ('legacy_rho015_bitwise_parameters','legacy_beta005_bitwise_parameters')]:
        text = text.replace(left,right)
    text = text.replace("for case,sub,output in [('A5','round_a','A_out'),('C2','round_c','C_out'),('D1','round_d','D_out')]:",
        "for case,sub,output in [('A5','round_a','A_out'),('C2','round_c','C_out'),('D1','round_d','D_out'),('E1','round_e','E_out')]:")
    old = """            engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/output/case),
                seed=3351,device='cpu',resume=False),ref)"""
    new = """            for seed in ((3351,3352,3353) if case=='E1' else (3351,)):
                engine.train(argparse.Namespace(cache=str(temp/'cache'),out=str(temp/output/case),
                    seed=seed,device='cpu',resume=False),ref)"""
    assert old in text
    text = text.replace(old,new)
    text = text.replace("'--d-out',str(temp/'D_out'),'--gpus'", "'--d-out',str(temp/'D_out'),'--e-out',str(temp/'E_out'),'--gpus'")
    text = text.replace("assert len(summary['rho_native_runs'])==5", "assert len(summary['beta_native_runs'])==6")
    text = text.replace("assert {r['case'] for r in summary['rho_native_runs']}=={'C2','D1','E1'}",
        "assert {r['case'] for r in summary['beta_native_runs']}=={'E1','F1'}\n"
        "        assert len(summary['support_training_runs'])==12\n"
        "        assert len([r for r in summary['paired_deltas'] if r['comparison']=='F1-E1'])==3")
    text = text.replace("assert len(summary['paired_deltas'])==3", "assert len(summary['paired_deltas'])==6")
    text = text.replace("rho_comparison_cases=['C2','D1','E1']", "beta_comparison_cases=['E1','F1']")
    old = """            assert torch.equal(a['rng']['torch'],b['rng']['torch'])"""
    new = old+"""
            assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
            for parameter,state in a['optimizer']['state'].items():
                for key,value in state.items():
                    other = b['optimizer']['state'][parameter][key]
                    assert torch.equal(value,other) if torch.is_tensor(value) else value==other
        for row in summary['support_training_runs']:
            expected = 1.5*row['beta']/(1+row['beta'])
            assert abs(row['support_weight']-expected)<1e-14
            assert abs(row['pair_weighted']-1.5*row['pair'])<1e-12
            assert abs(row['support_weighted']-expected*row['support'])<1e-12
            assert row['task_epoch_reconstruction_error']<1e-5"""
    assert old in text
    text = text.replace(old,new)
    text = text.replace("legacy_rng_equal=True,", "legacy_rng_equal=True,legacy_optimizer_equal=True,support_logs_valid=True,")
    (ROOT/'analysis_roundF_20261020/validate_release.py').write_text(text,encoding='utf-8',newline='\n')
    print('Prepared F CLI validator')


if __name__=='__main__':
    main()
