"""Exercise all 114 stages and 1800 noise rows on synthetic data, including restart."""
import os,sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ['CPF_CPU_THREADS']='1'
from test_formulas import synthetic_cache
from run import parser,main
from trustcpf.common import ROOT,read_json,write_json

with tempfile.TemporaryDirectory(prefix='trust_strong_full_') as td:
    root=Path(td);cache=root/'cache';cache.mkdir();synthetic_cache(cache)
    cfg=dict(read_json(ROOT/'configs/default.json'),synthetic=True,backbone_epochs=1,teacher_epochs=1,adapt_epochs=1,
        utility_epochs=1,fit_episodes=2,eval_episodes=2,utility_models=2,summaries=2,graphs=2,backgrounds=2,
        audit_backgrounds=2,support_images=4,batch_size=16,edges_per_type=2,rank=4,message_rank=2,
        backbone_rank=4,backbone_adapter_rank=2,eval_batch_size=64)
    write_json(root/'cfg.json',cfg)
    common=['--cache',str(cache),'--work',str(root/'work'),'--config',str(root/'cfg.json'),'--device','cpu']
    for seed in [3351,3352,3353]:main(parser().parse_args(['run-seed']+common+['--seed',str(seed)]))
    a=parser().parse_args(['status']+common);status=main(a)
    assert status['complete'],status
    # Identical rerun must not rewrite stage outputs.
    before={str(p):p.stat().st_mtime_ns for p in (root/'work').rglob('best.pt')}
    main(parser().parse_args(['run-seed']+common+['--seed','3351']))
    assert before=={str(p):p.stat().st_mtime_ns for p in (root/'work').rglob('best.pt')}
    for p in (root/'work').rglob('noise_*.json'):
        if not (p.parent/'metrics.json').exists():continue
        values=read_json(p);assert len(values)==75
        clean=read_json(p.parent/'metrics.json')['test']
        for row in values:
            if row['rate']==0:assert row['test']==clean
    write_json(ROOT/'validation/full_queue_synthetic.json',dict(status=status,restart_preserved_best_weights=True,
        clean_noise_rows_exact=True,synthetic_only=True,formal_epochs_not_run=True))
    print('FULL_SYNTHETIC_QUEUE_PASSED',status,flush=True)
