"""CPU reproduction on the archived real Competition effects; no image training."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trustcpf import r5,compact,competition,align
from trustcpf.common import ROOT,load_torch,read_json,write_json,sha

source=Path(sys.argv[1]);results=[]
for seed in (3351,3352,3353):
    w=source/'work'/('seed%d'%seed);types=load_torch(w/'inputs/bank.pt')['types'].numpy()
    xs=[load_torch(w/('responses/fold%d.pt'%j))['features'].double().numpy() for j in range(1,6)]
    ys=[load_torch(w/('measurements/fold%d.pt'%j))['target'][:,0] for j in range(1,6)]
    xr=load_torch(w/'responses/reference.pt')['features'].double().numpy()
    source_pred=load_torch(w/'utility/predictions.pt')['unll'];trials={name:[] for name in r5.METHODS}
    for alpha in (.1,1.,10.,100.):
        for name,model in r5.paired_models(xs[:3],ys[:3],types,alpha).items():
            v=compact.predict(model,xs[3],types)
            trials[name].append(dict(alpha=alpha,q4_error=competition.centered_type_error(v,ys[3],types,model['scales'])))
    entry=dict(seed=seed,source_prediction_sha256=sha(w/'utility/predictions.pt'),methods={})
    for name in r5.METHODS:
        choice=min(trials[name],key=lambda x:(x['q4_error'],-x['alpha']));alpha=choice['alpha']
        m3=r5.paired_models(xs[:3],ys[:3],types,alpha)[name];m5=r5.paired_models(xs,ys,types,alpha)[name]
        p5=compact.predict(m3,xs[4],types);pr=compact.predict(m5,xr,types)
        assert np.isfinite(p5).all() and np.isfinite(pr).all()
        if name=='absolute':
            assert alpha==read_json(w/'utility/selection.json')['unll']['selected_alpha']
            np.testing.assert_allclose(p5,source_pred['q5'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(pr,source_pred['reference'],rtol=1e-10,atol=1e-12)
            for strength in (0.,.5,1.,2.):
                p,_=align.policy(pr,types,strength)
                old=load_torch(w/'policies.pt')['scanned']['unll_k%g_real'%strength]['pi']
                np.testing.assert_array_equal(p.float().numpy(),old.float().numpy())
        entry['methods'][name]=dict(alpha=alpha,q4_error=choice['q4_error'],
            q5_prediction_finite=True,reference_prediction_finite=True,
            beta_shape=list(m3['beta'].shape),intercepts=m3['beta'][[0,9,18]].tolist())
    results.append(entry)
write_json(ROOT/'validation/real_input_reproduction.json',dict(passed=True,seeds=results,
    scope='Real effect/response CPU fit and all absolute policy strengths; no new server recognition training'))
print('REAL_INPUT_REPRODUCTION_OK: 3 seeds; absolute fit/alpha/policies match; relative coefficients finite')
