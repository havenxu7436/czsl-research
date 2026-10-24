"""Knowledge attachment graph, CLIP semantic bank, and budget constrained sigmoid."""
import sys
from pathlib import Path
import torch
from .common import ROOT, read_json, write_json, unit, save_torch, sha


def graph_records(labels, knowledge):
    attrs, objs = labels['attrs'], labels['objs']
    rows, parents = [], sorted(knowledge['oi'])
    # Node targets are attrs first, objects next; type indices OI/OE/AE=0/1/2.
    for h in parents:
        for o in knowledge['oi'][h]:
            rows.append({'type': 0, 'target': len(attrs) + objs.index(o), 'text': h,
                         'prompt': 'a photo of ' + h, 'owner': o, 'slot': h})
    for kind, labels_, offset, ty in [('oe', objs, len(attrs), 1), ('ae', attrs, 0, 2)]:
        for i, c in enumerate(labels_):
            for k, text in enumerate(knowledge[kind][c]):
                rows.append({'type': ty, 'target': offset+i, 'text': text,
                             'prompt': 'a photo of footwear with ' + text, 'owner': c, 'slot': k})
    return rows


def assemble_bank(text, source, rows):
    anchors = torch.cat([text['ea'], text['eo']]).float()
    types = torch.tensor([r['type'] for r in rows], dtype=torch.long)
    target = torch.tensor([r['target'] for r in rows], dtype=torch.long)
    degree = torch.tensor([sum(s['type']==r['type'] and s['target']==r['target'] for s in rows)
                           for r in rows], dtype=torch.float32)
    meta = torch.tensor([[1., 0., float(r['type'] != 0), 0.] for r in rows])
    return {'anchors': anchors, 'source': unit(source.float()), 'types': types,
            'target': target, 'degree': degree, 'metadata': meta, 'rows': rows}


def prepare_bank(args):
    import numpy as np
    sys.path.insert(0, str(ROOT/'vendor'))
    import clip
    cache = Path(args.cache)
    manifest = read_json(cache/'features_manifest.json')
    knowledge_dir = Path(args.knowledge or ROOT/'assets/knowledge')
    knowledge = {k: read_json(knowledge_dir/(k+'_dict.json')) for k in ('oi','oe','ae')}
    rows = graph_records(manifest['labels'], knowledge)
    model, _ = clip.load(str(args.checkpoint), device=args.device, jit=False)
    model = model.float().eval().requires_grad_(False)
    with torch.no_grad():
        source = torch.cat([unit(model.encode_text(clip.tokenize([r['prompt'] for r in rows[i:i+64]]).to(args.device)).float()).cpu()
                            for i in range(0,len(rows),64)])
    with np.load(cache/'text_features.npz') as z:
        text = {k:torch.from_numpy(z[k].copy()) for k in ('ea','eo','ec')}
    bank = assemble_bank(text, source, rows)
    save_torch(cache/'graph.pt', bank)
    write_json(cache/'knowledge_manifest.json', {'graph_sha256':sha(cache/'graph.pt'),
        'files':{k:sha(knowledge_dir/(k+'_dict.json')) for k in knowledge},
        'prompts':[r['prompt'] for r in rows], 'edges':len(rows), 'dimension':source.shape[1]})
    print('GRAPH_READY',len(rows),source.shape[1],flush=True)


def solve_budget(reward, types, tau, fraction=.5, prior=.5):
    """Unique convex optimum; multipliers are nonnegative, per-type upper budgets."""
    pi, multipliers = torch.zeros_like(reward), {}
    tau = max(float(tau), 1e-8)
    prior = torch.as_tensor(prior, device=reward.device, dtype=reward.dtype).clamp(1e-6,1-1e-6)
    prior_logit = torch.logit(prior).expand_as(reward)
    for ty in types.unique().tolist():
        mask = types == ty
        budget = fraction * int(mask.sum())
        def value(gamma): return torch.sigmoid(prior_logit[mask] + (reward[mask]-gamma)/tau)
        if fraction == 0:
            multipliers[str(ty)] = 0.; continue
        if value(0).sum() <= budget:
            gamma = 0.
        else:
            low, high = 0., max(1., float(reward[mask].abs().max()))
            while value(high).sum() > budget: high *= 2
            for _ in range(60):
                mid = (low+high)/2
                if value(mid).sum() > budget: low=mid
                else: high=mid
            gamma=high
        pi[mask] = value(gamma)
        multipliers[str(ty)] = gamma
    return pi, multipliers


def solve_budget_equal(reward, types, tau, budgets):
    """Strictly concave entropy objective with exact per-type expected counts."""
    if tau <= 0 or not torch.isfinite(reward).all():
        raise ValueError('Positive temperature and finite rewards required')
    r=reward.double();pi=torch.zeros_like(r);multipliers={}
    for ty in types.unique().tolist():
        mask=types==ty;n=int(mask.sum());b=float(budgets[str(ty)])
        if not 0<=b<=n:raise ValueError('Invalid expected edge count')
        if b==0 or b==n:
            pi[mask]=b/n;multipliers[str(ty)]=None;continue
        low=float(r[mask].min())-100*tau
        high=float(r[mask].max())+100*tau
        for _ in range(80):
            mid=(low+high)/2
            if float(torch.sigmoid((r[mask]-mid)/tau).sum())>b:low=mid
            else:high=mid
        gamma=(low+high)/2
        pi[mask]=torch.sigmoid((r[mask]-gamma)/tau)
        multipliers[str(ty)]=gamma
    return pi.to(reward.dtype),multipliers


def permute_bank(bank, seed=7101, rate=1., relation=None, deletion=False):
    """Fixed random order, nested corruption prefixes, types remain separate."""
    import copy
    out = copy.deepcopy(bank)
    gen = torch.Generator().manual_seed(seed)
    for ty in range(3):
        if relation is not None and ty != relation: continue
        ids = torch.where(bank['types']==ty)[0]
        ids = ids[torch.randperm(len(ids), generator=gen)]
        n = int(round(rate*len(ids)))
        if not n: continue
        if deletion:
            if 'availability' not in out: out['availability']=torch.ones(len(bank['types']))
            out['availability'][ids[:n]]=0
        else:
            # A block rotation deranges owner groups and preserves the exact source multiset.
            def owner(e):
                return str(bank['rows'][int(e)]['slot']) if ty==0 else str(int(bank['target'][e]))
            n=max(2,n)
            while n<=len(ids):
                chosen=ids[:n].tolist(); groups={owner(e) for e in chosen}
                largest=max(sum(owner(e)==g for e in chosen) for g in groups)
                if 2*largest<=n: break
                n+=1
            chosen=sorted(ids[:n].tolist(),key=lambda e:(owner(e),e))
            donors=chosen[largest:]+chosen[:largest]
            out['source'][chosen]=bank['source'][donors]
    return out
