"""Natural splits, image augmentation and composition-disjoint teacher splits."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from .common import ROOT, read, load, sha

NAMES = ('ut-zap50k','mit-states','cgqa')

def index_data(root, name):
    root = Path(root).resolve(); split = root/'compositional-split-natural'
    pairs = {}
    for phase in ('train','val','test'):
        rows = [line.split() for line in (split/(phase+'_pairs.txt')).read_text().splitlines() if line.strip()]
        if any(len(row)!=2 for row in rows): raise ValueError('Expected attribute/object pairs: '+str(split))
        pairs[phase] = sorted(set(map(tuple, rows)))
    union = sorted(set(sum([list(v) for v in pairs.values()], [])))
    allowed_pairs=set(union);train_pairs=set(pairs['train'])
    attrs = sorted({a for a,o in union}); objs = sorted({o for a,o in union})
    ai = {a:i for i,a in enumerate(attrs)}; oi = {o:i for i,o in enumerate(objs)}
    aliases = read(ROOT/'assets/path_aliases.json') if name=='ut-zap50k' else {}
    metadata = root/'metadata_compositional-split-natural.t7'
    rows = []; missing=[]
    for r in load(metadata):
        a,o,phase = r['attr'],r['obj'],r['set']
        if a=='NA' or phase not in pairs or (a,o) not in allowed_pairs: continue
        if phase=='train' and (a,o) not in train_pairs: raise ValueError('Training label outside TRAIN')
        rel=aliases.get(r['image'],r['image']); p=root/'images'/rel
        if not p.is_file(): missing.append(str(p))
        rows.append(dict(image=rel,attr=ai[a],obj=oi[o],pair=ai[a]*len(objs)+oi[o],phase=phase))
    if missing: raise FileNotFoundError('Missing %d images, first: %s'%(len(missing),missing[:8]))
    ids={k:sorted(ai[a]*len(objs)+oi[o] for a,o in p) for k,p in pairs.items()}
    for phase in ('val','test'):
        allowed=set(ids['train'])|set(ids[phase])
        if any(r['pair'] not in allowed for r in rows if r['phase']==phase):
            raise ValueError('Phase candidate mismatch: '+phase)
    return dict(name=name,root=str(root),attrs=attrs,objs=objs,pairs=ids,rows=rows,
                metadata_sha256=sha(metadata),split_sha256={s:sha(split/(s+'_pairs.txt')) for s in pairs},
                images={s:sum(r['phase']==s for r in rows) for s in pairs})

def transforms(train):
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    ops = [T.RandomResizedCrop(224,scale=(.8,1.),interpolation=InterpolationMode.BICUBIC),T.RandomHorizontalFlip()] if train else [
        T.Resize(224,interpolation=InterpolationMode.BICUBIC),T.CenterCrop(224)]
    return T.Compose(ops+[T.ToTensor(),T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])

class Images(Dataset):
    def __init__(self,index,ids,train=False):
        self.index,self.ids,self.transform=index,list(ids),transforms(train)
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        idx=self.ids[i];r=self.index['rows'][idx]
        with Image.open(Path(self.index['root'])/'images'/r['image']) as im: x=self.transform(im.convert('RGB'))
        return x,r['attr'],r['obj'],r['pair'],idx

def folds(index,seed):
    """Five balanced query folds with constructive primitive coverage.

    An edge (composition) can forbid at most one fold at each endpoint:
    the fold containing all its other incident edges. Thus at most two
    of five folds are forbidden, and a valid assignment always exists.
    Singleton-incident edges remain common Inner edges in every fold.
    """
    pairs=sorted(set(index['pairs']['train'])); no=len(index['objs'])
    counts=Counter(('a',p//no) for p in pairs)+Counter(('o',p%no) for p in pairs)
    common=[p for p in pairs if counts['a',p//no]==1 or counts['o',p%no]==1]
    common_set=set(common);rest=[p for p in pairs if p not in common_set]
    if len(rest)<5:raise ValueError('Five query folds require at least five non-singleton compositions')
    fixed={v for p in common for v in (('a',p//no),('o',p%no))}
    assigned={v:np.zeros(5,dtype=np.int64) for v in counts}
    qs=[[] for _ in range(5)];sizes=np.zeros(5,dtype=np.int64);rng=np.random.RandomState(seed)
    for raw in rng.permutation(rest):
        p=int(raw);vertices=(('a',p//no),('o',p%no));valid=np.ones(5,dtype=bool)
        for v in vertices:
            if v not in fixed:valid &= assigned[v]+1<counts[v]
        available=np.flatnonzero(valid)
        if not len(available):raise RuntimeError('Primitive coverage invariant violated')
        smallest=available[sizes[available]==sizes[available].min()]
        group=int(rng.choice(smallest));qs[group].append(p);sizes[group]+=1
        for v in vertices:assigned[v][group]+=1
    out=[]
    for j,q in enumerate(qs):
        qset=set(q);inner=sorted(set(pairs)-qset);train=[];query=[]
        aa={p//no for p in inner};oo={p%no for p in inner}
        if not q or any(p//no not in aa or p%no not in oo for p in q):
            raise RuntimeError('Constructed fold failed primitive coverage verification')
        for i,r in enumerate(index['rows']):
            if r['phase']=='train': (query if r['pair'] in qset else train).append(i)
        out.append(dict(fold=j+1,role=('fit','fit','fit','dev','audit')[j],inner_pairs=inner,
                        query_pairs=sorted(q),train_ids=train,query_ids=query,
                        split_algorithm='coverage_greedy_v2',common_pairs=common,primitive_coverage=True))
    return out

def balanced_sample(index,ids,per_pair,seed,limit_pairs=None):
    rng=np.random.RandomState(seed);groups={}
    for i in ids: groups.setdefault(index['rows'][i]['pair'],[]).append(i)
    keys=sorted(groups)
    if limit_pairs and len(keys)>limit_pairs:
        # Cover every primitive before filling the remaining pair budget.
        no=len(index['objs']);chosen=[];covered=set()
        for p in rng.permutation(keys):
            primitives={('a',int(p)//no),('o',int(p)%no)}
            if not primitives<=covered: chosen.append(int(p));covered |= primitives
        rest=[p for p in keys if p not in chosen]
        chosen+=list(map(int,rng.permutation(rest)[:max(0,limit_pairs-len(chosen))]))
        keys=sorted(chosen)
    selected=[];weights=[]
    for p in keys:
        take=list(map(int,rng.choice(groups[p],min(per_pair,len(groups[p])),replace=False)))
        selected+=take;weights += [1./len(keys)/len(take)]*len(take)
    return selected,np.array(weights,dtype=np.float64)

def partition_objects(x):
    """Deterministic balanced two-cluster partition, linear in vocabulary size."""
    x=np.asarray(x,dtype=np.float64);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    n=len(x);size=n//2
    if size<1: raise ValueError('At least two objects required')
    # Farthest from object 0 initializes a separated pair of centers.
    c0=x[0]; c1=x[np.argmin(x@c0)];last=None
    for _ in range(100):
        diff=((x-c0)**2).sum(1)-((x-c1)**2).sum(1)
        left=np.sort(np.argsort(diff,kind='stable')[:size]);right=np.setdiff1d(np.arange(n),left)
        if last is not None and np.array_equal(last,left):break
        last=left;c0=x[left].mean(0);c1=x[right].mean(0)
    if 0 not in left:left,right=right,left
    return [left.tolist(),right.tolist()]


EXPECTED={'ut-zap50k':(16,12,22998,3214,2914,83),'mit-states':(115,245,30338,10420,12995,1262),'cgqa':(413,674,26920,7280,5098,5592)}


def verify_split(index):
    got=(len(index['attrs']),len(index['objs']),*(index['images'][s] for s in ('train','val','test')),len(index['pairs']['train']))
    if got!=EXPECTED[index['name']]:raise ValueError('Natural split counts differ: '+str(got))
    phases=[set(index['pairs'][s]) for s in ('train','val','test')]
    # Natural VAL/TEST include seen pairs; only their unseen subsets are disjoint.
    if (phases[1]-phases[0])&(phases[2]-phases[0]):raise ValueError('Unseen VAL/TEST composition overlap')
    paths=[r['image'] for r in index['rows']]
    if len(paths)!=len(set(paths)):raise ValueError('Duplicate image across natural splits')


def teacher_folds(index,seed):
    result=folds(index,seed)
    for s in result:
        rng=np.random.RandomState(seed+s['fold']*17);by_pair={}
        for i in s['train_ids']:by_pair.setdefault(index['rows'][i]['pair'],[]).append(i)
        fit=[];held=[]
        for p,items in sorted(by_pair.items()):
            order=list(map(int,rng.permutation(items)));n=min(2,max(1,len(order)//5)) if len(order)>1 else 0
            held+=order[:n];fit+=order[n:]
        s['all_inner_ids']=s['train_ids'];s['train_ids']=fit;s['audit_seen_ids']=held
        if set(fit)&set(held) or set(fit)&set(s['query_ids']):raise AssertionError('Teacher image leakage')
        if set(index['rows'][i]['pair'] for i in fit)!=set(s['inner_pairs']):raise AssertionError('Inner image holdout removed a composition')
    return result


def context_groups(x):
    if len(x)>16:return partition_objects(x)
    import itertools
    x=np.asarray(x,dtype=np.float64);x/=np.linalg.norm(x,axis=1,keepdims=True);n=len(x);best=None
    for rest in itertools.combinations(range(1,n),n//2-1):
        left=(0,)+rest;right=tuple(j for j in range(n) if j not in left)
        score=sum(float(((x[list(g)]-x[list(g)].mean(0))**2).sum()) for g in (left,right))
        item=(score,left,right)
        if best is None or item<best:best=item
    return [list(best[1]),list(best[2])]
