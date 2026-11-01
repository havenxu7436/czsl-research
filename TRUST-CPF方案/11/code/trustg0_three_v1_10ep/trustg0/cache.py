"""Frozen CLIP features: global + patch memmaps, shared by all repetitions."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import ROOT,read,write,load,save,sha,seed_all,lock
from .data import Images
from .model import unit

CLIP_SHA='b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'


def clip_load(path,device):
    sys.path.insert(0,str(ROOT/'vendor'));import clip
    m,_=clip.load(str(path),device=device,jit=False)
    return m.float().eval().requires_grad_(False),clip.tokenize


def prompts(index):
    attrs,objs=index['attrs'],index['objs']
    if index['name']=='ut-zap50k':
        labels=read(ROOT/'assets/labels.json');names=read(ROOT/'assets/names.json')
        amap=dict(zip(labels['attrs'],names['a']));omap=dict(zip(labels['objs'],names['o']))
        a=[amap[v] for v in attrs];o=[omap[v] for v in objs]
        return dict(ea=['a photo of footwear made of '+v for v in a],eo=['a photo of '+v for v in o],
                    ec=['a photo of '+v+' made of '+u for v in o for u in a])
    clean=lambda s:s.replace('_',' ').replace('.',' ')
    return dict(ea=['a photo of a '+clean(a)+' object' for a in attrs],eo=['a photo of a '+clean(o) for o in objs],
                ec=['a photo of a '+clean(a)+' '+clean(o) for o in objs for a in attrs])


def knowledge_records(index):
    know={k:read(ROOT/'assets/knowledge'/index['name']/(k+'_dict.json')) for k in ('oi','oe','ae')}
    norm=lambda s:s.lower().replace('_',' ').strip()
    attrs,objs=index['attrs'],index['objs'];oi={norm(v):j for j,v in enumerate(objs)};rows=[]
    for parent,children in sorted(know['oi'].items()):
        for child in children:
            if norm(child) in oi:rows.append(dict(type=0,target=len(attrs)+oi[norm(child)],text=parent,prompt='a photo of '+parent))
    for kind,names,offset,ty in [('oe',objs,len(attrs),1),('ae',attrs,0,2)]:
        desc={norm(k):v for k,v in know[kind].items()}
        for j,name in enumerate(names):
            for v in desc[norm(name)]:
                prompt='a photo of footwear with '+v if index['name']=='ut-zap50k' else 'a photo of '+name.replace('_',' ')+': '+v
                rows.append(dict(type=ty,target=offset+j,text=v,prompt=prompt))
    return rows


def visual_tokens(v,images):
    x=v.conv1(images).flatten(2).transpose(1,2)
    x=torch.cat((v.class_embedding.to(x.dtype).expand(len(x),1,-1),x),1)
    x=v.ln_pre(x+v.positional_embedding.to(x.dtype));x=v.transformer(x.transpose(0,1)).transpose(0,1)
    x=unit((v.ln_post(x)@v.proj).float())
    return x[:,0],x[:,1:]


def query_patches(patches,queries,tau=.07):
    weights=(torch.einsum('qd,bld->bql',queries,patches)/tau).softmax(-1)
    return unit(torch.einsum('bql,bld->bqd',weights,patches))


@torch.no_grad()
def prepare(ds,checkpoint,cfg,device):
    ds=Path(ds);index=read(ds/'index.json');folder=ds/'cache';folder.mkdir(exist_ok=True)
    with lock(folder/'prepare.lock'):
        if (folder/'complete.json').exists():return
        seed_all(5351);model,tokenize=clip_load(checkpoint,device)
        if model.visual.proj.shape!=(1024,768):raise ValueError('Expected ViT-L/14')
        def encode(items):
            return torch.cat([unit(model.encode_text(tokenize(items[j:j+64],truncate=True).to(device)).float()).cpu() for j in range(0,len(items),64)])
        if not (folder/'text.pt').exists() or not (folder/'bank.pt').exists():
            ps=prompts(index);text={k:encode(v) for k,v in ps.items()};text['ec']=text['ec'].reshape(len(index['objs']),len(index['attrs']),-1)
            save(folder/'text.pt',text);write(folder/'prompts.json',ps)
            rows=knowledge_records(index);source=encode([r['prompt'] for r in rows]);anchors=torch.cat((text['ea'],text['eo']))
            target=torch.tensor([r['target'] for r in rows]);types=torch.tensor([r['type'] for r in rows])
            degree=torch.tensor([sum(s['type']==r['type'] and s['target']==r['target'] for s in rows) for r in rows],dtype=torch.float32)
            bank=dict(anchors=anchors,source=source,target=target,types=types,degree=degree,rows=rows,na=len(index['attrs']))
            for ty,want in [(0,set(range(len(index['attrs']),len(anchors)))),(1,set(range(len(index['attrs']),len(anchors)))),(2,set(range(len(index['attrs']))))]:
                if set(target[types==ty].tolist())!=want:raise ValueError('Knowledge owner coverage differs')
            save(folder/'bank.pt',bank)
        # A receipt is committed only after both arrays have been flushed.
        n=len(index['rows']);shapes={'g':(n,768),'patch':(n,256,768)}
        arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='r+' if (folder/(k+'.npy')).exists() else 'w+',dtype=np.float16,shape=s) for k,s in shapes.items()}
        receipt=folder/'progress.json';start=read(receipt)['images'] if receipt.exists() else 0
        loader=DataLoader(Images(index,range(start,n)),batch_size=cfg['cache_batch'],shuffle=False,num_workers=cfg['workers'],pin_memory=True)
        for image,a,o,p,ids in loader:
            image=image.to(device);g,patch=visual_tokens(model.visual,image)
            if start==0:
                torch.testing.assert_close(g,unit(model.encode_image(image).float()),atol=2e-5,rtol=2e-5)
            arrays['g'][ids.numpy()]=g.cpu().numpy().astype(np.float16);arrays['patch'][ids.numpy()]=patch.cpu().numpy().astype(np.float16)
            for v in arrays.values():v.flush()
            start=int(ids[-1])+1;write(receipt,dict(stage='cache',images=start,total=n))
            if start%480<cfg['cache_batch'] or start==n:print('CACHE',index['name'],start,'/',n,flush=True)
        files={p.name:sha(p) for p in [folder/'g.npy',folder/'patch.npy',folder/'text.pt',folder/'bank.pt']}
        write(folder/'complete.json',dict(images=n,sha256=files,clip=CLIP_SHA))


class FeatureCache:
    def __init__(self,ds,device):
        self.path=Path(ds)/'cache';self.device=device
        self.g=np.load(self.path/'g.npy',mmap_mode='r');self.patch=np.load(self.path/'patch.npy',mmap_mode='r')
        self.text=load(self.path/'text.pt');self.eo=self.text['eo'].to(device)

    @torch.no_grad()
    def batch(self,ids,objects=True):
        g=torch.from_numpy(np.array(self.g[ids],dtype=np.float32)).to(self.device)
        if not objects:return g,None
        patches=torch.from_numpy(np.array(self.patch[ids],dtype=np.float32)).to(self.device)
        fo=torch.cat([query_patches(patches,self.eo[j:j+64]) for j in range(0,len(self.eo),64)],1)
        return g,fo
