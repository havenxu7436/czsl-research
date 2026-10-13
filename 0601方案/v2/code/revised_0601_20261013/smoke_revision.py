"""Real TRAIN images on CUDA; independent normalization and gradient checks."""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).absolute().parents[1]))
from config import cfg as defaults
from dataset import CompositionDataset
from models.kcc_cpf import DDM, LegacyDDM


def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False


def norm(grads):
    return sum(float(g.detach().float().square().sum()) for g in grads if g is not None)**0.5


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--cfg',required=True); a.add_argument('--out',required=True)
    args = a.parse_args()
    dest = Path(args.out)
    if dest.exists(): raise FileExistsError('Preserve smoke evidence')
    c = defaults.clone(); c.merge_from_file(args.cfg)
    ds = CompositionDataset(phase='train',split=c.DATASET.splitname,open_world=False,cfg=c)
    batch = next(iter(torch.utils.data.DataLoader(ds,batch_size=4,shuffle=False,num_workers=0)))
    batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    off = c.clone()
    off.MODEL.kcc_feature=False; off.MODEL.kcc_logic=False; off.MODEL.kcc_beta=0.0
    seed(c.TRAIN.seed)
    original = LegacyDDM(ds,off).cuda().eval()
    native = {k:v.detach().cpu() for k,v in original.state_dict().items()}
    rng = torch.get_rng_state(),torch.cuda.get_rng_state_all()
    del original; torch.cuda.empty_cache()
    seed(c.TRAIN.seed)
    model = DDM(ds,c).cuda().eval()
    assert torch.equal(rng[0],torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(rng[1],torch.cuda.get_rng_state_all()))
    assert all(torch.equal(v,model.state_dict()[k].detach().cpu()) for k,v in native.items())
    del native
    model.val_objs,model.val_attrs,model.val_pairs=model.train_objs,model.train_attrs,ds.train_pairs
    # Delayed-beta branch is checked at an active epoch AND at epoch1.
    model.set_epoch(c.TRAIN.max_epoch)
    out,z,nodes=model.formula_train_terms(batch,True)
    weight=model.rev_lambda_raw.sigmoid() if hasattr(model,'rev_lambda_raw') else model.kcc_lambda
    conditional=z['conditional']+(weight*model.rev_prior[None] if model.kcc_calib else 0)
    expected=F.log_softmax(z['obj'].float(),-1)[:,:,None]+F.log_softmax(conditional.float(),-1)
    assert torch.allclose(z['score'],expected,atol=2e-6)
    assert torch.allclose(z['score'].exp().sum((1,2)),torch.ones(4,device='cuda'),atol=2e-6)
    assert torch.allclose(z['score'].exp().sum(2),z['obj'].softmax(-1),atol=2e-6)
    with torch.no_grad(): infer=model.val_forward(batch)['pred']
    assert torch.allclose(infer,z['score'][:,model.train_objs,model.train_attrs],atol=2e-6)
    assert torch.allclose(out['loss_total'],out['loss_cpf']+out['loss_logic_weighted'])
    rows=torch.arange(4,device='cuda')
    target=z['log_conditional'][rows,batch['obj']] if model.rev_attr_mode=='gt' else torch.logsumexp(z['score'],1)
    assert torch.allclose(out['loss_conditional_attr'],F.nll_loss(target,batch['attr']))
    gradients={}
    visual=[v for n,v in model.named_parameters() if n.startswith(('objc_embedder.','attrc_embedder.')) and v.requires_grad]
    backbone=[v for n,v in model.named_parameters() if n.startswith('feat_extractor.') and v.requires_grad]
    gradients['visual_task']=norm(torch.autograd.grad(out['loss_cpf'],visual,retain_graph=True,allow_unused=True))
    if backbone: gradients['backbone_task']=norm(torch.autograd.grad(out['loss_cpf'],backbone,retain_graph=True,allow_unused=True))
    if model.full_enabled:
        head=list(model.full_head.parameters())
        gradients['head_task']=norm(torch.autograd.grad(out['loss_cpf'],head,retain_graph=True,allow_unused=True))
        gradients['head_attr']=norm(torch.autograd.grad(out['loss_conditional_attr'],head,retain_graph=True,allow_unused=True))
        if nodes is not None: gradients['AE_to_head']=norm(torch.autograd.grad(out['loss_ae'],head,retain_graph=True,allow_unused=True))
    if nodes is not None:
        for term in ('loss_oi','loss_oe','loss_ae'):
            gradients[term+'_visual']=norm(torch.autograd.grad(out[term],visual,retain_graph=True,allow_unused=True))
        assert torch.allclose(nodes['pair'].sum((1,2)),torch.ones(4,device='cuda'),atol=2e-6)
        if model.rev_delay:
            model.set_epoch(1)
            delayed,_=model.revision_logic(z,batch)
            assert delayed['loss_logic_weighted'].item()==0
            assert delayed['loss_logic_raw'].requires_grad
    if hasattr(model,'rev_lambda_raw'):
        gradients['lambda_task']=norm(torch.autograd.grad(out['loss_cpf'],[model.rev_lambda_raw],retain_graph=True))
    assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
    out['loss_total'].backward()
    # Native visual composition must remain active (unlike old G/H raw head).
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.compose.parameters())
    report=dict(passed=True,revision='unified_20261013',variant=c.TRAIN.kcc_variant,gradients=gradients,
                normalized=True,same_native_initialization=True,native_rng_preserved=True,
                train_infer_identity=True,train_images=4,test_images_read=0,optimizer_steps=0,
                scope='Wiring check, not a trained accuracy result')
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('REVISION_REAL_CUDA_SMOKE_PASSED',json.dumps(report),flush=True)


if __name__=='__main__': main()
