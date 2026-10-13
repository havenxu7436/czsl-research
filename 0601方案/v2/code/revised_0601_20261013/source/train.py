import argparse
import sys
import hashlib

def checkpoint_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right
from models.kcc_cpf import DDM
from dataset import CompositionDataset
import evaluator_ge
from tqdm import tqdm
from utils import utils
from config import cfg
from torch.utils.tensorboard import SummaryWriter

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None


def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
    """Decays learning rate following milestones in cfg.
    """
    milestones = cfg.TRAIN.lr_decay_milestones
    it = bisect_right(milestones, epoch)
    gamma = cfg.TRAIN.decay_factor ** it
    
    gammas = [gamma] * len(group_lrs)
    assert len(optimizer.param_groups) == len(group_lrs)
    i = 0
    for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
        param_group["lr"] = lr * gamma_group
        i += 1
        print(f"Group {i}, lr = {lr * gamma_group}")


def save_checkpoint(model_or_optim, name, cfg):
    """Saves checkpoint.
    """
    state_dict = model_or_optim.state_dict()
    path = os.path.join(
        f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.train_name}/{name}.pth')
    torch.save(state_dict, path)


def train(epoch, model, optimizer, trainloader, logger, device, cfg):
    model.train()
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)
    if not cfg.TRAIN.finetune_backbone and cfg.TRAIN.aug_epoch <= epoch:
        freeze(model.feat_extractor)

    list_meters = [
        'loss_total', 'loss_cpf', 'loss_oi', 'loss_oe', 'loss_ae', 'loss_logic_raw', 'loss_logic_weighted', 'full_delta_rms', 'loss_conditional_attr'
    ]
    if cfg.MODEL.use_obj_loss:
        list_meters.append('loss_aux_obj')
        list_meters.append('acc_aux_obj')
    if cfg.MODEL.use_attr_loss:
        list_meters.append('loss_aux_attr')
        list_meters.append('acc_aux_attr')
    if cfg.MODEL.use_emb_pair_loss:
        list_meters.append('emb_loss')
    if cfg.MODEL.use_composed_pair_loss:
        list_meters.append('composed_unseen_loss')
        list_meters.append('composed_seen_loss')

    dict_meters = { 
        k: utils.AverageMeter() for k in list_meters
    }

    acc_attr_meter = utils.AverageMeter()
    acc_obj_meter = utils.AverageMeter()
    acc_pair_meter = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    end_time = time.time()


    start_iter = (epoch - 1) * len(trainloader)
    for idx, batch in enumerate(tqdm(trainloader)):
        it = start_iter + idx + 1
        data_time.update(time.time() - end_time)

        for k in batch:
            if isinstance(batch[k], list): 
                continue
            batch[k] = batch[k].to(device, non_blocking=True)
        r = np.random.rand(1)
        if cfg.TRAIN.aug_epoch <= epoch and 0 <= r and r < 0.5:
            out = model(batch, flag=True)
        else:
            out = model(batch)
        loss = out['loss_total']
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        if bool(getattr(cfg.MODEL, 'kcc_logic', False)) and cfg.MODEL.kcc_beta > 0:
            if not out['loss_logic_raw'].requires_grad:
                raise ValueError('Logic loss disconnected')
            if idx == 0 or (idx + 1) % cfg.TRAIN.disp_interval == 0:
                print('BF_LOGIC epoch={} iter={} cpf={:.8g} oi={:.8g} oe={:.8g} ae={:.8g} weighted={:.8g}'.format(
                    epoch, idx + 1, *(float(out[k].detach()) for k in ('loss_cpf','loss_oi','loss_oe','loss_ae','loss_logic_weighted'))), flush=True)

        optimizer.zero_grad()
        loss.backward()
        if bool(getattr(cfg.MODEL, 'kcc_feature', False)) and idx == 0:
            head_g = sum(float(p.grad.detach().square().sum()) for p in model.full_head.parameters() if p.grad is not None) ** 0.5
            if not np.isfinite(head_g) or head_g <= 0:
                raise ValueError('Feature head has no finite gradient')
            print('FULL_GRADIENT epoch={} norm={} conditional_ce={} delta_rms={}'.format(epoch, head_g, float(out['loss_conditional_attr'].detach()), float(out['full_delta_rms'].detach())), flush=True)
        optimizer.step()

        if 'acc_attr' in out:
            acc_attr_meter.update(out['acc_attr'])
            acc_obj_meter.update(out['acc_obj'])
        acc_pair_meter.update(out['acc_pair'])
        for k in out:
            if k in dict_meters:
                dict_meters[k].update(out[k].item())
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (idx + 1) % cfg.TRAIN.disp_interval == 0:
            print(
                f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
                f'Loss: {dict_meters["loss_total"].avg:.3f}, '
                f'Acc_Obj: {acc_obj_meter.avg:.2f}, '
                f'Acc_Attr: {acc_attr_meter.avg:.2f}, '
                f'Acc_Pair: {acc_pair_meter.avg:.2f}, '
                f'Batch_time: {batch_time.avg:.3f}, Data_time: {data_time.avg:.3f}',
                flush=True)

            for k in out:
                if k in dict_meters:
                    logger.add_scalar('train/%s' % k, dict_meters[k].avg, it)
            logger.add_scalar('train/acc_pair', acc_pair_meter.avg, it)

            batch_time.reset()
            data_time.reset()
            acc_pair_meter.reset()
            if 'acc_attr' in out:
                acc_attr_meter.reset()
                acc_obj_meter.reset()
            for k in out:
                if k in dict_meters:
                    dict_meters[k].reset()

def cal_hit_rate(pre, gt, k=3):
    # print(pre)
    # print(gt)
    m = min(k, len(gt))
    count = 0
    for p in pre:
        if p in gt:
            count += 1
    return count / m


@torch.no_grad()
def validate_ge(epoch, model, testloader, evaluator, device, topk=1):
    model.eval()
    dset = testloader.dataset

    val_attrs, val_objs = zip(*dset.pairs)
    model.val_attrs = torch.tensor(
        [dset.attr2idx[a] for a in val_attrs],
        dtype=torch.long,
        device=device
    )
    model.val_objs = torch.tensor(
        [dset.obj2idx[o] for o in val_objs],
        dtype=torch.long,
        device=device
    )
    model.val_pairs = dset.pairs

    all_attr_gt = []
    all_obj_gt = []
    all_pair_gt = []
    all_pred = []

    for data in tqdm(testloader, desc="Testing"):
        for key, value in data.items():
            if torch.is_tensor(value):
                data[key] = value.to(device, non_blocking=True)

        out = model(data)
        all_pred.append({
            pair: score.detach().cpu()
            for pair, score in out["scores"].items()
        })
        all_attr_gt.append(data["attr"].cpu())
        all_obj_gt.append(data["obj"].cpu())
        all_pair_gt.append(data["pair"].cpu())

    all_attr_gt = torch.cat(all_attr_gt)
    all_obj_gt = torch.cat(all_obj_gt)
    all_pair_gt = torch.cat(all_pair_gt)

    all_pred_dict = {
        pair: torch.cat([batch_pred[pair] for batch_pred in all_pred])
        for pair in dset.pairs
    }

    results = evaluator.score_model(
        all_pred_dict,
        all_obj_gt,
        bias=1e3,
        topk=topk
    )
    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=topk
    )

    stats["a_epoch"] = epoch
    keys = [
        "AUC", "best_hm", "best_seen", "best_unseen",
        "closed_attr_match", "closed_obj_match"
    ]
    print(
        f"Val/Test Epoch {epoch}: " +
        " | ".join(
            f"{key}={stats[key]:.4f}"
            for key in keys if key in stats
        )
    )

    del model.val_attrs
    del model.val_objs
    del model.val_pairs

    return stats["AUC"], stats["best_hm"]



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main_worker(gpu, cfg):
    """Main training code.
    """
    seed = cfg.TRAIN.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Use GPU {gpu} for training', flush=True)
    torch.cuda.set_device(gpu)
    device = f'cuda:{gpu}'

    # Log directory for tensorboard.
    log_dir = f'{cfg.TRAIN.log_dir}/{cfg.config_name}_{cfg.TRAIN.train_name}'
    logger = SummaryWriter(log_dir=log_dir)

    # Directory to save checkpoints.
    ckpt_dir = f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.train_name}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    cfg.TRAIN.batch_size = cfg.TRAIN.batch_size
    print('Batch size on each gpu: %d' % cfg.TRAIN.batch_size)
    
    print('Prepare dataset')
    trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, open_world=cfg.EVAL.open_world, cfg=cfg)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAIN.batch_size, shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker)
    valset = CompositionDataset(
        phase='val', split=cfg.DATASET.splitname, open_world=cfg.EVAL.open_world, cfg=cfg)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
        num_workers=cfg.TRAIN.num_workers)
    testset = CompositionDataset(
        phase='test', split=cfg.DATASET.splitname, open_world=cfg.EVAL.open_world, cfg=cfg)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
        num_workers=cfg.TRAIN.num_workers)

    # model = COT(trainset, cfg)
    model = DDM(trainset, cfg)
    model.to(device)

    # # Load checkpoint
    # if cfg.MODEL.load_checkpoint is not None:
    #     model.load_state_dict(torch.load(cfg.MODEL.load_checkpoint))

    if not cfg.TRAIN.finetune_backbone:
        freeze(model.feat_extractor)
    total_params = utils.count_parameters(model)
    print(f"Total Trainable Params: {total_params}")

    evaluator_val_ge = evaluator_ge.Evaluator(valset, model)
    evaluator_test_ge = evaluator_ge.Evaluator(testset, model)
    
    torch.backends.cudnn.benchmark = False  # Same reproducibility rule for A_ref/B/E_ref/F.

    params_word_embedding = []
    params_encoder = []
    params = []
    params_revision_scalar = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name == 'rev_lambda_raw':
            params_revision_scalar.append(p)
        elif 'attr_embedder' in name or 'obj_embedder' in name:
            if cfg.TRAIN.lr_word_embedding > 0:
                params_word_embedding.append(p)
        elif name.startswith('feat_extractor'):
            params_encoder.append(p)
        else:
            params.append(p)

    if cfg.TRAIN.lr_word_embedding > 0:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder},
            {'params': params_word_embedding, 'lr': cfg.TRAIN.lr_word_embedding},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)  
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr_word_embedding, cfg.TRAIN.lr]
    else:
        optimizer = optim.Adam([
            {'params': params_encoder, 'lr': cfg.TRAIN.lr_encoder},
            {'params': params, 'lr': cfg.TRAIN.lr},
        ], lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.wd)
        group_lrs = [cfg.TRAIN.lr_encoder, cfg.TRAIN.lr]

    if params_revision_scalar:
        # Weight decay on eta would implicitly pull sigmoid(eta) toward0.5.
        optimizer.add_param_group({'params': params_revision_scalar, 'lr': cfg.TRAIN.lr, 'weight_decay': 0.0})
        group_lrs.append(cfg.TRAIN.lr)

    start_epoch = cfg.TRAIN.start_epoch
    epoch = start_epoch

    best_records = {
        'val/best_auc': 0.0,
        'val/best_hm': 0.0,
        'test/auc_at_best_val': 0.0,
        'test/hm_at_best_val': 0.0,
    }

    # auc, best_hm = validate_ge(epoch, model, valloader, evaluator_val_ge, device, topk=cfg.EVAL.topk)

    best_records["val/best_auc"] = -float("inf")
    best_epoch = -1

    resume_path = os.environ.get('FULL0601_RESUME')
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        resume_best = os.path.join(ckpt_dir, 'best_auc.pth')
        if not os.path.isfile(resume_best) or checkpoint_sha256(resume_best) != state['best_checkpoint_sha256']:
            raise ValueError('Best checkpoint changed after the last durable epoch; preserve evidence, do not silently resume inconsistent state')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        best_records, best_epoch = state['best_records'], state['best_epoch']
        epoch = state['epoch'] + 1
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        print('FULL_RESUME completed_epoch={} next_epoch={}'.format(state['epoch'], epoch), flush=True)

    while epoch <= cfg.TRAIN.max_epoch:
        if epoch == cfg.TRAIN.aug_epoch:
            print('Start Minority Attribute Augmentation')
            freeze(model.feat_extractor)
            trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=cfg.TRAIN.batch_size // 2, shuffle=True,
            num_workers=cfg.TRAIN.num_workers,
            pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

        epoch_start_time = time.time()
        train(epoch, model, optimizer, trainloader, logger, device, cfg)
        max_gpu_usage_mb = torch.cuda.max_memory_allocated(device=device) / 1048576.0
        print(f'Max GPU usage in MB till now: {max_gpu_usage_mb}')

        if cfg.TRAIN.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        if epoch < cfg.TRAIN.start_epoch_validate:
            epoch += 1
            continue

        if epoch % cfg.TRAIN.eval_every_epoch == 0:
            print("Validation set ===>")

            auc, best_hm = validate_ge(
                epoch,
                model,
                valloader,
                evaluator_val_ge,
                device,
                topk=cfg.EVAL.topk
            )

            logger.add_scalar(
                "val/auc",
                auc,
                epoch * len(trainloader)
            )
            logger.add_scalar(
                "val/best_hm",
                best_hm,
                epoch * len(trainloader)
            )

            if auc > best_records["val/best_auc"]:
                best_records["val/best_auc"] = auc
                best_records["val/best_hm"] = best_hm
                best_epoch = epoch

                save_checkpoint(model, "best_auc", cfg)

                print(
                    f"Saved new best checkpoint: "
                    f"epoch={epoch}, AUC={auc:.6f}, HM={best_hm:.6f}"
                )

        last_path = os.path.join(ckpt_dir, 'last_resume.pth')
        tmp_path = last_path + '.pending'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'epoch': epoch, 'best_records': best_records, 'best_epoch': best_epoch,
                    'best_checkpoint_sha256': checkpoint_sha256(os.path.join(ckpt_dir, 'best_auc.pth')),
                    'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, tmp_path)
        os.replace(tmp_path, last_path)
        print('FULL_EPOCH_FINISHED epoch={} seconds={:.2f}'.format(epoch, time.time()-epoch_start_time), flush=True)
        epoch += 1

    best_path = os.path.join(
        cfg.TRAIN.checkpoint_dir,
        f"{cfg.config_name}_{cfg.TRAIN.train_name}",
        "best_auc.pth"
    )

    if not os.path.isfile(best_path):
        raise FileNotFoundError(
            f"Best checkpoint was not created: {best_path}"
        )

    print(
        f"Loading validation-best checkpoint from epoch {best_epoch}: "
        f"{best_path}"
    )

    model.load_state_dict(
        torch.load(best_path, map_location=device)
    )

    if bool(getattr(cfg.TRAIN, "kcc_test_at_end", False)):
        print("Test set with validation-best checkpoint ===>")

        test_auc, test_hm = validate_ge(
            best_epoch,
            model,
            testloader,
            evaluator_test_ge,
            device,
            topk=cfg.EVAL.topk
        )

        best_records["test/auc_at_best_val"] = test_auc
        best_records["test/hm_at_best_val"] = test_hm

        logger.add_scalar(
            "test/auc_at_best_val",
            test_auc,
            best_epoch * len(trainloader)
        )
        logger.add_scalar(
            "test/hm_at_best_val",
            test_hm,
            best_epoch * len(trainloader)
        )

    logger.close()
    
    print('Done: %s' % cfg.config_name)
    print('New Best AUC:',best_records['val/best_auc'])
    print('New Best HM:',best_records['val/best_hm'])
    print('New Best AUC Test:',best_records['test/auc_at_best_val'])
    print('New Best HM Test:',best_records['test/hm_at_best_val'])
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    print(cfg)

    seed = cfg.TRAIN.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    print('Random seed:', seed)
    cfg.TRAIN.seed = seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    main_worker(0, cfg)


if __name__ == "__main__":
    main()
