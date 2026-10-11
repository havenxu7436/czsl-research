import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .common import candidates, empty_output, move, sha, write_json


def evaluate(model, cfg, phase, setting, output, save_scores=False, force_off=False):
    from dataset import CompositionDataset
    import evaluator_ge
    cfg = cfg.clone()
    cfg.EVAL.open_world = (setting == "ow")
    if phase != "val" and not bool(getattr(cfg.TRAIN, "transition_test_allowed", False)):
        raise ValueError("Test evaluation requires a separately frozen final protocol")
    out = empty_output(output)
    ds = CompositionDataset(phase=phase, split=cfg.DATASET.splitname,
                            open_world=(setting == "ow"), cfg=cfg)
    ai, oi, pairs = candidates(model, ds)
    device = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.TRAIN.test_batch_size,
                                    shuffle=False, num_workers=cfg.TRAIN.num_workers)
    model.eval()
    ev = evaluator_ge.Evaluator(ds, model)
    n, nc = len(ds), len(pairs)
    print("EVALUATE", phase, setting, n, nc,
          "score_GiB", 4 * n * nc / (1024 ** 3), flush=True)
    truth = {k: np.empty(n, np.int64) for k in ("attr", "obj", "pair")}
    top1 = np.empty(n, np.int64)
    diagnostics, batches = {}, 0
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="scores_tmp_", dir=str(out)) as temp:
        score_file = out / "scores.dat" if save_scores else Path(temp) / "scores.dat"
        mm = np.memmap(str(score_file), mode="w+", dtype=np.float32, shape=(n, nc))
        offset = 0
        with torch.no_grad():
            for batch in dl:
                batch = move(batch, device)
                if hasattr(model, "router"):
                    result, diag = model.score(batch, ai, oi, force_off=force_off,
                                                return_diag=True)
                    for key, value in diag.items():
                        if key == "max_abs_residual":
                            diagnostics[key] = max(diagnostics.get(key, 0), value)
                        else:
                            diagnostics[key] = diagnostics.get(key, 0) + value * result.size(0)
                else:
                    result = model.raw_forward(batch)["pred"]
                if not torch.isfinite(result).all():
                    raise ValueError("Non-finite evaluation scores")
                values = result.detach().float().cpu().numpy()
                sl = slice(offset, offset + len(values))
                mm[sl] = values
                top1[sl] = values.argmax(1)
                for key in truth:
                    truth[key][sl] = batch[key].detach().cpu().numpy()
                offset += len(values)
                batches += 1
                if batches % 25 == 0:
                    print("EVAL_IMAGES", offset, "/", n, flush=True)
        if offset != n:
            raise ValueError("Evaluation image count mismatch")
        mm.flush()
        matrix = torch.from_numpy(mm)
        pred = {pair: matrix[:, i] for i, pair in enumerate(pairs)}
        ga, go, gp = [torch.from_numpy(truth[k]) for k in ("attr", "obj", "pair")]
        results = ev.score_model(pred, go, bias=1e3, topk=1)
        stats = ev.evaluate_predictions(results, ga, go, gp, pred, topk=1)
        metrics = {}
        for key, value in stats.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            a = np.asarray(value)
            if a.size == 1:
                metrics[key] = float(a.item())
        for key in ("AUC", "best_hm", "best_seen", "best_unseen", "hm_seen", "hm_unseen"):
            if key not in metrics or not np.isfinite(metrics[key]):
                raise ValueError("Invalid metric: " + key)
        record = {"run_id": Path(cfg.TRAIN.kcc_run_dir).name,
                  "dataset": cfg.DATASET.name, "variant": cfg.TRAIN.kcc_variant,
                  "seed": int(cfg.TRAIN.seed), "phase": phase, "setting": setting,
                  "score_mode": "legacy", "metrics_raw": metrics,
                  "n_images": n, "n_candidates": nc, "score_shape": [n, nc],
                  "scores_saved": save_scores, "eval_seconds": time.time() - start,
                  "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
                  "diagnostics": {k: v if k == "max_abs_residual" else v / n
                                  for k, v in diagnostics.items()},
                  "config_sha256": __import__("hashlib").sha256(cfg.dump().encode()).hexdigest(),
                  "bank_manifest_sha256": sha(Path(cfg.MODEL.description_bank) / "manifest.json"),
                  "checkpoint_sha256": getattr(model, "evaluation_checkpoint_sha256", None),
                  "force_off": bool(force_off or getattr(model, "off_mode", False))}
        write_json(out / "metrics.json", record)
        write_json(out / "candidates.json", pairs)
        write_json(out / "image_ids.json", [item[0] for item in ds.data])
        np.savez_compressed(str(out / "labels.npz"), **truth, top1_at_bias0=top1)
        (out / "config_eval_effective.yml").write_text(cfg.dump(), encoding="utf-8")
        del results, pred, matrix, mm
    print("METRICS", json.dumps(metrics), flush=True)
    return record
