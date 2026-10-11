# CPF 与 LOGICZSL 基线修复执行手册

## 0. 目标与边界

本手册用于完成两个阶段：

1. 重建并跑通 CPF：启用 text-enhanced object learning 与 object-guided attribute learning，恢复验证、测试和最佳 checkpoint 保存；
2. 先跑通 LogiCZSL 工程中的 Troika 非逻辑基线，再在知识文件齐全后修复 Logic-Troika。

需要明确区分：

- 修复后的 CPF 属于根据现有代码和论文结构得到的重建实现；在作者缺失文件和实现细节未补齐前，不能保证逐位复现论文数值；
- Troika 非逻辑基线不是完整 LOGICZSL；
- 论文完整 Logic-Troika 依赖压缩包中缺失的 OI/OE/AE 知识文件；在缺失文件时构造的假知识只能用于软件链路测试，不能作为论文复现结果。

本手册假设服务器目录为：

```text
/data/$USER/czsl_repro/
├── data/
├── inputs/
├── pretrained/
├── runs/
└── src/
```

---

## 1. 建立服务器修复副本

不要直接修改原始解压目录：

```bash
export CZSL_ROOT="/data/$USER/czsl_repro"
export SRC_ROOT="$CZSL_ROOT/src"
export DATA_ROOT="$CZSL_ROOT/data"
export RUN_ROOT="$CZSL_ROOT/runs"

export CPF_ORIG="$SRC_ROOT/CRMask_copy"
export LOGIC_ORIG="$SRC_ROOT/LogiCZSL/models/Troika"
export CPF_FIX="$SRC_ROOT/CPF_fixed"
export LOGIC_FIX="$SRC_ROOT/LOGICZSL_fixed"

test ! -e "$CPF_FIX" || { echo "$CPF_FIX 已存在，停止"; return 1; }
test ! -e "$LOGIC_FIX" || { echo "$LOGIC_FIX 已存在，停止"; return 1; }

cp -a "$CPF_ORIG" "$CPF_FIX"
cp -a "$LOGIC_ORIG" "$LOGIC_FIX"

cd "$CPF_FIX"
git init
git add .
git commit -m "original CPF archive"

cd "$LOGIC_FIX"
git init
git add .
git commit -m "original LogiCZSL archive"
```

每完成一个阶段都执行：

```bash
git add .
git commit -m "描述本阶段修改"
```

---

# 第一部分：CPF

## 2. CPF 环境检查

```bash
conda activate cpf

python - <<'PY'
import torch, torchvision, numpy, scipy, yacs
print('torch=', torch.__version__)
print('torchvision=', torchvision.__version__)
print('cuda_available=', torch.cuda.is_available())
print('cuda_runtime=', torch.version.cuda)
print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
PY
```

不要重新安装其他项目的 `requirements.txt` 覆盖该环境。

## 3. 确定 CPF 唯一模型入口

现有训练入口使用 `ddm_v2.py`，但它的普通训练路径没有启用完整 TEO/OGA。`models/ddm.py` 更接近组合后的实现。

修改 `train.py` 和 `test.py`：

```python
from models.ddm import DDM
```

可以执行：

```bash
cd "$CPF_FIX"

sed -i 's/from models\.ddm_v2 import DDM/from models.ddm import DDM/' train.py
sed -i 's/from models\.ddm_v2 import DDM/from models.ddm import DDM/' test.py
```

同时修复 `test.py` 中缺失的导入：

```bash
sed -i 's/from dataset_visualization import CompositionDataset/from dataset import CompositionDataset/' test.py
sed -i 's/import evaluator_ge_v/import evaluator_ge/' test.py
sed -i 's/evaluator_ge_v\.Evaluator/evaluator_ge.Evaluator/' test.py
```

检查修改结果：

```bash
grep -nE 'from models|from dataset|evaluator_ge' train.py test.py
```

## 4. 修复 CPF 退化注意力

打开：

```text
models/ddm_utils.py
```

把 `image_soft_weights()` 整个替换为：

```python
def image_soft_weights(query, key):
    """
    query: [B, D]
    key:   [B, L, D]
    return: [B, L]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = torch.einsum("bld,bd->bl", key, query)
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)
```

把 `choose_texts()` 替换为：

```python
def choose_texts(query, key):
    """
    query: [B, D]
    key:   [N, D]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = query @ key.t()
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)
```

把 `choose_visual_tokens()` 替换为：

```python
def choose_visual_tokens(query, key):
    """
    query: [B, D]
    key:   [B, L, D]
    """
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = torch.einsum("bld,bd->bl", key, query)
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)
```

必须确认下面两行已删除：

```python
query = torch.ones(size=query.size()).cuda()
key = torch.ones(size=key.size()).cuda()
```

检查：

```bash
grep -n "torch.ones" models/ddm_utils.py
python -m py_compile models/ddm_utils.py
```

第一条命令不应输出有效代码行。

## 5. 把 attribute 分支改成 object-guided

`models/ddm.py` 当前用 `attr_guide_texts` 选择中间视觉 token。为了形成 object-conditioned attribute representation，应改用增强后的 object representation `vis_obj_new`。

```bash
cd "$CPF_FIX"

grep -n "choose_visual_tokens(attr_guide_texts" models/ddm.py
sed -i 's/choose_visual_tokens(attr_guide_texts, /choose_visual_tokens(vis_obj_new, /g' models/ddm.py
grep -n "choose_visual_tokens(vis_obj_new" models/ddm.py
```

训练和验证路径正常应共出现 6 处。

原来生成 `attr_guide_texts` 的两行暂时可以保留，虽然它们不再参与计算。完成基线验证后再清理死代码。

## 6. 补齐 CPF evaluator

CPF 压缩包缺少 `evaluator_ge.py`。可以先从同一批代码中的 LogiCZSL evaluator 提取兼容实现：

```bash
cd "$CPF_FIX"

{
  printf '%s\n' \
    'import numpy as np' \
    'import torch' \
    'from scipy.stats import hmean' \
    'device = "cuda" if torch.cuda.is_available() else "cpu"' \
    ''
  sed -n '27,364p' "$LOGIC_FIX/test.py"
} > evaluator_ge.py
```

检查：

```bash
python -m py_compile evaluator_ge.py
python -c 'import evaluator_ge; print("evaluator import passed")'
```

这一步用于先跑通 Seen、Unseen、HM 和 AUC。正式论文实验时，应再与标准 CZSL evaluator 核对宏平均、closed-world mask 和 open-world mask。

## 7. 替换 CPF 的 `validate_ge()`

把 `train.py` 中原来的 `validate_ge()` 整个替换为：

```python
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
```

这个实现不再依赖 `out['obj_scores']` 和 `out['attr_scores']`，也不再停留在原来的 attribute hit-rate 调试逻辑。

## 8. 恢复 CPF 最佳 checkpoint

在训练循环的验证部分，把只调用但不接收结果的：

```python
validate_ge(...)
```

改成：

```python
auc, best_hm = validate_ge(
    epoch,
    model,
    valloader,
    evaluator_val_ge,
    device,
    topk=cfg.EVAL.topk
)

logger.add_scalar("val/auc", auc, epoch * len(trainloader))
logger.add_scalar("val/best_hm", best_hm, epoch * len(trainloader))

if auc > best_records["val/best_auc"]:
    best_records["val/best_auc"] = auc
    best_records["val/best_hm"] = best_hm
    save_checkpoint(model, "best_auc", cfg)
    print(f"Saved new best checkpoint: epoch={epoch}, AUC={auc:.6f}")
```

删除每个 epoch 都在 test 集上评估的调用。训练完成后再加载验证集最佳 checkpoint：

```python
best_path = os.path.join(
    cfg.TRAIN.checkpoint_dir,
    f"{cfg.config_name}_{cfg.TRAIN.train_name}",
    "best_auc.pth"
)

if os.path.isfile(best_path):
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_auc, test_hm = validate_ge(
        epoch,
        model,
        testloader,
        evaluator_test_ge,
        device,
        topk=cfg.EVAL.topk
    )
    best_records["test/auc_at_best_val"] = test_auc
    best_records["test/hm_at_best_val"] = test_hm
```

同时把 `main_worker()` 中的：

```python
torch.backends.cudnn.benchmark = True
```

改成：

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

## 9. 修复 CPF `test.py`

确认开头导入为：

```python
from models.ddm import DDM
from dataset import CompositionDataset
import evaluator_ge
```

确认 evaluator 创建为：

```python
evaluator_test_ge = evaluator_ge.Evaluator(testset, model)
```

把 checkpoint 加载改为：

```python
model.load_state_dict(
    torch.load(path_to_model, map_location=device),
    strict=True
)
```

检查：

```bash
cd "$CPF_FIX"
conda activate cpf

python -m compileall -q .
python -c 'import train; print("CPF train import passed")'
```

不要直接执行 `import test`，因为当前 `test.py` 是顶层执行脚本，导入时会立即解析命令行参数。

## 10. CPF 单 epoch 冒烟

第一轮只跑 UT-Zappos：

```bash
conda activate cpf
cd "$CPF_FIX"

python train.py \
  --cfg config/ut-zappos.yml \
  DATASET.root_dir "$DATA_ROOT/ut-zappos" \
  TRAIN.log_dir "$RUN_ROOT/cpf_smoke/logs" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf_smoke/checkpoints" \
  TRAIN.num_workers 0 \
  TRAIN.batch_size 4 \
  TRAIN.test_batch_size 8 \
  TRAIN.max_epoch 1 \
  TRAIN.final_max_epoch 1 \
  TRAIN.start_epoch_validate 1 \
  TRAIN.aug_epoch 999
```

成功标准：

- 至少完成一个训练 epoch；
- `loss_total` 有限，不出现 NaN；
- validation 输出 AUC 和 best HM；
- 生成非空的 `best_auc.pth`；
- 不再出现 evaluator、维度或 key 缺失错误。

查找 checkpoint：

```bash
find "$RUN_ROOT/cpf_smoke/checkpoints" -name best_auc.pth -ls
```

独立测试：

```bash
CPF_CKPT=$(find "$RUN_ROOT/cpf_smoke/checkpoints" -name best_auc.pth | head -n 1)

python test.py \
  --cfg config/ut-zappos.yml \
  --load "$CPF_CKPT" \
  DATASET.root_dir "$DATA_ROOT/ut-zappos" \
  TRAIN.num_workers 0 \
  TRAIN.test_batch_size 8
```

## 11. CPF 正式训练

冒烟通过后，先保持配置中其他超参数不变，只恢复正式 epoch、batch 和 worker：

```bash
python train.py \
  --cfg config/ut-zappos.yml \
  DATASET.root_dir "$DATA_ROOT/ut-zappos" \
  TRAIN.log_dir "$RUN_ROOT/cpf/ut-zappos/logs" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf/ut-zappos/checkpoints" \
  TRAIN.num_workers 4 \
  TRAIN.batch_size 64 \
  TRAIN.test_batch_size 64 \
  TRAIN.max_epoch 10 \
  TRAIN.final_max_epoch 10 \
  TRAIN.aug_epoch 999 \
  2>&1 | tee "$RUN_ROOT/cpf/ut-zappos/train.log"
```

显存不足时依次把 batch size 降到 32、16、8，不要同时改模型和损失。

---

# 第二部分：LogiCZSL/Troika

## 12. LogiCZSL 环境检查

不要安装压缩包里的 `requirements.txt`。其中包含作者整台机器的软件和无效的绝对路径。

```bash
conda activate logiczsl

python - <<'PY'
import torch, torchvision, numpy, scipy, pandas, cv2, einops, transformers
print('torch=', torch.__version__)
print('torchvision=', torchvision.__version__)
print('cuda_available=', torch.cuda.is_available())
print('cuda_runtime=', torch.version.cuda)
print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
PY
```

## 13. 补回缺失的 `model/ddm_utils.py`

压缩包只有旧 Python 版本的 `.pyc`，没有源文件。创建：

```text
model/ddm_utils.py
```

内容为：

```python
import torch
import torch.nn.functional as F


def choose_texts(query, key):
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = query @ key.t()
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)


def choose_visual_tokens(query, key):
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    score = torch.einsum("bld,bd->bl", key, query)
    score = score / (query.size(-1) ** 0.5)
    return F.softmax(score, dim=-1)


def image_soft_weights(query, key):
    return choose_visual_tokens(query, key)
```

检查：

```bash
cd "$LOGIC_FIX"
python -m py_compile model/ddm_utils.py
```

## 14. 先从 dataset 中关闭不完整逻辑分支

在 `dataset.py` 中删除或注释第 133–169 行附近的整个 `Logic` 数据加载块，包括：

```text
<your_oi_dict.json>
<your_oe_dict.json>
ae_dict.json
compatibility.csv
```

然后把 `__getitem__()` 整个替换为：

```python
def __getitem__(self, index):
    image, attr, obj = self.data[index]
    img = self.transform(self.loader(image))

    if self.phase == "train":
        pair_idx = self.train_pair_to_idx[(attr, obj)]
    else:
        pair_idx = self.pair2idx[(attr, obj)]

    return [
        img,
        self.attr2idx[attr],
        self.obj2idx[obj],
        pair_idx,
    ]
```

train、val、test 由此统一返回：

```text
image, attribute_index, object_index, pair_index
```

## 15. 修复 Troika 模型工厂

把 `model/model_factory.py` 整个替换为：

```python
from model.troika import Troika


def get_model(config, dataset, attributes, classes, offset, **kwargs):
    if config.model_name != "troika":
        raise NotImplementedError(
            f"Unrecognized model name: {config.model_name}"
        )

    return Troika(
        config,
        dataset=dataset,
        attributes=attributes,
        classes=classes,
        offset=offset,
    )
```

保留 `**kwargs` 是为了后续接 Logic-Troika 时不再次破坏调用接口。

## 16. 修复 `train.py` 的模型创建

删除 `train.py` 中读取以下字段的代码：

```python
train_dataset.hc_parent
train_dataset.oda_child
train_dataset.ada_child
```

模型创建改为：

```python
model = get_model(
    config,
    dataset=train_dataset,
    attributes=attributes,
    classes=classes,
    offset=offset,
).cuda()
```

同时删除无用且可能导致导入错误的：

```python
from transformers.utils.fx import torch_flip
```

## 17. 修复训练/评估模式切换

在每个 epoch 开始后立即加入：

```python
model.train()
optimizer.zero_grad(set_to_none=True)
```

即：

```python
for i in range(config.epoch_start, config.epochs):
    model.train()
    optimizer.zero_grad(set_to_none=True)
```

否则第一轮验证调用 `model.eval()` 后，后续 epoch 会继续以 eval 模式训练。

## 18. 恢复 Troika 最佳 checkpoint

把原来注释的最佳模型保存逻辑替换为：

```python
if config.val_metric == "best_loss":
    is_best = val_result["best_loss"] < best_loss
else:
    is_best = val_result[config.val_metric] > best_metric

if is_best:
    if config.val_metric == "best_loss":
        best_loss = val_result["best_loss"]
    else:
        best_metric = val_result[config.val_metric]

    best_epoch = i + 1
    torch.save(
        model.state_dict(),
        os.path.join(config.save_path, "val_best.pt")
    )
    print(f"Saved val_best.pt at epoch {best_epoch}")
```

删除每个 epoch 的：

```python
print("Evaluating test dataset:")
evaluate(model, test_dataset, config)
```

训练完成后增加：

```python
best_path = os.path.join(config.save_path, "val_best.pt")
if os.path.isfile(best_path):
    model.load_state_dict(
        torch.load(best_path, map_location="cuda")
    )
    print("Evaluating test dataset with validation-best checkpoint:")
    evaluate(model, test_dataset, config)
```

## 19. 避免 FP16/FP32 类型冲突

`model/troika.py` 的 CLIP 可能为 FP16，而新建 adapter/disentangler 默认为 FP32。

为了先稳定跑通，在 `Troika.__init__()` 最后增加：

```python
self.float()
self.dtype = torch.float32
```

这会增加显存占用，因此冒烟阶段使用 batch size 4。确认整体链路正确后，再统一新增模块 dtype 或引入 AMP。

同时删除 `model/common.py` 中无用的：

```python
from turtle import shape
from stringprep import b1_set
```

## 20. 准备 CLIP ViT-L/14

```bash
mkdir -p "$CZSL_ROOT/pretrained"

wget -c \
  -O "$CZSL_ROOT/pretrained/ViT-L-14.pt" \
  'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt'

echo 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  '"$CZSL_ROOT/pretrained/ViT-L-14.pt" |
sha256sum -c -
```

服务器无法访问下载源时，在其他电脑下载后用 `scp` 上传，不要用不明模型权重替代。

## 21. Troika 导入检查

```bash
conda activate logiczsl
cd "$LOGIC_FIX"

python -m compileall -q .
python - <<'PY'
from dataset import CompositionDataset
from model.model_factory import get_model
from model.troika import Troika
print("Troika imports passed")
PY
```

## 22. 创建 Troika 单 epoch 配置

```bash
cd "$LOGIC_FIX"

cp config/ut-zappos.yml config/ut-zappos-smoke.yml

sed -i -E \
  -e 's/^  epochs:.*/  epochs: 1/' \
  -e 's/^  train_batch_size:.*/  train_batch_size: 4/' \
  -e 's/^  eval_batch_size:.*/  eval_batch_size: 8/' \
  config/ut-zappos-smoke.yml
```

注意：该工程会先解析命令行，再用 YAML 覆盖同名配置，所以不能只通过 `--epochs 1` 修改 YAML 已包含的 epoch。

## 23. Troika 单 epoch 冒烟

```bash
mkdir -p "$RUN_ROOT/logiczsl_smoke/ut-zappos"

python train.py \
  --yml_path config/ut-zappos-smoke.yml \
  --dataset_path "$DATA_ROOT/ut-zappos" \
  --clip_arch "$CZSL_ROOT/pretrained/ViT-L-14.pt" \
  --save_path "$RUN_ROOT/logiczsl_smoke/ut-zappos" \
  --seed 1617 \
  --num_workers 0 \
  2>&1 | tee "$RUN_ROOT/logiczsl_smoke/ut-zappos/train.log"
```

成功标准：

- 模型构造不再出现额外逻辑参数导致的 `TypeError`；
- 不再查找 `<your_oi_dict.json>`；
- 完成一个训练 epoch；
- 验证输出 Seen、Unseen、HM、AUC；
- 生成非空 `val_best.pt`；
- 训练 loss 有限。

检查：

```bash
ls -lh "$RUN_ROOT/logiczsl_smoke/ut-zappos/val_best.pt"
grep -Ei 'nan|inf|traceback|error' "$RUN_ROOT/logiczsl_smoke/ut-zappos/train.log"
```

## 24. Troika 正式训练

冒烟通过后使用原正式 YAML：

```bash
mkdir -p "$RUN_ROOT/logiczsl/ut-zappos"

python train.py \
  --yml_path config/ut-zappos.yml \
  --dataset_path "$DATA_ROOT/ut-zappos" \
  --clip_arch "$CZSL_ROOT/pretrained/ViT-L-14.pt" \
  --save_path "$RUN_ROOT/logiczsl/ut-zappos" \
  --seed 1617 \
  --num_workers 4 \
  2>&1 | tee "$RUN_ROOT/logiczsl/ut-zappos/train.log"
```

显存不足时修改 YAML 内的 `train_batch_size`，并同步记录实际 batch size。

---

# 第三部分：完整 Logic-Troika

## 25. 必须向导师索取的文件

当前压缩包缺少三个数据集的：

```text
knowledge/
├── oi_dict.json
├── oe_dict.json
├── ae_dict.json
└── compatibility.csv
```

建议给导师发送：

> 当前 LogiCZSL 压缩包中的 `dataset.py` 仍使用 `<your_oi_dict.json>`、`<your_oe_dict.json>` 占位符。请补发 MIT-States、UT-Zappos、C-GQA 对应的 `oi_dict.json`、`oe_dict.json`、`ae_dict.json`、`compatibility.csv`，以及与这些文件匹配的 Logic-Troika `dataset.py/model_factory.py/train.py/test.py` 版本。

没有这些文件时，可以构造 identity/fake knowledge 检查程序是否执行，但不能报告为 LOGICZSL 论文结果。

## 26. 知识文件的数据契约

拿到文件后至少检查：

- `oi_dict.json`：每个 object 恰好属于一个 parent；所有 object 均被覆盖；
- `oe_dict.json`：key 覆盖所有 object，每个 object 恰有 3 条 description；
- `ae_dict.json`：key 覆盖所有 attribute，每个 attribute 恰有 3 条 description；
- `compatibility.csv`：元素数量应为 `N_object × M_attribute`；
- object、attribute 名称必须与数据划分文件规范化后一致；
- 不得利用 val/test pair 标签或频率生成知识。

建议知识加载时不要依赖 JSON 插入顺序，而是显式按照 `self.objs` 和 `self.attrs` 重排。

## 27. 统一 dataset 字段名

在逻辑 dataset 中建立兼容别名：

```python
self.hc_parent = self.oi_parent
self.hc_dict2idx = self.oi_dict2idx

self.oda_child = [
    text
    for obj in self.objs
    for text in self.oe_dict[obj]
]

self.ada_child = [
    text
    for attr in self.attrs
    for text in self.ae_dict[attr]
]
```

训练样本返回：

```text
image, attr_idx, obj_idx, pair_idx, hc_parent_idx, obj_idx, attr_idx
```

其中最后三个标签分别监督 HC、ODA 和 ADA 分支。

## 28. Logic-Troika 不能扩展主 prompt 数量

配置中只有三个主任务 prompt：

```text
composition、attribute、object
```

当前 `logic_copy.py` 访问：

```python
token_tensor[3]
token_tensor[4]
token_tensor[5]
```

会发生越界。正确方案是：

- `construct_token_tensors()` 始终只处理三个主任务 prompt；
- HC、ODA、ADA 文本单独经过 CLIP text encoder；
- 分别和 object/attribute visual feature 计算逻辑 logits；
- 不把逻辑概念伪装成第四到第六个主视觉分支。

## 29. Logic-Troika 统一输出

模型输出必须统一为：

```python
[
    comp_logits,       # B × pair
    attr_logits,       # B × M
    obj_logits,        # B × N
    hc_parent_logits,  # B × H
    oda_child_logits,  # B × 3N
    ada_child_logits,  # B × 3M
]
```

`logit_infer()` 改为：

```python
comp_logits, attr_logits, obj_logits, *_ = predict
```

评估仍然只使用前三个任务分支，逻辑分支用于训练约束和诊断。

## 30. 修复逻辑损失没有反向传播的问题

当前 `logic_copy.py` 虽然计算了：

```python
loss = loss_ce + loss_logic
```

但最后返回的却是 `loss_ce`。必须改为：

```python
return [loss, loss_comp, loss_attr, loss_obj]
```

否则 `loss_logic` 数值即使非零，也不会影响参数更新。

## 31. 逻辑 warm-up 必须使用全局 step

当前训练传入 `step=bid`，每个 epoch 都从 0 开始，逻辑分支会在每轮前 120 个 batch 重复关闭。

改成：

```python
global_step = 0

for epoch in range(config.epoch_start, config.epochs):
    model.train()

    for bid, batch in enumerate(train_dataloader):
        predict = model(batch, train_pairs)
        loss, loss_comp, loss_attr, loss_obj = model.loss_calu(
            predict,
            batch,
            step=global_step
        )
        global_step += 1
```

## 32. Logic-Troika 的最小验收

正式训练前，必须记录：

```text
loss_comp
loss_attr
loss_obj
loss_hc
loss_oda
loss_ada
loss_logic
```

通过条件：

- warm-up 前 `loss_logic` 被正确关闭；
- warm-up 后 `loss_logic > 0`；
- `loss_logic` 对模型参数产生非零梯度；
- HC/ODA/ADA logits shape 与预期一致；
- 没有 object/attribute 索引错位；
- 去掉 logic loss 后能够恢复到 Troika 基线性能附近。

---

# 第四部分：执行与排错顺序

## 33. 严格执行顺序

不要同时修改两套工程。按顺序执行：

1. CPF import 检查；
2. CPF UT-Zappos 单 epoch；
3. CPF 独立 test；
4. CPF 正式 UT-Zappos；
5. Troika import 检查；
6. Troika UT-Zappos 单 epoch；
7. Troika `val_best.pt` 加载与测试；
8. 向导师索取 LogiCZSL knowledge 文件；
9. 修复 Logic-Troika；
10. Logic-Troika 单 batch；
11. Logic-Troika 单 epoch；
12. 最后扩展到 MIT-States 和 C-GQA。

## 34. 每次实验需要保存的信息

```bash
nvidia-smi
git status
git log --oneline -5
python -m pip freeze
```

同时保存：

- 完整启动命令；
- YAML 配置副本；
- Git commit SHA；
- 随机种子；
- GPU 型号；
- train/val/test 日志；
- validation-best checkpoint；
- AUC、HM、Seen、Unseen；
- 最大显存占用和训练时间。

## 35. 常见失败判断

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| `No module named evaluator_ge` | CPF evaluator 未补齐 | 第 6 节 |
| `No module named model.ddm_utils` | LogiCZSL 缺失源码未创建 | 第 13 节 |
| `<your_oi_dict.json>` 不存在 | 非逻辑基线仍加载逻辑数据 | 第 14 节 |
| `unexpected keyword hc_parents` | model factory 仍导入普通 Troika 却传逻辑参数 | 第 15、16 节 |
| `token_tensor[3]` 越界 | 把逻辑概念错误加入主 prompt | 第 28 节 |
| `mat1 and mat2 must have the same dtype` | CLIP FP16 与新增 FP32 模块冲突 | 第 19 节 |
| 第二个 epoch 指标异常 | 验证后没有恢复 `model.train()` | 第 17 节 |
| 有 logic loss 但结果和无 logic 完全相同 | 返回的是 `loss_ce` 而不是总 loss | 第 30 节 |
| 每个 epoch 前 120 batch 都没有逻辑损失 | 使用了 epoch 内 `bid` 而非全局 step | 第 31 节 |
| loss 正常但指标严重异常 | pair/object/attribute 索引或 JSON 顺序错位 | 第 26、27 节 |
| CUDA OOM | batch、FP32 ViT-L 或 evaluation batch 太大 | 依次降低 train/eval batch |

## 36. 反馈报错时需要提供的内容

请保存并提供：

1. 实际执行的完整命令；
2. `git diff --stat`；
3. `git diff` 中与报错文件有关的部分；
4. 从 `Traceback (most recent call last)` 到最后一行的完整输出；
5. `nvidia-smi`；
6. 数据集名称及当前 YAML；
7. 报错发生在 import、第一 batch、反向传播、validation 还是 test。

不要只截取 Traceback 的最后一行，否则很难判断真正的调用位置。

