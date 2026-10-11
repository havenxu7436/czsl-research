# CPF 与 LOGICZSL 论文数据复现实验方案

更新日期：2026-09-29  
适用服务器：单卡 NVIDIA GeForce RTX 3090  
当前主要数据集：UT-Zappos50K（后续扩展到 MIT-States、C-GQA）

## 1. 实验目标与结果命名

本实验分成四个层级，报告结果时不得混用：

| 层级 | 名称 | 目的 | 能否与论文主表直接比较 |
|---|---|---|---|
| S0 | Smoke test | 验证环境、数据、训练、验证、保存与加载链路 | 否 |
| B1 | Repaired baseline | 在修复后的代码上复现 CPF 或 Troika 基线 | 条件允许时可以 |
| P1 | Paper-aligned reproduction | 尽可能严格采用论文的数据、模型、训练与评估协议 | 可以 |
| R1 | Reconstructed LOGICZSL | 使用已补齐并冻结的重构 OI/OE/AE 逻辑树复现 Logic-Troika | 只能作为重建实验，不能冒充作者原始结果 |

当前进度：

- CPF UT-Zappos50K S0 已通过；
- CPF 已完成训练、验证、最佳 checkpoint 保存、重新加载和测试；
- Troika 非逻辑基线的核心代码修复已完成，尚需完成论文参数配置核验、单 Epoch 冒烟和正式训练；
- UT-Zappos50K 的重构 OI/OE/AE 逻辑树与 `compatibility.csv` 已生成并上传到 `/data/xhw/czsl_repro/reconstructed_knowledge/ut-zap50k`；
- 逻辑树“文件补齐”已经解除 reconstructed 实验的数据阻塞，但 Logic 分支接线、知识校验、冒烟测试和配对正式实验仍未完成；
- LOGICZSL 作者实验所用的原始 OI/OE/AE 文件没有随代码公开，因此当前知识必须标注为 `reconstructed`，不能据此声称严格复现作者绝对数值。

论文与官方代码来源：

- [CPF：A Conditional Probability Framework for Compositional Zero-shot Learning](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf)
- [CPF 官方仓库](https://github.com/Pieux0/CPF)
- [LOGICZSL：Exploring Logic-induced Representation for Compositional Zero-shot Learning](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf)
- [LOGICZSL 官方仓库](https://github.com/Pieux0/LOGICZSL)

---

## 2. 复现纪律

### 2.1 模型选择纪律

必须仅使用验证集选择 checkpoint 和超参数：

1. 每个 epoch 在验证集计算 AUC；
2. 保存验证 AUC 最优的 checkpoint；
3. 训练全部结束后加载该 checkpoint；
4. 测试集只用于最终报告；
5. 禁止选择“测试集最高的 epoch”。

### 2.2 单位纪律

代码日志通常输出 `[0, 1]` 小数，而论文表格使用百分数。例如：

```text
日志 AUC = 0.2932
论文表格形式 = 29.32
```

所有对照表统一使用百分数：

```text
百分数指标 = 日志小数 × 100
```

### 2.3 单次复现与多随机种子

论文表通常没有报告标准差，因此分开报告：

- `paper-match`：采用代码或论文指定的主随机种子，和论文单点数值比较；
- `robustness`：至少运行 3 个种子，报告 `mean ± std`；
- 多种子平均值不得伪装成论文的单次结果。

建议种子：

- CPF 主种子：`3351`（压缩包配置）；
- CPF 稳健性种子：`0, 1, 2`；
- Troika/Logic-Troika 主种子：`0`；
- Troika/Logic-Troika 稳健性种子：`0, 1, 2`，必须成对运行。

---

## 3. 服务器目录与实验产物

统一使用：

```bash
export CZSL_ROOT=/data/xhw/czsl_repro
export DATA_ROOT="$CZSL_ROOT/data"
export SRC_ROOT="$CZSL_ROOT/src"
export RUN_ROOT="$CZSL_ROOT/runs"
export PRETRAINED_ROOT="$CZSL_ROOT/pretrained"

export CPF_FIX="$SRC_ROOT/CPF_fixed"
export LOGIC_FIX="$SRC_ROOT/LogiCZSL/models/Troika"
export KNOWLEDGE_ROOT="$CZSL_ROOT/reconstructed_knowledge/ut-zap50k"
```

每个正式实验目录应包含：

```text
run_name/
├── checkpoints/
├── logs/
├── command.sh
├── config_used.yml
├── environment.txt
├── pip-freeze.txt
├── conda-explicit.txt
├── git-status.txt
├── data-manifest.sha256
├── train.log
├── eval_cw.log
├── eval_ow.log
├── metrics.json
└── notes.md
```

禁止让正式训练写入 `cpf_smoke` 或其他冒烟测试目录。

---

## 4. 冻结环境、代码和数据

每次正式训练前执行一次。以 CPF 正式实验为例：

```bash
conda activate cpf

export EXP_ROOT=/data/xhw/czsl_repro/runs/cpf_ut_zappos_seed3351
mkdir -p "$EXP_ROOT"

{
  date -Is
  hostname
  uname -a
  nvidia-smi
  python --version
  python - <<'PY'
import torch, torchvision, numpy, scipy
print("torch=", torch.__version__)
print("torchvision=", torchvision.__version__)
print("numpy=", numpy.__version__)
print("scipy=", scipy.__version__)
print("cuda_runtime=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
} > "$EXP_ROOT/environment.txt" 2>&1

python -m pip freeze > "$EXP_ROOT/pip-freeze.txt"
conda list --explicit > "$EXP_ROOT/conda-explicit.txt"

cd /data/xhw/czsl_repro/src/CPF_fixed
{
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  git diff 2>/dev/null || true
} > "$EXP_ROOT/git-status.txt"
```

记录配置和关键代码校验和：

```bash
cd /data/xhw/czsl_repro/src/CPF_fixed

cp -a config/ut-zappos.yml "$EXP_ROOT/config_used.yml"

sha256sum \
  train.py \
  test.py \
  dataset.py \
  evaluator_ge.py \
  models/ddm.py \
  models/ddm_utils.py \
  config/ut-zappos.yml \
  > "$EXP_ROOT/code-manifest.sha256"
```

Logic-Troika reconstructed 正式实验还必须冻结知识文件；知识发生任何改动都必须建立新版本和新 run，不能覆盖旧结果：

```bash
export KNOWLEDGE_ROOT=/data/xhw/czsl_repro/reconstructed_knowledge/ut-zap50k

sha256sum \
  "$KNOWLEDGE_ROOT/oi_dict.json" \
  "$KNOWLEDGE_ROOT/oe_dict.json" \
  "$KNOWLEDGE_ROOT/ae_dict.json" \
  "$KNOWLEDGE_ROOT/compatibility.csv" \
  > "$EXP_ROOT/knowledge-manifest.sha256"

cp -a "$KNOWLEDGE_ROOT/generation_manifest.json" \
  "$EXP_ROOT/generation_manifest.json"
```

若 `generation_manifest.json` 尚不存在，不能启动 reconstructed 正式实验；应先补记生成模型、提示词、温度、生成日期、人工修订和知识版本号。

---

## 5. 数据完整性检查

### 5.1 论文数据规模

以下规模来自 CPF/LOGICZSL 补充材料采用的标准 compositional split：

| 数据集 | 属性数 | 对象数 | 全组合数 | 训练组合/图像 | 验证 Seen/Unseen 组合 | 验证图像 | 测试 Seen/Unseen 组合 | 测试图像 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UT-Zappos50K | 16 | 12 | 192 | 83 / 22,998 | 15 / 15 | 3,214 | 18 / 18 | 2,914 |
| MIT-States | 115 | 245 | 28,175 | 1,262 / 30,338 | 300 / 300 | 10,420 | 400 / 400 | 12,995 |
| C-GQA | 413 | 674 | 278,362 | 5,592 / 26,920 | 1,252 / 1,040 | 7,280 | 888 / 923 | 5,098 |

只有这些数字匹配后，才进入正式训练。

### 5.2 UT-Zappos 文件检查

```bash
export ZAP_ROOT=/data/xhw/czsl_repro/data/ut-zap50k
export SPLIT_ROOT="$ZAP_ROOT/compositional-split-natural"

test -s "$ZAP_ROOT/metadata_compositional-split-natural.t7"
test -s "$SPLIT_ROOT/train_pairs.txt"
test -s "$SPLIT_ROOT/val_pairs.txt"
test -s "$SPLIT_ROOT/test_pairs.txt"

wc -l \
  "$SPLIT_ROOT/train_pairs.txt" \
  "$SPLIT_ROOT/val_pairs.txt" \
  "$SPLIT_ROOT/test_pairs.txt"

find -L "$ZAP_ROOT/images" -type f | wc -l
```

核对 metadata：

```bash
python - <<'PY'
from collections import Counter
import torch

path = "/data/xhw/czsl_repro/data/ut-zap50k/metadata_compositional-split-natural.t7"
records = torch.load(path, map_location="cpu")

valid = [
    x for x in records
    if x.get("attr") != "NA"
    and x.get("set") in {"train", "val", "test"}
]

counts = Counter(x["set"] for x in valid)
attrs = sorted({x["attr"] for x in valid})
objs = sorted({x["obj"] for x in valid})

print("records=", len(valid))
print("split_counts=", dict(counts))
print("num_attrs=", len(attrs))
print("num_objs=", len(objs))
print("expected images: train=22998 val=3214 test=2914")
print("expected attrs=16 objs=12")
PY
```

保存数据清单：

```bash
sha256sum \
  "$ZAP_ROOT/metadata_compositional-split-natural.t7" \
  "$SPLIT_ROOT/train_pairs.txt" \
  "$SPLIT_ROOT/val_pairs.txt" \
  "$SPLIT_ROOT/test_pairs.txt" \
  > "$EXP_ROOT/data-manifest.sha256"
```

注意：图像通过符号链接重组不影响实验，但必须保证 `missing images = 0`，并记录26个重名候选的消歧方法。

---

## 6. 指标定义与评估协议

### 6.1 四个主指标

- `Seen`：Seen composition 上的准确率；
- `Unseen`：Unseen composition 上的准确率；
- `HM`：seen/unseen 最佳调和平均；
- `AUC`：校准扫描形成的 Seen-Unseen 曲线下面积。

调和平均：

```text
HM = 2 × Seen × Unseen / (Seen + Unseen)
```

### 6.2 Closed World 与 Open World

- `CW`：候选类别仅包含预定义的 seen/unseen composition 集合；
- `OW`：候选类别包含属性与对象的笛卡尔积；
- CW 与 OW 结果不得放在同一列比较；
- LOGICZSL OW 使用 compatibility/feasibility 筛选时，必须记录文件校验和与阈值选择方式。

### 6.3 checkpoint 选择

标准流程：

```text
训练集训练
    ↓
每个 epoch 在验证集计算 AUC
    ↓
保存验证 AUC 最优 checkpoint
    ↓
训练结束后加载该 checkpoint
    ↓
在测试集计算 AUC/HM/Seen/Unseen
```

---

## 7. CPF：UT-Zappos50K 正式复现

### 7.1 论文对齐配置

| 项目 | 目标值 | 来源 |
|---|---:|---|
| Backbone | ViT-B，768维 | 论文 |
| 文本向量 | GloVe，300维 | 论文与配置 |
| 输入尺寸 | 224 | 论文与代码 |
| Epoch | 10 | 论文与配置 |
| Optimizer | Adam | 论文 |
| 主学习率 | 1e-4 | 论文与配置 |
| Weight decay | 5e-5 | 压缩包配置 |
| LR decay factor | 0.1 | 论文与配置 |
| Temperature | 0.05 | 论文与配置 |
| 对象损失权重 | 0.4 | 论文与 `ddm.py` |
| 属性损失权重 | 0.6 | 论文与 `ddm.py` |
| Batch size | 64 | UT-Zappos 配置 |
| Seed | 3351 | UT-Zappos 配置 |

必须确认模型入口为：

```python
from models.ddm import DDM
```

必须确认：

- `models/ddm_utils.py` 中没有把 query/key 替换成全1张量；
- attribute 分支使用 object-guided representation；
- 验证集保存 `best_auc.pth`；
- 最终测试加载验证最优 checkpoint；
- 测试集没有参与 epoch 选择。

### 7.2 当前冒烟结果

当前 S0 使用 `batch_size=4`、`num_workers=0`、`epoch=1`，因此不能与论文正式数值直接比较。

| 集合 | AUC | HM | Seen | Unseen |
|---|---:|---:|---:|---:|
| 当前验证集 | 36.05 | 51.42 | 64.54 | 66.54 |
| 当前测试集 | 29.32 | 45.43 | 52.79 | 65.52 |
| 论文 CPF UT-Zappos CW | 41.40 | 55.70 | 66.40 | 71.10 |
| 当前测试与论文差值 | -12.08 | -10.27 | -13.61 | -5.58 |

这些差值仅用于确认“一轮尚未收敛”，不能作为复现失败判断。

### 7.3 正式训练命令

```bash
conda activate cpf
cd /data/xhw/czsl_repro/src/CPF_fixed

export DATA_ROOT=/data/xhw/czsl_repro/data
export RUN_ROOT=/data/xhw/czsl_repro/runs
export FORMAL_ROOT="$RUN_ROOT/cpf_ut_zappos_seed3351"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$FORMAL_ROOT/logs" "$FORMAL_ROOT/checkpoints"

python -u train.py \
  --cfg config/ut-zappos.yml \
  DATASET.root_dir "$DATA_ROOT/ut-zap50k" \
  TRAIN.log_dir "$FORMAL_ROOT/logs" \
  TRAIN.checkpoint_dir "$FORMAL_ROOT/checkpoints" \
  TRAIN.num_workers 4 \
  TRAIN.batch_size 64 \
  TRAIN.test_batch_size 64 \
  TRAIN.seed 3351 \
  TRAIN.start_epoch_validate 1 \
  TRAIN.eval_every_epoch 1 \
  TRAIN.max_epoch 10 \
  TRAIN.final_max_epoch 10 \
  2>&1 | tee "$FORMAL_ROOT/train.log"
```

在 RTX 3090 上第一次使用 batch 64 时，先观察前10个 iteration。若 OOM，应记录错误后设计 AMP 或等效 batch 实验，不得静默改成 batch 4 后仍称为论文对齐实验。

预期 checkpoint：

```bash
export CPF_CKPT="$FORMAL_ROOT/checkpoints/ut-zappos-czsl_final_predcition_mul/best_auc.pth"
test -s "$CPF_CKPT"
ls -lh "$CPF_CKPT"
sha256sum "$CPF_CKPT"
```

### 7.4 独立 CW 测试

```bash
python -u test.py \
  --cfg config/ut-zappos.yml \
  --load "$CPF_CKPT" \
  DATASET.root_dir "$DATA_ROOT/ut-zap50k" \
  TRAIN.num_workers 4 \
  TRAIN.test_batch_size 64 \
  EVAL.open_world False \
  2>&1 | tee "$FORMAL_ROOT/eval_cw.log"
```

### 7.5 独立 OW 测试

```bash
python -u test.py \
  --cfg config/ut-zappos.yml \
  --load "$CPF_CKPT" \
  DATASET.root_dir "$DATA_ROOT/ut-zap50k" \
  TRAIN.num_workers 4 \
  TRAIN.test_batch_size 64 \
  EVAL.open_world True \
  2>&1 | tee "$FORMAL_ROOT/eval_ow.log"
```

OW 结果只有在 evaluator 的候选集合、calibration sweep 和论文一致时才可比较。

---

## 8. CPF 论文目标值

### 8.1 Closed World

| 数据集 | AUC | HM | Seen | Unseen |
|---|---:|---:|---:|---:|
| MIT-States | 11.2 | 26.8 | 41.3 | 34.8 |
| UT-Zappos50K | 41.4 | 55.7 | 66.4 | 71.1 |
| C-GQA | 8.2 | 23.9 | 39.6 | 23.5 |

### 8.2 Open World

| 数据集 | AUC | HM | Seen | Unseen |
|---|---:|---:|---:|---:|
| MIT-States | 4.4 | 15.1 | 40.8 | 14.4 |
| UT-Zappos50K | 31.2 | 47.6 | 64.6 | 56.1 |
| C-GQA | 2.10 | 9.5 | 38.4 | 6.8 |

### 8.3 CPF 待填写对照表

| Setting | Dataset | Seed | Paper AUC | Repro AUC | ΔAUC | Paper HM | Repro HM | ΔHM | Paper Seen | Repro Seen | Paper Unseen | Repro Unseen |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CW | UT-Zappos | 3351 | 41.4 |  |  | 55.7 |  |  | 66.4 |  | 71.1 |  |
| OW | UT-Zappos | 3351 | 31.2 |  |  | 47.6 |  |  | 64.6 |  | 56.1 |  |

---

## 9. Troika 非逻辑基线：论文对齐前必须处理的差异

当前 LOGICZSL 压缩包配置与论文存在明显冲突：

| 项目 | 压缩包 `ut-zappos.yml` | 论文协议 |
|---|---:|---:|
| Epoch | 4 | 15 |
| Batch size | 128 | 64 |
| Learning rate | 5e-4 | 3.15e-6 |
| Weight decay | 1e-5 | 1e-5 |
| Context length | 8 | 8 |
| Backbone | CLIP ViT-L/14 | CLIP ViT-L/14 |
| GPU | 未锁定 | NVIDIA A800 80GB |

此外，压缩包训练变换使用了 `Resize + CenterCrop + RandomPerspective + RandomRotation`，论文描述的是随机裁剪到224并随机水平翻转。正式 paper-aligned 实验必须先对齐变换。

`load_args()` 会在 argparse 之后加载 YAML，并覆盖同名命令行参数。因此不能仅在命令行传 `--epochs 15` 或 `--lr 3.15e-6`；必须建立单独 YAML。

压缩包缺失的 `model/ddm_utils.py` 已从随包提供的 Python 3.7 字节码恢复。paper-aligned 实验必须使用恢复出的实际活动实现（矩阵乘法后直接 `softmax`），不能使用早期按接口猜测的 `normalize + sqrt(dim)` 版本。训练前检查：

```bash
cd /data/xhw/czsl_repro/src/LogiCZSL/models/Troika
python -m py_compile model/ddm_utils.py
grep -nE '^def |softmax' model/ddm_utils.py
if grep -nE 'normalize|sqrt|\*\* *0\.5' model/ddm_utils.py; then
  echo "ERROR: ddm_utils.py 仍是猜测版本"
  exit 1
fi
```

建议复制：

```bash
cd /data/xhw/czsl_repro/src/LogiCZSL/models/Troika
cp -a config/ut-zappos.yml config/ut-zappos-paper.yml
cp -a config/ut-zappos-ow.yml config/ut-zappos-paper-ow.yml
```

至少修改以下字段：

```yaml
train:
  optimizer: Adam
  scheduler: StepLR
  step_size: 5
  gamma: 0.5
  lr: 0.00000315
  weight_decay: 0.00001
  context_length: 8
  train_batch_size: 64
  gradient_accumulation_steps: 1
  epochs: 15

test:
  eval_batch_size: 64
  open_world: false
```

OW 配置只把：

```yaml
test:
  open_world: true
```

论文没有充分说明 StepLR 的全部细节，因此保留压缩包中的 `step_size=5, gamma=0.5` 时，必须在报告中披露。

### 9.1 FP32 修复与论文一致性

当前为解决 FP16/FP32 冲突，临时执行了全模型 FP32：

```python
self.float()
self.dtype = torch.float32
self.text_encoder.dtype = torch.float32
```

它适合先跑通，但会增加显存，并不等同于作者在 A800 上的精度策略。实验必须区分：

- `Troika-fp32-compat`：RTX 3090 上的兼容性基线；
- `Troika-paper-aligned`：修复统一 dtype/AMP 后，尽量维持 batch 64 的正式基线。

如果通过梯度累积实现 effective batch 64，需要记录 micro-batch。由于模型包含 BatchNorm，`micro-batch × accumulation` 并不与真实 batch 64 完全等价。

### 9.2 Troika paper-aligned 训练命令

仅在 `ut-zappos-paper.yml`、数据增强和 dtype 已核验后运行：

```bash
conda activate logiczsl
cd /data/xhw/czsl_repro/src/LogiCZSL/models/Troika

export DATA_ROOT=/data/xhw/czsl_repro/data
export RUN_ROOT=/data/xhw/czsl_repro/runs
export CLIP_ROOT=/data/xhw/czsl_repro/pretrained/clip
export TROIKA_RUN="$RUN_ROOT/troika_ut_zappos_paper_seed0"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$TROIKA_RUN"

python -u train.py \
  --clip_arch "$CLIP_ROOT/ViT-L-14.pt" \
  --dataset_path "$DATA_ROOT/ut-zap50k" \
  --save_path "$TROIKA_RUN" \
  --yml_path ./config/ut-zappos-paper.yml \
  --num_workers 4 \
  --seed 0 \
  2>&1 | tee "$TROIKA_RUN/train.log"
```

应使用验证 AUC 最优的：

```text
val_best.pt
```

### 9.3 Troika 独立 CW/OW 评估

CW：

```bash
python -u test.py \
  --clip_arch "$CLIP_ROOT/ViT-L-14.pt" \
  --dataset_path "$DATA_ROOT/ut-zap50k" \
  --save_path "$TROIKA_RUN" \
  --yml_path ./config/ut-zappos-paper.yml \
  --num_workers 4 \
  --seed 0 \
  --load_model "$TROIKA_RUN/val_best.pt" \
  2>&1 | tee "$TROIKA_RUN/eval_cw.log"
```

OW：

```bash
python -u test.py \
  --clip_arch "$CLIP_ROOT/ViT-L-14.pt" \
  --dataset_path "$DATA_ROOT/ut-zap50k" \
  --save_path "$TROIKA_RUN" \
  --yml_path ./config/ut-zappos-paper-ow.yml \
  --num_workers 4 \
  --seed 0 \
  --load_model "$TROIKA_RUN/val_best.pt" \
  2>&1 | tee "$TROIKA_RUN/eval_ow.log"
```

同一个 checkpoint 分别用于 CW 和 OW，不为 OW 偷看测试集重新选模型。

---

## 10. LOGICZSL 重建实验（UT-Zappos 逻辑树已补齐）

### 10.1 严格复现状态

论文的 LOGICZSL 使用 GPT-4 生成三类 primitive concept tree：

- OI：Object Induction；
- OE：Object Extension；
- AE：Attribute Extension。

官方仓库只要求根据提示生成并转成 JSON，没有提供作者实验时的原始 JSON。目前已经根据 UT-Zappos 的 16 个属性、12 个对象完成重构逻辑树并上传，因此可以继续开展 `Reconstructed LOGICZSL` 实验；但生成模型版本、提示词、输出与人工修订都可能和作者实验不同，所以仍不能声称严格复现作者 Logic-Troika 的绝对数值。

当前状态必须表述为：

```text
协议：尽量 paper-aligned
知识：reconstructed，非作者原始文件
实验名称：Logic-Troika reconstructed
主要结论：同环境、同 seed 相对 Troika 的 paired gain
```

不能写成：

```text
official LOGICZSL reproduction
exact author knowledge
strict paper reproduction
```

### 10.2 重建知识文件规范

UT-Zappos 重建版当前统一放在：

```text
/data/xhw/czsl_repro/reconstructed_knowledge/ut-zap50k/
├── oi_dict.json
├── oe_dict.json
├── ae_dict.json
├── compatibility.csv
├── generation_manifest.json    # 正式实验前必须补齐并冻结
└── validate_knowledge.py       # 建议随知识版本保存
```

当前 OI/OE/AE 与 compatibility 文件已生成并上传；`generation_manifest.json`、校验日志和 SHA256 清单属于正式实验准入材料，不能因“文件已经上传”而省略。

校验要求：

1. 所有12个对象必须且只能属于一个 OI 父类；
2. `oe_dict.json` 每个对象恰好有3条视觉区分描述；
3. `ae_dict.json` 每个属性恰好有3条视觉近义描述；
4. JSON key 必须与 split 文件中的原始标签完全一致；
5. compatibility 矩阵必须记录属性顺序、对象顺序和生成依据；
6. 禁止使用测试标签准确率人工修改知识；
7. `generation_manifest.json` 必须记录模型名称、版本、提示、温度、日期和人工修订。

### 10.3 已上传知识的准入检查

先验证文件存在、非空且 JSON 语法正确：

```bash
export KNOWLEDGE_ROOT=/data/xhw/czsl_repro/reconstructed_knowledge/ut-zap50k

for file in oi_dict.json oe_dict.json ae_dict.json compatibility.csv; do
  test -s "$KNOWLEDGE_ROOT/$file" || {
    echo "MISSING OR EMPTY: $KNOWLEDGE_ROOT/$file"
    exit 1
  }
done

python -m json.tool "$KNOWLEDGE_ROOT/oi_dict.json" >/dev/null
python -m json.tool "$KNOWLEDGE_ROOT/oe_dict.json" >/dev/null
python -m json.tool "$KNOWLEDGE_ROOT/ae_dict.json" >/dev/null

wc -l "$KNOWLEDGE_ROOT/compatibility.csv"
sha256sum \
  "$KNOWLEDGE_ROOT/oi_dict.json" \
  "$KNOWLEDGE_ROOT/oe_dict.json" \
  "$KNOWLEDGE_ROOT/ae_dict.json" \
  "$KNOWLEDGE_ROOT/compatibility.csv" \
  | tee "$KNOWLEDGE_ROOT/knowledge-files.sha256"
```

随后运行结构与标签覆盖验证器。验收条件不是“JSON 能打开”，而是：12 个对象、16 个属性全部覆盖，OE/AE 每个标签恰有 3 条描述，没有未知标签、重复键、空字符串或依赖测试结果进行的人工筛选。验证输出保存为：

```text
/data/xhw/czsl_repro/reconstructed_knowledge/ut-zap50k/validation_report.txt
```

### 10.4 代码隔离

不要在当前 Troika 基线目录直接重新启用 Logic。建立独立副本或 Git 分支：

```bash
cd /data/xhw/czsl_repro/src/LogiCZSL/models
cp -a Troika Troika_logic_reconstructed
```

实验命名必须包含：

```text
logic_troika_reconstructed_ut_zappos_seed0
```

不能命名成 `official_logiczsl`。

### 10.5 Logic 分支接线准入

逻辑树已经补齐不代表 Logic 代码已经可训练。只在 `Troika_logic_reconstructed` 中完成以下接线，基线目录继续保持 Logic 关闭：

1. 删除 `dataset.py` 中 `<your_oi_dict.json>`、`<your_oe_dict.json>` 等占位符；
2. 将 OI/OE/AE/compatibility 路径显式指向 `KNOWLEDGE_ROOT`，禁止依赖当前工作目录猜路径；
3. 恢复 Logic 数据字段和 Logic-Troika 模型入口，但不得改变 T0 基线的模型、数据增强和 evaluator；
4. 配置并记录论文值：`q=3.0`、`alpha1=0.1`、`alpha2=0.1`、`beta=0.01`、Joe/Jae 描述数均为 3；
5. 初始化时打印知识目录、四个文件 SHA256、对象/属性覆盖数以及启用的 Logic 组件；
6. 先完成单 batch 前向/反向，再运行单 Epoch 冒烟，禁止直接启动 15 Epoch 正式训练。

占位符清零检查：

```bash
cd /data/xhw/czsl_repro/src/LogiCZSL/models/Troika_logic_reconstructed

if grep -RInE \
  --exclude-dir='__pycache__' \
  --exclude='*.pyc' \
  '<your_|TODO.*(oi|oe|ae)|fake knowledge|identity knowledge' .; then
  echo "ERROR: Logic/knowledge 仍存在占位实现"
  exit 1
fi

python -m py_compile \
  dataset.py \
  train.py \
  test.py \
  model/model_factory.py \
  model/troika.py \
  model/ddm_utils.py
```

### 10.6 配对实验矩阵

| 组别 | Logic | Knowledge | Seed | 用途 |
|---|---|---|---:|---|
| T0 | 关闭 | 无 | 0 | Troika 基线 |
| T1 | OI | reconstructed | 0 | OI 消融 |
| T2 | OE | reconstructed | 0 | OE 消融 |
| T3 | AE | reconstructed | 0 | AE 消融 |
| T4 | OI+OE+AE | reconstructed | 0 | 完整重建 Logic-Troika |
| T5 | OI+OE+AE | reconstructed | 1 | 稳健性复现 |
| T6 | OI+OE+AE | reconstructed | 2 | 稳健性复现 |

所有组别必须保持：

- 相同数据与 split；
- 相同 CLIP checkpoint；
- 相同训练 epoch、batch、lr、weight decay；
- 相同图像变换；
- 相同 evaluator；
- 相同 checkpoint 选择方法；
- 相同 seed 成对比较。

---

## 11. LOGICZSL 论文目标值

### 11.1 Closed World

| 数据集 | 方法 | AUC | HM | Seen | Unseen | ΔAUC vs Troika | ΔHM vs Troika |
|---|---|---:|---:|---:|---:|---:|---:|
| C-GQA | Troika | 12.4 | 29.4 | 41.0 | 35.7 | — | — |
| C-GQA | Logic-Troika | 15.3 | 33.3 | 44.4 | 39.4 | +2.9 | +3.9 |
| UT-Zappos | Troika | 41.7 | 54.6 | 66.8 | 73.8 | — | — |
| UT-Zappos | Logic-Troika | 45.8 | 57.8 | 69.6 | 74.9 | +4.1 | +3.2 |
| MIT-States | Troika | 22.1 | 39.3 | 49.0 | 53.0 | — | — |
| MIT-States | Logic-Troika | 23.4 | 40.5 | 50.8 | 53.9 | +1.3 | +1.2 |

### 11.2 Open World

| 数据集 | 方法 | AUC | HM | Seen | Unseen | ΔAUC vs Troika | ΔHM vs Troika |
|---|---|---:|---:|---:|---:|---:|---:|
| C-GQA | Troika | 2.7 | 10.9 | 40.8 | 7.9 | — | — |
| C-GQA | Logic-Troika | 3.4 | 12.6 | 43.7 | 9.3 | +0.7 | +1.7 |
| UT-Zappos | Troika | 33.0 | 47.8 | 66.4 | 61.2 | — | — |
| UT-Zappos | Logic-Troika | 36.2 | 50.8 | 69.6 | 63.7 | +3.2 | +3.0 |
| MIT-States | Troika | 7.2 | 20.1 | 48.8 | 18.7 | — | — |
| MIT-States | Logic-Troika | 8.7 | 22.4 | 50.7 | 21.4 | +1.5 | +2.3 |

LOGICZSL 重建实验最有价值的比较不是只看绝对数值，而是比较同环境、同 seed 下：

```text
ΔAUC_local = AUC(Logic-Troika reconstructed) - AUC(Troika local)
ΔHM_local  = HM(Logic-Troika reconstructed)  - HM(Troika local)
```

UT-Zappos 论文目标增益：

```text
CW: ΔAUC = +4.1，ΔHM = +3.2
OW: ΔAUC = +3.2，ΔHM = +3.0
```

---

## 12. 结果记录格式

每次评估向统一 CSV 增加一行：

```text
project,variant,dataset,setting,seed,epochs,batch_size,effective_batch,dtype,checkpoint,knowledge_version,knowledge_sha256,auc,hm,seen,unseen,paper_auc,paper_hm,gap_auc,gap_hm,notes
```

建议文件：

```text
/data/xhw/czsl_repro/runs/results.csv
```

差值定义：

```text
gap_auc = reproduced_auc - paper_auc
gap_hm  = reproduced_hm  - paper_hm
```

相对差值只作辅助：

```text
relative_gap_auc = gap_auc / paper_auc × 100%
```

不要只报告“达到论文的百分之多少”，因为低数值 AUC 的相对比例可能误导。

### 12.1 UT-Zappos 最终填写表

| Project | Variant | Setting | Seed | AUC | HM | Seen | Unseen | Paper AUC | Paper HM | ΔAUC | ΔHM | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CPF | paper-aligned | CW | 3351 |  |  |  |  | 41.4 | 55.7 |  |  |  |
| CPF | paper-aligned | OW | 3351 |  |  |  |  | 31.2 | 47.6 |  |  |  |
| Troika | repaired baseline | CW | 0 |  |  |  |  | 41.7 | 54.6 |  |  |  |
| Troika | repaired baseline | OW | 0 |  |  |  |  | 33.0 | 47.8 |  |  |  |
| Logic-Troika | reconstructed-v1 | CW | 0 |  |  |  |  | 45.8 | 57.8 |  |  | 重构逻辑树；记录 knowledge SHA256 |
| Logic-Troika | reconstructed-v1 | OW | 0 |  |  |  |  | 36.2 | 50.8 |  |  | 重构逻辑树；记录 knowledge SHA256 |

---

## 13. 如何判定复现程度

以下阈值是本项目内部诊断标准，不是论文官方标准：

| 状态 | 条件 |
|---|---|
| 高度接近 | 协议一致，AUC/HM 绝对差均不超过 1.0 个百分点 |
| 基本复现 | 协议一致，AUC/HM 绝对差大多不超过 2.0 个百分点，趋势一致 |
| 部分复现 | 绝对差 2–5 个百分点，或存在已披露的硬件/精度/配置差异 |
| 未复现 | 绝对差超过 5 个百分点，或评估集合、数据 split、checkpoint 选择不一致 |
| 不可直接比较 | 冒烟测试、epoch 不同、非官方知识、不同 backbone 或测试集参与选择 |

对于 LOGICZSL reconstructed，优先判断 paired gain 是否接近论文，而不是要求绝对值完全一致。

---

## 14. 数值差距排查顺序

若正式结果与论文差距明显，严格按照以下顺序排查，一次只改变一个因素：

1. 数据规模、split 文件与图像映射是否一致；
2. CW/OW 候选集合是否一致；
3. 指标是否从 `[0,1]` 正确换算为百分数；
4. evaluator 是否使用相同的 calibration/bias sweep；
5. 是否加载验证集最优 checkpoint；
6. backbone 与预训练 checkpoint 是否一致；
7. 图像增强和归一化是否一致；
8. epoch、batch size、学习率、weight decay、scheduler 是否一致；
9. FP16/FP32/AMP 是否一致；
10. 随机种子与 DataLoader worker 是否固定；
11. 修复代码是否改变了论文原方法；
12. LOGICZSL knowledge 是否为作者原文件还是 reconstructed 文件，版本号与 SHA256 是否和该 run 一致；

每次排查产生一个新的 run 目录，禁止覆盖旧实验。

---

## 15. 推荐执行顺序

### 阶段 A：完成 CPF UT-Zappos

- [x] batch 4、1 epoch 冒烟测试；
- [x] 验证集保存最佳 checkpoint；
- [x] 重新加载最佳 checkpoint 测试；
- [ ] seed 3351、batch 64、10 epoch 正式训练；
- [ ] 独立 CW 测试；
- [ ] 独立 OW 测试；
- [ ] 填写论文对照表；
- [ ] 可选运行 seed 0/1/2 并报告 mean ± std。

### 阶段 B：完成 Troika 非逻辑基线

- [x] dataset 非 Logic 四字段返回修复；
- [x] model factory 恢复标准 Troika；
- [x] 最佳 checkpoint 保存/加载逻辑修复；
- [x] `ddm_utils.py` 从压缩包字节码恢复；
- [x] 每个 epoch 恢复 `model.train()` 与梯度清零；
- [ ] 环境导入与整链静态检查；
- [ ] 小 batch 冒烟测试；
- [ ] 建立 paper-aligned YAML；
- [ ] 对齐训练数据增强；
- [ ] 解决 dtype/显存方案；
- [ ] seed 0 正式训练；
- [ ] CW/OW 独立评估；
- [ ] 填写 Troika 对照表。

### 阶段 C：LOGICZSL reconstructed

- [x] 导出 UT-Zappos 精确属性和对象标签；
- [x] 生成 OI/OE/AE 重构逻辑树与 compatibility 文件；
- [x] 上传到 `/data/xhw/czsl_repro/reconstructed_knowledge/ut-zap50k`；
- [ ] 补齐 `generation_manifest.json` 并生成 SHA256 清单；
- [ ] 运行知识验证器并保存 `validation_report.txt`；
- [ ] 在独立代码副本恢复 Logic 分支；
- [ ] 清除所有 knowledge 占位符并记录实际加载路径；
- [ ] 单 batch 前向/反向与单 Epoch 冒烟；
- [ ] 先跑 OI/OE/AE 单组件消融；
- [ ] 运行完整 Logic-Troika；
- [ ] 与同 seed Troika 计算 paired gain；
- [ ] 明确标注非作者原始知识。

### 阶段 D：扩展至 MIT-States 与 C-GQA

- [ ] 先完成数据规模核对；
- [ ] 复制已验证的实验模板；
- [ ] 每个数据集建立独立配置和 run 目录；
- [ ] 先 baseline，再 logic；
- [ ] CW 与 OW 分开报告；
- [ ] 最后形成总表与消融表。

---

## 16. 最终交付物清单

完整复现实验至少应保留：

1. 原始压缩包 SHA256；
2. 修复后代码 commit/hash；
3. conda 与 pip 环境快照；
4. GPU、CUDA、PyTorch 版本；
5. 数据集 metadata 和 split 校验和；
6. 预训练模型校验和；
7. 每个 run 的完整配置与启动命令；
8. 原始训练日志；
9. 验证最优 checkpoint；
10. CW/OW 独立评估日志；
11. 原始小数指标与百分数指标；
12. 论文值、复现值、绝对差值；
13. 多种子 mean ± std；
14. LOGICZSL knowledge 的版本号、四个文件 SHA256、generation manifest 与 validation report；
15. Logic 分支实际加载的知识路径、启用组件及 `q/alpha1/alpha2/beta`；
16. 所有协议偏差及其原因。

只有在以上材料可追溯时，结果才适合写入论文或向导师汇报。
