# CPF 与 LOGICZSL 服务器复现训练流程

本文针对以下两个压缩包逐文件核对后编写：

- `CRMask_copy.tar.gz`（计划用于 CPF / ViT-B）
- `LogiCZSL_copy.tar.gz`（计划用于 LOGICZSL / Logic-Troika）

## 0. 先说结论

这两个压缩包当前都不能原样复现论文结果。下面的服务器流程可以完成环境、数据、预训练权重、日志目录和命令矩阵的准备；但是在开始正式训练前，必须先通过第 5 节的代码完整性门禁。

当前压缩包最多覆盖的论文范围是：

- CPF：理论上对应 ViT-B 主结果，但包内正常训练路径没有启用完整 TEO/OGA，且评测、保存文件缺失。
- LOGICZSL：理论上对应 Logic-Troika，但缺 primitive concept tree 文件，且数据集、模型工厂、训练脚本和测试脚本来自不一致的代码分支。

当前压缩包不包含足以复现以下结果的完整实现：

- CPF 的 CLIP 版及论文全部消融实验。
- LOGICZSL 的 Logic-CSP 版及论文全部消融实验。

因此，正确顺序是：准备环境和数据 -> 通过门禁 -> 单 batch/单 epoch 冒烟 -> 三数据集正式训练 -> 用验证集选 checkpoint -> 分别进行 CW/OW 测试。

## 1. 服务器约定与硬件检查

以下命令假设服务器是 Ubuntu 22.04，使用 bash，代码和数据存放在 `/data/$USER/czsl_repro`。如果服务器没有 `/data`，只改第一行 `CZSL_ROOT`。

LOGICZSL 论文使用单张 NVIDIA A800 80GB；要严格对齐论文，应使用同级显存并保持 batch size 64。CPF 论文未报告训练显卡；ViT-B 建议至少 24GB 显存。建议预留至少 80GB 磁盘空间。

```bash
nvidia-smi
free -h
df -h /data

export CZSL_ROOT="/data/$USER/czsl_repro"
export INPUT_ROOT="$CZSL_ROOT/inputs"
export SRC_ROOT="$CZSL_ROOT/src"
export DATA_ROOT="$CZSL_ROOT/data"
export RUN_ROOT="$CZSL_ROOT/runs"

mkdir -p "$INPUT_ROOT" "$SRC_ROOT" "$DATA_ROOT" "$RUN_ROOT"
```

## 2. 上传并锁定代码版本

在本地 Windows PowerShell 中运行；把 `USER` 和 `SERVER` 替换为实际登录信息：

```powershell
scp "C:\Users\llenovo\Desktop\CZSL_code\CRMask_copy.tar.gz" USER@SERVER:/data/USER/czsl_repro/inputs/
scp "C:\Users\llenovo\Desktop\CZSL_code\LogiCZSL_copy.tar.gz" USER@SERVER:/data/USER/czsl_repro/inputs/
```

在服务器中校验本次分析所用压缩包：

```bash
cd "$INPUT_ROOT"

sha256sum -c <<'EOF'
ca82629189e697355a3513821472694f7463db43f2f761a8abad55c289800601  CRMask_copy.tar.gz
37ac9efdadfe8cfbd75e28beb2d2ed6c539ba384dea690daf7428957d8528b31  LogiCZSL_copy.tar.gz
EOF

tar -xzf CRMask_copy.tar.gz -C "$SRC_ROOT"
tar -xzf LogiCZSL_copy.tar.gz -C "$SRC_ROOT"

export CPF_DIR="$SRC_ROOT/CRMask_copy"
export LOGIC_DIR="$SRC_ROOT/LogiCZSL/models/Troika"

find "$CPF_DIR" -maxdepth 2 -type f | sort | sed -n '1,80p'
find "$LOGIC_DIR" -maxdepth 3 -type f | sort | sed -n '1,120p'
```

不要在两个解压目录上直接覆盖修改。收到完整代码或准备修复时，先复制一份：

```bash
cp -a "$CPF_DIR" "$SRC_ROOT/CPF_repro"
cp -a "$LOGIC_DIR" "$SRC_ROOT/LOGICZSL_repro"

export CPF_REPRO="$SRC_ROOT/CPF_repro"
export LOGIC_REPRO="$SRC_ROOT/LOGICZSL_repro"
```

## 3. 数据准备

### 3.1 推荐下载方式

三套数据必须使用 CZSL 社区通用的 `compositional-split-natural`，不能只下载原始图片。优先使用官方 CZSL 数据脚本：

```bash
cd "$SRC_ROOT"
git clone https://github.com/ExplainableML/czsl.git czsl-data-helper
cd "$SRC_ROOT/czsl-data-helper"
bash utils/download_data.sh "$DATA_ROOT"
```

`download_data.sh` 使用当前目录下的 `utils/`、`flags.py` 等相对路径，必须从 `czsl-data-helper` 根目录运行。它不是断点续跑脚本；若曾从其他目录启动并中断，先检查 `$DATA_ROOT` 内已有文件和下载状态，再决定是否重新运行。`mkdir ... File exists` 本身不致命，但 `cp: cannot stat 'utils/download_embeddings.py'` 表示工作目录错误，不能忽略。

如果脚本中的旧 C-GQA 链接返回 404，不要换用未知版本的数据。应获得同时包含以下内容的 updated C-GQA 数据包：

```text
cgqa/
├── images/
├── metadata_compositional-split-natural.t7
└── compositional-split-natural/
    ├── train_pairs.txt
    ├── val_pairs.txt
    └── test_pairs.txt
```

MIT-States、UT-Zappos 和 C-GQA 的最终统一结构必须为：

```text
$DATA_ROOT/
├── glove/glove.6B.300d.txt
├── mit-states/
│   ├── images/
│   ├── metadata_compositional-split-natural.t7
│   └── compositional-split-natural/{train,val,test}_pairs.txt
├── ut-zappos/
│   ├── images/
│   ├── metadata_compositional-split-natural.t7
│   └── compositional-split-natural/{train,val,test}_pairs.txt
└── cgqa/
    ├── images/
    ├── metadata_compositional-split-natural.t7
    └── compositional-split-natural/{train,val,test}_pairs.txt
```

如果下载脚本生成 `ut-zap50k`，为当前两份代码建立别名：

```bash
if [ -d "$DATA_ROOT/ut-zap50k" ] && [ ! -e "$DATA_ROOT/ut-zappos" ]; then
  ln -s "$DATA_ROOT/ut-zap50k" "$DATA_ROOT/ut-zappos"
fi
```

CPF 代码固定从数据集同级目录读取 GloVe，所以这个文件必须存在：

```bash
test -s "$DATA_ROOT/glove/glove.6B.300d.txt"
```

### 3.2 下载 CLIP ViT-L/14

```bash
mkdir -p "$CZSL_ROOT/pretrained"
cd "$CZSL_ROOT/pretrained"

wget -c \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt

echo 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836  ViT-L-14.pt' | sha256sum -c -

export CLIP_CKPT="$CZSL_ROOT/pretrained/ViT-L-14.pt"
```

CPF 的 ViT-B/16 权重由代码首次运行时自动从 `lukemelas/PyTorch-Pretrained-ViT` 下载。正式训练前最好先缓存：

```bash
mkdir -p "$HOME/.cache/torch/hub/checkpoints"
wget -c \
  https://github.com/lukemelas/PyTorch-Pretrained-ViT/releases/download/0.0.2/B_16.pth \
  -O "$HOME/.cache/torch/hub/checkpoints/B_16.pth"
```

### 3.3 验证三个数据集的划分

先创建 CPF 环境（第 4.1 节），然后运行：

```bash
conda activate cpf

DATA_ROOT="$DATA_ROOT" python - <<'PY'
import os
from collections import Counter
import torch

root = os.environ['DATA_ROOT']
expected = {
    'ut-zappos': ((16, 12), (83, 15, 15, 18, 18), (22998, 3214, 2914)),
    'mit-states': ((115, 245), (1262, 300, 300, 400, 400), (30338, 10420, 12995)),
    'cgqa': ((413, 674), (5592, 1252, 1040, 888, 923), (26920, 7280, 5098)),
}

for name, (class_exp, pair_exp, image_exp) in expected.items():
    d = os.path.join(root, name)
    split = os.path.join(d, 'compositional-split-natural')

    pair_lists = {}
    for phase in ('train', 'val', 'test'):
        with open(os.path.join(split, f'{phase}_pairs.txt')) as f:
            pair_lists[phase] = [tuple(x.split()) for x in f if x.strip()]

    attrs = sorted({a for pairs in pair_lists.values() for a, _ in pairs})
    objs = sorted({o for pairs in pair_lists.values() for _, o in pairs})
    allowed = set().union(*map(set, pair_lists.values()))

    meta = torch.load(
        os.path.join(d, 'metadata_compositional-split-natural.t7'),
        map_location='cpu',
    )
    counts = Counter()
    observed_pairs = {phase: set() for phase in ('train', 'val', 'test')}
    for item in meta:
        pair = (item['attr'], item['obj'])
        phase = item['set']
        if item['attr'] != 'NA' and phase != 'NA' and pair in allowed:
            counts[phase] += 1
            observed_pairs[phase].add(pair)

    train_pair_set = set(pair_lists['train'])
    seen_val = len(observed_pairs['val'] & train_pair_set)
    unseen_val = len(observed_pairs['val'] - train_pair_set)
    seen_test = len(observed_pairs['test'] & train_pair_set)
    unseen_test = len(observed_pairs['test'] - train_pair_set)

    got_classes = (len(attrs), len(objs))
    got_pairs = (len(pair_lists['train']), seen_val, unseen_val, seen_test, unseen_test)
    got_images = (counts['train'], counts['val'], counts['test'])

    print(name, 'classes=', got_classes, 'pairs=', got_pairs, 'images=', got_images)
    assert got_classes == class_exp, (name, got_classes, class_exp)
    assert got_pairs == pair_exp, (name, got_pairs, pair_exp)
    assert got_images == image_exp, (name, got_images, image_exp)

print('DATASET CHECK PASSED')
PY
```

只有打印 `DATASET CHECK PASSED` 才能继续。

### 3.4 LOGICZSL 额外知识文件

每个数据集还必须有作者实验中使用的 primitive concept tree 文件：

```text
$DATA_ROOT/<dataset>/knowledge/
├── oi_dict.json
├── oe_dict.json
├── ae_dict.json
└── compatibility.csv
```

其中 `oe_dict.json` 与 `ae_dict.json` 每个 primitive 应含 3 条描述；`compatibility.csv` 应与完整属性-物体笛卡尔积一致。当前压缩包没有这些文件，`dataset.py` 中还是 `<your_oi_dict.json>` 和 `<your_oe_dict.json>` 占位符。要复现论文数值，必须向作者取得论文实验实际使用的 JSON/CSV 及其 SHA256；重新让任意 LLM 生成不能视作严格复现。

取得文件后执行：

```bash
for d in mit-states ut-zappos cgqa; do
  test -s "$DATA_ROOT/$d/knowledge/oi_dict.json"
  test -s "$DATA_ROOT/$d/knowledge/oe_dict.json"
  test -s "$DATA_ROOT/$d/knowledge/ae_dict.json"
  test -s "$DATA_ROOT/$d/knowledge/compatibility.csv"
  sha256sum "$DATA_ROOT/$d/knowledge/"*
done
```

## 4. Python 环境

### 4.1 CPF 环境

不要依赖系统 Python。

```bash
conda create -n cpf python=3.8.18 pip -y
conda activate cpf

python -m pip install --upgrade 'pip<25'
python -m pip install \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117

python -m pip install \
  numpy==1.23.5 scipy==1.10.1 Pillow==9.5.0 \
  yacs==0.1.8 tqdm==4.66.5 tensorboard==2.13.0 \
  prettytable==3.9.0 gensim==4.3.2 fasttext-wheel==0.9.2

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

### 4.2 LOGICZSL 环境

压缩包里的 `requirements.txt` 是作者整台机器的环境导出，包含 Ubuntu 系统包、无关 VLM 包以及作者本机的 `-e /data/wupeng/...` 路径，不能直接安装。使用最小环境：

```bash
conda create -n logiczsl python=3.11.9 pip -y
conda activate logiczsl

python -m pip install --upgrade pip
python -m pip install \
  torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install \
  numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 \
  PyYAML==6.0.2 Pillow==10.4.0 opencv-python-headless==4.8.1.78 \
  einops==0.6.1 ftfy==6.2.3 regex==2024.11.6 \
  tqdm==4.67.1 transformers==4.45.2

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

## 5. 正式训练前的代码完整性门禁

### 5.1 CPF 当前阻塞项

当前包存在以下确定性问题：

1. `train.py` 导入不存在的 `evaluator_ge.py`。
2. `test.py` 导入不存在的 `dataset_visualization.py` 和 `evaluator_ge_v.py`。
3. `models/ddm_v2.py` 的正常训练/验证路径中，TEO/OGA 关键实现大段被注释；直接训练得到的不是论文完整 CPF。
4. `train.py` 中 AUC/HM 评测和 `save_checkpoint(...)` 被注释，训练完不会产生可用于论文评测的 checkpoint。
5. 论文规定所有数据集训练 10 epoch；包内 MIT-States 和 C-GQA 配置为 20 epoch。
6. 配置的学习率里程碑是 23、30 epoch；在论文 10 epoch 训练中不会触发，与论文“衰减 0.1”无法对齐。
7. MIT-States 配置使用随机 seed `-1`，论文未报告 seed，无法逐数值严格复现。

完整 CPF 代码到位后，以下命令必须全部成功：

```bash
conda activate cpf
cd "$CPF_REPRO"

test -f evaluator_ge.py
test -f dataset_visualization.py
test -f evaluator_ge_v.py

python -m compileall -q .
python -c 'import train; print("CPF IMPORT CHECK PASSED")'

grep -n 'save_checkpoint(model' train.py
grep -n "return stats\['AUC'\], stats\['best_hm'\]" train.py
```

还必须由代码审查确认 Eq. 1-8 对应的 TEO、OGA、三项 loss 和推理加和在训练与验证路径中真实启用；仅通过 import 检查不够。

### 5.2 LOGICZSL 当前阻塞项

当前包存在以下确定性问题：

1. 三个数据集的 primitive concept tree JSON/CSV 缺失，且文件名仍是占位符。
2. `dataset.py` 暴露 `oi_parent/oe_child/ae_child`，`train.py` 却读取 `hc_parent/oda_child/ada_child`。
3. `model_factory.py` 导入的是非 logic 版 `model.troika.Troika`，但又传入 logic 版构造参数，会触发 `TypeError`。
4. `test.py` 调用 `get_model(...)` 时又没有传 factory 要求的 logic 参数。
5. `logic_copy.py` 的 prompt 数量、token tensor 数量和 visual feature 数量不一致，直接切换 import 仍会越界。
6. 训练配置与论文冲突：包内有 4 epoch、batch 16/128、LR `5e-5/1e-4/5e-4`；论文规定 CGQA/UT 15 epoch、MIT 10 epoch、batch 64、LR `3.15e-6`。
7. 包内没有 Logic-CSP 实现，不能复现论文 Logic-CSP 行。

完整 LOGICZSL 代码和知识文件到位后，以下命令必须全部成功：

```bash
conda activate logiczsl
cd "$LOGIC_REPRO"

! grep -R '<your_' -n dataset.py model train.py test.py
python -m compileall -q .
python -c 'from dataset import CompositionDataset; from model.model_factory import get_model; print("LOGIC IMPORT CHECK PASSED")'
```

必须再用一个真实数据 batch 验证 `model(batch, train_pairs)` 返回 6 组 logits（composition、attribute、object、oi、oe、ae），且 `loss_calu` 能完成 forward/backward，才能开始单 epoch 冒烟。

## 6. 修复版到位后的 CPF 精确命令

以下命令用于完整、已修复并经过第 5.1 节核验的 CPF 代码。论文训练设置为：ViT-B/16、Adam、10 epoch、主学习率 `1e-4`、温度 `0.05`、`alpha_attr=0.6`、`alpha_obj=0.4`。

### 6.1 单 epoch 冒烟

```bash
conda activate cpf
cd "$CPF_REPRO"

mkdir -p "$RUN_ROOT/cpf/smoke/log" "$RUN_ROOT/cpf/smoke/ckpt"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg config/ut-zappos.yml \
  DATASET.root_dir "$DATA_ROOT/ut-zappos" \
  TRAIN.log_dir "$RUN_ROOT/cpf/smoke/log" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf/smoke/ckpt" \
  TRAIN.train_name utz_seed0 \
  TRAIN.seed 0 \
  TRAIN.max_epoch 1 \
  TRAIN.final_max_epoch 1 \
  TRAIN.num_workers 4 \
  EVAL.open_world False \
  2>&1 | tee "$RUN_ROOT/cpf/smoke/ut-zappos.log"
```

验收：loss 有限、显存稳定、val/test 输出 AUC/HM、checkpoint 实际写入。

### 6.2 三数据集正式训练

归档中的 UT-Zappos 与 C-GQA seed 是 3351；MIT-States 是随机 seed `-1`。为使流程确定，下面暂用 MIT seed 0。拿到作者 seed 后必须替换 `MIT_SEED`。

```bash
conda activate cpf
cd "$CPF_REPRO"

export MIT_SEED=0
export UT_SEED=3351
export CGQA_SEED=3351

mkdir -p "$RUN_ROOT/cpf/log" "$RUN_ROOT/cpf/ckpt"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg config/mit-states.yml \
  DATASET.root_dir "$DATA_ROOT/mit-states" \
  TRAIN.log_dir "$RUN_ROOT/cpf/log" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf/ckpt" \
  TRAIN.train_name mit_seed${MIT_SEED} \
  TRAIN.seed "$MIT_SEED" \
  TRAIN.max_epoch 10 \
  TRAIN.final_max_epoch 10 \
  EVAL.open_world False \
  2>&1 | tee "$RUN_ROOT/cpf/mit-states-seed${MIT_SEED}.log"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg config/ut-zappos.yml \
  DATASET.root_dir "$DATA_ROOT/ut-zappos" \
  TRAIN.log_dir "$RUN_ROOT/cpf/log" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf/ckpt" \
  TRAIN.train_name utz_seed${UT_SEED} \
  TRAIN.seed "$UT_SEED" \
  TRAIN.max_epoch 10 \
  TRAIN.final_max_epoch 10 \
  EVAL.open_world False \
  2>&1 | tee "$RUN_ROOT/cpf/ut-zappos-seed${UT_SEED}.log"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg config/cgqa-vit.yml \
  DATASET.root_dir "$DATA_ROOT/cgqa" \
  TRAIN.log_dir "$RUN_ROOT/cpf/log" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf/ckpt" \
  TRAIN.train_name cgqa_seed${CGQA_SEED} \
  TRAIN.seed "$CGQA_SEED" \
  TRAIN.max_epoch 10 \
  TRAIN.final_max_epoch 10 \
  EVAL.open_world False \
  2>&1 | tee "$RUN_ROOT/cpf/cgqa-seed${CGQA_SEED}.log"
```

每个数据集只能按验证集 AUC 选择 checkpoint，不能按 test AUC 选。假设修复版保存为 `best_auc.pth`，CW/OW 测试命令为：

```bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --cfg config/mit-states.yml \
  --load "$RUN_ROOT/cpf/ckpt/mit-states-czsl_mit_seed${MIT_SEED}/best_auc.pth" \
  DATASET.root_dir "$DATA_ROOT/mit-states" EVAL.open_world False

CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --cfg config/mit-states.yml \
  --load "$RUN_ROOT/cpf/ckpt/mit-states-czsl_mit_seed${MIT_SEED}/best_auc.pth" \
  DATASET.root_dir "$DATA_ROOT/mit-states" EVAL.open_world True
```

把 `mit-states.yml`、数据路径和 checkpoint 依次替换为 `ut-zappos.yml`、`cgqa-vit.yml` 即可完成另外两个数据集。CW/OW 使用同一训练 checkpoint，不应为了 OW 重新在测试组合上训练。

## 7. 修复版到位后的 Logic-Troika 精确命令

论文训练设置：ViT-L/14、Adam、StepLR、初始 LR `3.15e-6`、weight decay `1e-5`、prompt context length 8、batch size 64、CGQA/UT-Zappos 15 epoch、MIT-States 10 epoch、`alpha_attr=0.1`、`alpha_obj=0.1`、`beta=0.01`、每个 oe/ae primitive 3 条描述、logic 聚合系数 `q=3.0`。官方 README 示例使用 seed 0。

当前 YAML 会覆盖同名 CLI 参数，所以不能只在命令末尾追加 `--lr` 或 `--epochs`；必须准备与论文一致的 YAML。完整代码到位后，用下列脚本生成三份训练配置和三份 OW 评测配置：

```bash
conda activate logiczsl
cd "$LOGIC_REPRO"

python - <<'PY'
from copy import deepcopy
from pathlib import Path
import yaml

spec = {
    'mit-states': {'epochs': 10, 'cmt_layers': 3, 'dropout': 0.1},
    'ut-zappos': {'epochs': 15, 'cmt_layers': 2, 'dropout': 0.0},
    'cgqa': {'epochs': 15, 'cmt_layers': 2, 'dropout': 0.0},
}

for name, s in spec.items():
    with open(f'config/{name}.yml') as f:
        cfg = yaml.safe_load(f)

    cfg['model']['model_name'] = 'troika'
    cfg['model']['clip_model'] = 'ViT-L/14'
    cfg['model']['cmt_layers'] = s['cmt_layers']
    cfg['model']['cross_attn_dropout'] = s['dropout']
    cfg['model']['pair_loss_weight'] = 1.0
    cfg['model']['attr_loss_weight'] = 0.1
    cfg['model']['obj_loss_weight'] = 0.1
    cfg['model']['pair_inference_weight'] = 1.0
    cfg['model']['attr_inference_weight'] = 1.0
    cfg['model']['obj_inference_weight'] = 1.0

    cfg['train']['dataset'] = name
    cfg['train']['optimizer'] = 'Adam'
    cfg['train']['scheduler'] = 'StepLR'
    cfg['train']['lr'] = 3.15e-6
    cfg['train']['weight_decay'] = 1e-5
    cfg['train']['context_length'] = 8
    cfg['train']['train_batch_size'] = 64
    cfg['train']['gradient_accumulation_steps'] = 1
    cfg['train']['epochs'] = s['epochs']
    cfg['train']['epoch_start'] = 0
    cfg['train']['val_metric'] = 'AUC'

    cfg['test']['eval_batch_size'] = 64
    cfg['test']['open_world'] = False
    cfg['test']['topk'] = 1
    cfg['test']['text_encoder_batch_size'] = 1024
    cfg['test'].pop('threshold', None)

    with open(f'config/{name}-paper.yml', 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    ow = deepcopy(cfg)
    ow['test']['open_world'] = True
    with open(f'config/{name}-paper-ow.yml', 'w') as f:
        yaml.safe_dump(ow, f, sort_keys=False)
PY
```

注意：StepLR 的 `step_size` 与 `gamma` 在论文中没有给出；上面保留原 YAML 的 `step_size=5, gamma=0.5`。这也是严格复现仍需作者确认的项目。

### 7.1 单 epoch 冒烟

从 `ut-zappos-paper.yml` 复制一个 smoke 配置并把 epoch 改成 1：

```bash
python - <<'PY'
import yaml
p = 'config/ut-zappos-paper.yml'
with open(p) as f:
    x = yaml.safe_load(f)
x['train']['epochs'] = 1
with open('config/ut-zappos-smoke.yml', 'w') as f:
    yaml.safe_dump(x, f, sort_keys=False)
PY

mkdir -p "$RUN_ROOT/logiczsl/smoke"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch "$CLIP_CKPT" \
  --dataset_path "$DATA_ROOT/ut-zappos" \
  --save_path "$RUN_ROOT/logiczsl/smoke/ut-zappos" \
  --yml_path config/ut-zappos-smoke.yml \
  --num_workers 10 \
  --seed 0 \
  2>&1 | tee "$RUN_ROOT/logiczsl/smoke/ut-zappos.log"
```

验收：第一个 batch 能 forward/backward；logic loss 在代码规定的启用 step 后非零；`epoch_1.pt`、`config.pkl` 和 `config.json` 写入；val AUC/HM 可正常计算。

### 7.2 三数据集正式训练

```bash
mkdir -p "$RUN_ROOT/logiczsl/mit-states" \
         "$RUN_ROOT/logiczsl/ut-zappos" \
         "$RUN_ROOT/logiczsl/cgqa"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch "$CLIP_CKPT" \
  --dataset_path "$DATA_ROOT/mit-states" \
  --save_path "$RUN_ROOT/logiczsl/mit-states" \
  --yml_path config/mit-states-paper.yml \
  --num_workers 10 \
  --seed 0 \
  2>&1 | tee "$RUN_ROOT/logiczsl/mit-states/train.log"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch "$CLIP_CKPT" \
  --dataset_path "$DATA_ROOT/ut-zappos" \
  --save_path "$RUN_ROOT/logiczsl/ut-zappos" \
  --yml_path config/ut-zappos-paper.yml \
  --num_workers 10 \
  --seed 0 \
  2>&1 | tee "$RUN_ROOT/logiczsl/ut-zappos/train.log"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --clip_arch "$CLIP_CKPT" \
  --dataset_path "$DATA_ROOT/cgqa" \
  --save_path "$RUN_ROOT/logiczsl/cgqa" \
  --yml_path config/cgqa-paper.yml \
  --num_workers 10 \
  --seed 0 \
  2>&1 | tee "$RUN_ROOT/logiczsl/cgqa/train.log"
```

当前训练脚本每个 epoch 都保存 `epoch_N.pt`，但没有保存 `val_best.pt`。完整修复版应按验证集 AUC 自动保存 `val_best.pt`；不要人工查看测试集后选择 epoch。

### 7.3 CW 与 OW 评测

下面以 MIT-States 为例，另外两个数据集只替换目录和 YAML：

```bash
CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --clip_arch "$CLIP_CKPT" \
  --dataset_path "$DATA_ROOT/mit-states" \
  --save_path "$RUN_ROOT/logiczsl/mit-states" \
  --yml_path config/mit-states-paper.yml \
  --num_workers 10 \
  --seed 0 \
  --load_model "$RUN_ROOT/logiczsl/mit-states/val_best.pt" \
  2>&1 | tee "$RUN_ROOT/logiczsl/mit-states/test-cw.log"

CUDA_VISIBLE_DEVICES=0 python -u test.py \
  --clip_arch "$CLIP_CKPT" \
  --dataset_path "$DATA_ROOT/mit-states" \
  --save_path "$RUN_ROOT/logiczsl/mit-states" \
  --yml_path config/mit-states-paper-ow.yml \
  --num_workers 10 \
  --seed 0 \
  --load_model "$RUN_ROOT/logiczsl/mit-states/val_best.pt" \
  2>&1 | tee "$RUN_ROOT/logiczsl/mit-states/test-ow.log"
```

OW 的 feasibility 文件已在压缩包 `models/Troika/data/` 中提供；不要重新生成，除非确认数据划分和 GloVe 完全一致。

## 8. 论文主结果验收表

论文中的数值是百分数；如果代码输出 `0.153`，对应论文中的 `15.3`。

### CPF / ViT-B

| Setting | Dataset | AUC | HM | Seen | Unseen |
|---|---:|---:|---:|---:|---:|
| CW | MIT-States | 11.2 | 26.8 | 41.3 | 34.8 |
| CW | UT-Zappos50K | 41.4 | 55.7 | 66.4 | 71.1 |
| CW | C-GQA | 8.2 | 23.9 | 39.6 | 23.5 |
| OW | MIT-States | 4.4 | 15.1 | 40.8 | 14.4 |
| OW | UT-Zappos50K | 31.2 | 47.6 | 64.6 | 56.1 |
| OW | C-GQA | 2.10 | 9.5 | 38.4 | 6.8 |

### Logic-Troika

| Setting | Dataset | AUC | HM | Seen | Unseen |
|---|---:|---:|---:|---:|---:|
| CW | C-GQA | 15.3 | 33.3 | 44.4 | 39.4 |
| CW | UT-Zappos50K | 45.8 | 57.8 | 69.6 | 74.9 |
| CW | MIT-States | 23.4 | 40.5 | 50.8 | 53.9 |
| OW | C-GQA | 3.4 | 12.6 | 43.7 | 9.3 |
| OW | UT-Zappos50K | 36.2 | 50.8 | 69.6 | 63.7 |
| OW | MIT-States | 8.7 | 22.4 | 50.7 | 21.4 |

单次训练因 seed、CUDA/cuDNN、未报告的调度细节而有轻微波动。先以训练/评测链路完整和指标接近为目标；要宣称严格复现，必须补齐作者 seed、StepLR 参数、knowledge 文件哈希和完整发布代码。

## 9. 保存复现实验元数据

每次正式运行前执行：

```bash
mkdir -p "$RUN_ROOT/meta"
nvidia-smi -q > "$RUN_ROOT/meta/nvidia-smi.txt"
uname -a > "$RUN_ROOT/meta/uname.txt"

conda activate cpf
python -m pip freeze > "$RUN_ROOT/meta/cpf-pip-freeze.txt"

conda activate logiczsl
python -m pip freeze > "$RUN_ROOT/meta/logiczsl-pip-freeze.txt"

sha256sum "$INPUT_ROOT/CRMask_copy.tar.gz" \
          "$INPUT_ROOT/LogiCZSL_copy.tar.gz" \
          "$CZSL_ROOT/pretrained/ViT-L-14.pt" \
          > "$RUN_ROOT/meta/input-sha256.txt"
```

日志、配置、checkpoint 和最终 JSON 均保留；不要只保存论文表格里的四个数字。

## 10. 建议向作者一次性索取的文件

1. CPF 完整 ViT-B 代码：`evaluator_ge.py`、测试数据集/评测器、启用 TEO/OGA 的模型、checkpoint 保存逻辑。
2. CPF 的 CLIP 实现及 Table 3 配置。
3. LOGICZSL 三数据集的 `oi_dict.json`、`oe_dict.json`、`ae_dict.json`、`compatibility.csv` 及 SHA256。
4. LOGICZSL 可运行的 Logic-Troika model factory、train/test 和 YAML。
5. Logic-CSP 实现。
6. 三数据集最终 seed、StepLR 的 `step_size/gamma`、logic loss 启用 step。
7. 主表 checkpoint 或逐 epoch 验证日志，用于确认 checkpoint 选择规则。

拿到这些文件后，先重新执行第 5 节门禁和单 epoch 冒烟，再进行第 6、7 节正式训练。
