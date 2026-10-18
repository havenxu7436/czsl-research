# 运行环境与输入

## 软件环境

CPF 主线记录：Linux、Python 3.8.18、PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117、NumPy 1.24.4、SciPy 1.10.1，GPU 为 RTX 3090。
各源码快照与依赖记录保存在 `records/`。CLIP 探针包含固定版本的源码与依赖 wheel。

## 输入文件

| 输入 | 对应要求 |
|---|---|
| UT-Zappos、MIT-States、C-GQA 图像与 split/metadata | CPF 数据加载器布局与对应原始划分 |
| `glove.6B.300d.txt` | GloVe 300d；manifest 中的文件 SHA256 |
| ViT B16 预训练权重 | 对应训练配置的 backbone |
| 前序 checkpoint | 第 05–09 阶段等使用的选中权重；续训使用完整 model/Adam/RNG 状态 |
| CLIP ViT-L/14 权重 | 第 14 阶段 protocol 指定的模型与 SHA256 |

外部资源索引见 [external_inputs.csv](../provenance/external_inputs.csv)。图像全集、大模型权重与部分前序 checkpoint 独立于本仓库存储。

## 命令变量

| 变量 | 含义 |
|---|---|
| `CPF_PY` | 匹配上述环境的 Python 可执行文件 |
| `PROJECT` | 基础源码、数据与知识的工程根目录 |
| `WORK` | 本次运行的独立输出目录 |
| `REPLAY` | `materialize` 生成的执行包目录 |
| `CZSL_GPU` / `CZSL_GPUS` | 执行使用的物理 GPU 编号 |
| `CZSL_DEADLINE` | 带时区的 ISO8601 队列截止时间 |

```bash
export CPF_PY=/srv/czsl-env/bin/python
export PROJECT=/srv/czsl-project
export WORK=/srv/czsl-runs/0601-02
```

Bash 入口中的解释器、输入与输出路径由各包脚本定义。缓存和前序权重的固定输入路径记录在 manifest/protocol 中；运行环境的目录映射与这些记录对应。
第 14、15 阶段的输出目录名固定在 protocol 中，具体值见各阶段入口。

## 运行目录恢复

从仓库根目录执行，以0601方案 02为例：

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --line 0601方案 --stage 02 --dest /srv/czsl-reproduce-0601-02
python tools/reproduce.py doctor --line 0601方案 --stage 02 --project /srv/czsl-reproduce-0601-02/project
```

`materialize` 恢复累计源码、知识、配置与包内校验文件；基础工程位于 `project/`。
`doctor` 输出外部输入和固定路径检查结果。随后执行对应阶段的 prepare、训练或诊断入口。

## 评价记录

记录字段包括 dataset、split、seed、epoch、知识哈希、选模规则与 checkpoint 身份。
0601方案 01、02按首次严格最大 VAL CW AUC 选模，同一权重评价 VAL/TEST CW/OW。
分数、指标、逐轮日志和权重身份由各阶段的结果索引关联。

## VG-CPOD独立路线

[VG-CPOD v1](../VG-CPOD方案/v1/README.md)使用冻结CLIP ViT-L/14双编码器，运行输入为UT原图/split与CLIP权重；不需要GloVe或CPF checkpoint。`VG_DATA`、`VG_CLIP`、`VG_WORK`、`VG_GPU`分别指定数据、权重、输出和物理GPU。版本页提供独立materialize与训练入口，现有前两条路线的恢复方式不变。
