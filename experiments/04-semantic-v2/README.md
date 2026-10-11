# 04 · TRAIN 描述审核、UT v2 与逐图分析

状态：已训练＋诊断。

## 修改与结论

一次 TRAIN 审核修订后，四组×三 seed 从初始化训练；随后导出既有预测与回放。

C_text−A_unit 平均 CW +3.0670pp，C_text−shuffle CW +0.3789pp，未过预定0.5pp；OW信号保留。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 04 --dest /srv/czsl-reproduce-04
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-04
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

```bash
export UT_V2_ROOT="$PROJECT"
export UT_V2_WORK=/srv/czsl-runs/semantic-v2-new
bash "$REPLAY/ut_semantic_v2_20261006/prepare.sh"
bash "$REPLAY/ut_semantic_v2_20261006/run_paired.sh"
```

执行包：[ut_semantic_v2_20261006](<code/ut_semantic_v2_20261006>)。

运行时入口：`/srv/czsl-reproduce-04/ut_semantic_v2_20261006/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../03-semantic-v1/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[cache_manifest.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/cache_manifest.json>)、[descriptions.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/descriptions.json>)、[labels.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/labels.json>)、[ae_dict.json](<../../knowledge/ut-zap50k/definitions/reviewed-v2/ae_dict.json>)、[attributes.txt](<../../knowledge/ut-zap50k/definitions/reviewed-v2/attributes.txt>)、[item_review.json](<../../knowledge/ut-zap50k/definitions/reviewed-v2/item_review.json>)、[manifest.json](<../../knowledge/ut-zap50k/definitions/reviewed-v2/manifest.json>)、[objects.txt](<../../knowledge/ut-zap50k/definitions/reviewed-v2/objects.txt>)；完整清单见 inputs.json。

配置文件：[effective_config.yml](<configs/experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s0_e10/effective_config.yml>)、[effective_config.yml](<configs/experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s2026_e10/effective_config.yml>)、[effective_config.yml](<configs/experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s3351_e10/effective_config.yml>)、[effective_config.yml](<configs/experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s0_e10/effective_config.yml>)；已生成的运行配置和 protocol 也在结果中，完整文件见 inputs.json。

## 证据入口

以下链接指向归档原文件；训练、回放和诊断范围以上述阶段状态为准。

- [results/recovered/czsl_xhw_ekr_train_support_20261008_review/decision.json](<results/recovered/czsl_xhw_ekr_train_support_20261008_review/decision.json>)
- [results/experiments/semantic_v2/reports/paired_v2/paired_deltas.csv](<results/experiments/semantic_v2/reports/paired_v2/paired_deltas.csv>)

<details>
<summary>审核方式与实验协议</summary>

### 知识修订

仅审核 TRAIN 视图：28 张 contact sheet、84 条描述，保留 30 条、改写 54 条。
这是辅助模型审查，不是人工专家批准。v2 AE/OE 是修订文本；OI、标签保持原版本。
逐句证据、例外、术语来源与审查记录见本阶段 inputs.json 指向的知识文件。

### 配对训练

执行 `prepare.sh` 核对来源、数据划分和全部 TRAIN 图片，构建 v2 cache/bank 并冻结
4 组 × 3 seed 配置；随后 `run_paired.sh` 先做真实图 CUDA smoke，再进行 12 次从初始化训练，
每次 10 epoch。静态 K 校准关闭，不进行 TEST 图像评价，也不构建 USG 图。
v1 的知识、结果与 gate 不计入 v2 的新增成绩。

仅在原 v2 gate 通过时，`enter_ekr_pilot.sh` 才允许进入原条件分支 pilot。
本次 v2 gate 未通过；第 05 阶段是单独制定的名称锚点探索，不冒充该 gate 已通过。

</details>
