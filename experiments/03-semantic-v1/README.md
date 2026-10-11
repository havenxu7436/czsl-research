# 03 · UT v1 三 seed 语义归因与工程修复

状态：已执行历史归因；保留缺项边界。

## 修改与结论

补 A_raw/A_unit/C_text/C_shuffle，检查文本收益是否需要正确对应；修复审计兼容和运行故障。

技术修复不改变研究 gate；v1 未保存的绝对指标不从 v2 反推。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 03 --dest /srv/czsl-reproduce-03
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-03
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

执行包：[ekr_cpf_transition_20261005_v1](<code/ekr_cpf_transition_20261005_v1>)。

运行时入口：`/srv/czsl-reproduce-03/ekr_cpf_transition_20261005_v1/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../02-mvp-calibration/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

<details>
<summary>模型定义、对照与协议边界</summary>

### CPF 文本归因 → EKR 过渡工具包

版本：`ekr_cpf_transition_20261005_v1`。

本包接入 Week5 的真实 KCC0601 源码；不包含图像、GloVe、视觉预训练权重或训练好的 EKR。先在新目录安装，再使用现有 CPF Python 3.8 / torch 1.13.1 / torchvision 0.14.1 环境做 GPU smoke。

```bash
"$CPF_PY" install.py \
  --source ${PROJECT}/src/KCC0601_v1 \
  --destination ${PROJECT}/src/EKR0601_transition_v1
cd ${PROJECT}/src/EKR0601_transition_v1
"$CPF_PY" -m transition.cli --help
```

安装器校验 9 个源码文件的 SHA256，保留原训练器，仅替换新训练入口的模型 import；从原 `val_forward` 自动生成暴露分数和属性特征的 `raw_forward`，原 CPF/KCC 目录不修改。所有新输出拒绝覆盖，训练中断不伪装成可恢复 optimizer 的断点。

#### 命令

| 命令 | 用途 |
| --- | --- |
| `preflight` | 原环境、CUDA、数据和知识路径检查 |
| `build-bank` | 重建单条描述，验证与旧 C 聚合一致，固定打乱置换 |
| `make-configs` | 从原训练日志恢复有效配置，生成归因/分支/联合对照 |
| `smoke` | 实际 GPU 上检查与原 A/C 恒等、冻结和梯度 |
| `run-semantic` | 原 CPF 损失训练 A_raw/A_unit/C_text/C_shuffle |
| `train-branch` | 冻结 C 的增量训练，或另列 joint CE 对照 |
| `run-plan` | 按配置顺序执行，验证已完成项、补完训练后的评价 |
| `eval` | CW/OW 评价；可追加分支知识错配诊断 |
| `summarize` | 完整结果、配对百分点差、均值/样本标准差 |
| `gate` | 三 seed 文本内容归因的工程投入门槛 |
| `bias0` | 按图像配对原始 top-1，区别于扫描偏置的 AUC/HM |
| `noise-summary` | 检查完整扰动矩阵，分别统计噪声重复及训练seed |
| `freeze-final` | 冻结配置/权重/验证证据，之后才允许 test |

每个命令支持 `--help`。`run-plan` 中的已完成项通过 `--finish-only` 校验身份；训练未完成的非空目录会停止队列。修复只完成评价，不续训不存在的优化器状态。

`C_text` 直接使用原 C 缓存张量。分支 C0 使用完全相同的 checkpoint；R5 初始化和 `--force-off` 为零增量。`R7`/`C_shuffle` 是负对照，不是 USG。静态 K 偏置始终关闭。三条新训练主对照 R0/R3/R5 使用相同新联合 CE；它不同于原 CPF 三支损失。

</details>
