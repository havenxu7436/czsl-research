# CPF 文本归因 → EKR 过渡工具包

版本：`ekr_cpf_transition_20261005_v1`。配合《第五步_文本收益归因到EKR-CPF_完整操作方案.md》使用。

本包接入 Week5 的真实 KCC0601 源码；不包含图像、GloVe、视觉预训练权重或训练好的 EKR。先在新目录安装，再使用现有 CPF Python 3.8 / torch 1.13.1 / torchvision 0.14.1 环境做 GPU smoke。

```bash
/data/xhw/miniconda3/envs/cpf/bin/python install.py \
  --source /data/xhw/czsl_repro/src/KCC0601_v1 \
  --destination /data/xhw/czsl_repro/src/EKR0601_transition_v1
cd /data/xhw/czsl_repro/src/EKR0601_transition_v1
/data/xhw/miniconda3/envs/cpf/bin/python -m transition.cli --help
```

安装器校验 9 个源码文件的 SHA256，保留原训练器，仅替换新训练入口的模型 import；从原 `val_forward` 自动生成暴露分数和属性特征的 `raw_forward`，原 CPF/KCC 目录不修改。所有新输出拒绝覆盖，训练中断不伪装成可恢复 optimizer 的断点。

## 命令

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

## 检查边界

本地用 Python 3.12 / CPU torch 2.5.1 检查了数学恒等、幅度界、候选重排、分块、梯度、扰动隔离，以及真实源码安装、GloVe 聚合、配置/统计/门槛流程。真实 UT 归档的 bias=0 配对结果复核通过。服务器 Python 3.8 / CUDA 完整模型检查和实际训练尚未执行；必须看到 `GPU SMOKE PASSED` 才开始新训练。单条源码语法同时按 Python 3.8 解析检查。

`prepare_reference.py` 是构包时的本地审核辅助程序；常规服务器部署不运行它，不用更新预期哈希来绕过源码不一致。
