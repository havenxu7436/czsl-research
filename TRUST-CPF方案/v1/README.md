# v1 · TRUST-CPF / CLIP

**[CLIP修订实验方案](TRUST-CPF_CLIP_修订实验方案v1.md)** · **[公式及核心代码](TRUST-CPF公式及代码v1.md)** 

第四条独立研究路线：通过组合隔离的配对干预学习关系效用，再用异方差集成、风险收益和类型预算选择知识图。现已归档完整CLIP实现、配置、知识资产及正式UT结果：21组主实验、知识打乱、两项效用消融审计和600次噪声评估。

## 最新结果 · 2026-10-22

**[完整分析与下一轮方向](analysis/server-results-20261022/README.md)** · **[全部数据、权重与日志](results/server-results-20261022/README.md)** · [图像](results/server-results-20261022/images/README.md) · [重算](analysis/server-results-20261022/REPRODUCE.md)

| 模型 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| `r0` | 2.020 ± 2.232 | 2.010 ± 2.215 |
| `capacity` | 13.902 ± 0.885 | 11.240 ± 1.065 |
| `dense` | 11.169 ± 2.488 | 9.214 ± 2.309 |
| `drop` | 10.915 ± 1.545 | 9.047 ± 1.257 |
| `free` | 10.896 ± 1.519 | 9.011 ± 1.319 |
| `mean` | 10.080 ± 2.102 | 8.249 ± 1.692 |
| `robust` | 9.901 ± 1.890 | 8.172 ± 1.515 |

下一轮集中检验强 CPF 基底上的条件知识残差，三组 × 三 seed，使用 GPU **1、2、3**。当前归档的是 v1 已完成结果与下一轮公式规格。

## 目录

```text
TRUST-CPF方案/
  README.md
  v1/
    README.md
    TRUST-CPF_CLIP_修订实验方案v1.md
    TRUST-CPF公式及代码v1.md
    服务器实践命令.md
    code/trustcpf_clip_v1/          # 模型、阶段、训练、评价、脚本、测试、CLIP源码
    configs/trustcpf_clip_v1/       # 默认/消融配置、标签、名称、知识与输入哈希
    analysis/                      # 完整分析、CSV、统计图与重算脚本
    references/                    # 原始idea、早期数学方案及版本演进
    results/                       # 原始权重/日志/压缩包、截图、验证与映射
    inputs.json
    runbook.json
    external_inputs.json
```

## 实验定义

- 冻结CLIP ViT-L/14双编码器，224px、768维；UT-Zappos natural split，16属性、12对象、83训练组合。
- OI/OE/AE分别12/36/48条，共96条知识关系；关系原文和标签轴随包保存。
- 五折组合隔离教师：Q1–Q3拟合效用，Q4开发与结构参数选择，Q5机制审计；选定参数后以五折标签拟合最终效用模型。
- 主矩阵为7组×3seed＝21次最终识别适配：`r0/capacity/dense/drop/free/mean/robust`，seed为3351/3352/3353。五折教师、参考教师和效用拟合属于每seed的上游流程，另计训练成本。
- 额外队列包括`shuffled`全流程、纯语义与20%未测边审计、OI/OE/AE错配与删除实验。
- 首次严格最大VAL CW `AUC_native`选识别权重；同权重评价VAL/TEST CW/OW。实际工作点bias和温度取自VAL，TEST沿用。

## 从仓库恢复并训练

在仓库根目录执行：

```bash
git lfs pull
python tools/register_trustcpf_v1.py --check
REPLAY="$HOME/experiments/trustcpf-replay-v1"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage v1 --dest "$REPLAY"
TRUST_PKG="$REPLAY/trustcpf_clip_v1"
cd "$TRUST_PKG"
export CPF_PY="$(command -v python)"
export DATA_ROOT=/srv/czsl-project/data/ut-zap50k
export CLIP_CKPT="$HOME/.cache/clip/ViT-L-14.pt"
export CACHE="$HOME/experiments/trustcpf_clip_v1/cache/ut-zap50k"
export WORK_ROOT="$HOME/experiments/trustcpf_clip_v1/work"
source scripts/env.sh
"$CPF_PY" -m pip install -r requirements.txt
"$CPF_PY" -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_seed.sh 3351
```

`REPLAY`指定新的绝对目录；数据集与ViT-L/14权重见[外部输入](external_inputs.json)。`code/`、`configs/`按阅读和归档用途分区，执行时先materialize为完整包布局。已有VG-CPOD同协议缓存时可直接指定`CACHE`，再执行本包`run.py graph`补充知识向量。

三seed、多GPU、逐阶段恢复、知识扰动与结果导出命令见[服务器实践命令](服务器实践命令.md)；原始部署ZIP也保存在[results/deployment](results/deployment/)。

## 实现与评价入口

| 内容 | 入口 |
|---|---|
| 条件概率、图消息、三元残差、图概率平均 | [model.py](code/trustcpf_clip_v1/trustcpf/model.py) |
| 配对增益、支持摘要、异方差MLP与方差分解 | [utility.py](code/trustcpf_clip_v1/trustcpf/utility.py) |
| 五折测量、拟合、审计、调参、结构与适配 | [stages.py](code/trustcpf_clip_v1/trustcpf/stages.py) |
| 教师、训练恢复、选模、校准 | [engine.py](code/trustcpf_clip_v1/trustcpf/engine.py) |
| 知识解析、CLIP编码、预算闭式解、知识错配 | [graph.py](code/trustcpf_clip_v1/trustcpf/graph.py) |
| 全部配置与主矩阵 | [配置索引](configs/README.md) |
| 必须验证的指标与产物 | [分析协议](analysis/README.md) |
| 13项测试、CUDA、词表和公式检查 | [验证与结果索引](results/README.md) |
| idea→数学方案→CLIP实现 | [版本与来源](references/README.md) |
| 全文件与命令索引 | [inputs.json](inputs.json) · [runbook.json](runbook.json) |

## 校验与来源

```bash
python tools/register_trustcpf_v1.py --check
python tools/register_trustcpf_v1.py --restore-check
python tools/reproduce.py verify
```

53个冻结发布包成员通过[成员映射](results/release_member_map.json)恢复，源代码、配置、知识和原始文档保持字节身份。仓库可读文档统一为GitHub数学块并链接分区源码；原始版本保存在`results/package_support/`。[来源与许可](来源与许可.md)记录复用模块与CLIP许可证。
