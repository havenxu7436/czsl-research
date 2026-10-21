# v1 · TRUST-CPF / CLIP

**[CLIP修订实验方案](TRUST-CPF_CLIP_修订实验方案v1.md)** · **[公式及核心代码](TRUST-CPF公式及代码v1.md)** · **[服务器实践命令](服务器实践命令.md)**

第四条独立研究路线：通过组合隔离的配对干预学习关系效用，再用异方差集成、风险收益和类型预算选择知识图。当前归档完整CLIP实现、配置、知识资产与本机验证；正式UT训练由下列服务器队列执行。

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
    analysis/                      # 核心指标、对照与结果文件字段
    references/                    # 原始idea、早期数学方案及版本演进
    results/                       # CPU/CUDA/词表/公式验证、部署包、恢复映射
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
