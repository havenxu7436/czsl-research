# CZSL 研究：实验记录与复现

代码、配置、知识与原始结果，按研究依赖顺序组织。实验状态截至 2026-10-12；历史结果不改写。

```text
experiments/01–18/   每阶段一个 README，配套 code/、configs/、results/、inputs.json
src/                有身份区分的 CPF / LogiCZSL 基础源码快照
knowledge/          按数据集、知识类型和版本集中管理的原始知识资产
environment/        运行环境、外部输入与历史路径兼容要求
tools/              完整性校验、运行目录生成与依赖预检
provenance/         原路径映射、SHA256、数据身份与删除记录
```

## 开始复现

```bash
git lfs install
git clone -c core.longpaths=true https://github.com/havenxu7436/czsl-research.git
cd czsl-research
git lfs pull
python tools/reproduce.py verify
```

先读 [运行环境与外部输入](environment/README.md) 和 [知识版本说明](knowledge/README.md)，再按下面的阶段进入。用 `materialize` 在独立目录恢复完整包内布局和必要兼容文件，不能直接在当前的分区展示目录里启动旧脚本。

这不是无外部输入的离线训练包：图像全集、GloVe、ViT 预训练权重及部分历史训练 checkpoint 仍需准备。已存储的知识、配置和结果可按 SHA256 校验；CPU/结构校验不等于已经复现 GPU 训练指标。

## 实验顺序

| 阶段 | 入口 | 状态 |
|---|---|---|
| 01 | [基线与原始知识](experiments/01-baseline/README.md) | 历史基线；多数据集 |
| 02 | [MVP 与固定 prior 校准](experiments/02-mvp-calibration/README.md) | 已训练＋后处理 |
| 03 | [语义归因 v1](experiments/03-semantic-v1/README.md) | 保留历史缺项边界 |
| 04 | [审核知识与语义归因 v2](experiments/04-semantic-v2/README.md) | 三 seed 实验与逐图诊断 |
| 05 | [名称锚点 EKR](experiments/05-name-anchor-ekr/README.md) | 三组各 3 轮 |
| 06 | [EKR 机制诊断](experiments/06-ekr-diagnostics/README.md) | 既有权重回放 |
| 07 | [共享值对齐 EKR v2](experiments/07-ekr-v2/README.md) | 四组各 3 轮 |
| 08 | [TRAIN 支持探针与 SG-EKR](experiments/08-support-sgekr/README.md) | 诊断＋三组训练 |
| 09 | [3→10 轮预算回溯](experiments/09-budget10/README.md) | 续训，非新 run |
| 10 | [CPF＋LogiCZSL](experiments/10-cpf-logiczsl/README.md) | 四组各 10 轮 |
| 11 | [条件属性预检与缓存](experiments/11-conditional-preflight/README.md) | 工程/梯度检查 |
| 12 | [条件属性 pilot](experiments/12-conditional-pilot/README.md) | r1 失败、r2 五组训练 |
| 13 | [Generic 同结构知识对照](experiments/13-generic-controls/README.md) | 三项新训练＋一项复用 |
| 14 | [冻结 CLIP 探针](experiments/14-clip-probe/README.md) | r2 七配方，无 optimizer |
| 15 | [TRAIN 逐句知识审查](experiments/15-train-knowledge-audit/README.md) | 诊断，不是泛化实验 |
| 16 | [residual-0601](experiments/16-residual-0601/README.md) | 12 组×10 轮、48 份评价 |
| 17 | [formula-aligned-0601](experiments/17-formula-aligned-0601/README.md) | seed3351：14组×30轮、56份评价完成；后续推荐10轮新协议 |
| 18 | [residual 描述删除扩展](experiments/18-noise-extension/README.md) | 仅代码，尚无正式结果 |

两条 0601 路线分开保存。第 17 阶段不能继承第 16 阶段的成绩。每阶段的 `inputs.json` 定位知识、配置和原始结果；`runbook.json` 保存实验命令模板及来源，使用前须按环境说明核对路径和包内默认值。

第17阶段的[单seed结果与epoch评估](experiments/17-formula-aligned-0601/results/seed3351-30epoch-20261012/README.md)保留完整负结果、VAL曲线、实际30轮执行身份和56次原始scores复算。全部最佳VAL CW模型出现在第2–6轮；新决策推荐后续10轮，不修改旧30轮protocol，也不冒称原42次计划已全部完成。

## 完整性与历史

本次只改变展示布局、合并说明和去除重复容器，没有修改保留代码、配置、知识或数值结果的字节。所有原路径均在 [layout.json](provenance/layout.json) 标明新位置或移除理由。

旧版完整保留于 [archive-20261011](https://github.com/havenxu7436/czsl-research/tree/archive-20261011)。历史 README/PLAN 等校验依赖由工具从该标签恢复到独立运行目录；不要使用缺少此标签的浅克隆。重复 ZIP、排版中间文件及机器专属操作说明不再出现在当前版本，不重写 Git 历史。

新增实验建立新的阶段目录，记录假设、变更、数据/知识版本、seed、预算、选模、结果和决策。续训、回放、失败和计划分别标注，TEST 不参与选模或调参。
