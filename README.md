# CZSL 研究：实验记录与复现

按实验依赖顺序组织 CPF、知识增强、条件属性与 0601 评分接口的代码、配置和结果。

文档更新：2026-10-16。最新结果归档：[0601方案 v2 · seed3351](0601方案/v2/results/seed3351-10epoch-20261015/README.md)，32组×10 epoch、128份评价，使用20261014修正版。旧版02的168份评价及30/10/10 epoch预算保持独立。

新增：[v2 单 seed 数据分析](0601方案/v2/results/seed3351-10epoch-20261015/v2单seed数据分析.md)，包含模型比较、机制对照、设置敏感性、原设计四个问题，以及 VAL 固定 bias 的 TEST 校准和错误分解。

## 目录

```text
前期实验/      01–15 阶段的说明、代码、配置与结果
0601方案/      01–03 与 v2 的评分实现、配置与结果
src/           CPF / LogiCZSL 基础源码快照
knowledge/     按数据集、类型和版本组织的知识资产
environment/   环境版本、外部输入与路径配置
tools/         文件校验与运行目录恢复
provenance/    文件来源、路径映射与 SHA256
```

## 两条实验线

| 实验线 | 内容 | 入口 |
|---|---|---|
| 前期实验 | 01–15：CPF 基线、知识归因、条件属性与冻结探针 | [实验索引](前期实验/README.md) |
| 0601方案 | 01–03 与 v2：残差版、公式对齐版、描述删除模块与统一修订版 | [方案索引](0601方案/README.md) |

## 使用流程

```bash
git lfs install
git clone -c core.longpaths=true https://github.com/havenxu7436/czsl-research.git
cd czsl-research
git lfs pull
python tools/reproduce.py verify
```

1. [环境说明](environment/README.md)定义软件版本、外部输入与命令变量。
2. [知识说明](knowledge/README.md)定义资产分区与版本。
3. 各阶段 README 提供实验设置、关键流程、入口和结果链接。
4. `inputs.json` 索引知识、配置和结果；`runbook.json` 保存该阶段的执行命令。

`materialize` 将分区存储的文件恢复为独立运行目录中的原包布局。训练输入包括数据集图像、GloVe、预训练权重，以及对应阶段使用的前序 checkpoint。

评价中的 CW/OW 分别表示 closed-world/open-world。训练组按对应 VAL 规则选模，选中权重用于结果评价。

文件来源与 SHA256 见 `provenance/`。
