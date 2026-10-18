# CZSL 研究：实验记录与复现

按实验依赖顺序组织 CPF、知识增强、条件属性、0601评分接口与CLIP视觉分解的代码、配置和结果。

## 目录

```text
前期实验/      01–15 阶段的说明、代码、配置与结果
0601方案/      01–03 与 v2 的评分实现、配置与结果
VG-CPOD方案/   独立CLIP路线；v1方案、公式、代码、配置与验证
src/           CPF / LogiCZSL 基础源码快照
knowledge/     按数据集、类型和版本组织的知识资产
environment/   环境版本、外部输入与路径配置
tools/         文件校验与运行目录恢复
provenance/    文件来源、路径映射与 SHA256
```

## 三条实验线

| 实验线 | 内容 | 入口 |
|---|---|---|
| 前期实验 | 01–15：CPF 基线、知识归因、条件属性与冻结探针 | [实验索引](前期实验/README.md) |
| 0601方案 | 01–03 与 v2：残差版、公式对齐版、描述删除模块与统一修订版 | [方案索引](0601方案/README.md) |
| VG-CPOD方案 | v1：冻结CLIP双编码器、视觉门控CP修正、伙伴聚合与独立验证 | [方案索引](VG-CPOD方案/README.md) |

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

`materialize` 将分区存储的文件恢复为独立运行目录中的原包布局。VG-CPOD使用 `--line VG-CPOD方案 --stage v1`，只恢复第三条路线。训练输入包括数据集图像、GloVe、预训练权重，以及对应阶段使用的前序 checkpoint。

评价中的 CW/OW 分别表示 closed-world/open-world。训练组按对应 VAL 规则选模，选中权重用于结果评价。

文件来源与 SHA256 见 `provenance/`。
