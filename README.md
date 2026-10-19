# CZSL 研究：实验记录与复现

最新归档（2026-10-19）：[VG-CPOD 02 · 支持损失重加权与 CP 分组学习率](VG-CPOD方案/02/README.md)，A 组 12 次训练、48 份评价、24 份权重及预测；包含公式与代码修订、超参数策略和[三 seed 数据分析](VG-CPOD方案/02/analysis/README.md)。[01 原正式结果](VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/README.md)继续保留。

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

## 实验路线

| 实验路线 | 实验编号 | 入口 |
|---|---|---|
| 前期实验 | 01–15 | [实验索引](前期实验/README.md) |
| 0601方案 | 01–04 | [方案索引](0601方案/README.md) |
| VG-CPOD | 01–02 | [方案索引](VG-CPOD方案/README.md) |

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

`materialize` 将分区存储的文件恢复为独立运行目录中的原包布局。VG-CPOD 的 01 使用 `--line VG-CPOD方案 --stage v1`，02 使用 `--line VG-CPOD方案 --stage 02`。各实验的外部输入见对应索引；02 可直接复用服务器已有的冻结 CLIP 特征缓存。

评价中的 CW/OW 分别表示 closed-world/open-world。训练组按对应 VAL 规则选模，选中权重用于结果评价。

文件来源与 SHA256 见 `provenance/`。
