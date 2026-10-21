# CZSL 研究：实验记录与复现

最新归档（2026-10-21）：[VG-CPOD 10 · 残差幅值保留与门控温度对照](VG-CPOD方案/10/README.md)，包含I1/I2各三个seed、共享TRAIN均值、幅值保留与温度对照、native／exact双准则及十二项均值评估，共144份预测、24份checkpoint文件；附公式与代码修订、超参数策略及末节[数据分析](VG-CPOD方案/10/analysis/README.md)。前序[09 H组](VG-CPOD方案/09/README.md)、[08 G组](VG-CPOD方案/08/README.md)、[07 F组](VG-CPOD方案/07/README.md)、[06 E组](VG-CPOD方案/06/README.md)、[05 D组](VG-CPOD方案/05/README.md)、[04 C组](VG-CPOD方案/04/README.md)、[03 B组](VG-CPOD方案/03/README.md)、[02 A组](VG-CPOD方案/02/README.md)与[01原正式结果](VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/README.md)继续保留。

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
| VG-CPOD | 01–10 | [方案索引](VG-CPOD方案/README.md) |

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

`materialize` 将分区存储的文件恢复为独立运行目录中的原包布局。VG-CPOD 的 01 使用 `--line VG-CPOD方案 --stage v1`，02–10 使用对应编号并指定 `--line VG-CPOD方案`。04 同时恢复 C 包和 B3 补 seed 使用的原 B 包；05 恢复原 D 包、C 源码和三 seed 补充轮启动器；06 恢复原 E 包，包含双准则选权重和均值评估入口；07 恢复原 F 包，包含支持损失日志与 β 配对汇总；08 恢复原 G 包，包含静态与动态配对队列及排序诊断；09 恢复原 H 包，包含共享 TRAIN 均值统计、归一化残差门控及三 seed 调度；10 恢复原 I 包，包含残差幅值与温度对照、六训练及十二项均值评估。各实验的外部输入见对应索引；02–10 可直接复用服务器已有的冻结 CLIP 特征缓存。

评价中的 CW/OW 分别表示 closed-world/open-world。训练组按对应 VAL 规则选模，选中权重用于结果评价。

文件来源与 SHA256 见 `provenance/`。
