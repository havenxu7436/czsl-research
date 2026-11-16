# CZSL 研究

本仓库汇集组合零样本学习的方案演进、数学公式、核心代码与实验分析。

## 研究路线

| 路线 | 重点版本 | 研究重点 | 入口 |
|---|---|---|---|
| 前期验证 | 01–15 | 从 CPF 基线到冻结 CLIP 探针的关键证据 | [索引](前期实验/README.md) · [汇总](前期实验/RESULTS_SUMMARY.md) |
| 0601 | 02 公式对齐版 + v2 统一修订版 | 三 seed 复现与统一公式实现 | [路线说明](0601方案/README.md) |
| VG-CPOD | 12 动态读出 + 13 知识残差 | 更新尺度诊断与三 seed 语义对照 | [路线说明](VG-CPOD方案/README.md) |
| TRUST-CPF | 10 完整 ABC + 11 三数据集独立流水线 | 机制拆解与 UT/MIT/C-GQA 泛化实验 | [路线说明](TRUST-CPF方案/README.md) |

各路线包含公式、实现、配置与结果分析。CW/OW 分别表示 closed-world/open-world；训练组按对应 VAL 规则选模，并使用选中权重进行结果评价。

[实验环境与外部输入](environment/README.md) · [数据可用性](DATA_AVAILABILITY.md)
