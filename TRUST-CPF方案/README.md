# TRUST-CPF方案

冻结CLIP的组合识别研究：v1为关系效用与预算选图，v2转向强CPF基底上的条件知识残差。

**[CLIP修订实验方案](v1/TRUST-CPF_CLIP_修订实验方案v1.md)** · **[公式及核心代码](v1/TRUST-CPF公式及代码v1.md)** 

| 编号 | 实验 | 入口 |
|---|---|---|
| 02 | G0 CPF＋静态CP；Base / Knowledge / Shuffle × 三seed | [版本索引](v2/README.md) · [修订方案](v2/REVISED_PLAN.md) · [公式代码](v2/FORMULAS_AND_CODE.md) · [服务器命令](v2/SERVER_COMMANDS.md) |
| 01 | UT-Zappos；五折配对效用、异方差集成、七组×三seed主矩阵 | [版本索引](v1/README.md) · [代码](v1/code/README.md) · [配置](v1/configs/README.md) · [核心指标](v1/analysis/README.md) · [验证记录](v1/results/README.md) |

本路线独立保存执行代码、配置、知识快照与结果目录。复用来源包括既有UT reviewed-v2知识、VG-CPOD的CLIP特征与数据映射接口，版本身份见[来源与许可](v1/来源与许可.md)。当前已完成21组主实验、知识打乱、两项效用消融审计及600次噪声评估；完整结果于2026-10-22归档。

## 最新归档

[完整分析与下一轮方向](v1/analysis/server-results-20261022/README.md) · [原始数据、模型、日志与图像](v1/results/server-results-20261022/README.md) · [分析重算](v1/analysis/server-results-20261022/REPRODUCE.md)
