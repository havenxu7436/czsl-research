# TRUST-CPF 方案

Teacher-supervised Relation Utility for Semantic Transfer（TRUST）的最终证据由版本 10 的机制拆解和版本 11 的三数据集独立流水线组成。精简版保留 v1、02–09 的分析性 Markdown，只保留 10、11 的核心代码与配置。

## 重点版本

| 版本 | 完整性 | 主要结论 | 入口 |
|---|---|---|---|
| 10：A/B/C 幅度、混合与情境效用 | UT，3 seed，36/36 阶段、936 行识别结果 | 各效用变体与 Uniform 差异很小；Context 有轻微均值优势但未形成强机制证据 | [阶段](10/README.md) · [分析](10/analysis/README.md) |
| 11：三数据集独立流水线 | UT/MIT/C-GQA×3 seed；165 任务、450 段训练、1,620 行识别 | Context 相对 Uniform 只在 UT 有小幅提升，在 MIT/C-GQA 基本为零，跨数据集泛化证据有限 | [阶段](11/README.md) · [分析](11/analysis/README.md) |

版本 10（CW/OW AUC，3 seed 均值）：

| 组 | CW | OW |
|---|---:|---:|
| Uniform | 36.0506 ± 0.7217 | 26.6377 ± 1.5274 |
| Matched | 36.1478 ± 0.8664 | 26.7437 ± 1.7058 |
| Amplitude | 36.0475 ± 0.7103 | 26.6260 ± 1.5108 |
| Mix16 | 36.1193 ± 0.8915 | 26.7136 ± 1.7347 |
| Context | 36.1472 ± 0.7079 | 26.7471 ± 1.5190 |
| Context Shuffle | 35.9785 ± 0.7695 | 26.5752 ± 1.5447 |

版本 11 的 Context − Uniform（均值 ± seed 标准差）：

| 数据集 | ΔCW | ΔOW |
|---|---:|---:|
| UT | +0.1499 ± 0.2065 | +0.1676 ± 0.3682 |
| MIT | −0.0108 ± 0.1047 | −0.0122 ± 0.0431 |
| C-GQA | +0.0333 ± 0.0853 | +0.0106 ± 0.0121 |

## 历史索引

[v1](v1/README.md) · [02](02/README.md) · [03](03/README.md) · [04](04/README.md) · [05](05/README.md) · [06](06/README.md) · [07](07/README.md) · [08](08/README.md) · [09](09/README.md)

原始 checkpoint、训练日志、逐样本预测和打包数据已移除，数值结果与机制分析保留在 Markdown 中。
