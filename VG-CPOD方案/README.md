# VG-CPOD 方案

这条路线从 CLIP 组合评分逐步推进到动态视觉读出与条件知识残差。精简版保留 v1、02–11 的分析性 Markdown 作为演进证据，只保留最后两个完整机制实验的执行文件。

## 重点版本

| 版本 | 完整性 | 核心结论 | 入口 |
|---|---|---|---|
| 12：动态读出学习率与更新尺度 | K1/K2 各 3 seed，双准则与 12 项均值评价 | 学习率分组能控制动态读出的更新尺度，但没有改善整体泛化；此后冻结 Wd 学习率/门控搜索 | [阶段](12/README.md) · [结果](12/results/README.md) · [分析](12/analysis/README.md) |
| 13：条件知识残差 | L0 Base、L1 Knowledge、L2 Shuffle 各 3 seed，共 9 次完整训练 | Knowledge 优于 Base，但不及 Shuffle，未证明语义知识优势 | [阶段](13/README.md) · [公式代码](13/FORMULAS_AND_CODE.md) · [结果](13/results/README.md) · [分析](13/analysis/README.md) |

版本 13 三 seed 均值：

| 组 | CW AUC | OW AUC |
|---|---:|---:|
| L0 Base | 34.4384 ± 0.6282 | 25.1400 ± 0.9063 |
| L1 Knowledge | 35.0555 ± 0.7869 | 25.7402 ± 0.8549 |
| L2 Shuffle | **35.3067 ± 0.7471** | **25.9153 ± 0.5651** |

## 历史索引

[v1](v1/README.md) · [02](02/README.md) · [03](03/README.md) · [04](04/README.md) · [05](05/README.md) · [06](06/README.md) · [07](07/README.md) · [08](08/README.md) · [09](09/README.md) · [10](10/README.md) · [11](11/README.md)

历史版本的代码和原始数据已由后续版本覆盖；其阶段说明、结果表和分析结论仍保留。
