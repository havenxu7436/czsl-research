# 09 · Stage 第二批：教师阶段匹配与响应平均

完成与归档日期：2026-10-25。UT-Zappos natural split，CLIP ViT-L/14 特征与 CPF 强基底；seed 3351 / 3352 / 3353，实际物理 GPU 1 / 2 / 3。

26/26 阶段，15 个教师各追加 10 epoch，正式权重复用第一批 Uniform-best；81 个评分策略实例、972 行识别、744 行 Q5、486 份预测、165 行教师曲线。

[方案](PLAN.md) · [公式与核心代码](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置](configs/README.md) · [结果汇总](results/README.md) · [完整分析](analysis/README.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| old_single | 35.8118 ± 0.9746 | 26.5034 ± 1.2842 |
| matched_single | 35.9645 ± 0.8630 | 26.5362 ± 1.2052 |
| old_average | 35.9443 ± 0.8797 | 26.5268 ± 1.2462 |
| matched_average | 35.9471 ± 0.8600 | 26.4965 ± 1.2364 |

表格固定主 bank1701、Uniform-best，均值 ± 三 seed 样本标准差；AUC 为 0–100 尺度。三 bank 结果及五次置换比较见完整分析，多个 bank 不计作额外训练重复。

教师阶段匹配把旧预测器的正式识别下降基本修复到 Uniform 水平。三 bank 平均的 Matched-Average 为 35.936 CW / 26.483 OW，微小平均正值尚未形成稳定优势。
