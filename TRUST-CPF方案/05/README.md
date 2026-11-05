# 05 · Align 半密度对齐

实验完成及归档日期：2026-10-24。UT-Zappos natural split；冻结 CLIP ViT-L/14 特征和强 CPF；seed 3351 / 3352 / 3353。

18/18 阶段、9 份识别结果、3 份 Q5 审计；5 次实际适配、4 份同 seed Uniform 复用。三方法适配 3 epoch，216 行识别、72 行扫描、312 行 Q5 图 bank 风险。

[实验方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置与实验清单](configs/README.md) · [结果汇总](results/README.md) · [完整分析](analysis/README.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.4811 ± 1.1982 | 26.1445 ± 1.7200 |
| align | 35.3934 ± 1.0672 | 26.1525 ± 1.7185 |
| permuted | 35.4295 ± 1.1204 | 26.0651 ± 1.7411 |

主图 bank 1701，按 VAL CW native AUC 从 epoch 0–3 选同一权重评价 TEST CW/OW。数值为百分数尺度，均值 ± 三训练 seed 样本标准差。多个图 bank 用于描述图采样变化。

半密度对齐与 Q5 实际效用诊断已完成。Uniform 为 35.4811 / 26.1445；Align 为 35.3934 / 26.1525。非均匀策略只在 seed3352 启用。
