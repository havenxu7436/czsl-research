# v2 评价与结果文件

主矩阵为Base / Knowledge / Shuffle × seed3351 / 3352 / 3353。训练后统一报告三seed均值、样本标准差及Knowledge−Base、Knowledge−Shuffle的逐seed差值。

| 指标 | 对应问题 |
| --- | --- |
| CW/OW AUC native与exact | 组合识别及校准曲线；epoch1–10以VAL CW native AUC选模 |
| 固定VAL bias的TEST HM | 已见／未见工作点的迁移表现 |
| 未见组合的真实物体条件属性准确率、margin | 改善是否进入条件属性判别 |
| 独立物体top-1与联合物体命中率 | 物体分支固定时，联合决策如何变化 |
| VAL温度迁移后的TEST NLL/ECE | 概率校准 |
| alpha、残差幅值、参数量、时间与显存 | 分支作用与成本 |

`summary.csv`保存逐seed、组、split/world结果；`aggregates.json`保存三seed汇总；`paired_comparisons.json`保存两项配对差值。每组包含`initial.json`、`history.json`、`metrics.json`、权重、完整VAL/TEST logits和样本标识。`DONE.json`在最终评价和预测导出完成后生成，`status`应显示9/9。

Knowledge同时优于Base与Shuffle，支持正确知识附件提供条件判别信息；仅优于Base时，要结合Shuffle判断新增分支的作用。精确公式与实现见[公式文档](../FORMULAS_AND_CODE.md)。
