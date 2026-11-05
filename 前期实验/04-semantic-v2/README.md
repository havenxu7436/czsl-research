# 04 · TRAIN 描述审核与语义归因 v2

状态：12 次训练与逐图分析已完成。

## 实验设置

- TRAIN 审核：28 张 contact sheet、84 条描述，保留 30 条、改写 54 条。
- 四组：A_raw、A_unit、C_text、C_shuffle；seeds=0/3351/2026；每组 10 epoch。
- 知识：reviewed-v2 AE/OE；OI 与标签顺序沿用原版本。

## 流程

1. 构建 v2 cache/bank 并生成配对配置。
2. 执行真实图梯度检查，完成四组×三 seed 训练。
3. 导出 VAL CW/OW 指标、配对差分与逐图分析。

## 结果与文件

C_text−A_unit 的平均 CW AUC 为 +3.0670 pp；C_text−C_shuffle 为 +0.3789 pp。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/experiments/semantic_v2/reports/paired_v2/paired_deltas.csv（精简版未保留）
