# 08 · TRAIN 支持掩码与 SG-EKR

状态：支持诊断与三组×3 epoch 已完成。

## 实验设置

- 每个 TRAIN pair 的六图分成确定性 3/3 两半；两半支持度均≥0.005 的 AE 槽进入掩码。
- 组别：G_name、G_support、G_shuffle；seed=0。
- 固定 value；训练 query/key/null/gain，共 28,834 个参数。

## 流程

1. 从 TRAIN 视图构建支持掩码。
2. 完成三组 SG-EKR 训练。
3. 导出 VAL CW/OW 指标和支持槽统计。

## 结果与文件

支持描述组的指标未高于匹配控制组。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/artifacts/reports/results.csv（精简版未保留）
