# 05 · 名称锚点 EKR

状态：seed 0 的三组×3 epoch 已完成。

## 实验设置

- 冻结 A_unit；训练 N_name、N_clean、N_shuffle，三组各 28,834 个可训练参数。
- batch=64；AdamW；lr=0.001；weight_decay=0.0001；rank=32。
- 分支使用图像×候选对象名称查询属性槽，含 null 槽；温度 0.2，组合 CE 温度 0.05。

## 流程

1. 准备 v2 bank 和名称锚点权重，回放既有分数。
2. 完成三组分支训练，基础 CPF 保持冻结。
3. 按 VAL CW AUC 选模并评价 CW/OW。

## 结果与文件

N_clean 的指标未高于匹配的名称与错配对照。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/final_metrics.csv（精简版未保留）
