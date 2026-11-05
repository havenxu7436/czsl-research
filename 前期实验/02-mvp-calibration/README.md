# 02 · MVP 对照与固定 prior 校准

状态：训练与后处理结果已归档。

## 实验设置

- UT-Zappos：A/C/D/E；MIT-States、C-GQA：A/C。
- 后处理使用固定 A/C checkpoint 扫描 prior 系数 λ，共 20 次评价。

## 流程

1. 按对应组配置执行训练并保存 checkpoint。
2. 在固定权重上评价不同 λ 的 CW/OW 指标。
3. 汇总指标、epoch 曲线与 bias=0 逐图结果。

## 结果与文件

UT 文本组呈现指标增量；后处理选定 λ=0。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/tables/epoch_curves.csv（精简版未保留）
