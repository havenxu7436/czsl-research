# 07 · 共享值对齐 EKR v2

状态：四组×3 epoch 已完成。

## 实验设置

- 共同底座：冻结 A_unit 与 N_name。
- 组别：V_name_align、V_clean_align、V_shuffle_align、V_clean_fixed；seed=0。
- 共享低秩 value 映射 rank=32；主对照 48,034 个可训练参数，fixed 对照 28,834 个。

## 流程

1. 准备共同底座、bank 与四组配置。
2. 执行分支训练，保存 epoch 0 与训练轮次的 VAL CW 指标。
3. 按 VAL CW AUC 选模，评价 CW/OW 并汇总。

## 结果与文件

V_clean_align 选中 epoch 0；V_clean_fixed 相对共同底座有小幅增量。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/artifacts/reports/results.csv（精简版未保留）
