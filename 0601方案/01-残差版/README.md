# 01 · 残差版：A–H 与 sanity

状态：12 组×10 epoch、48 份评价已完成。

## 实验设置

- UT-Zappos；seed=3351；ViT B16、GloVe 300d；batch=64。
- 组别：A–H 与 H_no_pair/H_shuffle/H_random_prior/H_uniform_prior。
- G/H 在 native CPF composition 分数上加入 factorized head residual；λ=0.1、β=0.01、q=3。
- ViT、CPF 与新 head 联合优化；选模使用首次严格最大 VAL CW AUC。

## 流程

1. 准备缓存、逻辑 bank 与 12 组配置，执行 TRAIN 图像检查。
2. 各组从初始化训练 10 epoch。
3. 同一选中 checkpoint 评价 VAL/TEST CW/OW。

## 结果与文件

完整 H 的指标低于 A；主表与配对差分见 metrics.csv。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/extracted/summary/metrics.csv（精简版未保留）
- results/extracted/summary/summary.json（精简版未保留）
