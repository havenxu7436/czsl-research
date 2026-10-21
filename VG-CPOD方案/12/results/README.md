# 12 · K组训练结果

K1/K2各三seed，共六次训练、60个模型epoch。native/exact分别保存24份预测；十二项均值评估包含24个条件、96份预测。合计 **144份预测文件、24份checkpoint文件**，另保存共享TRAIN查询均值与十二套门控矩。

## 1. native选择主结果

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | 3351 | 5／5 | 44.3233 | 31.1919 | 33.4992 | 24.3035 |
| K1 | 3352 | 8／6 | 41.6102 | 30.0067 | 35.3735 | 23.9467 |
| K1 | 3353 | 6／6 | 45.0030 | 34.1870 | 34.0870 | 25.4192 |
| K2 | 3351 | 5／5 | 43.9091 | 30.9288 | 33.6030 | 24.5263 |
| K2 | 3352 | 8／6 | 41.5693 | 29.8716 | 35.3351 | 23.9131 |
| K2 | 3353 | 6／6 | 44.8551 | 34.2888 | 33.8095 | 25.0468 |

每次完成十轮、3600次更新。AUC单位为点；各运行还保存seen/unseen、HM、物体／条件属性准确率及NLL。

## 2. exact选择结果

| 实验 | seed | exact最佳epoch | VAL CW native | VAL CW exact | TEST CW native | TEST OW native |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | 3351 | 5 | 44.3233 | 44.4750 | 33.4992 | 24.3035 |
| K1 | 3352 | 6 | 41.2625 | 41.8439 | 35.5217 | 24.3444 |
| K1 | 3353 | 6 | 45.0030 | 45.1977 | 34.0870 | 25.4192 |
| K2 | 3351 | 5 | 43.9091 | 44.0249 | 33.6030 | 24.5263 |
| K2 | 3352 | 6 | 41.4512 | 42.0681 | 35.5648 | 24.3963 |
| K2 | 3353 | 6 | 44.8551 | 45.1697 | 33.8095 | 25.0468 |

两组native均选中5/8/6轮，exact均为5/6/6。seed3352选择不同模型；其余四次训练的两套预测逐元素一致。`best.pt`与`best_native.pt`为字节相同的别名，`best_exact.pt`独立保存，`last.pt`保存末轮模型、AdamW与随机状态。

## 3. 原始训练与汇总

- [K1完整运行](raw_export/roundK_20261021/K1/ut-zap50k/V1/)、[K2完整运行](raw_export/roundK_20261021/K2/ut-zap50k/V1/)：protocol、history、metrics、曲线、预测、checkpoint、exact_selection。
- [TRAIN均值](raw_export/roundK_20261021/feature_statistics/)、[运行配置](raw_export/roundK_20261021/configs/)、[全部任务日志](raw_export/roundK_20261021/logs/)。
- [三卡状态](raw_export/roundK_20261021/round_k_status.json)、[任务表](raw_export/roundK_20261021/round_k_jobs.json)、[主日志](raw_export/roundK_20261021/launcher.log)、[退出码0](raw_export/roundK_20261021/launcher.exitcode)。
- [结果总表](raw_export/roundK_20261021/round_k_summary.json)、[均值与SD](raw_export/roundK_20261021/round_k_mean_sd.csv)、[配对增量](raw_export/roundK_20261021/round_k_paired_deltas.csv)、[配对汇总](raw_export/roundK_20261021/round_k_paired_mean_sd.csv)。
- [实际Adam更新量](raw_export/roundK_20261021/readout_optimizer_summary.csv)、[训练修正幅值](raw_export/roundK_20261021/readout_training_summary.csv)、[支持损失](raw_export/roundK_20261021/support_training_summary.csv)。
- [exact结果](raw_export/roundK_20261021/exact_selection_summary.csv)、[exact配对](raw_export/roundK_20261021/exact_reference_paired_mean_sd.csv)、[选权重规则差异](raw_export/roundK_20261021/exact_selection_effects.csv)。
- [排序诊断](raw_export/roundK_20261021/ranking_diagnostics.json)、[逐组合增量](raw_export/roundK_20261021/per_pair_gains.csv)、[组合概率](raw_export/roundK_20261021/pair_probability_diagnostics.csv)。

## 4. 动态与TRAIN均值评估

| 实验 | seed | TEST CW dynamic−mean | TEST OW dynamic−mean |
| --- | --- | --- | --- |
| K1 | 3351 | +3.0896 | +1.2690 |
| K1 | 3352 | +3.5313 | +1.1468 |
| K1 | 3353 | +3.9086 | +1.8217 |
| K2 | 3351 | +1.4292 | +0.6647 |
| K2 | 3352 | +1.6509 | +0.5019 |
| K2 | 3353 | +0.9704 | +0.2953 |

[native评估](raw_export/roundK_20261021/mean_gate/)与[exact评估](raw_export/roundK_20261021/mean_gate_exact/)均按实验／seed保存。每项包含dynamic/train_mean两条件、四种设置的预测与曲线、[12,32]门控均值和图像方差，优化步数为零。

96份预测逐项重算指标，另48份dynamic复评与原预测逐元素核对，覆盖全部144份文件。四个native/exact同权重训练对应16份重复预测也已验证；重复文件按原归档保留。

## 5. 原包、来源与数据分析

- [服务器TAR](archives/vgcpod_roundK_results_20261021.tar.gz)、[发布ZIP](archives/vgcpod_v1_roundK_20261021.zip)，均附SHA256；二进制按仓库原规则使用Git LFS。
- [原发布验证](package_validation/K/)、[原说明字节](package_support/K/)、[原实施与分析报告](package_support/original_reports/)。
- [99个成员映射](../package_members.json)、[结果哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[历史引用](reused_reference_members.json)、[阶段清单](../file_manifest.json)。
- 历史对照使用[11 J组](../../11/results/README.md)、[10 I组](../../10/results/README.md)、[08 G组](../../08/results/README.md)及其前序引用。
- [仓库复核](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

K1/K2缩小了Wd实际更新和动态分数幅值，整体结果仍低于静态G0。误差交集、候选内排序与下一阶段视觉判别研究集中放在最后的数据分析中。
