# 11 · J组训练结果

本阶段完成J1三seed，共三次训练、30个模型epoch。native/exact分别保存12份评价与预测；六项均值评估包含12个条件、48份评价与预测。合计 **72份预测文件、12份checkpoint文件**，另保存共享TRAIN查询均值与六套门控矩。

## 1. native选择主结果

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| J1 | 3351 | 5／5 | 43.9794 | 31.0402 | 33.8136 | 23.7420 |
| J1 | 3352 | 6／6 | 41.2374 | 29.8925 | 35.9243 | 24.3884 |
| J1 | 3353 | 6／6 | 44.7703 | 33.6924 | 34.2198 | 25.4464 |

AUC为服务器保存的原始分数曲线积分，面积×100。每次训练均完成十轮、3600次更新；物体、条件属性、CW/OW、seen/unseen、HM和NLL见各运行JSON。

## 2. exact选择结果

| 实验 | seed | exact最佳epoch | VAL CW native | VAL CW exact | TEST CW native | TEST OW native |
| --- | --- | --- | --- | --- | --- | --- |
| J1 | 3351 | 5 | 43.9794 | 44.1396 | 33.8136 | 23.7420 |
| J1 | 3352 | 6 | 41.2374 | 41.3872 | 35.9243 | 24.3884 |
| J1 | 3353 | 6 | 44.7703 | 44.9359 | 34.2198 | 25.4464 |

三seed的native/exact分别选中5/6/6轮，对应参数与预测完全相同。`best.pt`与`best_native.pt`为逐字别名；`best_exact.pt`独立保存，`last.pt`保存末轮模型、AdamW和随机状态。

## 3. 原运行、均值与汇总

- [三seed完整运行](raw_export/roundJ_20261021/J1/ut-zap50k/V1/)：protocol、history、metrics、曲线、预测、checkpoint、exact_selection。
- [固定TRAIN查询均值](raw_export/roundJ_20261021/feature_statistics/)、[运行配置](raw_export/roundJ_20261021/configs/)。
- [三卡状态](raw_export/roundJ_20261021/round_j_status.json)、[任务表](raw_export/roundJ_20261021/round_j_jobs.json)、[主日志](raw_export/roundJ_20261021/launcher.log)、[退出码0](raw_export/roundJ_20261021/launcher.exitcode)。
- [结果总表](raw_export/roundJ_20261021/round_j_summary.json)、[均值与SD](raw_export/roundJ_20261021/round_j_mean_sd.csv)、[配对差值](raw_export/roundJ_20261021/round_j_paired_deltas.csv)、[配对汇总](raw_export/roundJ_20261021/round_j_paired_mean_sd.csv)。
- [两路训练修正RMS](raw_export/roundJ_20261021/readout_training_summary.csv)、[支持损失](raw_export/roundJ_20261021/support_training_summary.csv)、[同ρ/β对照](raw_export/roundJ_20261021/matched_native_summary.csv)。
- [exact选择结果](raw_export/roundJ_20261021/exact_selection_summary.csv)、[exact配对](raw_export/roundJ_20261021/exact_reference_paired_mean_sd.csv)。
- [排序诊断](raw_export/roundJ_20261021/ranking_diagnostics.json)、[逐组合增量](raw_export/roundJ_20261021/per_pair_gains.csv)、[逐组合概率](raw_export/roundJ_20261021/pair_probability_diagnostics.csv)。

## 4. 动态与TRAIN均值干预

| seed | TEST CW dynamic−mean | TEST OW dynamic−mean | 未见属性净正确数 |
| --- | --- | --- | --- |
| 3351 | +8.5720 | +3.5260 | +19 |
| 3352 | +8.1282 | +2.4147 | +38 |
| 3353 | +9.2352 | +4.8356 | +132 |

[native评估](raw_export/roundJ_20261021/mean_gate/)与[exact评估](raw_export/roundJ_20261021/mean_gate_exact/)各保存[12,32]门控均值、图像方差和dynamic/train_mean四种设置的预测与曲线。静态Wc和动态Wd分别参与均值条件。三seed的两条选择轨道均相同，因此对应干预指标相同。

## 5. 原包、来源与数据分析

- [服务器TAR](archives/vgcpod_roundJ_results_20261021.tar.gz)、[训练ZIP](archives/vgcpod_v1_roundJ_20261021.zip)，旁附SHA256，二进制沿用Git LFS。
- [原发布验证](package_validation/J/)、[原说明字节](package_support/J/)、[原实施与分析报告](package_support/original_reports/)。
- [91个成员映射](../package_members.json)、[结果哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[历史引用](reused_reference_members.json)、[阶段清单](../file_manifest.json)。
- 历史结果沿用[10 I组](../../10/results/README.md)、[09 H组](../../09/results/README.md)、[08 G组](../../08/results/README.md)及其前序引用。
- [仓库复核](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

J1相对I2的CW提高0.448点、OW下降0.328点；seed3351的OW下降2.445点。动态分数修正幅值约增大9.9倍，下一轮优先检验Wd独立学习率。具体配对差值、Adam状态及推导见末项数据分析。
