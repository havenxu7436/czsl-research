# 09 · H组训练结果

本阶段新增 H1 三 seed，共三次训练、30 个模型 epoch。native／exact 两种选择各保存 12 份评价与预测；六项均值替换评估包含 12 个条件、48 份评价与预测。合计 **72 份预测文件、12 份 checkpoint 文件**；TRAIN 查询均值与六套门控矩另行保存。

## native选择主结果

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | 3351 | 5／5 | 43.7328 | 31.6663 | 33.1104 | 24.3121 |
| H1 | 3352 | 8／8 | 41.7932 | 29.9935 | 34.9722 | 22.9596 |
| H1 | 3353 | 6／6 | 45.0315 | 34.4327 | 32.8046 | 24.3395 |

AUC 为服务器原始分数保存的 native AUC，面积×100。每次训练完成十轮、3600 次更新，物体／GT-object 条件属性、CW／OW、seen／unseen 以及 NLL 的完整指标见各运行 JSON。

## exact选择结果

| 实验 | seed | exact最佳epoch | VAL CW native | VAL CW exact | TEST CW native | TEST OW native |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | 3351 | 5 | 43.7328 | 43.8912 | 33.1104 | 24.3121 |
| H1 | 3352 | 8 | 41.7932 | 41.9348 | 34.9722 | 22.9596 |
| H1 | 3353 | 6 | 45.0315 | 45.1725 | 32.8046 | 24.3395 |

H1 的两条轨道均选择 5／8／6 epoch，对应模型参数与预测一致；best.pt 与 best_native.pt 为相同字节，best_exact.pt 作为独立 checkpoint 文件保留。last.pt 保存完整模型、优化器与随机状态。

## 原运行、TRAIN均值与汇总

- [H1三seed运行](raw_export/roundH_20261020/H1/ut-zap50k/V1/)包含 protocol、history、metrics、曲线、预测、四份 checkpoint 及 exact_selection。
- [共享TRAIN均值](raw_export/roundH_20261020/feature_statistics/)、[实际运行配置](raw_export/roundH_20261020/configs/H1.json)：均值形状 [12,768]，统计 22,998 张 TRAIN 图像，float64 累计、float32 模型 buffer。
- [三卡状态](raw_export/roundH_20261020/round_h_status.json)、[任务表](raw_export/roundH_20261020/round_h_jobs.json)、[主日志](raw_export/roundH_20261020/launcher.log)、[退出码0](raw_export/roundH_20261020/launcher.exitcode)。
- [主结果](raw_export/roundH_20261020/round_h_summary.json)、[均值与SD](raw_export/roundH_20261020/round_h_mean_sd.csv)、[配对差值](raw_export/roundH_20261020/round_h_paired_deltas.csv)、[配对汇总](raw_export/roundH_20261020/round_h_paired_mean_sd.csv)。
- [同ρ／β主表](raw_export/roundH_20261020/matched_native_summary.csv)、[支持损失](raw_export/roundH_20261020/support_training_summary.csv)、[exact指标](raw_export/roundH_20261020/exact_selection_summary.csv)、[exact历史配对](raw_export/roundH_20261020/exact_reference_paired_mean_sd.csv)。
- [排序诊断](raw_export/roundH_20261020/ranking_diagnostics.json)、[诊断差值](raw_export/roundH_20261020/ranking_diagnostic_paired_mean_sd.csv)、[逐组合概率](raw_export/roundH_20261020/pair_probability_diagnostics.csv)、[逐组合增量](raw_export/roundH_20261020/per_pair_gains.csv)。

## H1动态与TRAIN均值替换

| seed | H1动态−均值 TEST CW AUC | TEST OW AUC | 未见属性正确数变化 |
| --- | --- | --- | --- |
| 3351 | +1.9423 | +1.1414 | +24 |
| 3352 | +1.7953 | +1.1629 | +26 |
| 3353 | +2.7006 | +2.2436 | +26 |

[native均值评估](raw_export/roundH_20261020/mean_gate/)和[exact均值评估](raw_export/roundH_20261020/mean_gate_exact/)分别保存 TRAIN 门控均值 [12,32]、图像方差及 dynamic/train_mean 两个条件的四种评价、预测与曲线。每项统计全部 22,998 张 TRAIN 图像，优化步数为零；条件各用自己的 VAL bias。

## 原交付、来源与分析

- [原服务器TAR](archives/vgcpod_roundH_results_20261020.tar.gz)、[原H训练ZIP](archives/vgcpod_v1_roundH_20261020.zip)，旁附 SHA256；二进制沿用 Git LFS。
- [原发布验证](package_validation/H/)、[原包说明字节](package_support/H/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包 80 个成员映射](../package_members.json)、[H结果成员与哈希](raw_export_members.json)、[服务器源码核对](server_source_identity.json)、[历史引用](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原 TAR 的 C/D/E/F/G 历史结果与源码映射到 [04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)、[07](../../07/results/README.md)、[08](../../08/results/README.md)；A5/A6 引用 [02](../../02/results/README.md)。
- [仓库复算与归档验证](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

H1−G1 的 TEST CW／OW AUC 为 −0.902／−1.186 点，H1 dynamic−mean 为 +2.146／+1.516 点；完整训练效果与同一权重的推理干预分别分析。I1/I2 为后续建议，本阶段归档为已完成 H1。
