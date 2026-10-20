# 07 · F组训练结果

本阶段新增 F1 三次训练、30 个模型 epoch。native／exact 两种选择共保存 24 份 VAL／TEST CW／OW 评价与预测；六项均值替换评估共 12 个条件、48 份评价与预测。合计 72 份预测导出、12 份 checkpoint 文件。

| seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- |
| 3351 | 8／8 | 43.7359 | 30.9921 | 33.7102 | 25.5140 |
| 3352 | 8／8 | 41.7244 | 29.8303 | 34.9602 | 22.8710 |
| 3353 | 6／6 | 45.0561 | 34.4285 | 33.4192 | 24.6694 |

AUC 使用服务器保存的 native AUC，面积×100。每次训练完成 10 epochs、3600 次更新；主权重按首次严格最大 VAL CW native AUC 选择。本轮 exact 与 native 的选中参数相同，12份对应预测数组逐元素一致。

## 运行、日志与β汇总

- [三个seed原运行](raw_export/roundF_20261020/F1/ut-zap50k/V1/)保存协议、完整历史、指标、曲线、预测与四份 checkpoint；exact 结果在各 seed 的 `exact_selection/`。
- [三卡状态](raw_export/roundF_20261020/round_f_status.json)、[任务表](raw_export/roundF_20261020/round_f_jobs.json)、[主日志](raw_export/roundF_20261020/launcher.log)、[退出码0](raw_export/roundF_20261020/launcher.exitcode)。三次训练及六项评估均完成。
- [原运行汇总](raw_export/roundF_20261020/round_f_summary.csv)、[均值与SD](raw_export/roundF_20261020/round_f_mean_sd.csv)、[同seed增量](raw_export/roundF_20261020/round_f_paired_deltas.csv)、[配对均值](raw_export/roundF_20261020/round_f_paired_mean_sd.csv)。
- [E1/F1 β主表](raw_export/roundF_20261020/beta_native_summary.csv)、[β均值](raw_export/roundF_20261020/beta_native_mean_sd.csv)、[支持损失训练表](raw_export/roundF_20261020/support_training_summary.csv)。
- [exact选择](raw_export/roundF_20261020/exact_selection_summary.csv)、[选权重差异](raw_export/roundF_20261020/exact_selection_effects.csv)、[逐组合增量](raw_export/roundF_20261020/per_pair_gains.csv)。

每个 seed 的 `best.pt` 与 `best_native.pt` 为逐字节相同的兼容入口；`best_exact.pt` 保留第二种选择的身份，`last.pt` 保存完整训练状态。12份 checkpoint 文件对应三次训练。

## 动态与TRAIN均值替换

| seed | 动态−均值 TEST CW AUC | TEST OW AUC | 未见条件属性正确数变化 |
| --- | --- | --- | --- |
| 3351 | +0.0268 | -0.0425 | +7 |
| 3352 | -0.0851 | -0.0512 | +6 |
| 3353 | +0.1051 | +0.0842 | +8 |

[native评估](raw_export/roundF_20261020/mean_gate/)和[exact评估](raw_export/roundF_20261020/mean_gate_exact/)各保存 TRAIN 均值及图像方差、dynamic／train_mean 两种条件的指标、预测与曲线。统计使用22,998张TRAIN图像，均值形状 `[12,32]`，优化步数为零。条件各自使用自己的 VAL bias。

## 原交付、来源与分析

- [原服务器TAR](archives/vgcpod_roundF_results_20261020.tar.gz)、[原F训练ZIP](archives/vgcpod_v1_roundF_20261020.zip)，旁附 SHA256；二进制沿用 Git LFS。
- [原发布验证](package_validation/F/)、[原包说明字节](package_support/F/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包69成员映射](../package_members.json)、[新增结果成员与哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[已有结果引用](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原 TAR 中 C／D／E 的结果与源码已与 [04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)核对并建立来源映射；A5／A6 三seed引用 [02](../../02/results/README.md)。
- [仓库复算核查](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。本轮分析额外从已保存概率计算两个分支因子交换，共24条标量评价，不增加训练或原始预测导出计数。

F1 相对 E1 的 TEST CW／OW AUC 为 −0.397／−0.829 点，未见条件属性 +1.816 个百分点。后续 G0／G1 属于分析建议；本阶段实际训练配置为 F1、β=0.05。
