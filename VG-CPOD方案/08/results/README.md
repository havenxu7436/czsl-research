# 08 · G组训练结果

本阶段新增G0/G1各三seed，共六次训练、60个模型epoch。native与exact两种选择分别保存24份评价／预测；六项G1均值替换评估包含12个条件、48份评价／预测。合计 **96份预测导出、24份checkpoint文件**。

## native选择主结果

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | 3351 | 8／8 | 43.9071 | 31.0988 | 34.1754 | 26.2127 |
| G0 | 3352 | 8／8 | 41.6325 | 29.6697 | 35.2053 | 23.7875 |
| G0 | 3353 | 9／6 | 44.4907 | 34.0762 | 34.1545 | 25.5505 |
| G1 | 3351 | 8／8 | 44.0321 | 31.0817 | 34.2140 | 26.1916 |
| G1 | 3352 | 8／8 | 41.8392 | 29.7950 | 35.1813 | 23.5737 |
| G1 | 3353 | 9／6 | 44.7601 | 34.1473 | 34.1965 | 25.4049 |

AUC为服务器保存的native AUC、面积×100。每次训练均完成10轮、3600次更新。每个模型的best.pt与best_native.pt为相同字节，last.pt保存完整训练状态。

## exact选择结果

| 实验 | seed | exact最佳epoch | VAL CW native | VAL CW exact | TEST CW native | TEST OW native |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | 3351 | 8 | 43.9071 | 44.0981 | 34.1754 | 26.2127 |
| G0 | 3352 | 8 | 41.6325 | 41.8349 | 35.2053 | 23.7875 |
| G0 | 3353 | 6 | 44.4458 | 45.1364 | 33.7866 | 25.0849 |
| G1 | 3351 | 8 | 44.0321 | 44.1745 | 34.2140 | 26.1916 |
| G1 | 3352 | 8 | 41.8392 | 42.0713 | 35.1813 | 23.5737 |
| G1 | 3353 | 6 | 44.6295 | 45.2592 | 33.6163 | 24.7593 |

seed3351/3352两种选择均为epoch8；G0/G1的seed3353分别选中native epoch9与exact epoch6。对应权重和预测独立保留。G0/G1在前两个seed上的16份exact/native预测逐元素一致，第三个seed使用不同权重。

## 原运行、日志与汇总

- [G0原运行](raw_export/roundG_20261020/G0/ut-zap50k/V1/)与[G1原运行](raw_export/roundG_20261020/G1/ut-zap50k/V1/)包含protocol、history、metrics、曲线、预测和四份checkpoint，exact结果位于各seed的exact_selection目录。
- [三条队列状态](raw_export/roundG_20261020/round_g_status.json)、[六项任务表](raw_export/roundG_20261020/round_g_jobs.json)、[主日志](raw_export/roundG_20261020/launcher.log)、[退出码0](raw_export/roundG_20261020/launcher.exitcode)。
- [主结果](raw_export/roundG_20261020/round_g_summary.csv)、[均值与SD](raw_export/roundG_20261020/round_g_mean_sd.csv)、[配对增量](raw_export/roundG_20261020/round_g_paired_deltas.csv)、[配对汇总](raw_export/roundG_20261020/round_g_paired_mean_sd.csv)。
- [三档β主表](raw_export/roundG_20261020/beta_native_summary.csv)、[支持损失表](raw_export/roundG_20261020/support_training_summary.csv)、[exact指标](raw_export/roundG_20261020/exact_selection_summary.csv)、[选择差异](raw_export/roundG_20261020/exact_selection_effects.csv)。
- [排序诊断](raw_export/roundG_20261020/ranking_diagnostics.json)、[诊断配对汇总](raw_export/roundG_20261020/ranking_diagnostic_paired_mean_sd.csv)、[逐组合概率](raw_export/roundG_20261020/pair_probability_diagnostics.csv)。

## G1动态与TRAIN均值替换

| seed | G1动态−均值 TEST CW AUC | TEST OW AUC | 未见属性正确数变化 |
| --- | --- | --- | --- |
| 3351 | +0.1244 | +0.0333 | +0 |
| 3352 | -0.0390 | -0.2495 | -4 |
| 3353 | +0.3191 | +0.1047 | -1 |

[native均值评估](raw_export/roundG_20261020/mean_gate/)和[exact均值评估](raw_export/roundG_20261020/mean_gate_exact/)各保存TRAIN均值、图像方差及dynamic/train_mean两种条件的四种评价、预测与曲线。统计使用22,998张TRAIN图像，均值形状[12,32]，优化步数为零；条件各用自己的VAL bias。

## 原交付、来源与分析

- [原服务器TAR](archives/vgcpod_roundG_results_20261020.tar.gz)、[原G训练ZIP](archives/vgcpod_v1_roundG_20261020.zip)，旁附SHA256；二进制沿用Git LFS。
- [原发布验证](package_validation/G/)、[原包说明字节](package_support/G/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包76成员映射](../package_members.json)、[新增结果成员与哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[已有结果引用](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原TAR中C/D/E/F结果与源码引用[04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)、[07](../../07/results/README.md)已有归档；A5/A6引用[02](../../02/results/README.md)。
- [仓库复算验证](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。静态CP折叠为保存权重上的算子等价验证，未增加训练或原预测导出。

G1−G0的TEST CW/OW native AUC为+0.019/−0.127点；β中间值与门控训练收益分别分析。H1为后续建议，本阶段归档为已完成G0/G1。
