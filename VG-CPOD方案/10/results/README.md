# 10 · I组训练结果

本阶段新增I1/I2各三seed，共六次训练、60个模型epoch。native／exact两条选择分别保存24份评价与预测；十二项均值评估包含24个条件、96份评价与预测。合计 **144份预测文件、24份checkpoint文件**，共享TRAIN查询均值与十二套门控矩另行保存。

## 1. native选择主结果

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | 3351 | 8／10 | 43.6373 | 31.1230 | 33.9122 | 26.1414 |
| I1 | 3352 | 8／8 | 41.7728 | 29.8443 | 35.1903 | 23.6263 |
| I1 | 3353 | 6／6 | 44.8723 | 34.5470 | 33.3681 | 24.5984 |
| I2 | 3351 | 8／8 | 43.7343 | 31.1052 | 33.9697 | 26.1872 |
| I2 | 3352 | 8／8 | 41.6859 | 29.7517 | 35.1811 | 23.7244 |
| I2 | 3353 | 6／6 | 44.8780 | 34.5049 | 33.4632 | 24.6480 |

AUC使用服务器原始分数保存的native曲线积分，面积×100。每次训练完成十轮、3600次更新，完整物体／条件属性、CW／OW、seen／unseen及NLL指标保存在各运行JSON中。

## 2. exact选择结果

| 实验 | seed | exact最佳epoch | VAL CW native | VAL CW exact | TEST CW native | TEST OW native |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | 3351 | 10 | 43.6287 | 43.8530 | 34.2815 | 26.3127 |
| I1 | 3352 | 8 | 41.7728 | 41.9822 | 35.1903 | 23.6263 |
| I1 | 3353 | 6 | 44.8723 | 45.1963 | 33.3681 | 24.5984 |
| I2 | 3351 | 8 | 43.7343 | 43.8698 | 33.9697 | 26.1872 |
| I2 | 3352 | 8 | 41.6859 | 41.8783 | 35.1811 | 23.7244 |
| I2 | 3353 | 6 | 44.8780 | 45.1829 | 33.4632 | 24.6480 |

I1/3351的native选择第8轮，exact选择第10轮。其余五次训练的两条选择轨道epoch相同，对应参数与预测一致。`best.pt`与`best_native.pt`为逐字别名；`best_exact.pt`独立保存；`last.pt`保存末轮完整模型、优化器和随机状态。

## 3. 原运行、共享均值与汇总

- [I1三seed](raw_export/roundI_20261021/I1/ut-zap50k/V1/)与[I2三seed](raw_export/roundI_20261021/I2/ut-zap50k/V1/)：protocol、history、metrics、曲线、预测、四份checkpoint和exact_selection。
- [共享TRAIN均值](raw_export/roundI_20261021/feature_statistics/)、[运行配置](raw_export/roundI_20261021/configs/)：22,998张图像、12个查询，float64累计，float32模型buffer。
- [三卡状态](raw_export/roundI_20261021/round_i_status.json)、[任务表](raw_export/roundI_20261021/round_i_jobs.json)、[主日志](raw_export/roundI_20261021/launcher.log)、[退出码0](raw_export/roundI_20261021/launcher.exitcode)。
- [主结果](raw_export/roundI_20261021/round_i_summary.json)、[均值与SD](raw_export/roundI_20261021/round_i_mean_sd.csv)、[配对差值](raw_export/roundI_20261021/round_i_paired_deltas.csv)、[配对汇总](raw_export/roundI_20261021/round_i_paired_mean_sd.csv)。
- [同ρ／β主表](raw_export/roundI_20261021/matched_native_summary.csv)、[支持损失](raw_export/roundI_20261021/support_training_summary.csv)、[exact指标](raw_export/roundI_20261021/exact_selection_summary.csv)、[exact历史配对](raw_export/roundI_20261021/exact_reference_paired_mean_sd.csv)。
- [排序诊断](raw_export/roundI_20261021/ranking_diagnostics.json)、[诊断配对汇总](raw_export/roundI_20261021/ranking_diagnostic_paired_mean_sd.csv)、[逐组合概率](raw_export/roundI_20261021/pair_probability_diagnostics.csv)、[逐组合增量](raw_export/roundI_20261021/per_pair_gains.csv)。

## 4. 动态与TRAIN均值替换

| 实验 | seed | 动态−均值 TEST CW | TEST OW | 未见属性正确数变化 |
| --- | --- | --- | --- | --- |
| I1 | 3351 | +0.2081 | -0.1352 | +8 |
| I1 | 3352 | +0.4597 | +0.1403 | -7 |
| I1 | 3353 | +0.2832 | -0.0747 | +2 |
| I2 | 3351 | +0.3251 | -0.0534 | +10 |
| I2 | 3352 | +0.4003 | +0.3004 | -3 |
| I2 | 3353 | +0.5592 | +0.0484 | +5 |

[native评估](raw_export/roundI_20261021/mean_gate/)与[exact评估](raw_export/roundI_20261021/mean_gate_exact/)保存TRAIN门控均值[12,32]、方差以及dynamic/train_mean两条件的四种评价、预测与曲线。每项统计全部TRAIN图像，优化步数为零，各条件使用自己的VAL bias。

## 5. 原交付、来源与数据分析

- [原服务器TAR](archives/vgcpod_roundI_results_20261021.tar.gz)、[原I训练ZIP](archives/vgcpod_v1_roundI_20261021.zip)，旁附SHA256；二进制沿用Git LFS。
- [原发布验证](package_validation/I/)、[原包说明字节](package_support/I/)、[原实施和分析报告字节](package_support/original_reports/)。
- [原包85个成员映射](../package_members.json)、[结果成员与哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[历史引用](reused_reference_members.json)、[阶段SHA256](../file_manifest.json)。
- 原TAR的C/D/E/F/G/H历史结果映射到[04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)、[07](../../07/results/README.md)、[08](../../08/results/README.md)、[09](../../09/results/README.md)；A5/A6引用[02](../../02/results/README.md)。
- [仓库复算与归档核对](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

I2相对H1的TEST CW／OW回升0.576／0.983点，相对G0仍低0.307／0.330点。I1/I2差异较小；下一轮建议J1检验静态与动态CP读出分别学习的效果。
