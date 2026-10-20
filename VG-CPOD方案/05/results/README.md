# 05 · D组与补充轮训练结果

本阶段新增归档 6 次训练、60 个模型 epoch、24 份训练后评价／预测、12 份 best/last 权重。另有 C2/D1 各三 seed 的同权重干预，共 6 个 checkpoint、12 个条件、48 份评价／预测；本阶段预测文件共 72 份。

| 组 | seed | 最佳 epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 3352 | 8 | 41.9346 | 29.7615 | 35.4659 | 24.6193 |
| C2 | 3352 | 8 | 41.8360 | 29.7651 | 35.3923 | 24.7416 |
| C2 | 3353 | 6 | 44.5015 | 34.0340 | 33.5573 | 24.3978 |
| D1 | 3351 | 10 | 43.8745 | 30.8834 | 34.7717 | 26.6923 |
| D1 | 3352 | 8 | 41.4983 | 29.7811 | 35.3665 | 24.4136 |
| D1 | 3353 | 9 | 44.5294 | 34.0270 | 34.2359 | 25.2524 |

AUC 为服务器保存的 native AUC，面积×100。六次训练均为 10 epochs、3600 次更新；best 按首次严格最大 VAL CW AUC 选择。原训练文件保留原字节。

## 运行与队列状态

- [D1三个seed](raw_export/roundD_20261020/D1/)、[C1补seed](raw_export/roundC_20261020/C1/)、[C2两个补seed](raw_export/roundC_20261020/C2/)：完整配置协议、历史、指标、best/last、预测和曲线。
- [首轮D状态](raw_export/roundD_20261020/round_d_status.json)、[日志](raw_export/roundD_20261020/launcher.log)、[退出码](raw_export/roundD_20261020/launcher.exitcode)。首轮新训练 D1/3351，C2 两份结果已完成并复用。
- [补充轮状态](raw_export/next_20261020/status.json)、[任务计划](raw_export/next_20261020/jobs.json)、[日志](raw_export/next_20261020/launcher.log)、[退出码0](raw_export/next_20261020/launcher.exitcode)。补充轮完成 D1/3352、3353 与 C1/3352，以及两份新 D1 的均值评估。
- [最终逐运行汇总](raw_export/roundD_20261020/round_d_summary.csv)、[同seed增量](raw_export/roundD_20261020/round_d_paired_deltas.csv)、[配对均值](raw_export/roundD_20261020/round_d_paired_mean_sd.csv)、[两组交互](raw_export/next_20261020/interaction_by_seed.csv)。

汇总含 C1/C2 首 seed，原始运行引用 [04](../../04/results/README.md)，故联合汇总有 8 份运行记录；A5/A6 的完整三 seed 基座引用 [02](../../02/results/README.md)。

## 动态与TRAIN均值替换

| checkpoint组 | seed | 选中epoch | 动态−均值 CW AUC | 动态−均值 OW AUC |
| --- | --- | --- | --- | --- |
| C2 | 3351 | 8 | +0.2004 | +0.0439 |
| C2 | 3352 | 8 | -0.1101 | -0.2958 |
| C2 | 3353 | 6 | +0.0685 | -0.1179 |
| D1 | 3351 | 10 | +0.1892 | +0.0328 |
| D1 | 3352 | 8 | -0.0851 | -0.1634 |
| D1 | 3353 | 9 | +0.1654 | +0.0908 |

[六个checkpoint的完整评估](raw_export/roundD_20261020/mean_gate/)保存 TRAIN 均值与图像方差，以及 dynamic/train_mean 两套指标、VAL/TEST CW/OW 预测与曲线。TRAIN 共 22,998 张，每个均值形状 `[12,32]`；优化步数为零。各条件分别用自身 VAL bias。

## 原始交付与仓库索引

- [首轮D服务器归档](archives/vgcpod_roundD_results_20261020.tar.gz)、[完成轮服务器归档](archives/vgcpod_next_results_20261020.tar.gz)。原 TAR 保留当时完整导出，包括已引用的 C 首 seed。
- [原D训练包](archives/vgcpod_v1_roundD_20261020.zip)、[原补充轮启动包](archives/vgcpod_next_v1_20261020.zip)；各文件旁附 SHA256，二进制沿用 Git LFS。
- [原发布验证](package_validation/)、[原说明字节](package_support/)、[服务器源码身份](server_source_identity.json)。
- [新增结果成员与哈希](raw_export_members.json)、[复用04的原成员](reused_stage04_members.json)、[阶段SHA256清单](../file_manifest.json)。
- [仓库复算及README核查](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

原版说明中的 E1 ρ=0.15 和双 native/exact checkpoint 保存为后续方案。本阶段保存实际完成的 D/C 训练及同权重干预。
