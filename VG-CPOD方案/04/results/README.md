# 04 · C组与B3补seed训练结果

本轮归档 C1/3351、C2/3351 和 B3/3352，共 3 次训练、30 个模型 epoch、12 份评价与预测、6 份 best/last 权重。C 队列实际完成两次 C 训练，B3/3352 在启动前已完成，状态记录为 `already_complete`。

| 组 | seed | 最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---|---|---|---|---|
| C1 | 3351 | 8 | 43.6114 | 30.6882 | 34.2699 | 26.3286 |
| C2 | 3351 | 8 | 43.3917 | 30.3144 | 34.0359 | 26.2493 |
| B3 | 3352 | 8 | 41.9110 | 29.7450 | 35.3642 | 24.4934 |

AUC 为保存的 AUC_native，面积×100。各训练共 10 epochs、3600 次更新，best 按首次严格最大 VAL CW AUC 选择；同一权重评价 VAL/TEST CW/OW。TEST HM 使用对应 VAL 的 bias。

- [C1完整运行](raw_export/roundC_20261020/C1/ut-zap50k/V1/seed3351/)、[C2完整运行](raw_export/roundC_20261020/C2/ut-zap50k/V1/seed3351/)、[B3/3352完整运行](raw_export/roundB_20261019/B3/ut-zap50k/V1/seed3352/)：metrics、history、protocol、best/last、四份预测和曲线。
- [C队列完成状态](raw_export/roundC_20261020/round_c_status.json)、[任务计划](raw_export/roundC_20261020/round_c_jobs.json)、[总日志](raw_export/roundC_20261020/launcher.log)、[退出码0](raw_export/roundC_20261020/launcher.exitcode)。
- [原逐运行汇总](raw_export/roundC_20261020/round_c_summary.csv)、[原同seed配对增量](raw_export/roundC_20261020/round_c_paired_deltas.csv)、[原配置聚合](raw_export/roundC_20261020/round_c_mean_sd.csv)、[原逐组合增量](raw_export/roundC_20261020/per_pair_gains.csv)。联合汇总含原 B3/3351，因此为 4 个运行记录；其原始运行保存在 03。
- [服务器完整导出包](archives/vgcpod_roundC_results_20261020.tar.gz)与[SHA256](archives/vgcpod_roundC_results_20261020.tar.gz.sha256)：保留全部 C、当时的 B 结果和两套运行源码。
- [实际C训练代码ZIP](archives/vgcpod_v1_roundC_20261020.zip)与[SHA256](archives/vgcpod_v1_roundC_20261020.zip.sha256)。归档、权重与预测沿用仓库 Git LFS 规则。
- [原发布验证](package_validation/)、[原说明文档字节](package_support/)、[C/B源码身份](server_source_identity.json)。
- [原始结果成员与哈希](raw_export_members.json)、[阶段SHA256清单](../file_manifest.json)、[完整数据分析](../analysis/README.md)。
- [仓库验证结果](../analysis/repository_validation.json)：已恢复原 C/B 运行包，复算 32 份预测；10 张 CSV 表与原分析逐字节一致。

归档解出的 B 结果只新增 B3/3352 及其日志；原 B1/B2/B3/3351 继续引用[03](../../03/results/README.md)，A5/A6 引用[02](../../02/results/README.md)。原导出 TAR 内保留当时完整目录。本轮日志逐字节保存，逐 epoch 数值以各运行的 `history.json` 为准。
