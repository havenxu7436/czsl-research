# 03 · B组训练结果

2026-10-19：B1/B2/B3 各完成 seed3351、10 epochs，共3次训练、30个模型epoch、12份评价与预测、6份best/last权重。

| 组 | seed | 最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 3351 | 10 | 42.6663 | 29.6301 | 33.2738 | 25.8394 |
| B2 | 3351 | 5 | 44.0890 | 31.1216 | 33.4501 | 24.6562 |
| B3 | 3351 | 8 | 43.5622 | 30.7067 | 34.3301 | 26.3164 |

选中权重由首次严格最大 VAL CW AUC_native 决定，使用同一权重评价 VAL/TEST 的 CW/OW。TEST HM 使用对应 VAL 的 bias。

- [原始运行目录](raw_export/roundB_20261019/)：metrics、history、protocol、checkpoints、predictions、curves和日志。
- [队列完成状态](raw_export/roundB_20261019/round_b_status.json)、[退出码](raw_export/roundB_20261019/launcher.exitcode)。
- [逐运行汇总](raw_export/roundB_20261019/round_b_summary.csv)、[原导出同seed配对增量](raw_export/roundB_20261019/round_b_paired_deltas.csv)、[原逐组合增量](raw_export/roundB_20261019/per_pair_gains.csv)。
- [服务器完整导出包](archives/vgcpod_roundB_results_20261019.tar.gz)与[SHA256](archives/vgcpod_roundB_results_20261019.tar.gz.sha256)。
- [实际训练代码ZIP](archives/vgcpod_v1_roundB_20261019.zip)与[SHA256](archives/vgcpod_v1_roundB_20261019.zip.sha256)。二进制归档、权重及预测由 Git LFS 保存。
- [原版发布验证](package_validation/)、[原说明文档字节](package_support/)、[服务器源码身份](server_source_identity.json)。
- [文件清单](raw_export_members.json)、[阶段SHA256清单](../file_manifest.json)、[完整数据分析](../analysis/README.md)。

每组结果位于 `raw_export/roundB_20261019/B*/ut-zap50k/V1/seed3351/`；已有A5/A6参照位于[02的原始运行目录](../../02/results/raw_export/roundA_20261019/)。
