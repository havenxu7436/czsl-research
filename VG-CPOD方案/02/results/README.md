# 02 · A 组训练结果

2026-10-19：12 次训练均完成 10 epochs，共 48 份评价、48 份预测和 24 份 best/last 权重。A1/A5/A6 各三个 seed，A2/A3/A4 各一个 seed。

| 组 | seed | 最佳 epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 3351 | 8 | 42.2460 | 29.2961 | 34.4951 | 25.9873 |
| A1 | 3352 | 6 | 42.0909 | 28.5317 | 35.5823 | 26.3542 |
| A1 | 3353 | 6 | 44.0273 | 33.3991 | 33.8244 | 23.3620 |
| A2 | 3351 | 10 | 39.6746 | 24.0615 | 32.5305 | 25.8365 |
| A3 | 3351 | 5 | 40.0189 | 26.1160 | 32.7563 | 23.6505 |
| A4 | 3351 | 5 | 34.3910 | 24.4658 | 31.8526 | 21.3055 |
| A5 | 3351 | 8 | 43.2281 | 30.2911 | 34.0224 | 26.1687 |
| A5 | 3352 | 8 | 41.7200 | 29.7282 | 35.6233 | 24.8439 |
| A5 | 3353 | 6 | 44.2855 | 33.6656 | 33.6657 | 24.7534 |
| A6 | 3351 | 8 | 34.4555 | 21.7217 | 30.1066 | 24.8260 |
| A6 | 3352 | 9 | 34.8987 | 21.3374 | 30.1658 | 25.0190 |
| A6 | 3353 | 10 | 35.2883 | 22.0559 | 30.0650 | 24.5065 |

每次运行按首次严格最大 VAL CW AUC_native 选模。HM 使用对应 VAL 的 bias，TEST 使用同一份选中权重。

- [原始运行目录](raw_export/roundA_20261019/)：metrics、history、protocol、checkpoints、predictions、curves 与日志。
- [初始六组 launcher 状态](initial_launch/)与[确认补跑状态](raw_export/roundA_20261019/round_a_status.json)。
- [完整服务器导出包](archives/vgcpod_roundA_confirm_20261019.tar.gz)与[SHA256](archives/vgcpod_roundA_confirm_20261019.tar.gz.sha256)，由 Git LFS 保存。
- [训练代码 ZIP](archives/vgcpod_v1_roundA_20261019.zip)与[SHA256](archives/vgcpod_v1_roundA_20261019.zip.sha256)。
- [原版发布检查记录](package_validation/)与[原始说明文档字节](package_support/)。
- [三 seed 数据分析](../analysis/README.md)和[文件 SHA256 清单](../file_manifest.json)。

归档的服务器包已包含最初六组结果；seed3351 的 78 个模型成员与初次下载文件一致。A1/A5/A6 的多 seed 结论与 A2/A3/A4 的单 seed 探索结果分开报告。
