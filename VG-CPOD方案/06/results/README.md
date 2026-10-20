# 06 · E组训练结果

本阶段新增 E1 三次训练、30 个模型 epoch。native／exact 两种选择共保存 24 份 VAL／TEST CW／OW 评价与预测，另有六项同权重均值替换评估、12 个条件、48 份评价与预测。合计 72 份预测导出、12 份 checkpoint 文件。

| seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---|---|---|---|
| 3351 | 8／8 | 43.8332 | 30.9922 | 34.2948 | 26.4335 |
| 3352 | 8／8 | 41.8212 | 29.7720 | 35.4295 | 24.4062 |
| 3353 | 6／6 | 45.0071 | 34.5988 | 33.5570 | 24.7015 |

AUC 为服务器保存的 native AUC，面积×100。每次训练均完成 10 epochs、3600 次更新；主 best 按首次严格最大 VAL CW native AUC 选择。本轮 exact 选择的模型参数与主选择一致，12 份对应预测数组逐元素相同。

## 运行与队列状态

- [三个seed原运行](raw_export/roundE_20261020/E1/ut-zap50k/V1/)：配置协议、完整历史、指标、曲线、预测、四份 checkpoint；第二套选择位于各 seed 的 `exact_selection/`。
- [任务状态](raw_export/roundE_20261020/round_e_status.json)、[三卡任务表](raw_export/roundE_20261020/round_e_jobs.json)、[日志](raw_export/roundE_20261020/launcher.log)、[退出码0](raw_export/roundE_20261020/launcher.exitcode)。三次训练及六项均值评估全部完成。
- [逐运行汇总](raw_export/roundE_20261020/round_e_summary.csv)、[均值及SD](raw_export/roundE_20261020/round_e_mean_sd.csv)、[同seed增量](raw_export/roundE_20261020/round_e_paired_deltas.csv)、[配对均值](raw_export/roundE_20261020/round_e_paired_mean_sd.csv)。
- [三档ρ主结果](raw_export/roundE_20261020/rho_native_summary.csv)、[三档均值](raw_export/roundE_20261020/rho_native_mean_sd.csv)、[exact选择汇总](raw_export/roundE_20261020/exact_selection_summary.csv)、[选择差值](raw_export/roundE_20261020/exact_selection_effects.csv)。

每个 seed 保存 `best.pt`、`best_native.pt`、`best_exact.pt`、`last.pt`；前两份为逐字节相同的兼容入口。本轮 native／exact 模型内容相同，文件序列化与选择身份分别保留。以上为 12 份文件，新增训练次数仍为三次。

## 动态与TRAIN均值替换

| seed | 选中epoch | 动态−均值 TEST CW AUC | 动态−均值 TEST OW AUC | 未见条件属性净正确数 |
|---|---|---|---|---|
| 3351 | 8 | +0.1609 | +0.0588 | +2 |
| 3352 | 8 | +0.2684 | +0.0624 | −2 |
| 3353 | 6 | −0.0755 | −0.0926 | +4 |

[native选择评估](raw_export/roundE_20261020/mean_gate/)与[exact选择评估](raw_export/roundE_20261020/mean_gate_exact/)分别保存 TRAIN 均值、总体图像方差，以及 dynamic／train_mean 两套指标、预测和曲线。TRAIN 共 22,998 张，均值形状 `[12,32]`，各条件重新选择自身 VAL bias。本轮两套选择给出的均值替换差异相同。

## 原始交付与仓库索引

- [原服务器TAR](archives/vgcpod_roundE_results_20261020.tar.gz)、[原E训练ZIP](archives/vgcpod_v1_roundE_20261020.zip)，旁附 SHA256，二进制沿用仓库 Git LFS 规则。
- [原发布验证](package_validation/E/)、[原说明字节](package_support/E/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包66成员映射](../package_members.json)、[新增结果成员与哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[复用旧阶段成员映射](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原 TAR 同时保留下载时的 C／D 参考结果及源码；参考成员均已与 [04](../../04/results/README.md)、[05](../../05/results/README.md)核对并建立来源映射。A5／A6 三 seed 引用 [02](../../02/results/README.md)。
- [仓库复算及README核查](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

本轮全部实际实验均为 E1、β=0.1；分析中的 F1、β=0.05 为后续建议。
