# L组完整结果

9/9组完成；90个模型epoch，18份最终VAL/TEST预测、18个best/last权重和3个G0初始化权重。VAL 3,214图，TEST 2,914图；预测与逐图像记录的样本顺序一致。

| 内容 | 入口 |
| --- | --- |
| 九组summary、聚合与配对结果 | [work/](raw_export/trustcpf_conditional_v2/work/) |
| 3351 / 3352 / 3353 | [seed3351](raw_export/trustcpf_conditional_v2/work/seed3351/) · [seed3352](raw_export/trustcpf_conditional_v2/work/seed3352/) · [seed3353](raw_export/trustcpf_conditional_v2/work/seed3353/) |
| 全部训练日志 | [logs/](raw_export/trustcpf_conditional_v2/logs/) |
| 环境与缓存设置 | [setup.json](raw_export/trustcpf_conditional_v2/setup.json) |
| 原始服务器压缩包与原发布ZIP | [archives/](archives/) |
| 原始服务器181成员哈希 | [raw_export_members.json](raw_export_members.json) |
| 重命名后的部署ZIP | [vgcpod_v1_roundL.zip](deployment/vgcpod_v1_roundL.zip) · [SHA256](deployment/vgcpod_v1_roundL.zip.sha256) |
| 原发布40成员映射 | [original_release_member_map.json](original_release_member_map.json) |
| 当前40成员执行包映射 | [package_members.json](../package_members.json) |
| 代码逐文件身份与差异 | [server_source_identity.json](server_source_identity.json) |
| 36行统一指标 | [metrics_verified.csv](../analysis/confirmed/metrics_verified.csv) |
| 完整数据分析 | [analysis/](../analysis/README.md) |

原始结果包保留历史运行目录名，仅作为数据身份。全部权重、预测与压缩包通过Git LFS保存；克隆后执行`git lfs pull --include="VG-CPOD方案/13/**"`。
