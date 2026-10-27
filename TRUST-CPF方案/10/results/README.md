# 完整实验结果与原始归档

- [work 全部内容](raw_export/trustcpf_abc_v1_10ep/work/)：权重、效用模型、响应、干预测量、辅助图、逐图像概率与全部表格。
- [正式结果](raw_export/trustcpf_abc_v1_10ep/work/summary.csv)、[正式配对](raw_export/trustcpf_abc_v1_10ep/work/paired.csv)、[Q4](raw_export/trustcpf_abc_v1_10ep/work/q4_curves.csv)、[Q5](raw_export/trustcpf_abc_v1_10ep/work/q5_curves.csv)、[方向曲线](raw_export/trustcpf_abc_v1_10ep/work/directional_curves.csv)。
- [完成状态](raw_export/trustcpf_abc_v1_10ep/work/status.json)、[训练曲线](raw_export/trustcpf_abc_v1_10ep/work/training_curves.csv)、[日志](raw_export/trustcpf_abc_v1_10ep/logs/)。
- [服务器原始成员清单](raw_export_members.json)：10,027 个文件、8,358,832,790 字节，逐文件核对原 TAR.GZ。
- [原包分片](archives/archive.json)：6,809,726,045 字节，26 个不超过 256 MiB 的 Git LFS 分片。
- [发包 ZIP](deployment/trustcpf_abc_v1_10ep.zip) 与 [SHA256](deployment/trustcpf_abc_v1_10ep.zip.sha256)。

原始 Markdown 以 .md.txt 保存原字节，阅读版位于阶段根目录；恢复工具还原原文件名。原始 RELEASE.json 记录 2026-10-26 发包时的状态，2026-10-27 实际完成状态见 work/status.json。

```bash
python tools/register_trustcpf_abc.py --restore-archive "$HOME/archives/trust-10"
python tools/register_trustcpf_abc.py --restore-results "$HOME/experiments/trust-10-saved"
```

第一条重建并核验同一 TAR.GZ 字节流；第二条恢复原目录结构，直接用于查看保存结果。
