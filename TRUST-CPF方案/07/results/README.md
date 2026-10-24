# R5 完整实验结果

27/27 阶段、15 份结果标签；1 次新适配，6 份前轮匹配结果复用，8 份同轮 Uniform 复用。原始成员共 1,005 个，逐文件核对原始 TAR.GZ 与本地提取结果。

- [原始 work](raw_export/trustcpf_r5_v1_10ep/work/)：权重、预测、效用、测量、扫描、曲线与 [完成状态](raw_export/trustcpf_r5_v1_10ep/work/status.json)。
- [日志](raw_export/trustcpf_r5_v1_10ep/logs/)、[原始执行快照](raw_export/trustcpf_r5_v1_10ep/) 与 [成员清单](raw_export_members.json)。
- [部署 ZIP](deployment/trustcpf_r5_v1_10ep.zip) 与 [SHA256](deployment/trustcpf_r5_v1_10ep.zip.sha256)。
- [归档分片索引](archives/archive.json)：1,258,749,962 字节；两个不超过 1 GiB 的 Git LFS 分片，拼接可还原原始归档。

原始 Markdown 使用 .md.txt 名称保存原字节，阅读版位于阶段根目录，恢复工具自动还原原名。

```bash
python tools/register_trustcpf_followup.py --stage 07 --restore-archive "$HOME/archives/trust-07-saved"
```
