# 完整实验结果

42/42 阶段、15 份识别结果、3 份 Q5 审计；6 次实际适配、9 份同 seed Uniform 复用。15 个新教师；五方法适配 10 epoch，360 行识别、96 行扫描、312 行 Q5 图 bank 风险、165 行训练曲线。

- [原始 work](raw_export/trustcpf_competition_v1_10ep/work/)：权重、epoch0／最佳预测、图结构、效用测量、审计和完整训练历史。
- [服务器日志](raw_export/trustcpf_competition_v1_10ep/logs/) 与 [完成状态](raw_export/trustcpf_competition_v1_10ep/work/status.json)。
- [全部原始成员](raw_export_members.json)：1197 个文件，保留路径、大小及 SHA256。
- [部署 ZIP](deployment/trustcpf_competition_v1_10ep.zip) 与 [SHA256](deployment/trustcpf_competition_v1_10ep.zip.sha256)。
- [原始归档索引](archives/archive.json)：1,552,879,972 字节，两个不超过 1 GiB 的 LFS 分片。

二进制通过 Git LFS 保存。服务器原始 Markdown 改用 .md.txt 文件名保留字节，阅读版在阶段根目录；恢复工具还原原文件名。

```bash
python tools/register_trustcpf_followup.py --stage 06 --restore-archive "$HOME/archives/trust-06-saved"
```

工具按索引顺序合并并核验归档 SHA256，恢复的是同一原始字节流。
