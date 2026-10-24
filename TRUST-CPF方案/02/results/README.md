# 强基底完整矩阵：完整结果

114/114 个阶段、30 份模型或读出结果、12 份效用审计、1,800 条噪声评估。30 份结果由三 seed 的 r0、六类适配、shuffled 以及两种读出组成；读出与 r0 不计为新适配训练。

- [原始 work 目录](raw_export/trustcpf_strong_v1/work/)：权重、预测、测量、效用审计、结构、逐 epoch 历史及完成状态。
- [服务器日志](raw_export/trustcpf_strong_v1/logs/) 与 [执行快照](raw_export/trustcpf_strong_v1/)。
- [原始部署 ZIP](deployment/trustcpf_strong_v1.zip) 及 [SHA256](deployment/trustcpf_strong_v1.zip.sha256)。
- [原始服务器归档及分片索引](archives/archive.json)：`trustcpf_strong_v1_results_20261023_162717.tar.gz`，4,908,964,816 字节。
- [全部 985 个原始成员](raw_export_members.json)：每个文件的原路径、仓库路径、大小与 SHA256。

二进制资产由 Git LFS 保存。原始 Markdown 以 `.md.txt` 保留字节，阅读版位于本阶段根目录；恢复工具自动还原原文件名。

## 还原服务器原包

在仓库根目录执行，目标目录须为新的绝对路径：

```bash
python tools/register_trustcpf_strong_series.py --stage 02 --restore-archive "$HOME/archives/trust-02"
```

工具按索引顺序合并，并验证每片与整个归档的 SHA256。无需重新打包实验结果。
