# 模型响应条件化：完整结果

21/21 个阶段、9 份识别结果、6 份效用审计、18 份模型响应。识别结果中 6 份 Drop / original 对照复用 02 阶段，新增 3 次 conditioned 适配。

- [原始 work 目录](raw_export/trustcpf_modelcond_v1/work/)：权重、预测、测量、效用审计、结构、逐 epoch 历史及完成状态。
- [服务器日志](raw_export/trustcpf_modelcond_v1/logs/) 与 [执行快照](raw_export/trustcpf_modelcond_v1/)。
- [原始部署 ZIP](deployment/trustcpf_modelcond_v1.zip) 及 [SHA256](deployment/trustcpf_modelcond_v1.zip.sha256)。
- [原始服务器归档及分片索引](archives/archive.json)：`trustcpf_modelcond_v1_results_20261023_205829.tar.gz`，292,568,914 字节。
- [全部 325 个原始成员](raw_export_members.json)：每个文件的原路径、仓库路径、大小与 SHA256。

二进制资产由 Git LFS 保存。原始 Markdown 以 `.md.txt` 保留字节，阅读版位于本阶段根目录；恢复工具自动还原原文件名。

## 还原服务器原包

在仓库根目录执行，目标目录须为新的绝对路径：

```bash
python tools/register_trustcpf_strong_series.py --stage 03 --restore-archive "$HOME/archives/trust-03"
```

工具按索引顺序合并，并验证每片与整个归档的 SHA256。无需重新打包实验结果。
