# R5 完整实验结果

27/27 阶段、15 份结果标签；1 次新适配，6 份前轮匹配结果复用，8 份同轮 Uniform 复用。原始成员共 1,005 个，逐文件核对原始 TAR.GZ 与本地提取结果。

- 原始 work（精简版未保留）：权重、预测、效用、测量、扫描、曲线与 完成状态（精简版未保留）。
- 日志（精简版未保留）、原始执行快照（精简版未保留） 与 成员清单（精简版未保留）。
- 部署 ZIP（精简版未保留） 与 SHA256（精简版未保留）。

原始 Markdown 使用 .md.txt 名称保存原字节，阅读版位于阶段根目录，恢复工具自动还原原名。

```bash
python tools/register_trustcpf_followup.py --stage 07 --restore-archive "$HOME/archives/trust-07-saved"
```
