# 完整实验结果与原始归档

- work 全部内容（精简版未保留）：权重、效用模型、响应、干预测量、辅助图、逐图像概率与全部表格。
- 正式结果（精简版未保留）、正式配对（精简版未保留）、Q4（精简版未保留）、Q5（精简版未保留）、方向曲线（精简版未保留）。
- 完成状态（精简版未保留）、训练曲线（精简版未保留）、日志（精简版未保留）。
- 服务器原始成员清单（精简版未保留）：10,027 个文件、8,358,832,790 字节，逐文件核对原 TAR.GZ。
- 原包分片（精简版未保留）：6,809,726,045 字节，26 个不超过 256 MiB 的 历史 Git LFS 分片（精简版未保留）。
- 发包 ZIP（精简版未保留） 与 SHA256（精简版未保留）。

原始 Markdown 以 .md.txt 保存原字节，阅读版位于阶段根目录；恢复工具还原原文件名。原始 RELEASE.json 记录 2026-10-26 发包时的状态，2026-10-27 实际完成状态见 work/status.json。

```bash
python tools/register_trustcpf_abc.py --restore-archive "$HOME/archives/trust-10"
python tools/register_trustcpf_abc.py --restore-results "$HOME/experiments/trust-10-saved"
```

第一条重建并核验同一 TAR.GZ 字节流；第二条恢复原目录结构，直接用于查看保存结果。
