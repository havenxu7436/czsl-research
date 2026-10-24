# R5 完整分析复算

[confirmed](confirmed/) 保存确认数据，原始脚本在 [original_scripts](../references/original_scripts/)。复算包括 180 份预测、168 项 Q5 bank 决策、12 个效用模型、来源匹配、概率和 VAL 选模。

在仓库根目录执行（Python、NumPy、PyTorch；CPU）：

```bash
git lfs pull --include="TRUST-CPF方案/**"
python tools/analyze_trustcpf_followup.py --stage 07 --dest "$HOME/analysis/trust-07-recompute"
```

脚本按原目录恢复 07 与所需的 06 来源，运行原始冻结脚本，生成新分析并与确认版本按数值容差核对。此过程不启动训练。
