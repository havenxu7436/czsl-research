# 完整分析复算

确认结果位于 [confirmed](confirmed/)，原始脚本位于 [original_scripts](../references/original_scripts/)。脚本复算归档预测、配对干预、实际风险及效用拟合；本命令使用 CPU，依赖 Python、NumPy、PyTorch。

```bash
git lfs pull --include="TRUST-CPF方案/**"
python tools/analyze_trustcpf_followup.py --stage 05 --dest "$HOME/analysis/trust-05-recompute"
```

恢复工具创建新的外部目录，按原路径布局恢复只读输入和原始归档，运行冻结的分析脚本；重新生成的结果与确认版本按数值容差核对。分析过程不启动训练。
