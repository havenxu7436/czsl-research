# 分析重算

完整结果来自已保存的 VAL/TEST 预测、权重、配对效用测量及审计。已确认的 JSON 在 [confirmed](confirmed/)，原始分析代码在 [original_scripts](../references/original_scripts/)。

在仓库根目录执行（Python 3.11、PyTorch 2.11、NumPy；CPU）：

```bash
git lfs pull --include="TRUST-CPF方案/**"
python tools/analyze_trustcpf_strong_series.py --stage 02 --dest "$HOME/analysis/trust-02-recompute"
```

脚本恢复原分析所需的目录、前序结果和原始归档，再运行原脚本。输出保存在新目录，仓库 confirmed 文件保持原样。
