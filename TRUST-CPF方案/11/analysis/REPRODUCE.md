# CPU完整复算

```bash
git lfs pull --include="TRUST-CPF方案/11/**"
python tools/analyze_trustg0_three.py --dest "$HOME/analysis/trust-g0-11"
```

依赖Python、NumPy、PyTorch。从仓库恢复实际服务器结果，重算1,620行正式指标和378行Q5指标、效用标签、岭系数和策略预算，检查冻结权重、完整10轮训练及teacher划分；不启动训练。

[确认值](confirmed/)保存三数据集完整复算及联合诊断；[原始脚本](../references/original_analysis_scripts/)保存最初本机字节。主表与配对均值和确认值核对；时间和临时绝对路径不作科学指标。
