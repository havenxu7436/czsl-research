# 完整分析复算

CPU 复算使用 Python、NumPy、PyTorch 与 pandas。从仓库原始文件重建归档目录，运行保存的预测、效用与表格脚本，并与 [confirmed](confirmed/) 的确认数据逐项比较。

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/08/**,TRUST-CPF方案/07/results/raw_export/**"
python tools/analyze_trustcpf_state_stage.py --stage 08 --dest "$HOME/analysis/trust-08-recompute"
```

比较覆盖逐图像指标、岭回归、关系概率、Q5 决策与聚合表，容差为 10⁻⁹；运行耗时不作为数值比较字段。原脚本位于 [original_scripts](../references/original_scripts/)。本命令使用已保存预测与统计量，不启动新的训练。

## 本次入库验证

发布包还原核对 78 个清单文件。38 项测试中 37 项通过，原生 CLIP 提取按原设计跳过；六份分析数据与确认版本一致。

记录见 [还原清单](repository_qa/materialization.json)、[分析复算](repository_qa/analysis_reproduction.json)、[命令语法](repository_qa/command_checks.json)、[验证汇总](repository_qa/validation_summary.json) 和 [测试日志](repository_qa/restored_tests.log)。

[仓库文件与历史引用校验](repository_qa/repository_checks.json) 覆盖本次 5,836 个原始服务器成员及仓库索引；历史大文件按 Git LFS 指针的 SHA256 与大小核对，本次新增文件核对实际字节。
