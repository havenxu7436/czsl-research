# 完整分析复算

CPU 复算使用 Python、NumPy、PyTorch 与 pandas。从仓库原始文件重建归档目录，运行保存的预测、效用与表格脚本，并与 [confirmed](confirmed/) 的确认数据逐项比较。

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/09/**,TRUST-CPF方案/08/results/raw_export/**"
python tools/analyze_trustcpf_state_stage.py --stage 09 --dest "$HOME/analysis/trust-09-recompute"
```

比较覆盖逐图像指标、岭回归、关系概率、Q5 决策与聚合表，容差为 10⁻⁹；运行耗时不作为数值比较字段。原脚本位于 [original_scripts](../references/original_scripts/)。本命令使用已保存预测与统计量，不启动新的训练。

## 本次入库验证

发布包还原核对 93 个清单文件；服务器快照的原始清单为 89 个文件，两套来源均保存。CPU 套件 42 项中 40 项通过、2 项按设计跳过；CUDA 检查单独重跑 2/2 通过。首次全套运行出现一次 CUDA 续训逐位比较差异，独立重跑未复现，原因尚未定位。原始代码保持原字节，首次日志与重跑日志同时保留。六份分析数据及 15 个选中教师权重与确认版本一致。

记录见 [还原清单](repository_qa/materialization.json)、[分析复算](repository_qa/analysis_reproduction.json)、[命令语法](repository_qa/command_checks.json)、[验证汇总](repository_qa/validation_summary.json) 和 [首次完整测试日志](repository_qa/restored_tests_initial.log)、[CPU 测试日志](repository_qa/restored_tests_cpu.log)、[CUDA 独立重跑](repository_qa/cuda_recheck.log)。

[仓库文件与历史引用校验](repository_qa/repository_checks.json) 覆盖本次 5,836 个原始服务器成员及仓库索引；历史大文件按 Git LFS 指针的 SHA256 与大小核对，本次新增文件核对实际字节。

GitHub 实际网页核对通过：27 个页面、69 个独立公式、91 个行内公式、51 张表格；发布包与结果图共 11 个文件从远端重新下载后，SHA256 与本地归档一致。见 [网页核对记录](repository_qa/github/checks.json) 与 [远端下载核对](repository_qa/remote_lfs_download.json)。
