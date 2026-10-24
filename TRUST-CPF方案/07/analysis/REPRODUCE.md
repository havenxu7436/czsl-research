# R5 完整分析复算

[confirmed](confirmed/) 保存确认数据，原始脚本在 [original_scripts](../references/original_scripts/)。复算包括 180 份预测、168 项 Q5 bank 决策、12 个效用模型、来源匹配、概率和 VAL 选模。

在仓库根目录执行（Python、NumPy、PyTorch；CPU）：

```bash
git lfs pull --include="TRUST-CPF方案/**"
python tools/analyze_trustcpf_followup.py --stage 07 --dest "$HOME/analysis/trust-07-recompute"
```

脚本按原目录恢复 07 与所需的 06 来源，运行原始冻结脚本，生成新分析并与确认版本按数值容差核对。此过程不启动训练。

## 本次归档校验

已从仓库还原执行包并核对全部 74 个清单成员；35 项测试中 34 项通过，1 项按原设计跳过 CLIP 重提取。四份分析数据复算与确认版一致，数值容差为 10⁻⁹。仓库整体核验覆盖 11,816 个原始文件，未发现字节变化、缺失文件、断链或 Python 语法错误。

记录见 [代码还原](repository_qa/materialization.json)、[测试日志](repository_qa/restored_tests.log)、[分析复算](repository_qa/analysis_reproduction.json)、[整体核验](repository_qa/repository_verify.json)、[本地公式检查](repository_qa/markdown_math_checks.json)、[GitHub 渲染检查](repository_qa/github/checks.json) 和 [远程 LFS 下载核对](repository_qa/remote_lfs_download.json)。
