# 完整分析复算

[确认结果](confirmed/) 保存正式汇总、逐 seed 数值、数值审计、AUC 条件贡献及排版记录；[原始脚本](../references/original_analysis_scripts/) 保留原始字节。下列入口从仓库恢复结果与原归档，在 CPU 复算所有保存的概率、标签、岭系数和配对表，不启动训练。

```bash
git lfs pull --include="TRUST-CPF方案/10/**"
python tools/analyze_trustcpf_abc.py --dest "$HOME/analysis/trust-10-recompute"
```

依赖 Python、NumPy、pandas、PyTorch。Stage 09 比较只读取其普通文本 summary.csv，不要求再次训练教师。工具创建新的外部目录，运行 analyze.py、audit.py、supplement.py；与确认 JSON 按绝对/相对容差 1e-9 比较，忽略运行耗时及机器绝对路径。

复算包括 936 行正式识别、7920 行 Q4/Q5/方向结果、所有正式及查询配对、144 个干预块的混合与条件标签、27/51 参数拟合、冻结主干与 best 权重一致性、AUC 四情境贡献。

本次已从归档恢复并完整执行三个复算脚本，与确认值一致。见 [复算结果](repository_qa/analysis_reproduction.json)、[三轮文件核对](repository_qa/three_round_file_checks.json) 和 [公式、表格及链接核对](repository_qa/markdown_math_checks.json)。
