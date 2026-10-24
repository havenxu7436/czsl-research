# 02 · 强基底完整矩阵

实验完成于 2026-10-23，仓库归档于 2026-10-24。UT-Zappos natural split；冻结 CLIP ViT-L/14 和已有强 CPF / G0，seed 3351 / 3352 / 3353。

114/114 个阶段、30 份模型或读出结果、12 份效用审计、1,800 条噪声评估。30 份结果由三 seed 的 r0、六类适配、shuffled 以及两种读出组成；读出与 r0 不计为新适配训练。

[方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置与实验清单](configs/README.md) · [完整结果](results/README.md) · [完整分析及下一步](analysis/README.md) · [服务器与本地恢复命令](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC | 结果性质 |
| --- | --- | --- | --- |
| capacity | 35.8053 ± 0.9046 | 26.4004 ± 1.8470 | 本轮结果 |
| dense | 35.6455 ± 0.9262 | 26.2884 ± 1.8182 | 本轮结果 |
| drop | 34.9881 ± 0.8640 | 25.9079 ± 1.1168 | 本轮结果 |
| free | 34.9960 ± 0.8522 | 25.9213 ± 1.1036 | 本轮结果 |
| mean | 35.2306 ± 0.5996 | 26.0703 ± 1.0387 | 本轮结果 |
| r0 | 34.5117 ± 0.6007 | 25.1836 ± 1.2535 | 本轮结果 |
| readout_mean_graph | 35.3792 ± 0.6259 | 26.2110 ± 1.3869 | 同一权重的读出对照 |
| readout_mean_logit | 35.2062 ± 0.5636 | 26.1609 ± 1.1097 | 同一权重的读出对照 |
| robust | 35.2015 ± 0.5713 | 26.1491 ± 1.1088 | 本轮结果 |
| shuffled | 35.0331 ± 0.5245 | 25.5448 ± 1.3099 | 本轮结果 |

数值为原工程百分数尺度的 native AUC，均值 ± 三 seed 样本标准差。选模使用 VAL CW AUC，同一权重评价 CW/OW；HM、NLL、ECE 保留 VAL 确定的校准参数。

强 CPF 上 robust 达到 35.2015 / 26.1491，超过 r0；capacity 和 dense 的识别均值仍然更高。噪声、知识打乱、semantic / unmeasured 审计和读出对照均已归档。

## 恢复执行包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/02/**"
python tools/register_trustcpf_strong_series.py --stage 02 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 02 --dest "$HOME/experiments/trust-02-replay"
cd "$HOME/experiments/trust-02-replay/trustcpf_strong_v1"
python -m unittest discover -s tests -v
```

[文件清单](file_manifest.json) · [原包成员映射](package_members.json) · [原始服务器成员映射](results/raw_export_members.json) · [来源与许可](来源与许可.md) · [外部输入](external_inputs.json)
