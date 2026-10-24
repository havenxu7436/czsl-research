# 04 · Compact 紧凑效用监督

实验完成于 2026-10-23，仓库归档于 2026-10-24。UT-Zappos natural split；冻结 CLIP ViT-L/14 和已有强 CPF / G0，seed 3351 / 3352 / 3353。

21/21 个阶段、9 份识别结果、6 份效用审计、18 份复用响应与6份策略结构。识别结果中 3 份 Drop 复用 02 阶段，新增 Compact / shuffled_utility 各三次适配。

[方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置与实验清单](configs/README.md) · [完整结果](results/README.md) · [完整分析及下一步](analysis/README.md) · [服务器与本地恢复命令](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC | 结果性质 |
| --- | --- | --- | --- |
| compact | 34.9218 ± 0.7461 | 25.9313 ± 1.3800 | 本轮结果 |
| drop | 34.9881 ± 0.8640 | 25.9079 ± 1.1168 | 复用 02 阶段对照 |
| shuffled_utility | 34.7961 ± 0.6771 | 25.6643 ± 0.8965 | 本轮结果 |

数值为原工程百分数尺度的 native AUC，均值 ± 三 seed 样本标准差。选模使用 VAL CW AUC，同一权重评价 CW/OW；HM、NLL、ECE 保留 VAL 确定的校准参数。

Compact 的真实效用监督提高了 Q5 关系排序，相比打乱监督识别表现更好；相对 Drop 的 CW 均值仍近似持平。下一轮建议校准类型内效用幅度并匹配半密度训练分布，具体设计见分析末尾。

## 恢复执行包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/04/**"
python tools/register_trustcpf_strong_series.py --stage 04 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 04 --dest "$HOME/experiments/trust-04-replay"
cd "$HOME/experiments/trust-04-replay/trustcpf_compact_v1"
python -m unittest discover -s tests -v
```

[文件清单](file_manifest.json) · [原包成员映射](package_members.json) · [原始服务器成员映射](results/raw_export_members.json) · [来源与许可](来源与许可.md) · [外部输入](external_inputs.json)
