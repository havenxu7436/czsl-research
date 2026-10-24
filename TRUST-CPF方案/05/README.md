# 05 · Align 半密度对齐

实验完成及归档日期：2026-10-24。UT-Zappos natural split；冻结 CLIP ViT-L/14 特征和强 CPF；seed 3351 / 3352 / 3353。

18/18 阶段、9 份识别结果、3 份 Q5 审计；5 次实际适配、4 份同 seed Uniform 复用。三方法适配 3 epoch，216 行识别、72 行扫描、312 行 Q5 图 bank 风险。

[实验方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置与实验清单](configs/README.md) · [完整结果](results/README.md) · [完整分析](analysis/README.md) · [复算](analysis/REPRODUCE.md) · [服务器命令](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.4811 ± 1.1982 | 26.1445 ± 1.7200 |
| align | 35.3934 ± 1.0672 | 26.1525 ± 1.7185 |
| permuted | 35.4295 ± 1.1204 | 26.0651 ± 1.7411 |

主图 bank 1701，按 VAL CW native AUC 从 epoch 0–3 选同一权重评价 TEST CW/OW。数值为百分数尺度，均值 ± 三训练 seed 样本标准差。多个图 bank 用于描述图采样变化。

半密度对齐与 Q5 实际效用诊断已完成。Uniform 为 35.4811 / 26.1445；Align 为 35.3934 / 26.1525。非均匀策略只在 seed3352 启用。

## 恢复执行包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/05/**"
python tools/register_trustcpf_followup.py --stage 05 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 05 --dest "$HOME/experiments/trust-05-replay"
cd "$HOME/experiments/trust-05-replay/trustcpf_align_v1"
python -m unittest discover -s tests -v
```

[文件清单](file_manifest.json) · [发布包成员映射](package_members.json) · [服务器原始成员映射](results/raw_export_members.json) · [输入与完成状态](inputs.json) · [执行命令清单](runbook.json) · [来源与许可](来源与许可.md)
