# 06 · Competition 三目标与 10 epoch

实验完成及归档日期：2026-10-24。UT-Zappos natural split；冻结 CLIP ViT-L/14 特征和强 CPF；seed 3351 / 3352 / 3353。

42/42 阶段、15 份识别结果、3 份 Q5 审计；6 次实际适配、9 份同 seed Uniform 复用。15 个新教师；五方法适配 10 epoch，360 行识别、96 行扫描、312 行 Q5 图 bank 风险、165 行训练曲线。

[实验方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置与实验清单](configs/README.md) · [完整结果](results/README.md) · [完整分析](analysis/README.md) · [复算](analysis/REPRODUCE.md) · [服务器命令](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| unll | 35.7292 ± 0.4261 | 26.3156 ± 1.5334 |
| balanced | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| competition | 35.9500 ± 0.7624 | 26.4935 ± 1.4100 |
| permuted | 35.8408 ± 0.8400 | 26.5057 ± 1.3958 |

主图 bank 1701，按 VAL CW native AUC 从 epoch 0–10 选同一权重评价 TEST CW/OW。数值为百分数尺度，均值 ± 三训练 seed 样本标准差。多个图 bank 用于描述图采样变化。

Uniform 为 35.9343 / 26.5317；Competition 为 35.9500 / 26.4935。测量效用在独立 Q5 图像上提高实际 AUC；当前预测及向最终 reference 迁移仍限制终端收益。

## 恢复执行包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/06/**"
python tools/register_trustcpf_followup.py --stage 06 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 06 --dest "$HOME/experiments/trust-06-replay"
cd "$HOME/experiments/trust-06-replay/trustcpf_competition_v1_10ep"
python -m unittest discover -s tests -v
```

[文件清单](file_manifest.json) · [发布包成员映射](package_members.json) · [服务器原始成员映射](results/raw_export_members.json) · [输入与完成状态](inputs.json) · [执行命令清单](runbook.json) · [来源与许可](来源与许可.md)
