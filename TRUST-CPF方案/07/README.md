# 07 · R5 相对效用拟合 / 10 epoch

实验完成及归档日期：2026-10-24。UT-Zappos natural split；CLIP ViT-L/14 缓存与冻结强 CPF；seed3351 / 3352 / 3353，实际 GPU 0 / 1 / 2。

27/27 阶段、15 份识别结果、15 个复用教师、3 份 Q5 审计；360 行识别、144 行冻结扫描、168 行 Q5 图 bank 风险和 165 行训练曲线。

15 个结果标签对应 5 条不同适配轨迹：4 条来自前轮，1 条本轮新增。Uniform 三 seed 和 Absolute 三 seed 复用 06 阶段匹配结果，另 8 份零强度结果复用本轮 Uniform；本轮新训练为 seed3352 的 Permuted-Absolute。Q5 新增 48 项 Relative/Permuted-Relative bank 结果，复用 120 项已核对来源的审计。

[实验方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置](configs/README.md) · [完整结果](results/README.md) · [完整分析](analysis/README.md) · [分析复算](analysis/REPRODUCE.md) · [服务器与本地命令](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| absolute | 35.7292 ± 0.4261 | 26.3156 ± 1.5334 |
| relative | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| permuted_absolute | 35.8086 ± 0.5583 | 26.4218 ± 1.4415 |
| permuted_relative | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |

主图 bank1701，按 VAL CW native AUC 在 epoch0–10 选同一权重评价 TEST CW/OW；数值为 0–100 尺度，均值 ± 三训练 seed 样本标准差。

Relative 三 seed 均选 κ=0，正式结果等于 Uniform；固定非零强度的 Q5 收益也低于 Absolute。本轮未支持 R5 优于绝对效用拟合，独立图像上的局部效用机制信号仍保留。具体证据与后续提案见完整分析。

## 恢复执行包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/07/**"
python tools/register_trustcpf_followup.py --stage 07 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 07 --dest "$HOME/experiments/trust-07-replay"
cd "$HOME/experiments/trust-07-replay/trustcpf_r5_v1_10ep"
python -m unittest discover -s tests -v
```

[文件清单](file_manifest.json) · [发布包成员映射](package_members.json) · [服务器原始成员](results/raw_export_members.json) · [输入与完成状态](inputs.json) · [逐组来源](configs/experiment_inventory.json) · [命令清单](runbook.json) · [来源与许可](来源与许可.md)
