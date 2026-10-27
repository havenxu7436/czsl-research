# 08 · State 第一批：状态与联合效用拟合

完成与归档日期：2026-10-25。UT-Zappos natural split，CLIP ViT-L/14 特征与 CPF 强基底；seed 3351 / 3352 / 3353，实际物理 GPU 1 / 2 / 3。

17/17 阶段，3 条重新执行的 10 epoch Uniform 轨迹，126 个共享权重的评分策略实例；1512 行识别、600 行 Q5、756 份预测、33 行曲线。

[方案](PLAN.md) · [公式与核心代码](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置](configs/README.md) · [完整结果](results/README.md) · [完整分析](analysis/README.md) · [分析复算](analysis/REPRODUCE.md) · [上传、运行和下载](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| independent_absolute | 35.8380 ± 0.7773 | 26.4598 ± 1.2292 |
| pooled_absolute | 35.8118 ± 0.9746 | 26.5034 ± 1.2842 |
| pooled_relative | 35.8218 ± 1.0425 | 26.5162 ± 1.3039 |

表格固定主 bank1701、Uniform-best，均值 ± 三 seed 样本标准差；AUC 为 0–100 尺度。三 bank 结果及五次置换比较见完整分析，多个 bank 不计作额外训练重复。

联合拟合提高了 Q5 实际关系决策收益，正式 TEST 尚未超过 Uniform。三条 Uniform 重训确定性重现了旧最佳权重；评分策略共享这些权重。

## 还原发布包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/08/**"
python tools/register_trustcpf_state_stage.py --stage 08 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 08 --dest "$HOME/experiments/trust-08-replay"
cd "$HOME/experiments/trust-08-replay/trustcpf_state_v1_10ep"
python -m unittest discover -s tests -v
```

[执行与评分清单](configs/experiment_inventory.json) · [状态](inputs.json) · [文件 SHA256](file_manifest.json) · [发布包成员](package_members.json) · [服务器成员](results/raw_export_members.json) · [来源与许可](来源与许可.md)

后续 A/B/C 幅度、混合与情境机制实验已完成，完整方案、实现与结果见 [10 阶段](../10/README.md)。
