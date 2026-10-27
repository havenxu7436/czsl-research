# 09 · Stage 第二批：教师阶段匹配与响应平均

完成与归档日期：2026-10-25。UT-Zappos natural split，CLIP ViT-L/14 特征与 CPF 强基底；seed 3351 / 3352 / 3353，实际物理 GPU 1 / 2 / 3。

26/26 阶段，15 个教师各追加 10 epoch，正式权重复用第一批 Uniform-best；81 个评分策略实例、972 行识别、744 行 Q5、486 份预测、165 行教师曲线。

[方案](PLAN.md) · [公式与核心代码](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置](configs/README.md) · [完整结果](results/README.md) · [完整分析](analysis/README.md) · [分析复算](analysis/REPRODUCE.md) · [上传、运行和下载](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| uniform | 35.9343 ± 0.7712 | 26.5317 ± 1.3662 |
| old_single | 35.8118 ± 0.9746 | 26.5034 ± 1.2842 |
| matched_single | 35.9645 ± 0.8630 | 26.5362 ± 1.2052 |
| old_average | 35.9443 ± 0.8797 | 26.5268 ± 1.2462 |
| matched_average | 35.9471 ± 0.8600 | 26.4965 ± 1.2364 |

表格固定主 bank1701、Uniform-best，均值 ± 三 seed 样本标准差；AUC 为 0–100 尺度。三 bank 结果及五次置换比较见完整分析，多个 bank 不计作额外训练重复。

教师阶段匹配把旧预测器的正式识别下降基本修复到 Uniform 水平。三 bank 平均的 Matched-Average 为 35.936 CW / 26.483 OW，微小平均正值尚未形成稳定优势。

## 还原发布包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/09/**"
python tools/register_trustcpf_state_stage.py --stage 09 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 09 --dest "$HOME/experiments/trust-09-replay"
cd "$HOME/experiments/trust-09-replay/trustcpf_stage_v1_10ep"
python -m unittest discover -s tests -v
```

[执行与评分清单](configs/experiment_inventory.json) · [状态](inputs.json) · [文件 SHA256](file_manifest.json) · [发布包成员](package_members.json) · [服务器成员](results/raw_export_members.json) · [来源与许可](来源与许可.md)

后续 A/B/C 幅度、混合与情境机制实验已完成，完整方案、实现与结果见 [10 阶段](../10/README.md)。
