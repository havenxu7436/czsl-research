# 03 · 模型响应条件化

实验完成于 2026-10-23，仓库归档于 2026-10-24。UT-Zappos natural split；冻结 CLIP ViT-L/14 和已有强 CPF / G0，seed 3351 / 3352 / 3353。

21/21 个阶段、9 份识别结果、6 份效用审计、18 份模型响应。识别结果中 6 份 Drop / original 对照复用 02 阶段，新增 3 次 conditioned 适配。

[方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置与实验清单](configs/README.md) · [完整结果](results/README.md) · [完整分析及下一步](analysis/README.md) · [服务器与本地恢复命令](SERVER_COMMANDS.md)

| 方法 | TEST CW AUC | TEST OW AUC | 结果性质 |
| --- | --- | --- | --- |
| conditioned | 34.9164 ± 0.9616 | 25.9581 ± 0.9939 | 本轮结果 |
| drop | 34.9881 ± 0.8640 | 25.9079 ± 1.1168 | 复用 02 阶段对照 |
| original | 35.2015 ± 0.5713 | 26.1491 ± 1.1088 | 复用 02 阶段对照 |

数值为原工程百分数尺度的 native AUC，均值 ± 三 seed 样本标准差。选模使用 VAL CW AUC，同一权重评价 CW/OW；HM、NLL、ECE 保留 VAL 确定的校准参数。

模型响应使效用监督具有当前模型信息，但 conditioned 的识别 AUC 没有超过原 TRUST。后续据此将效用预测收缩为低维、带正则的 Compact 模型。

## 恢复执行包

在仓库根目录执行：

```bash
git lfs pull --include="TRUST-CPF方案/03/**"
python tools/register_trustcpf_strong_series.py --stage 03 --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 03 --dest "$HOME/experiments/trust-03-replay"
cd "$HOME/experiments/trust-03-replay/trustcpf_modelcond_v1"
python -m unittest discover -s tests -v
```

[文件清单](file_manifest.json) · [原包成员映射](package_members.json) · [原始服务器成员映射](results/raw_export_members.json) · [来源与许可](来源与许可.md) · [外部输入](external_inputs.json)
