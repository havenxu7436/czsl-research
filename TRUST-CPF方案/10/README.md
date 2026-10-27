# 10 · A/B/C：幅度、混合与情境效用 / 10 epoch

实验完成及归档日期：2026-10-27。UT-Zappos natural split；冻结 CLIP ViT-L/14 特征与强 CPF；seed 3351 / 3352 / 3353，物理 GPU 1 / 2 / 3。

36/36 阶段；三条新 Uniform 轨迹各训练完整 10 epoch，best epoch 为 5 / 2 / 2。复用 Stage 的 15 个教师，由其中九个 Q1–Q3 教师拟合一套共享效用预测器；所有策略共用各 seed 的同一 best 分类器。

936 行识别、3360 行 Q4、3504 行 Q5、1056 行测量方向曲线、33 行训练曲线；26 个正式策略 × 3 seed × 3 bank，共 234 份正式预测结果、468 个 VAL/TEST 概率文件。

[v8 方案](PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [代码](code/README.md) · [配置](configs/README.md) · [完整结果](results/README.md) · [完整分析](analysis/README.md) · [复算](analysis/REPRODUCE.md) · [恢复与服务器命令](SERVER_COMMANDS.md)

| 方法 | TEST CW native AUC | TEST OW native AUC |
| --- | --- | --- |
| uniform | 36.0506 ± 0.7217 | 26.6377 ± 1.5274 |
| matched | 36.1478 ± 0.8664 | 26.7437 ± 1.7058 |
| amplitude | 36.0475 ± 0.7103 | 26.6260 ± 1.5108 |
| mix16 | 36.1193 ± 0.8915 | 26.7136 ± 1.7347 |
| context | 36.1472 ± 0.7079 | 26.7471 ± 1.5190 |
| context_shuffle | 35.9785 ± 0.7695 | 26.5752 ± 1.5447 |

先在每个 seed 内平均三图库，再汇总三 seed；± 为样本标准差。AUC 为 0–100，配对差为百分点。三个识别 seed 共享同一预测器。

C 在三个 seed 的 CW/OW 上均超过 Uniform、关系置换均值和情境打乱；平均增量约为 +0.0965 / +0.1093，但均值与 Matched 接近。A 未改善；B 未超过 Matched。完整 Q4/Q5 与下一步建议见分析。

## 恢复同一执行包

```bash
git lfs pull --include="TRUST-CPF方案/10/**"
python tools/register_trustcpf_abc.py --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 10 --dest "$HOME/experiments/trust-10-replay"
cd "$HOME/experiments/trust-10-replay/trustcpf_abc_v1_10ep"
python -m unittest discover -s tests -p test_abc.py -v
```

[执行与评分清单](configs/experiment_inventory.json) · [输入及状态](inputs.json) · [文件 SHA256](file_manifest.json) · [发布成员](package_members.json) · [服务器成员](results/raw_export_members.json) · [来源与许可](来源与许可.md)
