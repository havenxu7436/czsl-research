# 11 · G₀ 三数据集完整独立流水线 / 10 epoch

> **精简归档说明：** 三数据集统计、19 张分析表、公式、核心代码和配置仍在；逐样本预测、checkpoint、训练日志和 raw export 已从 GitHub 精简版移除。

实验完成：2026-10-30；仓库归档：2026-11-01。UT-Zappos50K、MIT-States、C-GQA；CLIP ViT-L/14；seed 5351/5352/5353。底座为历史 VG-CPOD V1 的 G₀。

9/9 完整流水线，165/165 队列任务有原始完成标记；450 个 10-epoch 训练阶段。正式识别 1,620 行，Q5 378 行，1,998 份逐图像预测；17方法，每套45读出。三条流水线独立拟合效用模型。

[完整方案与公式](PLAN.md) · [核心实现](FORMULAS_AND_CODE.md) · [执行代码](code/README.md) · [配置与完整队列](configs/README.md) · [原始结果](results/README.md) · [完整分析及19张表](analysis/README.md) · [复算](analysis/REPRODUCE.md) · [恢复和训练](SERVER_COMMANDS.md)

| Context − Uniform | CW AUC 差 | OW AUC 差 | 正向 seed CW / OW |
| --- | --- | --- | --- |
| UT | +0.1499 ± 0.2065 | +0.1676 ± 0.3682 | 2/3 / 2/3 |
| MIT | −0.0108 ± 0.1047 | −0.0122 ± 0.0431 | 2/3 / 2/3 |
| C-GQA | +0.0333 ± 0.0853 | +0.0106 ± 0.0121 | 2/3 / 3/3 |

Context在UT、C-GQA有小幅正式收益，MIT接近持平。MIT/C-GQA的Q5有效用决策收益；C-GQA条件目标RMSE改善，但正式迁移有限。AUC为0–100，±为三seed样本标准差。

```bash
git lfs pull --include="TRUST-CPF方案/11/**"
python tools/register_trustg0_three.py --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 11 --dest "$HOME/experiments/trust-g0-11"
```

历史输入索引/runbook/发布成员/raw-export/文件指纹（精简版已移除） · [来源与许可](来源与许可.md)
