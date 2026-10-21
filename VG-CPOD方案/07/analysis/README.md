# 07 · 数据分析

日期：2026-10-20。框架：冻结 CLIP ViT-L/14，UT-Zappos。原运行见[结果索引](../results/README.md)，公式与修订见[主README](../README.md)。本页先分析已完成 F1，再记录 G0/G1 下一轮建议。

**F1 改善了未见属性泛化，但组合排序回落。下一轮优先做 β=0.075 的 G1，并补同配置的静态训练对照 G0。** 固定 ρ=0.15、η=0.25、rank32、原学习率与 10 epochs，三张 GPU 各负责一个 seed 的 G0/G1 两项训练。

F1 相对 E1：TEST 未见条件属性 **+1.816 个百分点**，CW／OW AUC **−0.397／−0.829 点**。物体预测同时提高；排序下降主要随条件属性分支变化出现。动态门控相对 TRAIN 均值替换的 CW／OW AUC 贡献为 **+0.016／−0.003 点**。下一轮集中检验损失权重的中间值及视觉动态的独立收益。

## 1. 三个 seed 完成，属性与排序的变化方向一致

下表为三个 seed 的均值。AUC 为面积乘 100，差值以点计；未见条件属性使用真实物体标签评估。

| 方案 | ρ | β | VAL CW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- | --- | --- |
| A5 | 0.1 | 0.1 | 43.078 | 34.437 | 25.255 | 23.638% |
| D1 | 0.125 | 0.1 | 43.301 | 34.791 | 25.453 | 24.343% |
| E1 | 0.15 | 0.1 | 43.554 | 34.427 | 25.180 | 23.374% |
| F1 | 0.15 | 0.05 | 43.505 | 34.030 | 24.351 | 25.189% |

![F1 与已有配置的验证和测试结果](confirmed/figures/05_f1_comparison.png)

F1 的 VAL CW／OW AUC 为 **43.505／31.750**，E1 为 **43.554／31.788**。F1 的 TEST CW／OW AUC 在三个 seed 上均下降，未见条件属性正确数在三个 seed 上均增加。E1 和 F1 对应 seed 选中同一 epoch，因此该对比在训练轮数上也对齐。

| seed | E1/F1 选中 epoch | CW AUC 变化 | OW AUC 变化 | 未见属性正确数变化 |
| --- | --- | --- | --- | --- |
| 3351 | 8/8 | -0.585 | -0.919 | +35 |
| 3352 | 8/8 | -0.469 | -1.535 | +26 |
| 3353 | 6/6 | -0.138 | -0.032 | +42 |

| 比较 | TEST CW AUC：均值 ± 配对 SD | TEST OW AUC：均值 ± 配对 SD | 未见属性增量 |
| --- | --- | --- | --- |
| F1-E1 | -0.397 ± 0.232 | -0.829 ± 0.756 | +1.816 个百分点 |
| F1-D1 | -0.762 ± 0.331 | -1.101 ± 0.484 | +0.846 个百分点 |
| F1-A5 | -0.407 ± 0.224 | -0.904 ± 0.969 | +1.551 个百分点 |

![F1 相对 E1 的同 seed 变化](confirmed/figures/06_f1_e1_paired.png)

F1−E1 的 exact AUC 变化为 **−0.395／−0.810 CW／OW**，与 native AUC 方向一致。F1 三个 seed 的 native 和 exact 均选择 epoch **8／8／6**，模型参数与预测导出逐元素一致。

固定各自 VAL bias 后，F1 的 OW HM 平均从 **39.792 升到 40.349**，增量 **+0.556 ± 2.705 点**；对应三个 seed 为 **-0.588／-1.389／+3.645 点**。F1 的 OW 已见准确率从 **39.557% 升到 43.239%**，未见准确率从 **40.543% 降到 37.828%**。该工作点更偏向已见类，HM 与 AUC 分别描述了工作点和整条曲线的变化。

## 2. 支持集权重的作用已经从数据中分离出来

E1、F1 唯一训练超参数差异为 β 从 0.1 到 0.05。选中 epoch 的 TRAIN 结果如下：

| 选中 epoch 的 TRAIN 均值 | E1 | F1 |
| --- | --- | --- |
| pair 原始损失 | 1.150114 | 1.149372 |
| support 原始损失 | 0.123752 | 0.146553 |
| pair 加权损失 | 1.725170 | 1.724058 |
| support 加权损失 | 0.016875 | 0.010468 |
| TRAIN 组合概率质量 | 0.893445 | 0.875791 |
| 裁剪步数比例 | 0.375926 | 0.379630 |

支持集系数从 0.136363636 降到 0.071428571 后，原始 support 损失上升、加权 support 损失下降；TRAIN 组合概率质量由 **89.345% 降到 87.579%**。pair 损失由 1.150114 变为 1.149372，组合内训练拟合基本持平。

TEST 未见图像上的 TRAIN 组合概率质量从 **73.698% 降到 70.770%**，未见条件属性准确率从 **23.374% 升到 25.189%**，NLL 从 **2.3872 降到 2.2884**。降低支持集权重确实改变了泛化方向。

## 3. 排序下降集中在条件分支的跨组合竞争

以下指标均在 TEST 的 1,891 张未见组合图像上计算，再取三 seed 均值。“CW 集合外”指该 split 的 CW 候选集之外、OW 候选集之内的组合；TEST CW 有 101 个组合，OW 有 192 个组合，其中 TRAIN 组合为 83 个。

| TEST 未见图像指标 | E1 | F1 | F1−E1 |
| --- | --- | --- | --- |
| 物体准确率 | 63.3527% | 63.7405% | +0.3878 |
| 物体 NLL | 0.8841 | 0.8754 | -0.0087 |
| GT-object 条件属性准确率 | 23.3739% | 25.1895% | +1.8156 |
| GT-object 条件属性 NLL | 2.3872 | 2.2884 | -0.0988 |
| TRAIN 组合概率质量 | 73.6985% | 70.7702% | -2.9283 |
| CW 集合外组合概率质量 | 11.1504% | 12.8369% | +1.6865 |
| 在全部未见候选中的真实组合 top-1 | 49.9207% | 48.0169% | -1.9038 |
| CW 集合外最高分压过 CW 内未见最高分 | 23.6207% | 25.7712% | +2.1505 |

F1 的物体准确率和物体 NLL 均改善。属性在真实物体条件下更准确，但与其他物体、属性共同竞争时，真实组合在全部未见候选中的 top-1 比例从 **49.921% 降到 48.017%**。

![属性预测与未见候选排序的分离](confirmed/figures/07_ranking_mechanism.png)

CW 集合外组合的总概率质量平均增加 **1.687 个百分点**；其最高分压过 CW 内未见组合最高分的图像比例增加 **2.151 个百分点**。三个 seed 中，原本正确的未见候选 top-1 共退化 177 张次，另有 69 张次修正，净减少 108 张次；177 次退化中，130 次转向 CW 集合外组合。该统计描述模型的候选排序变化。

### 3.1 交换已保存的两个概率因子

用同 seed、同选中 epoch 的已保存预测，构造 $`p_E(o\mid x)p_F(a\mid o,x)`$ 和 $`p_F(o\mid x)p_E(a\mid o,x)`$，重新评价组合排序。该计算使用已有预测，训练步数为零；每个组合条件在自己的 VAL 上校准 bias。

| 联合分数构成 | TEST CW exact AUC | TEST OW exact AUC |
| --- | --- | --- |
| E1 物体 × E1 条件属性 | 34.549 | 25.261 |
| F1 物体 × F1 条件属性 | 34.154 | 24.451 |
| E1 物体 × F1 条件属性 | 34.162 | 24.548 |
| F1 物体 × E1 条件属性 | 34.543 | 25.131 |

保留 E1 条件分支、换入 F1 物体分支后，OW exact AUC 为 **25.131**，接近 E1 的 25.261；保留 F1 条件分支、换回 E1 物体分支后为 **24.548**，接近 F1 的 24.451。该分解将下一轮重点放在条件分支对所有候选的相对排序上。物体分支继续沿用当前实现。

### 3.2 逐组合收益与损失落在不同关系上

| TEST 未见组合 | 图像数/seed | 属性正确数增量之和 | 未见候选 top-1 正确数增量之和 |
| --- | --- | --- | --- |
| Synthetic / Boots.Mid-Calf | 329 | +45 | +36 |
| Suede / Boots.Mid-Calf | 342 | +23 | -26 |
| Canvas / Shoes.Loafers | 92 | +14 | +4 |
| Hair.Calf / Shoes.Heels | 67 | +0 | -48 |
| Suede / Slippers | 149 | -5 | -46 |

最后两列均为三个 seed 的正确数差值之和，同一图像在三个模型中分别计数。Synthetic／Boots.Mid-Calf 同时提高属性和组合排序；Suede／Boots.Mid-Calf 的属性改善而组合排序下降；Hair.Calf／Shoes.Heels 的条件属性正确数持平，组合排序明显下降。下一轮继续同时跟踪这两类指标。

## 4. 当前视觉动态带来的排序增量很小

F1 的动态门控与 TRAIN 均值替换使用相同 checkpoint，统计均值覆盖全部 22,998 张 TRAIN 图像。各自 VAL bias 用于各自 TEST 评价。

| seed | F1 dynamic−mean CW AUC | OW AUC | 未见属性正确数变化 | 门控图像变化占比 |
| --- | --- | --- | --- | --- |
| 3351 | +0.027 | -0.042 | +7 | 1.88% |
| 3352 | -0.085 | -0.051 | +6 | 1.88% |
| 3353 | +0.105 | +0.084 | +8 | 1.51% |

三 seed 的 dynamic−mean 平均为：

- TEST CW AUC：**+0.0156 ± 0.0956 点**。
- TEST OW AUC：**−0.0032 ± 0.0758 点**。
- TEST 未见条件属性：**+0.3702 个百分点**，正确数变化分别为 +7／+6／+8。

![不同配置中的动态门控作用](confirmed/figures/08_f1_dynamic_effect.png)

TRAIN 门控图像变化占比为 **1.88%／1.88%／1.51%**，其余主要为物体／模式之间的固定差异。F1 均值替换相对 E1 均值替换的 TEST CW／OW AUC 仍下降 **0.295／0.816 点**，β 引起的排序变化在两种推理条件下都存在。

下一轮补充从初始化训练的静态模型，可直接量化学习视觉门控这一模块对最终模型的贡献。当前均值替换衡量的是已训练动态模型对图像变化的依赖，静态训练对照补齐模块训练层面的比较。

## 5. 公式核查：当前实现正确，β 调整继续沿用现有目标

设完整联合 log 概率为 $`s_{bji}`$、TRAIN 组合集合为 $`\mathcal{S}`$，真实标签为 $`(i_b,j_b)`$。当前目标满足：

```math
\begin{aligned}
Z_b&=\sum_{(j,i)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{support}}&=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_{\mathrm{pair}}&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],
\\[4pt]
\mathcal{L}_a+\mathcal{L}_o&=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_a+\mathcal{L}_o)]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

F1 新日志的 `pair_weighted`、`support_weighted` 和系数与该恒等式一致。20 个已有训练运行的 epoch 均值重构最大误差为 **8.31×10⁻⁸**，联合概率归一化最大误差为 **1.31×10⁻⁶**；F1 记录的平均 batch 绝对重构误差最大为 **1.18×10⁻⁷**。中心化门控、训练和两种选权重的导出均按当前公式完成。

下一轮 β=0.075 时：

```math
\mathcal{L}_{\mathrm{task}}(0.075)
=\frac{3}{2}\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}}
\approx1.5\mathcal{L}_{\mathrm{pair}}+0.104651163\mathcal{L}_{\mathrm{support}}.
```

该系数位于 E1 的 0.136363636 与 F1 的 0.071428571 之间。G1 用这一中间强度检验属性收益与组合排序的平衡，组合分类项系数保持 1.5。正交项和 AdamW 设置沿用原实现。

## 6. 下一轮具体做 G0／G1 六次训练

| 实验 | ρ | β | gate_input | η | 要回答的问题 |
|---|---|---|---|---|---|
| **G1：动态** | 0.15 | **0.075** | local | 0.25 | 中间 β 的属性与排序平衡 |
| **G0：静态** | 0.15 | **0.075** | **none** | 沿用配置 | 同一目标下视觉门控模块的训练收益 |

G1 相对 E1 只调整 β；G0 在 G1 配置上关闭门控输入，得到 $`\gamma_{bjk}=1`$。G0 使用同样的 CLIP、适配器、CP 分解、初始化规则与数据顺序；静态分支的 Bvis 冻结，orth 项为零，与现有代码的分支定义一致。

| 物理 GPU | 顺序执行的训练 | seed |
|---|---|---|
| 1 | G0 → G1，各 10 epochs | 3351 |
| 2 | G0 → G1，各 10 epochs | 3352 |
| 3 | G0 → G1，各 10 epochs | 3353 |

共 **6 个模型、60 个模型 epoch**；从对应 seed 的原始初始化开始。G1 仍完成 native／exact 两种选中权重的六项 TRAIN 均值门控评估。

固定参数：CLIP ViT-L/14 冻结，rank32、adapter rank8、κ20、τγ0.1、τv0.07、batch64；适配器 LR 1e-3→1e-5，CP LR 3e-4→3e-6；weight decay 1e-4、gradient clip1、动态分支 λorth0.001，10 epochs。

结果分析按以下顺序推进：

1. **G1−E1、G1−F1**：比较 β=0.1／0.075／0.05 的 CW／OW AUC、属性准确率、NLL 与支持集质量。候选排序继续使用既定 VAL CW native AUC 的三 seed 均值；exact、OW 与 TEST 指标并列报告。
2. **G1−G0**：同 seed 比较最终组合排序与条件属性，直接量化视觉门控模块的训练收益。
3. **G1 dynamic−mean**：判断新目标下图像变化对推理排序的作用。
4. 跟踪 CW 集合外候选超越比例、未见候选 top-1 和逐组合变化，连接属性改善与组合排序。

下一步代码工作集中于 G0/G1 配置、三卡双任务队列和上述指标汇总；核心模型与损失表达式继续沿用。这个实验集同时完成 β 中间值比较和模块训练对照，计算投入集中在六个模型。

## 7. 本次数据核查与复算文件

F 组退出码为 0，三项训练与六项均值门控评估均完成。核查 **20 个训练运行、6 个 exact 选中条件、18 个均值替换条件**，共复算 **248 份预测导出**，核对 **126 处训练源指纹**。72 份动态重评估导出与原预测逐元素一致，24 份 exact 导出与对应 native 导出逐元素一致。F 组源代码包的 68 份文件均与交付清单 SHA256 一致。

固定 VAL bias 的 HM 和未见条件属性复算一致；exact AUC 最大误差约 0.000049 点、native AUC 最大误差约 0.00864 点。正文使用服务器保存的主指标；因子交换单列，采用从保存联合 log 概率恢复的分布。每项对比均使用同 seed、同特征缓存。

- [完整指标](confirmed/tables/metrics.csv)、[三 seed 均值](confirmed/tables/means.csv)、[配对差值](confirmed/tables/comparisons.csv)、[配对均值与标准差](confirmed/tables/paired_means.csv)。
- [排序分解](confirmed/tables/ranking_diagnostics.csv)、[逐组合概率与正确数](confirmed/tables/pair_probability_diagnostics.csv)、[未见候选排序变化](confirmed/tables/unseen_ceiling_movements.csv)、[概率因子交换](confirmed/tables/head_swap_metrics.csv)。
- [训练历史](confirmed/tables/history.csv)、[TRAIN 门控统计](confirmed/tables/moments.csv)、[结构化分析](confirmed/analysis.json)、[机制分解](confirmed/mechanism.json)、[复算核查](confirmed/audit.json)、[新增日志核查](confirmed/supplementary_audit.json)。
- [复算脚本](../code/original_analysis/analysis_roundF_results_20261020/analyze_f.py)、[分解脚本](../code/original_analysis/analysis_roundF_results_20261020/analyze_mechanism.py)。

## 8. 仓库复算与归档核查

- [复算入口](recompute.py)顺序运行[预测复算](recompute_metrics.py)、[排序分解与因子交换](recompute_mechanism.py)、[图表](recompute_figures.py)。读取02/04/05/06已有结果、07新增结果和01缓存元数据，数值计算沿用原分析。
- [归档验证](repository_validation.json)记录原包与服务器成员、README数值、公式、链接和运行包恢复；[原始复算](confirmed/audit.json)、[新增日志核查](confirmed/supplementary_audit.json)、[机制分解](confirmed/mechanism.json)。

```bash
python VG-CPOD方案/07/analysis/recompute.py
python tools/reproduce.py verify
```

输出至 `analysis/recomputed/`；原分析在 `analysis/confirmed/`。本阶段已完成 F1 三seed、β=0.05，G0/G1 为后续实验建议。
