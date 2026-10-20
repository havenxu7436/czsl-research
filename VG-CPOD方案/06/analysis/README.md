# 06 · 数据分析

日期：2026-10-20。框架：冻结 CLIP ViT-L/14，UT-Zappos。新增运行与原导出见[结果索引](../results/README.md)，公式与修订见[主README](../README.md)。本页报告 E1 三 seed 结果，再记录 F1 的下一轮建议。

**下一轮转向训练目标的支持集权重：F1 固定 E1 的 ρ=0.15，将 β 从 0.1 调到 0.05，三个 seed 并行。** η 保持 0.25，门控公式、rank、学习率和 10 epochs 沿用本轮设置。

本轮 ρ=0.15 的 E1 提高了 VAL AUC，但相对 D1，TEST CW/OW AUC 平均下降 **0.364／0.272 点**，未见条件属性下降 **0.970 个百分点**。三个 seed 的 native 与 exact 选中相同权重。当前下一步最有针对性的变量是 **β 对已见组合概率质量的训练压力**。

## 1. 三档 ρ 的结果已形成清楚的轮廓

下表为三个 seed 的均值。AUC 使用服务器保存的 native AUC，面积乘 100；差值以点计。未见条件属性指标使用 GT-object。

| 方案 | ρ | VAL CW AUC | VAL OW AUC | TEST CW AUC | TEST OW AUC | 未见条件属性 |
| --- | --- | --- | --- | --- | --- | --- |
| A5：静态 | 0.1 | 43.078 | 31.228 | 34.437 | 25.255 | 23.638% |
| C2：动态 | 0.1 | 43.243 | 31.371 | 34.329 | 25.130 | 23.638% |
| D1：动态 | 0.125 | 43.301 | 31.564 | 34.791 | 25.453 | 24.343% |
| E1：动态 | 0.15 | 43.554 | 31.788 | 34.427 | 25.180 | 23.374% |

![三档 rho 的验证与测试表现](confirmed/figures/01_rho_sweep.png)

从 C2 到 D1，TEST 平均排序提高；从 D1 到 E1，VAL CW/OW 继续提高，而 TEST 回落。E1 相对 A5 的 TEST CW/OW AUC 为 **−0.010／−0.075 点**，基本回到静态基座水平。D1 保留为本轮三档中已有 TEST 排序最高的参照。

按既定 VAL CW 均值选择候选，E1 的 43.554 高于 D1 的 43.301 和 C2 的 43.243。下一轮固定该 ρ，单独改变损失中的 β，与 E1 形成一项参数差异的直接对照。

| seed | E1 选中 epoch | E1−D1 CW AUC | E1−D1 OW AUC | 未见属性正确数变化 |
| --- | --- | --- | --- | --- |
| 3351 | 8 | -0.477 | -0.259 | -6 |
| 3352 | 8 | +0.063 | -0.007 | -3 |
| 3353 | 6 | -0.679 | -0.551 | -46 |

| 比较（三 seed） | CW AUC 均值 ± 配对 SD | OW AUC 均值 ± 配对 SD | CW HM 增量 | OW HM 增量 |
| --- | --- | --- | --- | --- |
| E1-D1 | -0.364 ± 0.384 | -0.272 ± 0.272 | -0.279 | -0.184 |
| E1-C2 | +0.099 ± 0.140 | +0.051 ± 0.340 | +0.175 | -0.740 |
| E1-A5 | -0.010 ± 0.248 | -0.075 ± 0.352 | +0.019 | -0.846 |

![E1 相对 D1 的同 seed 差异](confirmed/figures/02_e1_d1_deltas.png)

E1−D1 的 exact AUC 均值为 **−0.326／−0.292 CW／OW**，方向与 native 一致。未见条件属性正确数在三个 seed 上分别减少 **6、3、46 张**。E1 相对 A5 也分别少 2、7、6 张。

## 2. native 与 exact 选权重结果完全相同

| seed | native 选中 epoch | exact 选中 epoch | 同权重预测比较 |
| --- | --- | --- | --- |
| 3351 | 8 | 8 | 逐元素一致 |
| 3352 | 8 | 8 | 逐元素一致 |
| 3353 | 6 | 6 | 逐元素一致 |

已核对每个 checkpoint 对应的首次最大值、模型参数，以及两条导出的 VAL/TEST 预测数组。native 与 exact 在本轮 E1 上均选择 8／8／6 轮，全部预测与指标一致。

因此，本轮 E1 内部没有出现两种选权重方式带来的结果差异。后续继续保留两份最优权重输出；主结果维持既定 native 选择规则。

不同配置仍可能选中不同 epoch，例如 D1/3353 为 epoch9、E1/3353 为 epoch6。本文的方案比较都使用各自按原规则选出的权重。

## 3. 增加 ρ 对未见属性与已见概率质量的影响

E1 的 TEST 未见条件属性均值为 **23.374%**，D1 为 **24.343%**，A5 为 **23.638%**。三个 seed 的未见条件属性都比 D1 低。

| seed | D1 未见图像的已见概率质量 | E1 对应质量 | D1 未见条件属性 | E1 未见条件属性 | D1/E1 裁剪步数比例 |
| --- | --- | --- | --- | --- | --- |
| 3351 | 72.621% | 72.832% | 24.590% | 24.273% | 34.2%/41.7% |
| 3352 | 73.830% | 73.986% | 23.585% | 23.427% | 27.5%/40.3% |
| 3353 | 73.945% | 74.277% | 24.855% | 22.422% | 23.3%/30.8% |

这里的“已见概率质量”是对完整 192 个组合的联合概率，在 83 个训练组合上求和，再对未见组合图像取平均。

E1 三个 seed 的该质量均高于 D1，平均由 **73.466% 增至 73.698%**；未见条件属性 NLL 平均由 **2.3763 增至 2.3872**。E1 的梯度裁剪步数比例也均增加，表明修正尺度调整已明显影响优化过程。

![已见概率质量与未见条件属性的变化](confirmed/figures/04_support_attributes.png)

这些观测把下一轮实验目标落在现有训练目标中的支持集质量项。F1 将直接改变这一项的系数，测量其对未见属性和 OW 平衡的影响。

E1 的 OW HM 相对 A5 平均下降 **0.846 点**，相对 D1 下降 **0.184 点**。下一轮同时记录 AUC、固定 VAL bias 的已见准确率、未见准确率与 HM，保持对排序和工作点的完整描述。

## 4. 动态门控的平均贡献仍以 CW 为主

| seed | E1 dynamic−mean CW AUC | E1 dynamic−mean OW AUC | 未见属性正确数变化 |
| --- | --- | --- | --- |
| 3351 | +0.161 | +0.059 | +2 |
| 3352 | +0.268 | +0.062 | -2 |
| 3353 | -0.075 | -0.093 | +4 |

E1 的 dynamic−TRAIN mean 三 seed 平均为：

- TEST CW AUC：**+0.118 ± 0.176 点**；exact AUC 为 +0.137 点。
- TEST OW AUC：**+0.010 ± 0.088 点**；exact AUC 为 +0.021 点。
- 未见条件属性：平均 +0.071 个百分点，正确图像数分别为 +2、−2、+4。

![三档 rho 下动态输入的作用](confirmed/figures/03_dynamic_effect.png)

E1 的门控图像变化占比仍低：

| seed | E1 门控图像变化占比 | dynamic−mean OW AUC |
| --- | --- | --- |
| 3351 | 2.20% | +0.059 |
| 3352 | 2.00% | +0.062 |
| 3353 | 1.59% | -0.093 |

TRAIN 均值替换后的 E1 相对 A5 的 TEST CW/OW AUC 为 **−0.128／−0.085 点**。动态与均值两种推理形式都体现了本轮训练配置的回落。下一轮 η、τγ 和均值评估方法保持当前设置，新增计算集中于训练目标系数。

## 5. β 的作用可以精确写成支持集压力

现有实现的概率定义、中心化公式和损失计算一致。β 的变化可直接解释为训练目标中的一个系数变化。

设完整联合 log 概率为 $s_{bji}$，训练组合集合为 $\mathcal{S}$，标签为 $(i_b,j_b)$。定义训练组合概率质量：

```math
Z_b=\sum_{(j,i)\in\mathcal{S}}\exp(s_{bji}),
\qquad
\mathcal{L}_{\mathrm{support}}=-\frac{1}{B}\sum_b\log Z_b.
```

当前训练组合归一化损失和条件／物体损失满足：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{pair}}
&=\frac{1}{B}\sum_b\left[-s_{bj_bi_b}+\log Z_b\right],\\
\mathcal{L}_{a}+\mathcal{L}_{o}
&=-\frac{1}{B}\sum_b s_{bj_bi_b}
=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

代入现有 β 归一化训练目标：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}
\left[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_{a}+\mathcal{L}_{o})\right]\\
&=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

因此 β=0.1 和 β=0.05 分别对应：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{task}}(0.1)
&=1.5\mathcal{L}_{\mathrm{pair}}+0.1363636\mathcal{L}_{\mathrm{support}},\\
\mathcal{L}_{\mathrm{task}}(0.05)
&=1.5\mathcal{L}_{\mathrm{pair}}+0.0714286\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

pair 项系数保持 1.5，支持集项系数变为原来的 **52.38%**。最小化支持集项会推动训练组合总概率质量增大；降低该系数直接减弱这一训练压力。正交项与权重衰减保持原值。

这是下一轮的明确实验变量：在同一模型、ρ 和优化设置下，测量减弱已见支持集集中压力的结果。本次训练历史复算上述恒等式的最大误差为 $3.36\times10^{-8}$。

## 6. 下一轮直接执行 F1 三 seed

**F1 = E1，固定 ρ=0.15，将 β 从 0.1 改为 0.05。**

| 物理 GPU | 实验 | seed |
|---|---|---|
| GPU1 | F1，β=0.05 | 3351 |
| GPU2 | F1，β=0.05 | 3352 |
| GPU3 | F1，β=0.05 | 3353 |

三项各 10 epochs，共 30 个模型 epoch。E1 已有三个 seed，直接提供同 seed 的 β=0.1 对照；D1 和 A5 继续提供已有排序与静态基座参照。

| 参数 | 下一轮设置 |
|---|---|
| 框架／特征 | 冻结 CLIP ViT-L/14，复用当前 UT-Zappos 缓存 |
| ρ | **0.15** |
| β | **0.05** |
| η / τγ | 0.25 / 0.1 |
| rank / adapter rank | 32 / 8 |
| 适配器 LR / CP LR | 1e-3 / 3e-4，余弦终点 1e-5 / 3e-6 |
| epochs / batch size | 10 / 64 |
| 正交项／weight decay／clip | 0.001 / 1e-4 / 1 |
| 读出／门控输入 | 全局 g／物体查询视觉特征 |
| 权重与评价 | native／exact 分别保存；各自 VAL bias 固定用于 TEST |

结果按以下顺序分析：

1. **F1−E1**：支持集质量是否下降，未见条件属性和 OW HM 如何变化。
2. **F1−D1、F1−A5**：组合排序收益与已见／未见平衡的整体水平。
3. **F1 dynamic−mean**：目标权重调整后，图像相关调制的预测作用如何变化。

当前模型公式继续沿用。代码下一步集中于新增 F1 配置与三卡任务、同 seed 汇总，并在日志中直接记录 support 项的系数及加权值。模型结构、CLIP、门控公式和均值评估实现保持当前版本。

## 7. 逐组合结果说明了第三个 seed 的变化

下表是 E1−D1 的 TEST 未见条件属性正确数变化：

| 未见组合 | 图像数 | seed3351 属性正确数变化 | seed3352 属性正确数变化 | seed3353 属性正确数变化 |
| --- | --- | --- | --- | --- |
| Suede / Boots.Mid-Calf | 342 | -6 | -1 | -30 |
| Canvas / Shoes.Loafers | 92 | +1 | +1 | -11 |
| Suede / Slippers | 149 | -6 | -1 | -14 |
| Synthetic / Boots.Mid-Calf | 329 | -1 | -3 | +8 |

seed3353 的减少主要分布在 Suede / Boots.Mid-Calf、Canvas / Shoes.Loafers 和 Suede / Slippers；Synthetic / Boots.Mid-Calf 有回升。F1 的逐组合分析继续跟踪这些属性／物体关系，以及它们对应的已见概率质量和条件属性 NLL。

## 8. 数据核查与复算文件

E 组退出码为 0，三项训练和六项均值评估均完成。分析包含 17 个训练运行、3 个 exact 选中条件、12 个均值替换条件，共检查复算 **176 份预测导出**，核对 **99 处训练源指纹**。

48 份动态重评估预测与对应原导出逐元素一致；E1 的 12 份 exact 选中预测与 native 选中预测也逐元素一致。固定 VAL bias 的 HM 和未见条件属性复算一致；exact AUC 最大误差约 0.000049 点，native AUC 最大误差约 0.00864 点。正文采用服务器保存指标，配对 SD 由同 seed 差值计算。

- [全部指标](confirmed/tables/metrics.csv)、[各组均值](confirmed/tables/means.csv)、[同 seed 差值](confirmed/tables/comparisons.csv)、[配对均值与标准差](confirmed/tables/paired_means.csv)。
- [条件属性与概率质量](confirmed/tables/diagnostics.csv)、[逐图像修正与退化](confirmed/tables/similarity.csv)、[逐组合增量](confirmed/tables/pair_deltas.csv)。
- [训练历史](confirmed/tables/history.csv)、[TRAIN 门控统计](confirmed/tables/moments.csv)、[结构化分析](confirmed/analysis.json)、[复算记录](confirmed/audit.json)。
- [分析脚本](../code/original_analysis/analysis_roundE_results_20261020/analyze_e.py)、[报告生成脚本](../code/original_analysis/analysis_roundE_results_20261020/build_report.py)。

## 9. 仓库复算与归档核查

- [仓库复算入口](recompute.py)：读取本阶段 E1、02 的 A5／A6、04／05 的 C1／C2／D1 和 01 缓存元数据，数值计算沿用原分析。
- [仓库验证记录](repository_validation.json)：核对原包成员、权重与预测数量、README 数值、公式、链接及恢复目录。
- [原始结构化分析](confirmed/analysis.json)、[原始复算审计](confirmed/audit.json)、[分析图表](confirmed/figures/)。

```bash
python VG-CPOD方案/06/analysis/recompute.py
python tools/reproduce.py verify
```

复算输出至 `analysis/recomputed/`。原始分析和服务器指标保留在本阶段目录。F1 为后续实验计划，本轮新增训练为 E1 三个 seed、β=0.1。
