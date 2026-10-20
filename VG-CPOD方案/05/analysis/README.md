# 05 · 数据分析

日期：2026-10-20。框架：冻结 CLIP ViT-L/14，UT-Zappos。新增运行与原导出见[结果索引](../results/README.md)，公式与修订见[主README](../README.md)。本页先报告本轮三 seed 结果，再记录 E1 的后续建议。

**下一轮直接运行 E1：沿用 D1 的中心化门控，将 ρ 从 0.125 改为 0.15，三张 GPU 各跑一个 seed。** η 保持 0.25，β、rank、学习率、训练轮数保持当前值。研究重点集中在视觉条件化训练下 CP 修正强度的作用。

本轮已补齐 D1 三 seed。相对 A5，D1 的 TEST CW/OW AUC 配对平均增量为 **+0.354／+0.197 点**，CW HM 为 +0.298 点，OW HM 为 **−0.662 点**。动态门控相对同权重 TRAIN 均值替换的平均增量为 **+0.090／−0.013 点 CW／OW AUC**。这些结果将后续重点指向训练阶段的修正尺度与最终分类器的排序、校准表现。

## 1. 三 seed 总体结果

A5 为 ρ=0.1 的静态 CP，A6 为对应无 CP 对照，C2 为 ρ=0.1 的中心化动态 CP，D1 为 ρ=0.125 的中心化动态 CP。`D1_mean` 复用 D1 的已选权重，在推理中把门控替换成 TRAIN 均值。

下表为跨 seed 的均值 ± 样本标准差。AUC 使用服务器保存的 native AUC，面积乘 100；差值以点计。条件属性指标使用 GT-object，并统计未见组合图像。

| 方案（三 seed） | VAL CW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- |
| A5 | 43.078 ± 1.289 | 34.437 ± 1.043 | 25.255 ± 0.792 | 23.638% |
| A6 | 34.881 ± 0.417 | 30.112 ± 0.051 | 24.784 ± 0.259 | 24.432% |
| C2 | 43.243 ± 1.339 | 34.329 ± 0.952 | 25.130 ± 0.985 | 23.638% |
| D1 | 43.301 ± 1.595 | 34.791 ± 0.566 | 25.453 ± 1.153 | 24.343% |
| D1_mean | 43.239 ± 1.128 | 34.702 ± 0.698 | 25.466 ± 1.074 | 24.273% |

D1 的 CW AUC 三 seed 均值为 34.791，OW 为 25.453，均高于 A5、C2。未见条件属性均值为 24.343%，A5 为 23.638%，A6 为 24.432%。当前最明显的改进是组合排序。

| seed | D1 选中轮数 | D1−A5 CW AUC | D1−A5 OW AUC | D1−A5 OW HM | 未见属性正确数变化 |
| --- | --- | --- | --- | --- | --- |
| 3351 | 10 | +0.749 | +0.524 | -0.071 | +4 |
| 3352 | 8 | -0.257 | -0.430 | -0.763 | -4 |
| 3353 | 9 | +0.570 | +0.499 | -1.151 | +40 |

![同 seed 的 TEST AUC 增量](confirmed/figures/01_paired_results.png)

seed3351、3353 的 AUC 提高，3352 下降；未见条件属性正确数依次为 **+4、−4、+40**。三个 seed 的平均净增为 13.33 张，主要由 seed3353 的变化贡献。

| 配对比较 | seed 数 | CW AUC 差值均值 ± SD | OW AUC 差值均值 ± SD | CW HM 差值均值 | OW HM 差值均值 |
| --- | --- | --- | --- | --- | --- |
| D1-A5 | 3 | +0.354 ± 0.537 | +0.197 ± 0.544 | +0.298 | -0.662 |
| D1-C2 | 3 | +0.463 ± 0.424 | +0.323 ± 0.600 | +0.454 | -0.556 |
| D1-C1 | 2 | +0.201 ± 0.425 | +0.079 ± 0.403 | +0.121 | +0.428 |
| C1-A5 | 2 | +0.045 ± 0.286 | -0.032 ± 0.272 | +0.240 | -0.845 |

上述标准差是同 seed 差值的标准差。D1−C1 使用已有的 seed3351、3352，其余三 seed 比较使用 3351、3352、3353。

exact AUC 的配对平均结果与 native 方向一致：D1−A5 为 **+0.273／+0.228 CW／OW**，D1−C2 为 **+0.385／+0.357**。本轮把 D1 作为继续优化的排序候选，A5 保留为已见／未见平衡对照。

## 2. 动态推理对平均结果的贡献较小

六个 checkpoint 的均值门控均由全部 **22,998 张 TRAIN 图像**统计，每个物体查询保留独立模式均值。动态与均值替换各自在自身 VAL 上选择 bias。

| seed | 动态−均值 CW AUC | 动态−均值 OW AUC | 未见属性正确数变化 |
| --- | --- | --- | --- |
| 3351 | +0.189 | +0.033 | +3 |
| 3352 | -0.085 | -0.163 | -1 |
| 3353 | +0.165 | +0.091 | +2 |

D1 的 dynamic−mean 三 seed 平均结果：

- TEST CW AUC：**+0.090 ± 0.152 点**；exact AUC 为 +0.088 点。
- TEST OW AUC：**−0.013 ± 0.133 点**；exact AUC 为 −0.012 点。
- 未见条件属性：平均 **+0.071 个百分点**，每个 seed 的净正确数为 +3、−1、+2。
- OW HM：平均 −0.006 点。

![D1 动态输入相对 TRAIN 均值替换的增量](confirmed/figures/02_dynamic_effect.png)

D1_mean 相对 A5 的平均 CW/OW AUC 仍有 **+0.264／+0.211 点**，未见条件属性提高 **0.635 个百分点**。D1 训练得到的分类器在固定物体模式系数下保留了平均收益。

训练得到的参数可能受多个因素共同影响，包括视觉条件化、门控正交项和优化尺度。现有干预直接测量的是同一权重下动态输入的预测作用；训练方案整体差异由 D1−C1 等配对衡量。

### 2.1 公式与数据一致

当前中心化公式保持每张图像、每个物体查询的模式均值为 1。将门控分解为 TRAIN 均值与图像残差：

```math
\begin{aligned}
\gamma_{bj}&=\bar\gamma_j+\varepsilon_{bj},\\
\Delta U_{bji}
&=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\bar\gamma_j\right)
+\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\varepsilon_{bj}\right).
\end{aligned}
```

第一项的固定系数可写成 $v'_j=v_j\odot\bar\gamma_j$，形成静态 CP 读出；第二项是样本相关调制。均值门控仍通过全局图像特征 $g_b$ 完成识别。

| D1 seed | TRAIN 跨图像门控标准差 | 门控总方差中的图像变化比例 |
| --- | --- | --- |
| 3351 | 0.01270 | 2.52% |
| 3352 | 0.01245 | 2.27% |
| 3353 | 0.01168 | 2.06% |

图像变化在 TRAIN 门控总方差中的比例约为 2.06%—2.52%，其余来自固定物体／模式差异。模式均值约束的最大统计误差低于 $10^{-10}$。模型公式与均值替换实现一致，当前模型公式继续沿用。

## 3. 两组四格交互的结论

seed3351、3352 现均有完整四格：

| CP 形式 | ρ=0.1 | ρ=0.125 |
|---|---|---|
| 静态 | A5 | C1 |
| 中心化动态 | C2 | D1 |

按各自 VAL 规则选出的结果，交互差为：

```math
I=(\mathrm{D1}-\mathrm{C2})-(\mathrm{C1}-\mathrm{A5}).
```

| seed | CW 交互差 | OW 交互差 | CW exact 交互差 | OW exact 交互差 |
| --- | --- | --- | --- | --- |
| 3351 | +0.488 | +0.283 | +0.394 | +0.367 |
| 3352 | +0.132 | -0.103 | +0.155 | -0.104 |

![两 seed 的修正强度与中心化模块交互](confirmed/figures/03_interactions.png)

CW 交互两次都为正，均值 +0.310 点；OW 一正一负，均值 +0.090 点。单独提高静态 CP 的 ρ，两个 seed 平均 CW/OW AUC 为 +0.045／−0.032 点；保留中心化动态训练后，提高 ρ 的三 seed 增量 D1−C2 为 +0.463／+0.323 点。

这些观测支持继续沿动态训练中的修正强度研究。下一轮只增加相邻的 ρ=0.15，形成 C2／D1／E1 的 0.1、0.125、0.15 三点比较。

ρ 调整属于训练尺度变化。对任意正数 $\rho'$，取 $W'_c=(\rho/\rho')W_c$ 可保持修正函数不变；训练时，初始化后的梯度、梯度裁剪、权重衰减与优化轨迹会随参数化尺度变化。该方向的目标是找到更合适的训练配置。

## 4. OW 排序与固定 bias 的平衡需要分开读取

D1 的 OW AUC 平均提高 0.197 点，固定 VAL bias 的 OW HM 在三个 seed 上均下降，平均下降 0.662 点。

| seed | A5 OW 已见准确率 | D1 OW 已见准确率 | A5 OW 未见准确率 | D1 OW 未见准确率 |
| --- | --- | --- | --- | --- |
| 3351 | 44.575% | 43.402% | 41.883% | 42.834% |
| 3352 | 44.770% | 43.988% | 37.811% | 37.070% |
| 3353 | 34.506% | 31.574% | 41.618% | 43.469% |

seed3351、3353 的未见准确率提高，同时已见准确率下降；seed3352 的已见、未见准确率均下降。因此，下一轮结果应同时记录 AUC 与固定 VAL bias 的已见准确率、未见准确率和 HM。

本次还直接用现有预测做了一次无训练复算：把 native VAL bias 换成 exact VAL bias，D1−A5 的平均 OW HM 仍为 **-0.866 点**。当前差异来自分数分布与跨 split 的平衡变化，单纯细化 bias 扫描没有消除该差异。下一轮仍采用原主评价规则。

## 5. 一个具体的选权重细节

D1/seed3353 的 native AUC 选中 epoch9，exact AUC 的峰值在 epoch6：

| D1/3353 epoch | VAL CW native AUC | VAL CW exact AUC |
| --- | --- | --- |
| 6 | 44.469564 | 45.186722 |
| 9 | 44.529430 | 44.709495 |
| 10 | 44.514647 | 44.679927 |

native 使用有限 bias 扫描，exact 使用完整临界点曲线。本例中 epoch9 的 native 比 epoch6 高约 0.060 点，而 epoch6 的 exact 比 epoch9 高约 0.477 点。现有 TEST 结果均来自原规则选中的 epoch9。

![各 seed 的 native／exact 验证曲线与原选中轮数](confirmed/figures/04_epoch_selection.png)

下一轮在一次训练中同步保存 `best_native.pt`、`best_exact.pt` 和 `last.pt`。主表继续按既定 native 规则选择，exact 权重作为附加输出。这项改动只增加保存点，使后续能直接分析完整积分指标对应的权重。

## 6. 下一轮直接执行的三卡实验

**E1 = 当前 D1，仅将 ρ 从 0.125 调整为 0.15。**

| 物理 GPU | 任务 | seed |
|---|---|---|
| GPU1 | E1，ρ=0.15 | 3351 |
| GPU2 | E1，ρ=0.15 | 3352 |
| GPU3 | E1，ρ=0.15 | 3353 |

三项各训练 10 epochs，共 30 个模型 epoch。已有 C2、D1 提供对应 seed 的 ρ=0.1、0.125 对照。本轮补 seed 和静态幅度对照已完成，新增计算直接投入 E1。

| 项目 | 设置 |
|---|---|
| 框架 | 冻结 CLIP ViT-L/14，复用当前 UT-Zappos 缓存 |
| ρ | **0.15** |
| η / τγ | **0.25 / 0.1** |
| 读出／门控输入 | 全局 g／物体查询视觉特征 |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| 适配器 LR / CP LR | 1e-3 / 3e-4，余弦终点 1e-5 / 3e-6 |
| epochs / batch size | 10 / 64 |
| λorth / weight decay / clip | 0.001 / 1e-4 / 1 |
| 主 checkpoint | VAL CW native AUC 首次严格最大值 |
| 附加输出 | exact 最优权重、动态与 TRAIN 均值替换评估 |

本轮核心问题是：提高到 ρ=0.15 后，CW/OW 排序增量、OW HM 和未见条件属性会怎样共同变化。候选设置按 VAL 汇总，TEST 保留为对应方案的完整结果。

代码改动集中于新增 E1 配置、三卡任务、额外 checkpoint 保存和三档 ρ 的同 seed 汇总。中心化门控、概率分解、训练损失和 CLIP 缓存实现继续沿用。模型层面的研究解释聚焦于“视觉条件化训练得到的 CP 分类器”，并同时报告动态和固定均值两种推理形式。

## 7. 逐组合表现解释了第三个 seed 的变化

下面列出 D1−A5 在 TEST 未见组合上的条件属性正确数变化：

| 未见组合 | 图像数 | seed3351 属性正确数变化 | seed3352 属性正确数变化 | seed3353 属性正确数变化 |
| --- | --- | --- | --- | --- |
| Suede / Boots.Mid-Calf | 342 | +6 | +2 | +27 |
| Canvas / Shoes.Loafers | 92 | +0 | +0 | +12 |
| Satin / Sandals | 113 | +2 | +0 | +7 |
| Synthetic / Boots.Mid-Calf | 329 | -2 | +0 | -9 |
| Suede / Slippers | 149 | -6 | -6 | +1 |

seed3353 的 +40 张净增主要来自 Suede / Boots.Mid-Calf、Canvas / Shoes.Loafers、Satin / Sandals，同时 Synthetic / Boots.Mid-Calf 有回落。后续继续按同 seed 逐组合比较，观察更高 ρ 对这些属性／物体关系的影响。

## 8. 本次数据核查与复算文件

本轮退出码为 0，D1/3352、D1/3353、C1/3352 全部完成。分析覆盖 14 个训练运行、6 个 TRAIN 均值替换条件，共复算 104 份预测文件，核对 72 处训练源指纹。六个动态重评估的 24 份预测数组与原导出逐元素一致。

固定 VAL bias 的 HM 和条件属性准确率复算一致；exact AUC 最大误差约 0.000049 点，native AUC 最大误差约 0.00864 点。正文使用服务器保存指标，跨 seed 标准差使用配对差值计算。

- [全部指标](confirmed/tables/metrics.csv)、[均值与标准差](confirmed/tables/means.csv)、[同 seed 增量](confirmed/tables/comparisons.csv)、[配对均值与标准差](confirmed/tables/paired_means.csv)。
- [四格交互](confirmed/tables/interactions.csv)、[条件属性诊断](confirmed/tables/diagnostics.csv)、[预测修正与退化计数](confirmed/tables/similarity.csv)、[逐组合增量](confirmed/tables/pair_deltas.csv)。
- [门控 TRAIN 统计](confirmed/tables/moments.csv)、[训练历史](confirmed/tables/history.csv)、[VAL bias 复算](confirmed/tables/val_bias_comparison.csv)。
- [结构化分析](confirmed/analysis.json)、[复算记录](confirmed/audit.json)、[主分析脚本](../code/original_analysis/analysis_next_results_20261020/analyze_next.py)、[bias 分析脚本](../code/original_analysis/analysis_next_results_20261020/analyze_bias.py)、[报告生成脚本](../code/original_analysis/analysis_next_results_20261020/build_report.py)。


## 9. 仓库复算与首轮记录

- [首轮D分析](pilot/analysis.json)、[首轮逐运行指标](pilot/tables/metrics.csv)保留 D1/3351 时的分析阶段。
- [三seed分析](confirmed/analysis.json)、[原复算记录](confirmed/audit.json)、[仓库复算核查](repository_validation.json)、[五份README公式与排版](readme_layout_check.json)。

在仓库根目录执行：

```bash
python VG-CPOD方案/05/analysis/recompute.py
python VG-CPOD方案/05/analysis/recompute_bias.py
```

计算读取 01 缓存元数据、02 的 A5/A6、04 的 C1/C2 首 seed 与本轮新增运行，不启动训练。结果输出 `analysis/recomputed/`，13 张 CSV 与确认版逐字节比较。
