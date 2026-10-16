# v2 单 seed 数据分析

UT-Zappos · seed 3351 · 每组 10 epoch · 2026-10-16

本文依次分析核心模型比较、机制对照、设置敏感性和原设计四个核心问题，最后补充 seen/unseen 校准和错误分解。

## 0. 数据口径与模型对应

- 共 32 个模型、128 份评价；每个模型按首次严格最大 **VAL CW AUC** 选中 checkpoint，同一权重评价 VAL/TEST 的 CW/OW。
- 下文以 VAL CW 判断是否达到主验证目标，同时独立报告 VAL OW、TEST CW/OW。表中 AUC 用百分数表示，差值用百分点；差值由原始精度计算后保留两位小数。
- 这是 seed 3351 的实测结论，不报告跨 seed 均值、显著性或误差条。
- v2 的 Logic 是 **JS 语义一致性约束＋描述标签锚点**；不是 LogiCZSL 原论文的模糊逻辑蕴含损失。本文统一称“语义约束”。

| 原设计角色 | v2 模型 | Text K | prior | 耦合 head | 语义约束 |
|---|---|---|---|---|---|
| A：CPF-only | `A_native` | 无 | 无 | 无 | 无 |
| 新概率接口无知识基准 | `R0` | 无 | 无 | 无 | 无 |
| B：external loss | `Rexternal` | 无 | 无 | 无 | 有 |
| C：richer text | `Rtext` | 有 | 无 | 无 | 无 |
| D：calibration | `Rprior` | 无 | 有 | 无 | 无 |
| E：MVP | `Rmvp` | 有 | 有 | 无 | 无 |
| F：naive combination | `Rmvp_logic` | 有 | 有 | 无 | 有 |
| G：feature K | `Rfeature` | 有 | 有 | 有 | 无 |
| H：Full Model | `Rfull` | 有 | 有 | 有 | 有 |

A–H 在此用于对照原设计的研究角色；所有 R 组采用 v2 联合概率与共享任务损失。`Rfull` 同时是 Template-only pair、每类 3 条描述、3 层 OI、mean 聚合、centered prior、固定 λ、默认属性损失的共同参照。

数据入口：[原始 AUC 汇总](raw_export/summary/summary.json)、[32 组指标](README.md)、[模型定义与命名](实验命名对照.md)、[实际开关](../../code/revised_0601_20261014_fix/prepare_revision.py)。

## 1. 核心模型比较

正值表示前者高于后者；每行均对应两个独立训练模型的选中 checkpoint。

| 序号 | 对比 | ΔVAL CW | ΔVAL OW | ΔTEST CW | ΔTEST OW | 对应问题与结论 |
|---:|---|---:|---:|---:|---:|---|
| 1 | `Rfull − A_native` | -14.12 | -1.75 | -9.73 | -7.90 | 整体修订效果：四项均下降；完整修订模型未提升原生 CPF。 |
| 2 | `R0 − A_native` | -8.40 | -4.28 | -6.43 | -6.04 | 无知识时概率／loss／attention 接口变化：四项均下降；下降在知识模块启用前已出现。 |
| 3 | `R0 − R0_native_attention` | +2.47 | -0.96 | -2.12 | -3.78 | 修订接口内切换视觉注意力设置：VAL CW 提高，VAL OW 与两项 TEST 降低；新注意力没有统一优势。 |
| 4 | `R0_native_attention − A_native` | -10.86 | -3.33 | -4.31 | -2.26 | 保留原生注意力后的其余接口变化：四项均下降；仅保留原生注意力未恢复原生表现。 |
| 5 | `Rtext − R0` | -6.25 | +1.45 | +0.21 | +1.06 | 仅替换 Text K：VAL CW 下降，OW 与 TEST 小幅提高；没有形成稳定的 CW 增益。 |
| 6 | `Rprior − R0` | -0.61 | +2.07 | +0.76 | +2.32 | 无 Text K 时加入 prior：VAL OW 与两项 TEST 提高；prior 的积极信号主要在 OW。 |
| 7 | `Rmvp − Rtext` | +0.51 | +0.78 | -4.74 | -3.77 | 有 Text K 时加入 prior：VAL 两项提高、TEST 两项下降；prior 收益依赖接入背景与选中轮次。 |
| 8 | `Rexternal − R0` | +1.21 | +1.37 | -0.13 | +0.55 | 无其他知识入口时加入 JS 语义约束：VAL CW/OW 提高，TEST CW 基本持平、OW 略升；收益较小。 |
| 9 | `Rmvp_logic − Rmvp` | +0.22 | +0.21 | +0.51 | +0.14 | MVP 上加入 JS 语义约束：四项小幅提高；本批最一致的语义约束增益，幅度为 0.14–0.51 点。 |
| 10 | `Rfeature − Rmvp` | -0.45 | -0.82 | +5.30 | +4.10 | 无语义约束时加入耦合残差 head：TEST CW/OW 明显提高，VAL 两项略降；测试收益清楚，主验证指标未提升。 |
| 11 | `Rfull − Rfeature` | +0.47 | +1.11 | -4.08 | -3.25 | feature 模型上加入 JS 语义约束：VAL 两项提高、TEST 两项下降；语义约束没有在 feature 模型上形成一致增益。 |
| 12 | `Rfull − Rmvp_logic` | -0.20 | +0.08 | +0.72 | +0.71 | 含语义约束的 MVP 上加入耦合残差 head：TEST CW/OW 约提高 0.7 点；VAL CW 略降，完整模型相对 naive combination 的收益有限。 |

### 1.1 整体修订与接口变化

`Rfull − A_native` 的 TEST CW/OW 分别为 −9.73/−7.90 点。无知识的 `R0` 已比 `A_native` 低 −6.43/−6.04 点，说明相当部分下降来自新接口本身。保留原生注意力的 `R0_native_attention` 仍低于原生模型，因而注意力设置不是唯一来源。

这里比较的是概率定义、任务损失、全组合 composer 等接口的联合变化，不把全部差异归到一个归一化项。

### 1.2 Text K 与 prior

`Rtext − R0` 的 VAL CW 为 −6.25，但 TEST CW/OW 为 +0.21/+1.06：默认文本替换没有形成稳定 CW 提升。

`Rprior − R0` 的 VAL OW 为 +2.07、TEST OW 为 +2.32，显示 prior 对 OW 有正向信号；在含 Text K 的背景下，`Rmvp − Rtext` 的 TEST CW/OW 转为 −4.74/−3.77。两条路径没有自然形成叠加收益。

### 1.3 语义约束

语义约束的三个接入背景结论不同：

- `Rexternal − R0`：VAL 改善，TEST CW 基本不变。
- `Rmvp_logic − Rmvp`：四项均有小幅改善。
- `Rfull − Rfeature`：VAL 改善，TEST 下降。

因此，当前语义约束更像背景相关的辅助目标，而不是对所有模型都有效的增益项。

### 1.4 耦合 head 与 Full Model

`Rfeature − Rmvp` 的 TEST CW/OW 为 +5.30/+4.10；在已有语义约束时，`Rfull − Rmvp_logic` 只剩 +0.72/+0.71。head 的测试收益依赖其余模块，完整叠加没有带来相应规模的额外收益。

`Rfull` 第 2 轮 VAL CW 为 26.50，第 5 轮为 26.46；`Rfeature` 选第 5 轮，`Rmvp_logic` 与 `Rfull` 选第 2 轮。验证近似持平时的选中轮次差异，是解释部分 TEST 大幅差异的重要线索。

## 2. 机制对照

### 2.1 指标变化

以下均为“对照模型 − 默认 Rfull”。

| 对照模型（相对 Rfull） | ΔVAL CW | ΔVAL OW | ΔTEST CW | ΔTEST OW | 结果 |
|---|---:|---:|---:|---:|---|
| `Rfull_shuffle` | -2.02 | -3.09 | -5.62 | -3.81 | 四项下降；类别—描述对应关系具有作用。 |
| `Rfull_random_prior` | -0.38 | -1.01 | +3.08 | +2.56 | VAL 下降、TEST 提高；真实 prior 没有获得一致优势。 |
| `Rfull_uniform_prior` | -0.86 | -1.28 | -0.19 | -0.21 | 四项略降；与 prior-off 的四套保存分数逐字节相同。 |
| `Rfull_nopair_head` | -0.30 | -1.19 | +2.96 | +1.84 | VAL 略降、TEST 提高；新增 pair 项未显示稳定必要性。 |
| `Rfull_prior_off` | -0.86 | -1.28 | -0.19 | -0.21 | 四项略降；真实 prior 在此背景下贡献小。 |
| `Rfull_nopair_all` | -0.11 | -0.47 | +0.36 | +0.45 | VAL 略降、TEST 略升；两项联合移除未造成明显退化。 |
| `Rfull_static_head` | -0.01 | -0.13 | -0.23 | -0.33 | 四项仅小幅变化；动态 query 的独立收益弱。 |

### 2.2 每个对照实际检验了什么

| 模型 | 实际改动 | 保留项 |
|---|---|---|
| `Rfull_shuffle` | OE/AE 按类别错配，Text K 与描述约束同步置换 | 名称、OI、pair 模板、主 prior |
| `Rfull_random_prior` | 固定随机 K 后 centered | 其余输入与模型 |
| `Rfull_uniform_prior` | K 全 1，centered 后为 0 | head、文本、语义约束 |
| `Rfull_nopair_head` | 仅移除新 head 的 pair 内积 | 三线性交互、CPF composer、Text K、prior、语义约束 |
| `Rfull_prior_off` | 关闭条件 prior | 新 head 的 pair 项 |
| `Rfull_nopair_all` | 同时移除 head pair 项与 prior | CPF composer 和对象—属性交互 |
| `Rfull_static_head` | 新 head 的两个视觉 query 改为常数 | 原生视觉分支 |

### 2.3 描述错配：语义对应关系有作用

`Rfull` 对 `Rfull_shuffle` 的 VAL CW/OW 和 TEST CW/OW 均更好。在描述数量、名称、OI、模板与主 prior 保持不变时，错配使性能下降。这支持类别与描述的语义对应关系产生作用，而非只有描述数量在起作用。

本模型用 GloVe 聚合而非增加 CLIP prompt token；这里检验的是同数量文本内容的对应关系，不是独立的 token 长度实验。

### 2.4 随机 prior：真实 prior 尚无一致优势

真实 prior 相对随机 prior 的 VAL CW/OW 高 0.38/1.01 点，TEST CW/OW 则低 3.08/2.56 点。因此，“收益来自真实 prior 语义而非随机 bias”的命题未得到一致支持。

此处随机的是 `K_oa` 矩阵，其他语义资产不变。

### 2.5 uniform prior 与 prior-off：常数先验不产生额外贡献

全 1 矩阵经过 centered 变换后为 0。`Rfull_uniform_prior` 与 `Rfull_prior_off` 的 **VAL CW、VAL OW、TEST CW、TEST OW 四套 scores.dat 均逐字节相同**。

这验证了常数 prior 在实现中等价于关闭 prior。默认真实 prior 相比关闭 prior，TEST CW/OW 只提高 0.19/0.21 点。

### 2.6 pair 移除与动态 head

去掉新增 head 的 pair 内积后，VAL CW 仅下降 0.30 点，TEST CW/OW 反而提高 2.96/1.84。动态改静态后，VAL CW 只下降约 0.015 点，TEST CW/OW 下降 0.23/0.33。

新增 pair 项和动态 query 都未呈现出强而一致的独立贡献；静态对照仍保留原生视觉分支。

### 2.7 Template-only 与完整 No-pair-knowledge

`Rfull` 本身使用 Template-only pair 资产，已有结果；没有独立的 pair-level LLM 模型与它比较。

`Rfull_nopair_head`、`Rfull_prior_off`、`Rfull_nopair_all` 分别移除两个机制或其组合。它们均保留 CPF composer，当前没有“删除所有组合知识、只保留对象／属性知识”的完整对照。

## 3. 设置与敏感性分析

### 3.1 Text-only／LLM-only／mixed prior

| prior 来源 | 对应模型 | 本批结果 |
|---|---|---|
| 文本／模板派生 prior | `Rprior`、`Rmvp`、`Rfeature`、`Rfull` 等 | 已完成 |
| 独立 LLM feasibility prior | `Rfull_llm_prior` | 无训练结果；缺独立生成的 feasibility 资产 |
| 文本与 LLM mixed prior | `Rfull_mixed_prior` | 无训练结果；依赖上述 LLM 资产 |

`Rtext` 是 Text K 表示替换组，不是“text-only prior”组。当前只能分析同一来源 prior 的接入方式和变换形式，尚无三种来源的质量排序。

### 3.2 描述删除 10／30／50%

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| 不删除 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| 删除 10% | `Rfull_noise10` | 5 | 26.82 | 21.47 | 32.05 | 24.84 |
| 删除 30% | `Rfull_noise30` | 5 | 30.19 | 24.10 | 31.49 | 24.17 |
| 删除 50% | `Rfull_noise50` | 5 | 32.11 | 23.78 | 31.76 | 25.69 |

**结论：删减描述总体优于默认完整描述。**删除 30%/50% 后，四项指标都提高；删除 50% 相比默认，VAL CW 为 +5.61，TEST CW/OW 为 +4.81/+5.23。

CW/OW 的测试曲线不是随删除比例单调上升，不能把 50% 解释为已确定的最佳比例。这些实验反映描述冗余或注入负担，未检验错误事实替换。

### 3.3 每类描述 1／3／5 条

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| 1 条 | `Rfull_count1` | 5 | 28.82 | 23.23 | 30.57 | 24.52 |
| 3 条 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| 5 条 | `Rfull_count5` | 5 | 25.03 | 17.62 | 30.21 | 23.04 |

1 条在四项指标上均优于默认 3 条；5 条的 VAL CW/OW 低于 3 条，但 TEST 高于 3 条。**增加描述数量没有带来一致收益，1 条是本批该组对照中更均衡的选择。**

### 3.4 OI 树深 1／2／3

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| 1 层：8 类 | `Rfull_depth1` | 5 | 27.25 | 22.08 | 30.31 | 23.31 |
| 2 层：8→5 类 | `Rfull_depth2` | 2 | 26.37 | 21.60 | 26.87 | 20.28 |
| 3 层：8→5→3 类 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |

1 层比默认 3 层的 VAL CW 高 0.75 点、TEST CW/OW 高 3.36/2.85；VAL OW 略低 0.20。2 层与 3 层接近且四项略低。**加深 OI 没有形成整体收益，较浅约束更有利于当前 CW 与测试表现。**

### 3.5 Mean／attention 文本聚合

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| mean | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| attention | `Rfull_attention_K` | 5 | 28.74 | 21.94 | 32.53 | 24.66 |

attention 的 VAL CW 为 +2.25、TEST CW/OW 为 +5.57/+4.20；VAL OW 略降 0.34。**有选择地聚合描述，比默认平均聚合更有效。**这里仅改 Text K 聚合，主 prior 固定；语义损失内部的描述 masked mean 未改动。

### 3.6 Centered／clamped-log prior

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| centered | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| clamped-log | `Rfull_clamped_log` | 5 | 26.27 | 21.48 | 30.93 | 23.76 |

clamped-log 的 VAL CW/OW 为 −0.23/−0.80，TEST CW/OW 为 +3.97/+3.30。**不同 prior 变换改变了验证—测试折中，主验证指标没有支持切换为 clamped-log。**

### 3.7 固定／可学习 prior 权重

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| 固定 λ | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| 有界可学习 λ | `Rfull_learn_lambda` | 2 | 25.94 | 21.84 | 26.54 | 19.91 |

可学习 λ 四项均低于固定 λ，TEST CW/OW 为 −0.41/−0.56。**当前参数化与训练预算下，学习标量 λ 没有收益。**

### 3.8 固定／延迟语义权重

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| 默认调度 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| delay=2、ramp=3 | `Rfull_delayed_beta` | 2 | 25.86 | 21.24 | 26.61 | 19.65 |

延迟调度四项均下降，TEST CW/OW 为 −0.34/−0.82。**本次延迟方案没有改善表现。**

### 3.9 GT-object／marginalized 属性损失

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| GT-object 条件属性损失 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| 对象边缘化属性损失 | `Rfull_marginal_attr` | 5 | 30.12 | 22.45 | 31.28 | 25.56 |

边缘化损失四项均提高：VAL CW 为 +3.63，TEST CW/OW 为 +4.33/+5.10。**这是最一致的结构性改善项之一，说明属性监督方式是当前模型的重要影响因素。**

### 3.10 动态／静态 head

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| 动态图像 query | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| 常数 query | `Rfull_static_head` | 2 | 26.48 | 22.16 | 26.72 | 20.13 |

两者近似持平，动态版仅小幅领先。**本批没有建立“新增 head 必须依赖图像动态 query 才有效”的证据。**

### 3.11 视觉注意力设置

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| R0：修订注意力 | `R0` | 6 | 32.22 | 19.76 | 30.26 | 22.32 |
| R0：原生注意力 | `R0_native_attention` | 5 | 29.75 | 20.71 | 32.38 | 26.10 |
| Full：修订注意力 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| Full：原生注意力 | `Rfull_native_attention` | 2 | 26.43 | 21.22 | 26.29 | 19.93 |

无知识 R0 中，修订注意力提高 VAL CW、降低两项 TEST；Full 中，修订注意力的四项均略高。**注意力设置的作用依赖模型背景，恢复原生注意力不是 Full 的有效修复项。**这两组不更换视觉 backbone。

### 3.12 描述温度 0.05／0.1／0.2

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| T=0.05 | `Rfull_descT005` | 5 | 26.56 | 21.29 | 30.89 | 23.91 |
| T=0.1 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| T=0.2 | `Rfull_descT020` | 2 | 26.39 | 21.45 | 26.61 | 20.27 |

0.05 的 VAL CW 与默认接近（+0.07），TEST CW/OW 提高 3.93/3.44；VAL OW 下降约 1.00。0.2 四项略低于默认。**较低温度显示测试改善信号，但验证结果不是全面改善。**该温度同时作用于 parent、OE、AE 分布，不是视觉注意力温度。

### 3.13 prior 与新增 head pair 项：二因素移除

| 设置 | 对应模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---:|---:|---:|---:|---:|
| pair 有／prior 有 | `Rfull` | 2 | 26.50 | 22.29 | 26.95 | 20.46 |
| pair 无／prior 有 | `Rfull_nopair_head` | 5 | 26.20 | 21.09 | 29.91 | 22.30 |
| pair 有／prior 无 | `Rfull_prior_off` | 2 | 25.64 | 21.01 | 26.76 | 20.25 |
| pair 无／prior 无 | `Rfull_nopair_all` | 2 | 26.39 | 21.82 | 27.32 | 20.91 |

- 保留 pair 时，加入 prior（`Rfull − Rfull_prior_off`）：VAL CW +0.86，TEST CW/OW +0.19/+0.21。
- 移除 pair 时，加入 prior（`Rfull_nopair_head − Rfull_nopair_all`）：VAL CW −0.19，TEST CW/OW +2.60/+1.39。
- 保留 prior 时，加入 pair（`Rfull − Rfull_nopair_head`）：VAL CW +0.30，TEST CW/OW −2.96/−1.84。
- 关闭 prior 时，加入 pair（`Rfull_prior_off − Rfull_nopair_all`）：VAL CW −0.75，TEST CW/OW −0.55/−0.66。

**prior 与 pair 的作用不满足简单相加；pair 项尤其没有获得一致的独立支持。**

## 4. 四个核心实验问题

### 4.1 结构化知识 K 是否提升 CPF？

**完整模型未提升 CPF；在固定 R0 接口内，个别知识入口有局部收益。**

- `Rfull − A_native`：TEST CW/OW 为 −9.73/−7.90。
- `Rfull − R0`：VAL CW 为 −5.72，TEST CW/OW 为 −3.30/−1.86，VAL OW 为 +2.53。
- `Rprior − R0`：OW 改善。
- `Rexternal − R0`：主验证指标改善，但 TEST CW 基本持平。

因此，结果支持“入口和使用方式决定收益”，不支持“默认完整知识方案整体提升 CPF”。

### 4.2 K 直接进入概率建模，是否优于仅作为 external loss？

以下均以只加 external 语义约束的 `Rexternal` 为参照：

| 对比（前者减后者） | ΔVAL CW | ΔVAL OW | ΔTEST CW | ΔTEST OW |
|---|---:|---:|---:|---:|
| `Rtext − Rexternal` | -7.46 | +0.09 | +0.34 | +0.51 |
| `Rprior − Rexternal` | -1.83 | +0.70 | +0.89 | +1.77 |
| `Rmvp − Rexternal` | -6.95 | +0.87 | -4.40 | -3.26 |
| `Rfeature − Rexternal` | -7.40 | +0.05 | +0.90 | +0.84 |
| `Rfull − Rexternal` | -6.93 | +1.16 | -3.17 | -2.41 |

**主验证指标未支持直接概率入口整体优于 external loss。**`Rprior` 的 OW、`Rfeature` 的 TEST 有局部优势；`Rmvp`、`Rfull` 在 TEST CW/OW 均低于 external-only。

其中 `Rexternal` 是 v2 的 JS 约束组；本结论不评价 LogiCZSL 原论文模糊逻辑损失。

### 4.3 Full Model 是否优于 naive combination？

`Rfull − Rmvp_logic`：

- VAL CW：−0.20；VAL OW：+0.08。
- TEST CW：+0.72；TEST OW：+0.71。

**测试集有小幅提升，主验证指标没有提升。**完整模型相对 naive combination 的优势有限。

原设计六条命题的本批回答：

| 原设计命题 | 对应模型比较 | ΔVAL CW | ΔTEST CW | ΔTEST OW | 回答 |
|---|---|---:|---:|---:|---|
| H > A | `Rfull − A_native` | -14.12 | -9.73 | -7.90 | 未达到 |
| H > B | `Rfull − Rexternal` | -6.93 | -3.17 | -2.41 | 未达到 |
| H > F | `Rfull − Rmvp_logic` | -0.20 | +0.72 | +0.71 | 测试小幅提升，主验证未提升 |
| G > E | `Rfeature − Rmvp` | -0.45 | +5.30 | +4.10 | 测试提升，主验证未提升 |
| H > G | `Rfull − Rfeature` | +0.47 | -4.08 | -3.25 | 主验证略升，测试下降 |
| E > A | `Rmvp − A_native` | -14.14 | -10.96 | -8.75 | 未达到 |
| G > A | `Rfeature − A_native` | -14.59 | -5.66 | -4.65 | 未达到 |

### 4.4 是否排除了描述数量、随机 prior 或数据泄漏解释？

| 问题 | 对应对照／检查 | 当前回答 |
|---|---|---|
| 是否只有描述数量产生作用？ | `Rfull` 对 `Rfull_shuffle` | 等数量错配后四项下降，支持语义对应关系有作用；该对照不单独测 token 长度。 |
| 真实 prior 是否优于随机 bias？ | `Rfull` 对 `Rfull_random_prior` | VAL 支持真实 prior，TEST 支持随机 prior；未形成一致优势。 |
| 常数 prior 是否产生虚假收益？ | `Rfull_uniform_prior` 对 `Rfull_prior_off` | 四套分数文件完全相同，常数 prior 等价于关闭。 |
| 知识是否使用了 VAL／TEST 数据构建？ | 资产生成代码、来源记录、冻结输入与选模规则 | 资产生成使用类别标签、固定描述和 GloVe；未读取 VAL／TEST 图像或成绩。当前流程核对未发现该类泄漏；完整数据来源独立审计未开展。 |

来源：[知识资产构建代码](../../code/revised_0601_20261014_fix/build_revision_assets.py)、[知识来源说明](../../inputs.json)、[已有导入与哈希核验](audit.json)。

## 5. 本次补充：seen/unseen 校准与错误分解

本节是 2026-10-16 从已保存的 scores.dat、candidates.json、labels.npz 和 split 只读重算的诊断，不改变 checkpoint 或原始 AUC。派生读数见 [后分析 JSON](v2单seed后分析读数_20261016.json)。

### 5.1 方法

- CW 候选限制为 TRAIN＋当前阶段合法组合；OW 使用全部 192 个组合。
- 在每个模型的 VAL 上搜索 bias ∈ [−5,5]，步长 0.05，以 HM 最大选定；并列时先取绝对值较小、再取较小 bias。bias 加到 unseen 候选分数。
- 将 VAL 选出的 bias 原样用于 TEST，再统计 seen/unseen、HM 和错误类型。没有用 TEST 选择 bias。
- 已核对 32 组的分数形状、标签映射与四评价 checkpoint 一致性。下表是一个固定工作点的 HM／准确率，不是 AUC，也不是在 TEST 上搜索的 best HM。

### 5.2 OW seen/unseen 校准

HM、Seen、Unseen 均为百分数。

| 模型 | VAL 选定 bias | TEST HM：bias=0 | TEST HM：VAL 固定 bias | TEST Seen | TEST Unseen |
|---|---:|---:|---:|---:|---:|
| `A_native` | 0.30 | 16.45 | 42.54 | 40.47 | 44.84 |
| `R0` | 5.00 | 1.57 | 39.92 | 48.97 | 33.69 |
| `Rprior` | 4.85 | 1.67 | 40.89 | 49.27 | 34.96 |
| `Rexternal` | 5.00 | 1.57 | 40.20 | 49.95 | 33.63 |
| `Rmvp` | 3.75 | 0.32 | 35.20 | 51.42 | 26.76 |
| `Rfeature` | 4.25 | 1.36 | 39.78 | 54.55 | 31.31 |
| `Rfull` | 3.70 | 0.63 | 35.91 | 52.10 | 27.39 |
| `Rfull_noise50` | 4.90 | 1.87 | 41.68 | 44.87 | 38.92 |
| `Rfull_attention_K` | 4.70 | 1.46 | 40.88 | 51.61 | 33.84 |
| `Rfull_marginal_attr` | 3.80 | 5.95 | 42.65 | 47.90 | 38.45 |

**结论：当前多数 R 模型在零 bias 时存在明显 seen/unseen 失衡；VAL 固定校准后 HM 显著恢复。**但校准没有改写其 AUC 曲线结论。

`Rfull_marginal_attr` 的固定校准 TEST HM 为 42.65，高于 `Rfull` 的 35.91；`Rfull_noise50` 为 41.68。`R0` 和 `Rexternal` 的 VAL 最优点落在搜索上界 +5，说明该网格下最优点触及边界。

原生余弦分数和 R 组 log-probability 的量纲不同，bias 数值不直接横比。

### 5.3 OW 错误分解

共 2914 张 TEST 图像。统计每个模型在上述 VAL 固定 bias 下的联合 top-1 预测：

- 仅对象错：属性正确、对象错误。
- 仅属性错：对象正确、属性错误。
- 两者都错：对象与属性同时错误。
- 四类互斥并覆盖全部图像；表内“对象／属性”由同一联合预测拆解，不是独立分类头的 argmax。

| 模型 | 组合正确 | 仅对象错 | 仅属性错 | 对象属性都错 | 组合准确率（%） |
|---|---:|---:|---:|---:|---:|
| `R0` | 1138 | 304 | 906 | 566 | 39.05 |
| `Rprior` | 1165 | 303 | 928 | 518 | 39.98 |
| `Rexternal` | 1147 | 309 | 898 | 560 | 39.36 |
| `Rmvp` | 1032 | 312 | 938 | 632 | 35.42 |
| `Rmvp_logic` | 1046 | 311 | 930 | 627 | 35.90 |
| `Rfeature` | 1150 | 301 | 877 | 586 | 39.46 |
| `Rfull` | 1051 | 309 | 936 | 618 | 36.07 |
| `Rfull_noise50` | 1195 | 279 | 949 | 491 | 41.01 |
| `Rfull_attention_K` | 1168 | 324 | 878 | 544 | 40.08 |
| `Rfull_marginal_attr` | 1217 | 247 | 847 | 603 | 41.76 |

**Rfull 的属性识别是更突出的错误来源。**仅属性错 936 张，约为仅对象错 309 张的 3.03 倍。边缘化属性损失将仅属性错降至 847 张、仅对象错降至 247 张；组合正确数从 1051 增至 1217。

### 5.4 无 bias 的对象边缘与 GT 对象属性读出

这两个诊断直接读取完整 OW 联合概率，均不加 seen/unseen bias：

| 模型 | 对象边缘分类准确率（%） | 固定 GT 对象的条件属性准确率（%） |
|---|---:|---:|
| `R0` | 65.51 | 32.40 |
| `Rprior` | 67.09 | 31.81 |
| `Rmvp` | 66.92 | 30.03 |
| `Rfeature` | 69.05 | 33.08 |
| `Rfull` | 66.88 | 30.37 |
| `Rfull_noise50` | 68.29 | 31.43 |
| `Rfull_attention_K` | 67.64 | 31.85 |
| `Rfull_marginal_attr` | 69.53 | 38.13 |

边缘化属性损失相对 `Rfull`：

- 对象边缘分类准确率：+2.64 点。
- 固定 GT 对象后的条件属性准确率：+7.76 点。

**这进一步支持属性损失是优先改进点。**此处 GT-object 读出只表示提供真实对象后的未校准属性预测，与上一表的校准联合准确率使用不同读出条件。

### 5.5 原方案剩余分析项目

| 项目 | 本次完成内容 | 尚缺内容 |
|---|---|---|
| Calibration／seen bias | VAL 固定 bias、TEST seen/unseen/HM | 同权重下 prior 开／关的完整读出归因 |
| Error decomposition | 对象单错、属性单错、双错；对象边缘与 GT-object 属性读出 | 独立定义并标注 compatibility error，不能把所有错组合都叫兼容性错误 |
| Failure case taxonomy | 保存文件含失败样本图像 ID，可供回查 | 图像人工核验：视觉模糊、语义歧义、知识不足、错误 prior |
| Cross-benchmark transfer | UT-Zappos 单 seed 结果整理完成 | MIT-States／C-GQA 的相同协议迁移 |
| prior 质量来源 | 文本派生、随机、uniform、关闭与变换对比 | 独立 LLM-only／mixed feasibility 资产与结果 |

## 6. 汇总结论

1. **原设计的整体提升目标未达成。**默认 Full 没有超过原生 CPF、external-only 或无知识 R0 的主验证指标。
2. **知识并非完全无效，但效果依赖入口。**prior 对 OW 有积极信号；语义约束在 MVP 上有小幅一致收益，接入 feature 模型后则出现验证／测试分歧。
3. **描述筛选和使用方式比堆叠模块更有价值。**减少描述、attention 聚合、边缘化属性损失是本批更清楚的改善项。
4. **动态 head、额外 pair 项和真实 prior 的独立必要性尚未建立。**静态对照近似持平；移除 pair 不导致明显退化；真实 prior 与随机 prior 的排序随验证／测试变化。
5. **错误分解把问题进一步定位到属性监督。**边缘化属性损失改善条件属性读出，并同时改善四项 AUC 和 VAL 固定校准后的 TEST HM。

这些结果对应“默认组合效果不足、部分入口有效、属性目标与描述聚合值得优先修正”，而不是“结构化知识本身无效”。

## 附录 A. 全部 32 组原始 AUC

本表保留四位小数，直接来自原始 summary；未重选 checkpoint。

| 模型 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---:|---:|---:|---:|---:|
| `A_native` | 5 | 40.6141 | 24.0402 | 36.6853 | 28.3609 |
| `R0` | 6 | 32.2161 | 19.7573 | 30.2555 | 22.3223 |
| `Rtext` | 5 | 25.9672 | 21.2094 | 30.4675 | 23.3836 |
| `Rprior` | 5 | 31.6019 | 21.8278 | 31.0198 | 24.6457 |
| `Rmvp` | 2 | 26.4760 | 21.9927 | 25.7268 | 19.6122 |
| `Rexternal` | 6 | 33.4303 | 21.1235 | 30.1276 | 22.8737 |
| `Rmvp_logic` | 2 | 26.6925 | 22.2013 | 26.2343 | 19.7523 |
| `Rfeature` | 5 | 26.0254 | 21.1747 | 31.0298 | 23.7132 |
| `Rfull` | 2 | 26.4968 | 22.2853 | 26.9538 | 20.4649 |
| `Rfull_nopair_head` | 5 | 26.2006 | 21.0944 | 29.9130 | 22.3015 |
| `Rfull_shuffle` | 2 | 24.4786 | 19.1959 | 21.3317 | 16.6590 |
| `Rfull_random_prior` | 5 | 26.1143 | 21.2755 | 30.0376 | 23.0252 |
| `Rfull_uniform_prior` | 2 | 25.6411 | 21.0103 | 26.7648 | 20.2543 |
| `Rfull_noise10` | 5 | 26.8238 | 21.4695 | 32.0476 | 24.8402 |
| `Rfull_noise30` | 5 | 30.1896 | 24.0974 | 31.4940 | 24.1673 |
| `Rfull_noise50` | 5 | 32.1087 | 23.7833 | 31.7630 | 25.6942 |
| `Rfull_count1` | 5 | 28.8246 | 23.2305 | 30.5733 | 24.5224 |
| `Rfull_count5` | 5 | 25.0347 | 17.6179 | 30.2060 | 23.0402 |
| `Rfull_depth1` | 5 | 27.2514 | 22.0830 | 30.3112 | 23.3130 |
| `Rfull_depth2` | 2 | 26.3656 | 21.5957 | 26.8656 | 20.2787 |
| `Rfull_attention_K` | 5 | 28.7426 | 21.9429 | 32.5257 | 24.6645 |
| `Rfull_clamped_log` | 5 | 26.2713 | 21.4815 | 30.9268 | 23.7615 |
| `Rfull_learn_lambda` | 2 | 25.9425 | 21.8428 | 26.5393 | 19.9063 |
| `Rfull_delayed_beta` | 2 | 25.8612 | 21.2366 | 26.6128 | 19.6459 |
| `Rfull_marginal_attr` | 5 | 30.1230 | 22.4465 | 31.2812 | 25.5608 |
| `Rfull_static_head` | 2 | 26.4818 | 22.1595 | 26.7191 | 20.1327 |
| `Rfull_native_attention` | 2 | 26.4342 | 21.2187 | 26.2887 | 19.9265 |
| `Rfull_prior_off` | 2 | 25.6411 | 21.0103 | 26.7648 | 20.2543 |
| `Rfull_nopair_all` | 2 | 26.3878 | 21.8185 | 27.3165 | 20.9135 |
| `Rfull_descT005` | 5 | 26.5622 | 21.2889 | 30.8862 | 23.9097 |
| `Rfull_descT020` | 2 | 26.3890 | 21.4492 | 26.6143 | 20.2682 |
| `R0_native_attention` | 5 | 29.7500 | 20.7136 | 32.3795 | 26.0986 |

## 附录 B. 数据与实现入口

- [原始结果页](README.md)
- [128 份原始指标汇总](raw_export/summary/summary.json)
- [逐轮 VAL CW 记录](训练记录.md)
- [原始分数、标签与样本 ID](raw_export/runs/)
- [本批实际 protocol](raw_export/protocol.json)
- [原始 split](raw_export/checks/split_pairs.json)
- [模型命名与开关对应](实验命名对照.md)
- [v2 公式与核心代码](../../复现公式与代码v2.md)
- [变体构造](../../code/revised_0601_20261014_fix/prepare_revision.py)
- [语义约束实际实现](../../code/revised_0601_20261014_fix/revision_model.py)
- [后分析参考协议](../../code/revised_0601_20261014_fix/posthoc_revision.py)
- [本次派生读数与 uniform／prior-off 哈希核验](v2单seed后分析读数_20261016.json)
- [本次只读分析脚本](../../../../tools/analyze_v2_seed3351.py)
- [分析发布与来源校验记录](../../../../provenance/updates/20261016_v2_seed3351_analysis.json)

在仓库根目录安装 NumPy 并拉取 Git LFS 数据后，可重新计算并核对派生 JSON：

```bash
python tools/analyze_v2_seed3351.py --verify-archived
```

脚本检查 32 组、128 份评价，核对报告中 13 个模型的 26 行诊断读数；只读归档分数，不启动训练、不修改原始导出。

原始 summary SHA256：`f7d5b0f8d3ec1fbb750d4636f93dff813496c354852dc144190c7086c837e6f5`。
