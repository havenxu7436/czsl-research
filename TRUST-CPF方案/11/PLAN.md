# TRUST-G₀ 三数据集总实验 v1 · 10 epoch

日期：2026-10-28。发布目录：`trustg0_three_v1_10ep`。服务器 GPU：**仅 0、1、2**。

## 1. 本轮目标与体系定位

本轮以历史 VG-CPOD V1（记作 G₀）为底座，完成 UT-Zappos50K、MIT-States、C-GQA 的独立流水线复现。论文主问题是：**由模型响应预测的关系效用，能否在相同关系预算和相同识别权重下提高组合识别，并迁移到不同数据集？**

G₀ 的模型文件与历史 `trustcpf_abc_v1_10ep/trustcpf/g0.py` 字节一致；沿用静态 CP 修正、全局视觉读出、任务辅助权重 0.075 和 CP 学习率系数 0.3。它与 Wu 等人 ICCV 2025 CPF 的论文复现是两个实验体系。本轮报告使用“G₀ / TRUST-G₀”，保留 TRUST-CPF 作为研究路线名称。

历史 ABC 的 Context 相对 Uniform，在 UT 的 CW/OW AUC 平均增加 0.097/0.109 个百分点，三个分类器 seed 均为正；其均值与 Matched 基本持平。历史三个分类器共用效用预测器。本轮使用 **5351、5352、5353 三套独立完整流水线**，每套分别训练教师、拟合效用模型、训练正式识别器。G₀ 架构保持一致，权重在各数据集重新训练。

参考：历史 `TRUST-CPF_ABC整批10epoch_完整结果分析_20261027.md`、`CPF基线复现诊断_历史源码与本轮结果核对_20261028.md`；已核对其他对话“主experiment (3)”及两条“主experiment (2)”中的划分修复、数据核验和三数据集安排。

## 2. 一次队列完成的实验

| 方法 | 训练权重 | 推理策略 | 对应问题 |
| --- | --- | --- | --- |
| G₀ | 原始 G₀ | 无关系残差 | 底座能力 |
| Capacity | 冻结 G₀，训练通用残差头 | 无知识附着 | 参数容量贡献 |
| Dense | 与 Uniform 相同 | 所有关系开启 | 分配与稠密使用的差异 |
| Uniform | 公共关系权重 | 每条边概率 0.5 | 等预算基线 |
| Matched | 与 Uniform 相同 | 27 参数全局效用预测 | 效用监督贡献 |
| Context | 与 Uniform 相同 | 51 参数情境效用预测 | 情境条件效用 |
| Context-Shuffle | 与 Uniform 相同 | 打乱拟合教师的情境输入 | 情境对应关系贡献 |
| Matched-Perm × 5 | 与 Uniform 相同 | 关系类型内置换概率 | 全局效用与具体关系的对应 |
| Context-Perm × 5 | 与 Uniform 相同 | 关系类型内置换概率 | 条件效用与具体关系的对应 |

合计 17 个方法。G₀、Capacity、Dense 为确定性读出，各评估一次；其余 14 个策略使用三个固定图库种子，每个图库 16 张图。因此每套流水线有 45 份读出，覆盖 VAL/TEST × CW/OW，共 180 行指标；三数据集 × 三 seed 共 **1620 行指标**。

每套流水线包含 15 个教师（3 次初始化 × 5 折）。Q1–Q3 的 9 个教师用于效用拟合，Q4 的 3 个教师提供开发诊断，Q5 的 3 个教师用于独立审计。本轮超参数固定，Q4 不执行调参。完整队列：3 个缓存任务、9 个正式训练任务、135 个教师任务、9 个效用拟合任务、9 个综合评估任务，共 **165 个任务**。

所有神经网络训练阶段均为 10 epoch：G₀ 10 epoch；关系参考训练 10 epoch；再从该参考权重开始适配 10 epoch。正式 Uniform 与 Capacity 分别执行两个关系阶段。所有效用策略复用同一份 Uniform 适配权重。教师固定取最后一轮；正式 G₀、Uniform 适配、Capacity 适配保留 epoch 0–10，并按 VAL CW native AUC 选模。效用模型通过闭式岭回归求解。

总计 144 次 G₀ 训练、306 次关系阶段训练，即 450 个 10-epoch 阶段。训练预算包含独立效用模型的教师成本；汇总文件记录各阶段实际耗时。

## 3. 数据协议与五折划分

| 数据集 | 属性 / 对象 | TRAIN / VAL / TEST 图像 | TRAIN 组合 | 知识边 OI / OE / AE |
| --- | --- | --- | --- | --- |
| UT-Zappos50K | 16 / 12 | 22998 / 3214 / 2914 | 83 | 12 / 36 / 48 |
| MIT-States | 115 / 245 | 30338 / 10420 / 12995 | 1262 | 245 / 735 / 345 |
| C-GQA | 413 / 674 | 26920 / 7280 / 5098 | 5592 | 674 / 2022 / 1239 |

读取官方 `compositional-split-natural` 与对应 metadata。启动时扫描图像，核对之前已验证的索引、metadata 和 split 哈希；UT 两条历史路径映射随包提供。自然 VAL/TEST 包含 seen 组合，只要求它们的 **unseen 子集互斥**；图像路径不能跨集合重复。

候选组合采用规范索引：

```math
\mathrm{id}(a,o)=aN_o+o.
```

模型张量为 `[batch, object, attribute]`，取值显式使用 `joint[:, id % N_o, id // N_o]`。CW 候选为 TRAIN 与当前自然 split 组合的并集；OW 为全部属性对象笛卡尔积，不做可行性过滤。

教师仅划分自然 TRAIN 的组合图。沿用 `coverage_greedy_v2`：度为 1 的原语所连组合进入所有折的公共 Inner；其余组合按最小合法折分配为 Query。对任一端点，最多只有一个折会包含它的全部其他邻边；一条边两个端点最多排除两个折，五折中始终存在合法位置。每折均检查 Query 原语在 Inner 中出现。

本次真实索引验证：UT 公共组合 0 个，MIT 6 个，C-GQA 178 个。C-GQA 每折 Query 为 1082 或 1083 个组合。五折 Query 互斥，公共组合与五折 Query 的并集恰好为自然 TRAIN。

每个 Inner 组合按固定 seed 留出少量 seen 审计图像，至少保留一张训练图像；Query 图像再分为效用测量与独立审计两部分。教师训练完全使用 Inner 训练图像，Q5 决策评估使用留出的 seen 图像及 Query 审计图像。所有索引保存在 `folds.json` 与 `response.pt`，可复查图像级交集。

## 4. G₀ 与关系残差的完整前向

记归一化操作为 n(v)=v/max(‖v‖₂,ε)。CLIP ViT-L/14 冻结，图像全局向量为 g，patch 向量为 vₗ；属性、对象、组合文本向量分别为 eₐ、eₒ、eₒₐ。UT 文本模板与历史实验相同；MIT/C-GQA 使用通用属性对象模板，完整提示词保存在缓存 `prompts.json`。

G₀ 对象与条件属性 logit：

```math
b_o(x)=\kappa g^\top e_o+(g^\top B_o)(e_o^\top C_o)^\top.
```

```math
b_{oa}(x)=\kappa g^\top e_{oa}+(g^\top B_c)(e_{oa}^\top C_c)^\top.
```

```math
u_a=A_u e_a,\quad v_o=A_v e_o,\quad
c_{oa}(x)=b_{oa}(x)+\rho\kappa\sum_{r=1}^{R}(g^\top W_c)_r(v_o)_r(u_a)_r.
```

这里 R=32、adapter rank=8、κ=20、ρ=0.15；静态门控恒为 1。联合 log 概率：

```math
s_{oa}(x)=\log\mathrm{softmax}_o(b(x))_o+
\log\mathrm{softmax}_a(c_o(x))_a.
```

给定训练候选集合 C，组合、对象和条件属性损失为：

```math
\ell_{\rm pair}=-s_{o_y a_y}+\log\sum_{(a,o)\in C}\exp s_{oa},\qquad
\ell_o=-\log p(o_y\mid x),\qquad
\ell_c=-\log p(a_y\mid x,o_y).
```

```math
\mathcal L_{\rm task}=\frac{1.5}{1.075}
\mathbb E[\ell_{\rm pair}+0.075(\ell_o+\ell_c)].
```

G₀ 使用 AdamW，adapter 学习率 10⁻³、CP 参数学习率 3×10⁻⁴，逐 step cosine，权重衰减 10⁻⁴，梯度范数上限 1。关系阶段冻结 G₀。

对象查询局部特征与关系视觉上下文：

```math
\alpha_{ol}(x)=\mathrm{softmax}_l(e_o^\top v_l/0.07),\quad
f_o(x)=\mathrm{n}\!\left(\sum_l\alpha_{ol}(x)v_l\right),\quad
h_o(x)=\mathrm{n}\!\left(\mathrm{n}(g)+0.5\mathrm{n}(f_o(x))\right).
```

对边 e，类型 t(e)∈{OI,OE,AE}、目标原语 j(e)、源文本向量 kₑ、目标初始向量 z⁰ⱼ：

```math
m_e=(I+L_{t(e)}R_{t(e)})(k_e-z^0_{j(e)}),\qquad
z_j(b)=\mathrm{n}\!\left(z_j^0+0.1\sum_{e:j(e)=j}\frac{b_e}{d_{j,t(e)}}m_e\right).
```

d 为目标与类型对应的原始边数，L、R 的内部秩为 16。实现用有序入边 gather 求和，避免构造大型稠密 incidence 矩阵。

定义 rank=64 的三线性项：

```math
T(x,o,a;z)=\sum_r
\mathrm{n}(P_vh_o(x))_r\mathrm{n}(P_oz_o)_r\mathrm{n}(P_az_a)_r.
```

```math
\widetilde b_o=b_o+g^\top(z_o-z_o^0),\qquad
\widetilde c_{oa}=c_{oa}+T(x,o,a;z)-T(x,o,a;z^0).
```

重新计算对象与条件属性 softmax 得到关系联合概率。关系残差倍率固定为 1。关闭所有边时恢复 G₀，数值测试容差为浮点舍入级别。关系训练使用 Bernoulli(0.5) 图，AdamW 学习率 3×10⁻⁴，首 epoch warmup 后 cosine，权重衰减 10⁻⁴，并加入：

```math
\mathcal L=\mathcal L_{\rm task}+10^{-3}\frac{\sum_t\|L_tR_t\|_F^2}{768^2}.
```

Capacity 沿用历史实现：对象使用 n(z⁰ₒ+0.1L₀R₀z⁰ₒ)，属性使用 n(z⁰ₐ+0.1(L₁R₁+L₂R₂)z⁰ₐ)，条件残差为通用 T 项；不使用知识源及关系附着。

## 5. 响应、效用监督与精确干预计算

每个教师的 Inner 支持集按组合均衡抽样，每组合最多两张图像。大数据集先覆盖所有原语，再补足 256 个组合；覆盖所需组合数可以超过 256。Query 测量同理，以 128 个组合为目标、每组合最多两张。每个被选组合总权重相同，组合内图像权重均分。

给定独立背景图 b，构造 b⁺ᵉ 与 b⁻ᵉ，其他边完全相同。测量使用 4 个背景图，候选集合固定为自然 TRAIN。记候选归一化 log 概率为 ℓ，单图效用：

```math
\Delta_e(x,y;b)=\ell_{b^{+e}}(y\mid x)-\ell_{b^{-e}}(y\mid x)
=\mathrm{NLL}_{b^{-e}}-\mathrm{NLL}_{b^{+e}}.
```

正值表示加入关系降低损失。对 Inner 的全部支持图像、目标原语拥有者图像分别计算四个统计量：效用均值、效用标准差、候选 log-score 差的中心化 RMS、预测熵差，得到 8 维响应 hₑ。

```math
\mathrm{RMS}_e^2=
\mathbb E_{x,b}\!\left[\frac{1}{|C|}\sum_{c\in C}
\left(D_e(x,c;b)-\frac{1}{|C|}\sum_{c'\in C}D_e(x,c';b)\right)^2\right].
```

其中 D 是两次干预的联合 log-score 差。均值、方差和 RMS 均按组合均衡权重计算。效用标签由 Query 测量图像产生：

```math
y_{je}=\mathbb E_{(x,y)\sim Q_j,b}[\Delta_e(x,y;b)].
```

情境组通过对象文本向量的确定性平衡二聚类获得。UT 使用历史精确枚举；MIT/C-GQA 使用确定性平衡迭代。路由只依赖冻结 G₀ 的对象概率：

```math
g(x)=\arg\max_{g\in\{0,1\}}\sum_{o\in\mathcal O_g}p_{G_0}(o\mid x).
```

分别在两个情境上计算 hⱼgₑ 与 yⱼgₑ，并记录 Query 加权质量 wⱼg。某情境缺少支持图像时，响应回退至全局响应；该情境没有 Query 测量图像时权重为 0。

逐边干预只改变一个原语节点。实现精确更新受影响的对象行或属性列。若背景候选 logit 为 a，干预变化为 δ，p=softmax(a)，则：

```math
Z'=1+\sum_{c\in I}p_c(\exp\delta_c-1),\qquad
\log p'_y-\log p_y=\delta_y-\log Z'.
```

```math
H(p')=\frac{H(p)-\sum_{c\in I}p_c(\exp\delta_c-1)\log p_c
-\sum_{c\in I}p_c\exp\delta_c\delta_c}{Z'}+\log Z'.
```

I 是变化列集合，未列入的 δ=0。对象干预更新对应对象下的条件属性分母；属性干预更新每个对象下该属性及其条件分母。这与完整重算相等；测试覆盖三种边类型、全开/全关/随机背景，并验证 C-GQA 完整维度。

## 6. 效用预测与等预算分配

每套流水线的 Q1–Q3 共 J=9 个教师，单独估计响应均值 m、标准差 s，以及各类型效用尺度 qₜ。qₜ 为各教师类型内中心化标签的均方根；输入标准差下限 10⁻⁶，效用尺度下限 10⁻¹²。

Matched 的类型分块设计向量：

```math
\phi_{je}=\bigoplus_{t=0}^{2}\mathbf1[t(e)=t]
\left[1,\,(h_{je}-m)\oslash s\right]\in\mathbb R^{27}.
```

```math
\widehat\beta=(\Phi^\top\Phi+J\lambda I)^{-1}\Phi^\top\widetilde y,
\qquad \widetilde y_{je}=y_{je}/q_{t(e)},\quad \lambda=10/3.
```

因此正则项为 30I。Context 在每种类型的全局 9 维向量后拼接 8 维情境差分；差分尺度按 Query 情境质量加权估计：

```math
s_{\Delta,k}^2=\frac{1}{JE}\sum_{j,g,e}w_{jg}(h_{jge,k}-h_{je,k})^2.
```

```math
\psi_{jge}=\bigoplus_{t=0}^{2}\mathbf1[t(e)=t]
\left[1,\,(h_{je}-m)\oslash s,\,(h_{jge}-h_{je})\oslash s_\Delta\right]\in\mathbb R^{51}.
```

```math
\widehat\beta_c=(\Psi^\top W\Psi+J\lambda I)^{-1}\Psi^\top W\widetilde y_c.
```

W 对每条关系重复对应教师情境质量；预测恢复原类型尺度。Context-Shuffle 在固定选出的 5/9 个拟合教师中交换两个情境输入，标签与质量保持原位置。

预测效用在每种关系类型内部标准化为 rₑ。分配解为以下严格凹目标的唯一最优解：

```math
\max_{0<\pi_e<1}\sum_{e\in E_t}
\left[r_e\pi_e-\pi_e\log\pi_e-(1-\pi_e)\log(1-\pi_e)\right],
\quad \sum_{e\in E_t}\pi_e=|E_t|/2.
```

一阶条件 rₑ−logit(πₑ)−γₜ=0，得到：

```math
\pi_e=\sigma(r_e-\gamma_t).
```

按类型二分求 γₜ。类型内效用完全相同时恢复 π=0.5。Matched 为一套概率，Context 为两套情境概率。固定类型内置换保持概率多重集、期望边数和幅度完全一致。

对每个图库，以共享均匀随机数实现不同策略的共同随机图。推理先对每张图的候选概率归一化，再做 16 图平均：

```math
\overline p(c\mid x)=\frac1{16}\sum_{m=1}^{16}p(c\mid x,b^{(m)}_{g(x)}),
\qquad s^{\rm out}(c\mid x)=\log\overline p(c\mid x).
```

每类型边数约束是期望约束；单张图按独立 Bernoulli 抽样，不要求每张图恰好包含半数边。

## 7. 指标、归因与交付结果

正式报告 CW/OW 的 native/exact AUC、HM、seen/unseen 上限、VAL bias 下 TEST HM、NLL、ECE。AUC/HM 单位均为 0–100。native 保持历史有限偏置扫描，exact 对所有分数翻转阈值积分。TEST HM 使用对应 VAL 选出的 bias；同时保存 TEST 可达最佳 HM，字段分别命名。

主比较为 Context−Uniform、Matched−Uniform、Context−类型内关系置换均值、Context−Context-Shuffle。G₀−Uniform 用于描述附加关系分支的整体作用，Capacity 与 Dense 用于解释额外容量和关系分配。Context−Matched 单列，用于判断情境是否提供均值增量或重复间一致性。论文采用全局或情境主线，依据三个数据集的实际配对结果选择。

统计单位为完整流水线 seed：先在同 seed 内平均三个图库，置换对照再平均五次置换，然后汇总三个 seed 的均值、样本标准差与逐 seed 差。图库、置换、教师初始化均不作为额外独立正式实验样本。Q5 每个教师使用一个固定 16 图库评估决策结果，另保存效用预测 Spearman、RMSE 和方向准确率。

主要文件：`summary.csv`（1620 行）、`main_mean_std.json`、`report.json`（配对差与完成状态）、`q5_decision_summary.json`、`compute_summary.json`；各 seed 下保留权重、10 epoch 轨迹、响应/效用标签、概率、逐图像分数及审计索引。`recognition/*/*/*.npz` 足以独立复算 AUC/HM/NLL/ECE。

运行中 `complete=false` 表示尚有阶段未完成。结束标准为 `ALL_G0_TRUST_EXPERIMENTS_COMPLETE`、`report.json.complete=true`、9 套流水线和 1620 行正式指标。

## 8. 实现文件与运行成本

| 内容 | 实现 |
| --- | --- |
| 历史 G₀ 前向与损失 | `trustg0/g0.py` |
| 图残差、Capacity、参数保存 | `trustg0/model.py` |
| 官方索引、覆盖五折、图像留出 | `trustg0/data.py` |
| CLIP 全局/patch 缓存 | `trustg0/cache.py` |
| 10 epoch 训练与断点续训 | `trustg0/train.py` |
| 精确局部干预与响应统计 | `trustg0/response.py` |
| 岭回归、情境预测、概率分配 | `trustg0/utility.py` |
| CW/OW 混合读出与统计 | `trustg0/evaluate.py`、`metrics.py` |
| 完整流程、汇总 | `trustg0/workflow.py` |
| 0/1/2 动态单卡队列 | `scripts/run_queue.py` |

CLIP 特征仅提取一次，所有 seed/教师共享 FP16 global 与 patch memmap；组合文本保持 FP32。三数据集缓存约 46 GiB，启动根据实际图像数和已有缓存计算剩余需求，并保留 12 GiB。每个模型只保存可训练/冻结模型参数，文本与图像缓存不重复写入检查点；最终下载包排除共享缓存。

队列只接受 0、1、2 的子集；每卡同时一个任务，自动分配已满足依赖的任务。已有完成任务会跳过；训练按完整 epoch 续跑，关系测量按图像批次续跑。任务失败时停止派发新任务，等待已启动任务结束，修复后重跑相同队列。已有训练、缓存和输出均可复用。

本地验证包含真实自然划分与 9 套五折检查、历史 G₀ 文件哈希、数值干预一致性、断点恢复、完整 10-epoch 合成流水线和 C-GQA 完整维度 GPU 前向/反向。真实三数据集训练由下述服务器命令执行；本地验证不替代真实数据集指标。
