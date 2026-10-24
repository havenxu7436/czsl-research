本文件保留六路线后续计划的原始研究安排。第一批 Align 已完成，见 [本阶段分析](analysis/README.md)；Competition 已完成，见 [06 阶段](../06/README.md)。R5 等后续路线仍为提案。

# TRUST-CPF 后续实验方案：从可迁移效用到组合识别收益

版本：2026-10-24。当前起点：Strong CPF＋TRUST-Compact v1；数据集：UT-Zappos；CLIP ViT-L/14；训练 seed：3351、3352、3353；物理 GPU 0/1/2。

**建议保留 TRUST 的效用监督主线，按“部署校准 → 组合竞争效用 → 跨情境确认”推进。立即执行 R1 的三组、三 seed 实验；后续方法深化优先选择 R4。R3 和 R5 用来解决实际测出的读出或预测问题，R6 用来建立与 Capacity 相同的表达起点。风险惩罚 R2 暂列扩展研究。**

研究问题集中为：**在强 CPF 上，能否从 TRAIN 内留出组合的识别收益学习关系使用价值，并通过更好的关系分配改善未见组合识别？**

本方案综合五轮完整结果、现有代码和用户提供的第四版 PDF。文中的新实验均为后续计划；已有数值来自本地归档分析。PDF 的六条路线作为候选方法材料，执行安排按实际数据重新组织。

## 1. 历史结果共同指向什么

### 1.1 五轮结果的连续证据

下表 AUC 均为现有工程的 0–100 显示尺度，数值为三个训练 seed 的均值。不同基底之间用于定位问题，同一轮匹配组之间用于判断机制贡献。

| 实验轮次 | 完整结果中的关键数字：CW / OW AUC | 对后续方案的作用 |
| --- | --- | --- |
| 弱基底 TRUST v1 | robust 9.901 / 8.172；capacity 13.902 / 11.240；drop 10.915 / 9.047 | 说明旧基底和效用选择同时存在问题；这组结果保留为发展记录 |
| 强基底条件残差 v2 | Base 34.438 / 25.140；Knowledge 35.055 / 25.740；Shuffle 35.307 / 25.915 | 新增分支能改善识别，真实附件的额外价值需要同结构对照 |
| Strong TRUST v1 | r0 34.512 / 25.184；robust 35.201 / 26.149；drop 34.988 / 25.908 | 强基底恢复后，TRUST 相对 CPF 有增益，相对同预算随机图有小幅收益 |
| 模型响应条件化 | conditioned 34.916 / 25.958；原 robust 35.201 / 26.149 | 加入八维响应的大 MLP 没有提高终端识别；响应信息的使用方式比扩大网络更关键 |
| Compact v1 | compact 34.922 / 25.931；drop 34.988 / 25.908；打乱标签 34.796 / 25.664 | 紧凑效用预测改善了代理决策，终端识别仍接近随机图 |

完整证据范围包括：弱基底 21 组主结果及恢复的 600 条 OW 噪声记录；v2 九组训练；Strong v1 的 30 组模型／读出结果、12 份效用审计、1800 条噪声记录；条件化与 Compact 的新增训练、复用对照、逐图像预测和训练历史。重复引用的对照仍计为同一批实验。

当前同工程最强的两个简单参照为：

| 方法 | CW AUC | OW AUC | 含义 |
| --- | --- | --- | --- |
| Capacity | 35.805 ± 0.905 | 26.400 ± 1.847 | 无关系附件的完整可训练三元分支 |
| Dense | 35.646 ± 0.926 | 26.288 ± 1.818 | 使用全部知识关系的现有图分支 |
| 原 TRUST robust | 35.201 ± 0.571 | 26.149 ± 1.109 | 原效用均值与风险扣分选图 |
| Compact | 34.922 ± 0.746 | 25.931 ± 1.380 | 27 参数的期望效用预测与固定预算选图 |

这里的强 CPF 是本项目的 CLIP 条件概率＋静态 CP 实现。论文实验应另列官方 CPF 的可比复现，明确骨干、特征、训练和评价协议。

### 1.2 已经值得保留的部分

**效用监督具有可学习的内容。** Compact 的 Q5 边均值 Spearman 为 0.207、0.310、0.361，平均 0.293；打乱标签平均为 −0.022。三个 seed 的同预算代理遗憾相对类型常数分别下降 30.3%、12.1%、12.9%。这些数字支持继续研究关系收益的预测与迁移。

**正确关系内容在强基底上出现过正向信号。** Strong robust 相对完整 shuffled 流程，OW 分别提高 0.541、1.087、0.184 点。这说明知识内容值得保留；由于 shuffled 同时重训了教师、效用模型和参考头，下一轮用概率置换进一步定位“把概率分给哪条边”的作用。

**部署方式是当前直接可改的问题。** 15 个真实知识教师和 3 个 reference 的采边中心均为 0.5，效用测量也使用该背景；现有部署的总期望边数却只有 19.05、21.89、19.33，约占 96 条边的 20%–23%。seed3352 的 AE 概率标准差仅 0.0084，与同密度随机图每次期望只改变 0.238 条 AE 边。

**组合竞争决定后续目标设计。** 多轮结果中，给定真实物体后，未见图像的属性错误约 89% 落入该物体的已见搭配。该分析指向已见／未见竞争；正式实验仍使用完整候选集和联合预测。

### 1.3 当前投入如何调整

| 既有发现 | 实验安排 |
| --- | --- |
| Compact−Drop 为 −0.066 CW、+0.023 OW | 先让已学排序产生明确的概率差异，检查真实关系分配收益 |
| 输入改成类型内标准化后，三个 Q5 排序均下降 | 保留当前八维输入标准化；R1 只校准部署输出尺度 |
| 多数方法选中 epoch 0–2，Dense 三个 seed 都选 epoch 0 | 保留 3 epoch 适配，同时单列 epoch 0；暂不延长训练 |
| Strong 噪声 OW ΔRAUC：mean +0.0012，robust −0.0328 | 主要资源用于平均识别收益；风险和抗噪作为后续独立贡献 |
| semantic／unmeasured 的多数审计未超过简单对照 | 聚焦模型条件下的跨组合效用，未测边迁移放到最终方法确认 |
| 同样参数数目的 Capacity 使用完整三元项，TRUST 使用差分项 | 将“关系选择”和“表达容量”分开解释，必要时执行 R6 |

## 2. 六条路线的取舍与三批执行安排

| PDF 路线 | 本方案定位 | 具体安排 |
| --- | --- | --- |
| R1：部署概率校准 | **立即执行** | 当前 Compact 保持不变，统一半密度，校准排序强度 |
| R4：组合竞争效用 | **后续方法主方向** | 将 TRAIN 内双组竞争面积改善作为关系监督，与 NLL 目标比较 |
| R3：混合读出效用 | 目标对齐分支 | 第一批顺带测单图／16 图效用；读出差异明显时替换第二批目标实验 |
| R5：相对效用监督 | 低成本拟合分支 | 实测分配有效、预测分配较弱时，直接在 Compact 上投影训练残差 |
| R6：Capacity 共同起点 | 表达层改进 | 选图已有收益而总体仍受表达限制时，接入同一个 Capacity |
| R2：方向敏感风险 | 扩展研究 | 均值主线确定后，再研究误差信息对概率更新的独立价值 |

常规执行只组织三批：

| 批次 | 核心问题 | 识别实验规模 | 主要复用 |
| --- | --- | --- | --- |
| A：TRUST-Align | 现有效用能否通过合适的部署概率改善识别？ | 3 方法 × 3 seed，最多 9 次短适配 | 全部旧教师、Compact、reference、特征和知识库 |
| B：TRUST-Competition | 面向组合竞争的监督是否比 NLL 更适合选关系？ | 5 方法 × 3 seed，共 15 组结果；其中 Uniform 可复用 A | 图像特征、知识库、最终 reference；新建双组风险教师 |
| C：方法确认 | 效用监督、模型响应和关系分配能否在新情境复现？ | 每个数据设置 6 方法 × 3 seed，共 18 组结果 | 已锁定实现与选择规则 |

A 是当前应交付代码并上服务器运行的完整队列。B 是默认的下一项方法研究。若 A 已给出清楚的收益，B 可作为论文中的监督目标比较；若 A 显示主要差异来自读出或预测，则用第 6 节对应方案替换 B，保持每批只有一个主要方法问题。

## 3. 公共模型与监督定义

### 3.1 保留现有强基底

关系类型为 OI、OE、AE，边数分别为 12、36、48。原词锚点为 $`e_v`$，附件向量为 $`h_e`$，边开关为 $`b_e`$，固定完整图入度为 $`d_{v,t}`$。关系分支保持：


```math
m_e=(I+L_{t(e)}R_{t(e)})(h_e-e_{v(e)}),
\qquad
z_v(b)=\mathop{\mathrm{norm}}\nolimits\left(e_v+0.1\sum_{e:v(e)=v}\frac{b_e m_e}{d_{v,t(e)}}\right).
```


令 $`T_\psi(x,o,a;z)`$ 为当前代码的归一化三元评分，保留配置增益 $`\lambda_g=\texttt{alpha}/\texttt{tg}`$：


```math
s_o=s_o^0+\lambda_g g(x)^\top(z_o-e_o),
\qquad
c_{oa}=c_{oa}^0+\lambda_g\left[T_\psi(x,o,a;z)-T_\psi(x,o,a;e)\right].
```


当前默认 $`\lambda_g=1`$；归一化沿用向量除以其 L2 范数、分母下限为 10⁻⁸ 的实现。A、B 均沿用现有残差结构、冻结 CPF 和关系头优化设置。


```math
J_{oa}=\log\mathop{\mathrm{softmax}}\nolimits_o(s)_o+
\log\mathop{\mathrm{softmax}}\nolimits_a(c_{o\cdot})_a,
\qquad
P_{\mathcal C}(c\mid x,b)=
\frac{\exp J_c(x,b)}{\sum_{c'\in\mathcal C}\exp J_{c'}(x,b)}.
```


张量轴沿用 `[batch, object, attribute]`，组合 ID 为 `object_id * n_attr + attribute_id`。正式读出为 16 张图各自在相应 CW／OW 候选集归一化后，再平均概率。

### 3.2 原效用是固定模型下的局部关系收益

对于同一教师、同一 Query、同一候选和权重，定义：


```math
R_D(\theta,b)=-\sum_{i\in D}w_i\log P_{\mathcal C}(y_i\mid x_i,b),
\qquad
F_{D,1}(\pi)=\mathbb E_{b\sim q_\pi}R_D(\theta,b),
```


```math
u_e(\rho)=\mathbb E_{b_{-e}\sim q_{\rho,-e}}
\left[R_D(\theta,b_{-e},0)-R_D(\theta,b_{-e},1)\right]
=-\left.\frac{\partial F_{D,1}(\pi)}{\partial\pi_e}\right|_{\pi=\rho}.
```


最后一步来自对 $`b_e=0,1`$ 两种情况展开期望。模型和样本保持固定时，该身份成立；它给出原测量中心附近的下降方向。

已核对 `stages.py`：每个 episode 先均匀抽 Query 组合、再抽组合内图像，权重为组合等权；随后独立地在每种类型内均匀抽边。因此，各边聚合均值估计同一个 Query 抽样总体下的期望效用，满足 R5 的共同目标条件。每条边的实际有限样本不同，聚合均值仍有采样误差。

### 3.3 当前 Compact 的确切实现

每条边使用已有八维模型响应 $`h_{je}`$；$`c,s`$ 为拟合折逐维统计，$`a_t`$ 为拟合折中该类型边均值的标准差：


```math
X_{je}=\mathop{\mathrm{onehot}}\nolimits(t(e))\otimes
\left[1,(h_{je}-c)\oslash s\right]\in\mathbb R^{27},
```


```math
\widehat\beta_\alpha=
\operatorname*{argmin}_\beta
\left\{\sum_{j,e}\left(X_{je}^\top\beta-\frac{\bar y_{je}}{a_{t(e)}}\right)^2+
\alpha\|\beta\|_2^2\right\},
\qquad
\widehat\mu_{je}=a_{t(e)}X_{je}^\top\widehat\beta_\alpha.
```


这是已经运行的 27 参数、float64 岭回归，包括截距正则；输入标准差下限为 10⁻⁶，标签尺度下限为 10⁻¹²。PDF 因未读取源码而将线性头列为可选实现；本方案直接以现有实现为基础。

第一批保留已选 $`\alpha=10,100,100`$ 及最终五折拟合模型。旧 Q5 保留为开发诊断；正式确认安排在新教师情境及第二数据集。

## 4. 第一批：TRUST-Align，一次完整队列

### 4.1 三个正式组

| 组名 | 概率规则 | 回答的问题 |
| --- | --- | --- |
| Uniform-0.5 | 每条边 $`\pi_e=0.5`$ | 测量背景中心下的关系分支能力 |
| TRUST-Align | 当前 Compact 预测，类型内尺度校准，统一强度 | 已学效用排序的部署价值 |
| Permuted-Align | 按类型置换 Align 的最终概率 | 概率与具体关系对应的价值 |

三个组固定类型期望边数为 **OI=6、OE=18、AE=24**。同密度解决平均关系用量的可比性；非均匀策略仍会改变背景分布，因此同时测量实际风险变化。

### 4.2 概率公式与求解

对于 reference 上的预测效用，计算：


```math
c_t=\frac1{N_t}\sum_{e\in E_t}\widehat\mu_e,
\qquad
s_t=\sqrt{\frac1{N_t}\sum_{e\in E_t}(\widehat\mu_e-c_t)^2},
```


```math
r_e=
\begin{cases}
(\widehat\mu_e-c_t)/s_t,&s_t\gt 10^{-12},\\[4pt]
0,&s_t\le10^{-12}.
\end{cases}
```


这里调整的是**部署输出尺度**，输入特征及回归权重保持原样。预设 $`\kappa\in\{0,0.5,1,2\}`$：


```math
\pi_\kappa=\operatorname*{argmax}_{\pi\in\mathcal P}
\left\{\kappa\sum_e r_e\pi_e-
\sum_eD_{\mathrm{KL}}\bigl(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\|\mathop{\mathrm{Bern}}\nolimits(1/2)\bigr)\right\},
```


```math
\mathcal P=\left\{\pi\in[0,1]^{96}:\sum_{e\in E_t}\pi_e=N_t/2\right\}.
```


KL 的导数为 $`\log(\pi_e/(1-\pi_e))`$，驻点满足：


```math
\pi_{\kappa,e}=\sigma(\kappa r_e-\gamma_t),
\qquad
\sum_{e\in E_t}\sigma(\kappa r_e-\gamma_t)=N_t/2.
```


目标严格凹。概率和随 $`\gamma_t`$ 连续严格下降，因此每类型乘子唯一；使用 float64 二分，区间从 $`\min_e\kappa r_e-40`$ 到 $`\max_e\kappa r_e+40`$，预算误差目标 $`10^{-10}`$。中心化后的 $`r`$ 仍需求乘子。$`\kappa=0`$ 时精确得到 Uniform。

对 $`\kappa\gt 0`$，对应原效用单位下的温度为 $`\tau_t=s_t/\kappa`$。该式把相同排序转换为不同程度的概率重分配，保留 TRUST 的效用预测—决策接口。

### 4.3 选模、置换和训练一次写清

1. 加载各 seed 对应的 `reference/last.pt` 和 `utility_all/compact/model.pt`。在 epoch 0，用图随机数种子 1701、每次 16 图，评价四个强度的 VAL CW native AUC，选择最大值，并列取较小强度。
2. 概率置换种子预先固定为 7101–7105。按 OI／OE／AE 分别打乱边概率，冻结评价五个实例；正式适配使用 7101。每个实例保留原概率多重集合、期望边数、熵和类型边数分布。
3. 冻结表同时保存固定 $`\kappa=1`$ 的真实／置换结果，以及置换组各自四个强度的 VAL 扫描。这给出固定强度和同等扫描机会的参照。正式 Permuted-Align 使用真实组已选概率集合，专门衡量对应关系。
4. 三组从同一 reference 开始，训练关系头 3 epoch；AdamW、学习率 $`3\times10^{-4}`$、weight decay $`10^{-4}`$、batch size 64、原调度与损失保持一致。每组重新初始化相同 seed、优化器和数据随机序列。
5. 按 VAL CW native AUC 在 epoch 0、1、2、3 中选权重。CW／OW 共用该权重；bias 与概率温度由各自 VAL 确定。
6. 最佳模型分别用图种子 1701、2701、3701 评估。1701 为与历史相同的主读数；另外两个 bank 用于量化图采样变化。三个 bank 各自保持 16 图，分别计算指标；附表汇总指标均值，不将它们合成 48 图模型。

冻结扫描主要使用 VAL。完成策略及权重选择后，统一评价 TEST。若 $`\kappa=0`$，真实组、置换组和 Uniform 的策略及训练随机序列完全一致，可复用结果，并记录该 seed 的选择为均匀图。

类型内置换可能改变目标节点的覆盖。因此本轮主张是“关系分配有价值”；关于同一目标节点内的语义精细选择，后续用节点内置换补充。

### 4.4 同队列的实际效用诊断

这部分复用三个 seed 的第五折教师，属于现有数据上的机制诊断。使用审计版 Compact，即仅由 Q1–Q3 拟合、Q4 选正则的模型，预测该教师响应；部署 reference 则继续使用最终五折模型。

每个 Q5 Query 组合按固定 seed 8101 将图像分成两个不重叠集合，每边每集合最多取 8 张；少于两张图的组合从此项诊断中排除并记录。两个集合分别用于 **测量关系收益** 和 **评价策略后的风险**，内部均采用组合等权。它们是同一留出组合上的不同图像；验证的是样本迁移及决策作用。

在测量集合上，对完整 96 条边使用 8 个独立背景块，每个块含 16 张 Bernoulli(0.5) 图。干预第一槽的一条边，其他槽和其他开关保持不变。同一次前向同时取得单图与混合标签：


```math
d^{(1)}_{e,k}=\sum_iw_i(\log p^1_{i,k}-\log p^0_{i,k}),
\qquad
d^{(16)}_{e,k}=\sum_iw_i\left[\log(S_{i,k}+p^1_{i,k})-\log(S_{i,k}+p^0_{i,k})\right],
```


```math
S_{i,k}=\sum_{m=2}^{16}P_{\mathcal C_{\mathrm{tr}}}(y_i\mid x_i,b_k^m),
\qquad
\widehat u^{(1)}_e=\frac18\sum_kd^{(1)}_{e,k},
\qquad
\widehat u^{(16)}_e=\frac{16}{8}\sum_kd^{(16)}_{e,k}.
```


概率在全部 TRAIN 组合上归一化，实际计算使用 logsumexp。单槽干预后乘以 16，是因为同一 $`\pi_e`$ 控制 16 个独立槽位的概率。

用测得的 $`\widehat u^{(1)}`$、$`\widehat u^{(16)}`$ 和 Compact 预测分别生成固定 $`\kappa=1`$ 的策略，并包含对应的类型内概率置换与 Uniform。在评价集合上用全新随机图计算单图 NLL、16 图混合 NLL。另保存同一路径上的局部策略：


```math
\pi^{\mathrm{local}}=\tfrac12\mathbf1+
\eta\left(\pi_{\kappa=1}-\tfrac12\mathbf1\right),
\qquad
\eta=\min\left(1,\frac{0.1}{\|\pi_{\kappa=1}-\tfrac12\mathbf1\|_\infty}\right).
```


零方向直接取 $`\eta=1`$。该插值保留类型预算和排序，将最大概率变化限定为 0.1；只作局部方向诊断，正式训练仍使用第 4.2 节策略。

在评价集合上预设 8 个独立 bank，每个 bank 16 图。对同一批概率分别计算：先求每图 NLL 再平均得到单图风险估计，先平均 16 图概率再求 NLL 得到混合风险估计。保存八个 bank 的配对风险差及其均值。

这项诊断一次回答三件事：关系收益是否可用于实际决策；Compact 与直接测量之间还有多少差距；单图目标与混合读出是否存在明显分歧。Q5 这里只包含伪未见样本，AUC 竞争分析安排到具有双组风险样本的第二批。

### 4.5 第一批最终表格

| 表格 | 核心列 |
| --- | --- |
| 主识别表 | 三组 CW／OW native、exact AUC；VAL 工作点 HM；逐 seed 的 Align−Uniform、Align−Permuted |
| 部署表 | 选中强度、类型概率标准差、期望边数、KL、节点覆盖、原型位移、选中 epoch |
| 机制表 | 实测／预测单图效用、混合效用的类型内排序；评价集合实际单图与混合 NLL 变化 |
| 成本表 | 响应与测量耗时、冻结扫描耗时、适配耗时、推理延迟、峰值显存 |

Capacity、Dense、原 robust 和原 Compact 在主表下方列为历史参照。第一批的直接判断来自 Align 与两个匹配组的配对差值，结合三个训练 seed、图 bank 和冻结／适配结果的一致性。

## 5. 第二批：以组合竞争效用深化 TRUST

### 5.1 目标与五组设计

**核心假设：关系对留出组合的价值，应反映它能否同时改善已见与未见组合的竞争，而不只反映伪未见样本的 NLL。**

这一批只改变效用监督对象。分类器、八维响应、27 参数预测器、半密度概率规则和 3 epoch 适配统一使用相同设置。

| 组名 | 效用监督 | 主要比较 |
| --- | --- | --- |
| Uniform-0.5 | 类型常数概率 | 随机关系分配基线，可复用第一批 |
| TRUST-U-NLL | 伪未见组 NLL 改善 | 原目标类别在新教师上的匹配版本 |
| TRUST-Balanced | 伪已见／伪未见各占一半的 NLL 改善 | 增加已见样本监督的作用 |
| TRUST-Competition | 平滑竞争面积改善 | 改变监督目标的作用 |
| Permuted-Competition | 置换 Competition 的最终概率 | 竞争效用与关系对应的作用 |

关键机制比较为 **Competition−Balanced**；关键部署比较为 **Competition−Uniform、Competition−Permuted**。U-NLL 与 Balanced 解释增加已见监督的作用。每组采用同样的正则候选、概率强度候选和选 epoch 规则。

### 5.2 建立双组风险教师

沿用五折 TRAIN 组合划分。第 $`j`$ 折令 $`S_j`$ 为 Inner 组合，$`U_j`$ 为留出组合，候选集为全部 TRAIN 组合 $`\mathcal C_{\mathrm{tr}}=S_j\mathbin{\dot\cup}U_j`$。

对 $`S_j`$ 中每个至少有两张图的组合，按固定划分种子 9101 留出约 20% 图像、至少一张且保留至少一张用于训练；构成伪已见风险图像池。其余 Inner 图像训练该折 G0 和关系头。$`U_j`$ 的全部图像构成伪未见风险池。只有一张图的 Inner 组合用于训练，在风险清单中单独记录。

每折重新训练 G0 10 epoch、冻结后训练关系教师 10 epoch，关系采样中心固定为 0.5。两部分任务内训练都只读取对应训练图像。旧教师读取过全部 Inner 图像，因此本批重新建立双组教师；CLIP 通用预训练权重及图像缓存继续复用。

每折分别从两个风险池均匀、不放回抽取最多 256 张图像，形成 $`D_{j,S}`$ 和 $`D_{j,U}`$，各组内部图像等权。第五折另留同组不重叠的评价图像；每组先将风险池一分为二，再各取最多 256 张作为测量／评价集合。保存全量候选、图像 ID、划分、样本权重和教师身份。

本批的 U-NLL 也采用伪未见组内图像等权，与另外两组共用风险图像。旧 Compact 使用组合等权，因此其历史值单列；本批目标比较在统一的新样本协议下进行。

响应输入从教师训练过的 Inner 图像产生，继续每组合最多两张图、4 个背景、原八维定义。最终部署沿用已有全 TRAIN reference；该模型及响应探针协议保持一致时，其原八维响应可以复用。

### 5.3 竞争面积的公式

以下在一个教师、一个风险集合和一张图上定义，省略 $`j`$。令 $`z_{ic}=\log P_{\mathcal C_{\mathrm{tr}}}(c\mid x_i,b)`$，对伪未见候选统一加偏置 $`\beta`$。

对已见图像 $`i\in D_S`$ 和未见图像 $`k\in D_U`$，定义：


```math
a_i=z_{i,y_i}-\max_{c\in U}z_{ic},
\qquad
b_k=\max_{c\in S}z_{kc}-z_{k,y_k},
```


```math
g_i^S=\mathbf1\!\left\{\operatorname*{argmax}_{c\in S}z_{ic}=y_i\right\},
\qquad
g_k^U=\mathbf1\!\left\{\operatorname*{argmax}_{c\in U}z_{kc}=y_k\right\}.
```


组内预测正确时，已见图像在 $`\beta\lt a_i`$ 一侧正确，未见图像在 $`\beta\gt b_k`$ 一侧正确。两者存在共同正确的偏置区间，等价于 $`a_i\gt b_k`$。

设 $`w_i^S=1/\vert D_S\vert `$、$`w_k^U=1/\vert D_U\vert `$，对全部临界偏置作梯形积分，有：


```math
A_{\mathrm{exact}}=
\sum_{i\in D_S}\sum_{k\in D_U}
w_i^Sw_k^U g_i^Sg_k^U H(a_i-b_k),
\qquad
H(v)=\mathbf1\{v\gt 0\}+\tfrac12\mathbf1\{v=0\}.
```


证明可由每次未见准确率上跳的面积贡献求和：在 $`b_k`$ 处，仍正确的已见样本满足 $`a_i\gt b_k`$，同时跳跃贡献半个梯形。组内错误样本始终没有面积贡献。该式与当前 `exact_curve` 的事件积分一致；工程 native AUC 继续按原有限偏置扫描单独计算。

为了得到连续的配对标签，定义组内真实类 margin：


```math
m_i^S=z_{i,y_i}-\max_{c\in S\setminus\{y_i\}}z_{ic},
\qquad
m_k^U=z_{k,y_k}-\max_{c\in U\setminus\{y_k\}}z_{kc}.
```


组内只有一个候选时对应门值直接设为 1。第一版固定平滑温度 **$`T=1`$ nat**，各教师、各背景共用，形成：


```math
\widetilde A(z)=
\sum_{i,k}w_i^Sw_k^U
\sigma(m_i^S/T)\,
\sigma(m_k^U/T)\,
\sigma((a_i-b_k)/T),
\qquad
R_{\mathrm{cmp}}(z)=1-\widetilde A(z).
```


三项分别表示已见组内判别、未见组内判别和跨组兼容的竞争区间。对所有未见候选统一加常数 $`d`$ 时，$`a_i`$、$`b_k`$ 同时减去 $`d`$，组内 margin 保持不变，因而该目标对统一未见偏置严格不变。

固定温度的平滑面积对整体分数缩放敏感。因此结果同时报告平滑面积、真实 exact/native AUC 和 margin 尺度，区分扩大分差与改善竞争排序。

### 5.4 同一次前向产生三套标签

在相同 $`D_S,D_U`$ 上定义：


```math
R_U=-\sum_{k\in D_U}w_k^U\log P_{\mathcal C_{\mathrm{tr}}}(y_k\mid x_k,b),
```


```math
R_{\mathrm{bal}}=-\tfrac12\sum_{i\in D_S}w_i^S\log P_{\mathcal C_{\mathrm{tr}}}(y_i\mid x_i,b)
-\tfrac12\sum_{k\in D_U}w_k^U\log P_{\mathcal C_{\mathrm{tr}}}(y_k\mid x_k,b).
```


对 $`q\in\{U,\mathrm{bal},\mathrm{cmp}\}`$，在同一背景上开关边 $`e`$：


```math
d_{je\ell}^{q}=R_q(\theta_j,b_{-e}^{\ell},0)-R_q(\theta_j,b_{-e}^{\ell},1),
\qquad
\bar y_{je}^{q}=\frac1K\sum_{\ell=1}^{K}d_{je\ell}^{q},
```


```math
\widehat v_{je}^{q}=
\frac1{K(K-1)}\sum_{\ell=1}^{K}(d_{je\ell}^{q}-\bar y_{je}^{q})^2.
```


前三折及开发折使用 $`K=8`$，第五折使用 $`K=16`$。每个背景都测完整 96 条边，所有类型、所有目标共用图像和随机背景。$`\widehat v`$ 表示固定风险集合下的背景测量方差；第五折另一个图像集合用于实际策略评价。

三套风险都由相同 on/off 分数计算。每组最多 256 张图时，竞争面积最多包含 65,536 个样本对，可分块精确求和。首版单图测量对应 R4，正式模型仍采用已有的 16 图读出；两种读出的风险变化一并保留。

### 5.5 效用拟合与正式部署

三套目标分别重算拟合折标签尺度 $`a_t^q`$，使用相同的全局八维输入标准化和 27 参数绝对岭回归。Q1–Q3 拟合，$`\alpha\in\{0.1,1,10,100\}`$，Q4 选择；第五折用于新教师条件下的排序及实际决策分析。

本批统一按 Q4 的标准化类型内误差选择正则。令 $`P_t=I-\mathbf1\mathbf1^\top/N_t`$：


```math
\mathcal E_4^q(\alpha)=
\frac13\sum_t\frac1{N_t}
\left\|P_t\frac{\widehat\mu_{4,t,\alpha}^{q}-\bar y_{4,t}^{q}}{a_t^q}\right\|_2^2.
```


并列取更大正则。该准则选择固定预算内的预测能力；三套目标共用该规则，训练损失仍为绝对岭回归，未加入 R5。旧 Compact 的正则与旧代理遗憾另作历史记录。

审计后按选定正则合并五折重拟合，重算输入和标签尺度，预测最终 reference 的关系效用。部署仍执行第一批的四个强度扫描、半密度预算、类型概率置换和 3 epoch 适配。各目标独立使用相同开发预算。

本批计 15 个识别结果；Uniform 的三个结果满足权重、训练随机序列和配置一致时直接复用，实际新增 **12 次短适配**。另需新建 15 个 G0＋关系教师组合，测量 15 份完整关系向量及响应。新增成本主要来自教师和效用测量。

## 6. 另外四条路线如何接入，保持每批工作集中

### 6.1 R3：以同一教师验证混合目标

第一批已经取得同图像、同背景的 $`u^{(1)}`$ 与 $`u^{(16)}`$。若混合效用引导的实际混合 NLL 改善更清楚，下一批优先比较单图监督与混合监督；沿用旧教师即可。

精确对象为：


```math
F_{D,M}(\pi)=\mathbb E_{b^{1:M}\sim q_\pi^{\otimes M}}
\left[-\sum_iw_i\log\left(\frac1M\sum_{m=1}^M P_{\mathcal C}(y_i\mid x_i,b^m)\right)\right],
```


```math
u_e^{(M)}(\rho)=-\partial_{\pi_e}F_{D,M}(\rho)
=M\,\mathbb E[d_{e,1}^{(M)}],
\qquad
\mathop{\mathrm{SE}}\nolimits^2(\widehat u_e^{(M)})=
\frac{M^2}{K(K-1)}\sum_k(d_{e,k}^{(M)}-\bar d_e^{(M)})^2.
```


直接取 $`M=16`$，背景块独立。对于每块完整 96 条边，其他槽需 15 次头前向，干预槽需 192 次，共 207 次／图像 batch；同块所有边共享伴随槽。重新拟合标签尺度和 Compact，输入及识别器保持一致。

正式组为 Uniform、单图效用、16 图效用、16 图概率置换，三个 seed 共 12 组；已有一致组可复用。该批完成读出目标比较后再决定是否研究竞争目标，避免把两种改动的效果混在一行。

### 6.2 R5：直接优化现有 Compact 的相对预测

若直接测得的效用能改善实际决策，而 Compact 指导的策略较弱，先完成这项低成本修改。

对教师 $`j`$、类型 $`t`$ 的完整矩阵 $`X_{j,t}`$，使用与原 Compact 相同的输入、标签尺度及平方损失权重：


```math
L_{\mathrm{rel}}(\beta)=
\sum_{j,t}\left\|P_t\left(X_{j,t}\beta-\bar y_{j,t}/a_t\right)\right\|_2^2+
\alpha\|\beta\|_2^2.
```


```math
\widehat\beta_{\mathrm{rel}}=
\mathop{\mathrm{solve}}\nolimits\left(
\sum_{j,t}X_{j,t}^\top P_tX_{j,t}+\alpha I,
\sum_{j,t}X_{j,t}^\top P_t\bar y_{j,t}/a_t
\right).
```


由于 $`P_t^2=P_t`$，该式直接由梯度为零得到。截距被投影消去，保留其列及正则可维持原 27 维接口。绝对／相对两组共享原输入标准化、标签尺度和未平均的求和损失，避免额外改变类型权重。

两组重新用第 5.5 节同一 Q4 准则选择正则，部署共用 R1 规则。只比较绝对、相对及各自概率置换，加上 Uniform，形成一个五组队列。CPU 拟合可先复算，GPU 主要增加短适配。

固定预算下，$`u_t^\top\pi_t=(P_tu_t)^\top\pi_t+\bar u_tB_t`$。因此类型共同偏移与选择无关；R5 改的是训练时需要拟合的误差。部署时仅减类型均值，本来就不会改变等式预算的最优概率。

### 6.3 R6：让关系从同一个 Capacity 起点生效

当已取得清楚的关系分配收益、而整体仍低于 Capacity 时，下一项表达改进采用 PDF 的共同端点设计。保留 Capacity 原型 $`q_v`$，关系增量更新为：


```math
z_v(b)=\mathop{\mathrm{norm}}\nolimits\left(q_v+0.1\sum_{e:v(e)=v}\frac{b_em_e}{d_{v,t(e)}}\right),
```


无激活入边时直接返回 $`q_v`$，联合分数采用：


```math
s_o=s_o^{\mathrm{cap}}+\lambda_g g(x)^\top(z_o-q_o),
\qquad
c_{oa}=c_{oa}^{\mathrm{cap}}+\lambda_g[T_\psi(z)-T_\psi(q)].
```


投影参数共用，零图逐分数恢复同一个 Capacity。各折需要本折合法数据训练的 Capacity，并重新计算响应与标签。

第一版使用均值接入效用，先建立知识增量。令 $`p_{0,i}`$ 为零图真实类概率，$`p_{e,i}`$ 为单边图概率，零点的 16 图混合效用为：


```math
u_e^{(16)}(0)=16\sum_iw_i\log\left(1+\frac{p_{e,i}/p_{0,i}-1}{16}\right).
```


它由“16 个槽位中恰有一次该边激活”的一阶概率推导。计算所有边只需零图一次、96 个单边图各一次；保存逐图像 log 概率以稳定计算。

接入采用绝对效用和二次正则：


```math
\max_{\pi}\;\widehat\mu^\top\pi-\frac1{2h}\|\pi\|_2^2,
\qquad
0\le\pi_e\le0.25,
\qquad
\sum_{e\in E_t}\pi_e\le N_t/8,
```


```math
\pi_e=\mathop{\mathrm{clip}}\nolimits\bigl(h(\widehat\mu_e-\gamma_t),0,0.25\bigr),
\qquad \gamma_t\ge0.
```


预算对应 OI/OE/AE 上限 1.5／4.5／6，第一版采用局部稀疏接入。步长写成 $`h=\eta/a`$，其中 $`a`$ 为拟合折绝对效用 RMS、下限取 10⁻¹²，$`\eta\in\{0,0.1,0.25,0.5\}`$ 按冻结 VAL 选择；$`\eta=0`$ 直接返回零图，正步长使用上述目标。R5 的纯相对标签与该上界预算分开使用。

共同起点下比较 Capacity 零图、Dense、同实际类型用量 Uniform、预测接入和概率置换，三 seed 共 15 组。该设计首先研究超出容量的知识增量，风险项可在均值结果之后再比较。

### 6.4 R2：保留为误差建模研究

R2 比旧的静态风险扣分更符合“概率更新承担误差”的含义：


```math
\max_{\pi\in\mathcal P}
\left\{\widehat\mu^\top(\pi-\pi^0)
-\sum_e\epsilon_e|\pi_e-\pi_e^0|
-\sum_t\tau_t\sum_{e\in E_t}D_{\mathrm{KL}}(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\|\mathop{\mathrm{Bern}}\nolimits(\pi_e^0))\right\}.
```


对应解为：


```math
\pi_e=\sigma\left(\mathop{\mathrm{logit}}\nolimits\pi_e^0+
\frac{S_{\epsilon_e}(\widehat\mu_e-\gamma_t)}{\tau_t}\right),
\qquad
S_a(v)=\mathop{\mathrm{sign}}\nolimits(v)\max(|v|-a,0).
```


这条路线需要跨教师的相对预测残差，边均值标准误只描述测量精度。现有噪声结果和校准结果尚未显示逐边风险信息的价值，因此当前主方案使用均值效用；后续固定同一均值，统一比较零风险、类型常数风险、逐边风险及风险置换，再确定风险是否进入最终方法。

## 7. 第三批：把方法结果组织成论文证据

### 7.1 确认设置

先锁定效用目标、预测器、概率规则、超参数选择范围和主要指标，再做两项完整确认：

| 设置 | 具体安排 | 对应价值 |
| --- | --- | --- |
| UT-Zappos 新模型情境 | 新训练 seed 4351／4352／4353，新 TRAIN 五折划分，官方 VAL／TEST 保持原协议 | 检查跨教师、跨留出组合结构的稳定性 |
| MIT-States | 同一 CLIP 框架和锁定的方法规则，三个 seed，重新建立该数据集的关系库及教师 | 检查超出鞋类材质组合的数据集适用性 |

MIT-States 与 UT-Zappos、C-GQA 均用于原 CPF 的 CW／OW 评价，因此适合作为第二个论文实验场景。选 MIT-States 是本方案的资源安排：先扩展一个数据集，再根据投稿范围加入 C-GQA。[CPF 官方论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf)

新 TRAIN 划分产生新的教师和留出组合情境，仍属于同一个 UT-Zappos 数据集；MIT-States 提供跨数据集证据。每个设置完整比较 **CPF、Capacity、Dense、Uniform、最终 TRUST、概率置换**，六方法、三 seed，共 18 组结果。CPF 可直接读出对应强基底权重，其余训练或适配按各自匹配协议执行。

MIT-States 的知识生成只读取合法原词词表和固定生成模板；来源、去重、类型定义及每原词附件数写入清单。图规模随该数据集原词变化，R1 预算仍为各类型边数的一半。新数据集先建立可比的 CPF／Capacity／Dense 参照，再运行同一队列。

### 7.2 三个核心贡献对应三类证据

| 研究内容 | 正式证据 | 低成本对照 |
| --- | --- | --- |
| 配对效用监督 | 真标签模型在新教师上的实际策略收益与最终 AUC | 同类型打乱标签，全部拟合和选模规则相同 |
| 模型响应条件化 | 当前教师响应比静态边偏好更有预测和决策价值 | 历史边均值、错配教师响应 |
| 有效关系分配 | 相同模型、期望边数和概率集合下正确对应更好 | 概率置换；需要细化到节点语义时补节点内置换 |

这些效用对照共用新教师、测量、八维响应和下游 evaluator，集中生成一张机制表。历史边均值按拟合教师等权平均；错配响应使用预先固定的教师置换，保持边及类型 ID 对应但替换模型状态。

现有 Compact 的输入只有类型与八维响应。删除八维响应后，每类型只有一个截距；R1 的类型等式预算使它精确产生 Uniform。**“无响应”识别结果直接复用 Uniform**，无需新增一个同义训练组。错配教师响应与历史边均值仍有独立的信息价值。

新方法的打乱标签拟合使用和真实模型相同的正则候选与选择准则；该组可先用于效用及冻结策略表。论文需要正式声称监督贡献时，再将其加入最终适配表，作为每个数据设置额外三组结果。

### 7.3 统计与主要指标

主要指标为 **TEST CW native AUC**，OW native AUC 为同时报告的泛化指标；exact AUC、VAL bias 下的 seen／unseen／HM、NLL／ECE 为配套结果。相同规则贯穿 A、B、C。

对每个匹配 seed 计算：


```math
\Delta_s=\mathop{\mathrm{AUC}}\nolimits_{s,\mathrm{TRUST}}-
\mathop{\mathrm{AUC}}\nolimits_{s,\mathrm{control}},
\qquad
\bar\Delta=\frac13\sum_s\Delta_s,
\qquad
\mathop{\mathrm{SD}}\nolimits(\Delta)=\sqrt{\frac12\sum_s(\Delta_s-\bar\Delta)^2}.
```


主表给出各方法均值±样本标准差，同时列出逐 seed 配对差值和配对差的标准差。三个图 bank 用于评价采图随机性，扰动 seed 用于噪声条件；训练 seed 仍是训练重复的单位。

最终确认表另做按 seen／unseen 分层的组合簇 bootstrap：每次在两组中分别有放回抽组合、保留其图像、对所有方法使用相同抽样，重新计算指标差。该区间描述当前模型和候选条件下的样本组成敏感性，与训练 seed 波动分列。

本路线的优先成果是：真实效用监督提高跨组合预测，正确关系分配提高识别，并在新情境中重现。效果判断依据配对差值、变化来源和新数据证据，不用某个任意 AUC 阈值代替研究结论。

### 7.4 噪声实验集中在最终模型

已有 600／1800 行噪声结果保留为旧方法的完整记录。新方法确定后，把噪声评估加入最后一次模型导出：Uniform、Dense、最终 TRUST；三个训练 seed；三个扰动 seed；OE 删除和 AE 附件错配两种场景；强度 0、0.25、0.5，共 **162 个扰动配置**，每个配置包含 CW／OW。

固定干净 VAL 的 bias、温度和识别权重，按照最终部署定义重新生成受扰知识下的响应及概率；同时保存固定原概率的读数，区分关系分支受扰和策略响应的作用。策略重算只能使用 TRAIN 支持输入。报告绝对 AUC、相对干净变化及实际扰动比例，作为最终知识使用行为的分析。

## 8. 结果如何决定下一项工作

把 PDF 的多个“验证关口”合并为同一结果包中的解释表，避免每个读数都启动一轮零散实验。

| 一批结果的主要现象 | 研究解释 | 下一项集中工作 |
| --- | --- | --- |
| Align 同时高于 Uniform 和概率置换 | 现有效用排序已产生识别价值 | 固定该部署方式，开展竞争目标比较及新情境确认 |
| 半密度三组一起改善、彼此接近 | 当前主要收益来自关系分支用量 | 查看实测策略及风险链条，选择目标或预测改写 |
| 实测效用策略降低实际风险，Compact 策略较弱 | 主要差距在效用预测 | R5 相对拟合；共用原输入与标签 |
| 单图收益明确，16 图混合收益较弱 | 主要差距在监督与读出对应 | R3 混合目标，复用旧教师 |
| 单图／混合 NLL 改善，AUC 变化仍弱 | 当前监督偏向概率拟合，组合竞争收益不足 | 按第 5 节执行 R4 |
| 冻结时存在收益，适配后减少 | 固定策略与更新后的分类器状态发生偏移 | 以 epoch 0 为主结果；后续单独研究响应与策略更新 |
| 关系分配有增益，但同工程 Capacity 持续更强 | 表达起点仍影响最终性能 | R6 的共同 Capacity 端点 |

多个现象同时出现时，优先处理在实测策略中已出现、且能解释最大终端差距的问题；一次只改一个主要环节。对于同一模型已有清楚收益的方案，优先做新情境确认，再扩展模块。

## 9. 计算安排与实现接口

### 9.1 三卡运行方式

| 物理 GPU | seed | 第一批工作 |
| --- | --- | --- |
| 0 | 3351 | 本 seed 的冻结扫描、教师风险诊断、三组适配、最终评价 |
| 1 | 3352 | 同上 |
| 2 | 3353 | 同上 |

每卡串行执行一个 seed 的完整队列。图像特征缓存只读共享，各 seed 独立保存 work、日志和测量文件。第二批继续按 seed 分卡，五折教师在该卡内顺序运行。第三批 seed 改为 4351／4352／4353。

### 9.2 成本依据

已有 Compact 每 seed 两次适配约 114–119 秒。相同硬件下，第一批三次适配可按约 3 分钟／seed 安排；新增冻结扫描与实际风险诊断单独计时。

第一批每个诊断教师的完整边测量，按每个图像 batch 计：


```math
8\times\bigl[(16-1)+2\times96\bigr]=1656
```


次关系头前向。各教师风险集合均有 $`B`$ 个 batch 时，三个教师合计 $`4968B`$ 次；batch 数不同则逐教师累加。风险评价和冻结扫描另计。基底特征及固定教师的 base 输出可缓存。

第二批前三折和开发折各 8 个背景、第五折 16 个背景，五折标签测量每 seed 共：


```math
(4\times8+16)\times2\times96=9216
```


次关系头前向／图像 batch。三个监督目标共享这些前向。实际耗时按第一折记录外推，同时计入 G0、关系教师、响应、适配、最终评价和额外存储。

第一批不重跑完整教师；第二批的新教师成本单列；论文效率表报告完整方法开销和复用后的增量开销两列。

### 9.3 第一批代码改动位置

| 现有位置 | 需要实现的内容 |
| --- | --- |
| `trustcpf/graph.py` | 类型标准化后的等式预算求解、强度与乘子保存、概率置换 |
| `trustcpf/compact.py` | 加载现有模型，输出 Align 结构；预测器保持原样 |
| `trustcpf/evaluation.py` | 接受显式图 bank，导出每图及混合概率、各 bank 的一致指标 |
| 新 `trustcpf/decision_audit.py` | 同样本完整边测量、单槽 16 图标签、独立图像上的实际策略评价 |
| `trustcpf/workflow.py` | 三组队列、冻结扫描、复用身份与最终汇总 |

第二批主要增加风险集合划分、`competition_risk` 和多目标测量。`strong.py` 与教师训练入口需接受明确的允许图像 ID；当前仅按组合过滤的接口要扩展为“组合＋图像行”过滤，并让 G0、关系头、响应探针共用该清单。

主要公式核验集中为：零强度恢复 Uniform、每类型预算成立；同概率置换保持概率集合；单槽混合导数在小图枚举下成立；竞争面积与 exact 事件积分一致、统一未见偏置不变；R5 共同类型偏移被消去。核验随相应实现加入，不另建独立实验轮次。

### 9.4 每批一次性交付的数据包

每个 seed 保存统一的协议清单：模型与特征身份、风险图像及候选顺序、样本权重、背景分布、图数、风险函数、拟合折与选择参数。关系效用文件明确保存所估计的对象，单图、混合、竞争标签使用不同目标名。

结果包集中包含主结果 CSV、逐 seed 配对差、冻结扫描、策略概率、教师风险诊断、VAL／TEST 预测、训练历史、最佳权重和阶段耗时。正文报告只保留三张核心表：**识别收益、效用到决策的变化、计算成本**；完整行级数据随归档提供。

## 10. 最终论文方向的定位

建议当前工作题目表述为：**TRUST-CPF：面向组合泛化的关系效用监督与决策学习**。

保留的核心链条是：


```math
\begin{aligned}
&\text{TRAIN 内组合留出}\;\longrightarrow\;\text{关系开关的配对收益}\\[4pt]
&\qquad\longrightarrow\;\text{模型响应条件化的效用预测}\\[4pt]
&\qquad\longrightarrow\;\text{关系概率分配}\;\longrightarrow\;\text{CZSL 识别改善}.
\end{aligned}
```


如果竞争目标有效，方法贡献可以进一步聚焦为“学习关系对已见／未见组合竞争的可迁移价值”。效用监督继续是主要研究点；风险贡献由后续独立实验证据决定。

Co-CGE 已使用图关系及组合可行性加权，CPF 已建立条件概率识别框架。TRUST 的研究重点应落在**以留出任务的实际收益监督关系使用价值，并证明这种价值可迁移到有效的关系决策**。类型温度、岭回归和 Bernoulli 采样作为实现工具，论文贡献由整条证据链支撑。[Co-CGE 原论文](https://arxiv.org/abs/2105.01017)；[CPF 官方页面](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.html)

当前立即执行的版本明确为 **TRUST-Align**：复用现有 Compact，三组、三 seed、最多九次短适配，同时完成一套实际效用诊断。下一项方法深化明确优先为 **TRUST-Competition**。整个计划始终围绕效用监督和关系决策展开。

## 资料与本地实现索引

以下报告及代码是本方案的可追溯依据。已有报告中的旧建议作为当时结果的解释，本文件给出当前统一的后续安排。

- [用户提供的六条路线第四版 PDF](references/six_routes/README.md)：R1 印刷页 5–6，R2 页 7–9，R3 页 10–11，R4 页 12–15，R5 页 16–17，R6 页 18–20；PDF 文件页码比印刷页码大 2。
- [弱基底 v1 完整结果](../v1/analysis/server-results-20261022/README.md)。
- [条件残差 v2 完整结果](references/history/conditional_v2_analysis.md.txt)。
- [Strong v1 完整结果及全部噪声、消融](../02/analysis/README.md)。
- [模型响应条件化完整结果](../03/analysis/README.md)。
- [Compact v1 完整结果与逐 seed 配对](../04/analysis/README.md)。
- [Compact 完整分析数据](../04/analysis/confirmed/analysis.json)。
- [Strong 完整分析数据](../02/analysis/confirmed/analysis.json)。
- [当前 Compact 回归与聚合实现](../04/code/trustcpf_compact_v1/trustcpf/compact.py)。
- [当前采样与配对测量实现](../04/code/trustcpf_compact_v1/trustcpf/stages.py)。
- [关系与 Capacity 评分实现](../04/code/trustcpf_compact_v1/trustcpf/model.py)。
- [当前 native／exact AUC 实现](../04/code/trustcpf_compact_v1/trustcpf/metrics.py)。
