## 5. 大路径决策与创新定位

**下一阶段的主问题：让模型学习“哪些局部视觉证据能区分两个属性在同一物体上的组合”。方法核心是改变比较与汇聚的顺序。** CLIP 视觉适配负责让 token 接受任务梯度，静态 CP 提供已有组合参数化；研究贡献集中在局部判别算子。

| 路径 | 当前依据 | 下一步安排 |
|---|---|---|
| 继续调整 Wd、门控幅度、温度、rank | 更新量已成功改变，组合排序与共同错误没有整体改善 | 结束这条参数搜索线 |
| 多层 CLIP＋Adapter＋条件注意力 | 工程可行，相关工作已充分覆盖 | 作为已有技术；不作为新方法的核心 |
| **同位置属性比较，再汇聚判别证据** | 当前约 31% 未见图像在真实物体内排错；现有主读出看不到可训练 patch | **下一阶段研究主线** |
| AKTR／知识精化 | 适合改变语义关系与组合先验 | 待视觉判别主线形成后，再针对剩余语义错误展开 |
| EU-VLRF／Trust／GFlow | 增加随机性、置信度或关系选择 | 本轮不扩展这些机制 |

数据已经确立的是“现有路线停滞、同物体内属性排序仍有空间”。进一步提出的研究假设是：**分别为每个属性寻找高响应区域，容易把共同响应当成区分证据；先在同一位置比较属性假设，可以使组合监督更直接地训练差异证据。** 这是下一阶段要回答的机制问题，现有缓存结果没有记录逐 patch 响应。

### 5.1 与近邻工作的实质差别

本次按公开原文核查至 2026-10-21。文献差异界定了可主张的研究内容；新方法的性能与跨数据集作用由正式实验给出。

| 已有工作／思想 | 已有内容 | 本方案具体研究什么 |
|---|---|---|
| [FACT 本地复核](VG-CPOD与FACT_创新性复核_含D组_20261020.md) | CP 交互及低秩算子解释 | CP 作为共同评分器；改变其 patch 级输出如何成为属性判别分数 |
| [CANet](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Learning_Conditional_Attributes_for_Compositional_Zero-Shot_Learning_CVPR_2023_paper.html)、[CPF](https://arxiv.org/html/2507.17377v1) | 物体条件属性学习、条件概率及条件视觉提取 | 在同一物体条件下，以属性对的逐位置差值作为证据单位 |
| [MSCI](https://arxiv.org/html/2505.10289v1) | 多层 CLIP 特征、跨模态交互与适配 | 首版只读最后一层，研究比较算子；视觉适配本身计入共同基座 |
| [FILIP](https://arxiv.org/abs/2111.07783) | 视觉／文本 token 的细粒度晚交互 | 将两个属性假设放在同一 patch 上比较，再边缘化局部位置 |
| [TsCA](https://www.ijcai.org/proceedings/2025/0624.pdf) | patch、primitive、composition 之间的条件传输与一致性 | 学习有符号的属性对证据；不以跨集合传输距离为目标 |
| [Flow-CLIP](https://arxiv.org/html/2610.21937v1) | 图像条件局部修正；跨图像比较同一文本对的 margin | 本方案在单张图像内部、汇聚之前比较属性；不把跨图像 margin 换名作为创新 |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) | 两张 softmax 注意力图相减 | 本方案先相减未汇聚响应，再计算两个对称位置配分函数的对数比 |
| [HodgeRank](https://link.springer.com/article/10.1007/s10107-010-0419-x) | 将成对比较投影为一致的全局分数 | 用作已知的概率接口，不作为原创数学贡献 |

因此，“新增 Adapter”“换成 patch”“使用图”“加入对比损失”都不是本方案的贡献表述。**贡献候选是：物体条件下，先逐位置比较、再边缘化的组合判别算子，以及它对未见组合内部排序的作用。** 与上述已核查工作的差异落实在公式和对照模型上；现阶段将其称为有明确差异的研究方案。

## 6. 新机制的数学定义与正式实验

### 6.1 共同基座：同一个 CLIP、同一个局部评分器

继续使用 OpenAI CLIP ViT-L/14。冻结其预训练主权重，在视觉 Transformer 中训练 bottleneck64 的 Adapter。最后一层经 `ln_post`、视觉投影和 L2 归一化后，得到 CLS 向量 $g\in\mathbb{R}^{d}$ 与 patch 向量 $p_n\in\mathbb{R}^{d}$；224 输入下 $N=256$、$d=768$。首版直接使用最后一层，集中研究判别机制。

设属性、物体、组合的固定文本向量分别为 $e_a^a,e_o^o,e_{ao}^c\in\mathbb{R}^{d}$。沿用 G0 的静态评分器，只将输入写成任意视觉向量 $v$：

$$
\begin{aligned}
u_a&=A_u e_a^a,\qquad v_o=A_v e_o^o,\\
R(v,a,o)
&=\kappa v^{\top}e_{ao}^c
+(v^{\top}B_c)\bigl((e_{ao}^c)^{\top}C_c\bigr)^{\top}
+\rho\kappa v^{\top}W_c(u_a\odot v_o).
\end{aligned}
$$

维度为 $A_u,A_v\in\mathbb{R}^{r\times d}$、$B_c,C_c\in\mathbb{R}^{d\times q}$、$W_c\in\mathbb{R}^{d\times r}$。固定 $r=32$、$q=8$、$\kappa=20$、$\rho=0.15$，与当前静态头一致。记局部响应为 $r_{nao}=R(p_n,a,o)$。

所有属性共用物体条件的位置权重：

$$
w_{no}=\frac{\exp\!\left(\kappa p_n^{\top}e_o^o\right)}
{\sum_{k=1}^{N}\exp\!\left(\kappa p_k^{\top}e_o^o\right)},
\qquad \sum_n w_{no}=1.
$$

位置权重随视觉 Adapter 训练而变化。它只依赖图像和候选物体；推理逐候选物体计算。R1/R2 完全共用这一评分器、权重定义和可训练参数，差别只有下面的计算顺序。

### 6.2 现有形式对照：分别汇聚，再比较

每个属性单独得到局部汇聚分数：

$$
\ell_{ao}(x)=\log\sum_n w_{no}\exp(r_{nao}),
\qquad m^{\mathrm{sep}}_{ab,o}=\ell_{ao}-\ell_{bo}.
$$

这个分数可以利用局部高响应，但两个属性的高响应可能来自不同位置。这是正式对照 R1，具有与新方法相同的可训练局部视觉能力。

### 6.3 核心改变：逐位置比较，再边缘化

首先在每个位置比较两个属性假设：

$$
\delta_{n,ab,o}=r_{nao}-r_{nbo}.
$$

将局部位置 $n$ 看成需要边缘化的隐变量，给两个相反假设定义对称能量：

$$
\begin{aligned}
Z^+_{ab,o}&=\sum_n w_{no}\exp\!\left(\frac{\delta_{n,ab,o}}{2}\right),\\
Z^-_{ab,o}&=\sum_n w_{no}\exp\!\left(-\frac{\delta_{n,ab,o}}{2}\right),\\
q(+\mid x,a,b,o)&=\frac{Z^+_{ab,o}}{Z^+_{ab,o}+Z^-_{ab,o}},\\
m_{ab,o}&=\log\frac{q(+\mid x,a,b,o)}{1-q(+\mid x,a,b,o)}
=\log Z^+_{ab,o}-\log Z^-_{ab,o}.
\end{aligned}
$$

这里的 $q$ 是所定义二元能量模型的概率；全属性预测使用下一节的一致分数。$1/2$ 使单 patch 情形严格恢复原始响应差。实现用 `logsumexp(log_w ± delta/2)`，指数尺度直接沿用已有 logits 单位。

该定义满足四个可直接验证的性质。

**（一）方向一致。** 交换属性次序得到 $m_{ba,o}=-m_{ab,o}$；$m_{aa,o}=0$。

**（二）消除逐位置的共同响应。** 固定 $w_{no}$，若给该物体所有属性响应加上同一位置项 $h_{no}(x)$，则

$$
r'_{nao}=r_{nao}+h_{no}(x)
\quad\Longrightarrow\quad
\delta'_{n,ab,o}=\delta_{n,ab,o}
\quad\Longrightarrow\quad
m'_{ab,o}=m_{ab,o}.
$$

共同项可以随位置变化。分别做非线性汇聚再相减，一般没有这一性质。这是**响应层面、位置权重固定时**的不变性；图像变化也会改变 $w$，需要通过完整模型测量。

**（三）区分证据可以反向训练视觉 token。** 定义

$$
\pi^{\pm}_{n,ab,o}
=\frac{w_{no}\exp(\pm\delta_{n,ab,o}/2)}{Z^{\pm}_{ab,o}},
\qquad
\left.\frac{\partial m_{ab,o}}{\partial\delta_{n,ab,o}}\right|_{w}
=\frac{\pi^+_{n,ab,o}+\pi^-_{n,ab,o}}{2}.
$$

该偏导非负且对位置求和等于 1，支持和反对某一属性的强差异位置都会接收梯度。完整反向传播同时经过 $w$ 和 $r$。

**（四）小响应极限回到普通线性读取。** 对整体缩放后的局部差 $t\delta$，令 $t\to0$：

$$
m(t\delta)=t\,\mathbb{E}_{w}[\delta]
+\frac{t^3}{24}\operatorname{cum}_{3,w}(\delta)+O(t^5).
$$

首项是普通位置加权平均，非线性项来自局部差异分布的高阶奇数累积量。一般情况下，新算子与“先汇聚再比较”不等价；单位置或某些对称差异分布下会退化为平均差。这给出明确的等价边界。

另一个边界也能写成公式：若加入的是属性特有、位置无关的偏置 $b_{ao}$，则

$$
r'_{nao}=r_{nao}+b_{ao}
\quad\Longrightarrow\quad
m'_{ab,o}=m_{ab,o}+b_{ao}-b_{bo}.
$$

因此，此机制针对的是共同局部响应；seen/unseen 类别偏好仍由原有训练目标与 VAL 校准处理。

一个只说明代数差异的三位置例子：$r_a=(4,0,0)$、$r_b=(0,3,0)$，位置权重均匀；共同项 $h=(0,5,0)$。

| 算法 | 添加共同项前的 $a-b$ 分数 | 添加共同项后 |
|---|---:|---:|
| 分别汇聚后相减 | +0.9411 | −2.6825 |
| 先比较再边缘化 | +0.4274 | +0.4274 |

这是构造数组上的公式核验，作用是说明两个函数确实不同。

### 6.4 将成对证据还原为条件概率

一般的 $m_{ab,o}$ 不保证三角一致性，不能直接把每条边都当成同一个多类概率的 log-odds。对固定、连通的属性比较图 $G=(\mathcal{A},E)$，给边 $(a,b)$ 约定方向。关联矩阵 $B\in\mathbb{R}^{|E|\times A}$ 的对应行在 $a$ 列为 $+1$、$b$ 列为 $-1$，其中 $A=|\mathcal{A}|$。

按最小二乘得到统一条件 logits：

$$
U_o=\underset{\mathbf{1}^{\top}U=0}{\operatorname{argmin}}
\;\frac12\lVert BU-m_o\rVert_2^2.
$$

令 $L=B^{\top}B$。连通性给出 $\operatorname{null}(L)=\operatorname{span}(\mathbf{1})$，从而有唯一解：

$$
U_o=Dm_o,
\qquad
D=\left(L+\frac{\mathbf{1}\mathbf{1}^{\top}}{A}\right)^{-1}B^{\top}.
$$

$D$ 在训练前一次计算，并检查零和与正规方程。它不含可学习参数。完整图时进一步简化为

$$
U_{ao}=\frac1A\sum_{b\in\mathcal{A}}m_{ab,o}.
$$

这一步采用已有的最小二乘排序思想。对 R1，若输入 $m_o^{\mathrm{sep}}=B\ell_o$，同一映射精确恢复 $U_o=\ell_o-\overline{\ell}_o\mathbf{1}$，softmax 与直接使用 $\ell_o$ 完全相同。因此，R1/R2 的差别可严格归于边证据的计算，而非额外图模型。

为使 MIT-States 上的计算量可控，首版统一采用固定图：按冻结属性文本向量的余弦距离，取每个属性最近的 $\min(8,A-1)$ 个邻居，做无向并集，再并入同一距离上的最小生成树。距离相同时按标签索引排序。图只使用属性词表，跨 seed、物体、TRAIN/VAL/TEST、CW/OW 共用。8 为首版计算规格，当前不安排邻居数搜索。

完整图的成对局部运算量为 $O(ON A^2)$；上述稀疏图为 $O(ON|E|)$，且 $|E|\leq 8A+A-1$。属性分数仍覆盖全部 $A$ 个词。候选组合只在最终评价时筛选，比较图不随 CW/OW 候选变化。

联合概率沿用现有框架：

$$
\begin{aligned}
S_o(x)&=\kappa g^{\top}e_o^o
+(g^{\top}B_o)\bigl((e_o^o)^{\top}C_o\bigr)^{\top},\\
p(o\mid x)&=\operatorname{softmax}_o S_o(x),\\
p(a\mid o,x)&=\operatorname{softmax}_a U_{ao}(x),\\
p(a,o\mid x)&=p(o\mid x)p(a\mid o,x).
\end{aligned}
$$

沿用当前 TRAIN 支持集 $\mathcal{C}_{\mathrm{tr}}$、$s=\log p(a,o\mid x)$，对真实组合 $y=(a,o)$：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{pair}}
&=\mathbb{E}\!\left[-s(x,y)+\log\sum_{c\in\mathcal{C}_{\mathrm{tr}}}e^{s(x,c)}\right],\\
\mathcal{L}_{\mathrm{support}}
&=-\mathbb{E}\log\sum_{c\in\mathcal{C}_{\mathrm{tr}}}e^{s(x,c)},\\
\mathcal{L}
&=1.5\,\mathcal{L}_{\mathrm{pair}}
+\frac{9}{86}\,\mathcal{L}_{\mathrm{support}}.
\end{aligned}
$$

最后一式就是 G0 在 $\beta=0.075$ 时的任务损失恒等式。新模型通过同一分类目标训练证据算子；预测保存格式、VAL 选模和 native/exact 评价继续复用。

### 6.5 一套完整的机制比较

| 模型 | 共同 CLIP／静态头之外的区别 | 回答的问题 |
|---|---|---|
| R0 | 可训练视觉 Adapter；$U_{ao}=R(g,a,o)$，全局读取 | 将梯度接入 CLIP 后，普通静态头的表现 |
| R1 | 相同 Adapter；最后层局部响应分别汇聚为 $\ell_{ao}$ | 普通局部视觉证据学习的收益 |
| **R2** | 相同 Adapter、局部响应、物体权重；使用 $\delta\to m\to U$ | **先比较后汇聚是否改善未见组合判别** |

R1/R2 参数量相同，训练图像、增强、初始化、优化器、训练步数和选模规则相同。R2 的额外代价是成对响应与固定线性映射，需在结果中同时报告实际耗时和峰值显存。

安排 **UT-Zappos、MIT-States × R0/R1/R2 × seed3351/3352/3353，共 18 次完整训练**。每张 GPU 负责一个 seed，依次执行两数据集的三个模型。UT 与现有 G0/D1 建立联系；MIT 检查状态词和更多物体下的迁移。每次训练固定 10 epochs、有效 batch64；显存通过 microbatch 和梯度累积处理。

固定首版配方：视觉 Adapter 学习率 $5\times10^{-5}$，现有后编码适配矩阵 $10^{-3}$，静态 CP 参数 $3\times10^{-4}$；使用余弦衰减至各自初始值的 1%，AdamW weight decay $10^{-4}$。三个模型共用此配方。$\rho=0.15$、$\beta=0.075$、rank32 和图邻居数8 全部固定。这份配方是正式比较的起点，不把上一轮冻结特征头的学习率原样用于 CLIP 内部。

UT 使用现有文本模板；MIT 使用适合状态—物体组合的模板，三个模型共用。数据集以已有标准训练／验证／测试划分接入，不按 TEST 错误选图、选边或选择负例。

**主比较为 R2−R1。** R2 相对旧 G0 的收益同时包含“视觉适配、局部特征和新算子”，应与 R2−R1 分开解释。结果同时给出：

1. 三 seed CW/OW AUC 与固定 VAL bias 的 HM。
2. 真实物体内的未见属性排序、跨物体竞争，以及共同错误的纠正量和新增错误量。
3. R1/R2 的边 margin、投影后的最终 margin、位置梯度分布；报告投影残差，判断有效判别信息是否保留下来。
4. 正常图像上的差异证据可视化，以及固定 $w$ 的响应共同项不变性检查。后者用于核验实现，前者用于解释真实学习结果。

其中 1–3 来自这 18 次完整训练及保存的预测；无需继续串行追加温度、幅度或学习率小网格。论文定位还应与 CPF/MSCI 等已有方法按相同骨干和候选协议比较；本轮 R0/R1/R2 首先把新机制的贡献与工程适配收益明确分开。

### 6.6 下一次代码工作的明确范围

1. **在线视觉训练路径。** 复用本机 `CZSL/src/LOGICZSL_repro/model/troika.py` 中视觉 Adapter 与 CLIP 前向接口，输出 CLS 和最后层 patch；旧 `g/fa/fo` 缓存保留用于历史结果读取。
2. **统一评分器与三种读取。** 同一个静态 CP 参数集合实现 R0、R1、R2；将 $\delta\to m\to Dm$ 写成独立、可微的算子，不引入新的动态门控。
3. **固定比较图与分块计算。** 保存词表、文本向量、边方向和 $D$；按候选物体、边和 microbatch 分块计算，避免物化完整 $[B,O,A,A,N]$ 张量。累积训练梯度时使用重计算控制激活内存，`logsumexp` 和图投影采用 FP32。
4. **保留评价体系，补充机制记录。** 继续保存图像身份、候选顺序、完整 logits、checkpoint、优化器状态和三 seed 汇总；增加 TRAIN/VAL 固定样本的 patch 响应及最终模型的边证据诊断。
5. **一次提交上述正式矩阵。** 三张 GPU 独立运行 seed 队列，用 tmux 管理、独立目录保存，全部完成后统一比较。

本次已完成公式性质和非等价性的数值核验，明细见 `analysis_roundK_results_20261021/proposed_math_checks.json`。这部分运行的是代数检查；训练代码尚未改动。下一次实现以这里的一套机制和正式比较为范围。
