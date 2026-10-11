# CGE 原论文总结与 USG-CPF 优化方案

修订日期：2026-10-08。依据：CGE 的 CVPR 2021 终稿、本机 USG-CPF 归档，以及 CZSL 目录中截至 2026-10-08 的语义归因、机制诊断、十轮复核结果、条件概率讨论和 B/F 逻辑方案。

本文保留 CGE 的方法总结，并将优化建议更新为一个可与现有研究衔接、可独立验证的方案。文中“建议设置”尚未成为真实训练结果；已完成结果均注明来源和预算。

## 1. 本轮修订后的判断

CGE 最值得借鉴的是：**用共享的图表示生成组合分类器，让 seen 图像监督同时塑造可迁移的属性、对象和组合表示。** 它为“图应当真正进入预测”提供了清楚的实现范例。

结合你后续的讨论，USG 的首要问题应收紧为：

> 在同一强 CPF 基线和同一知识输入下，外部关系的确定性聚合、学习边权和结构采样，能否分别改善候选对象下的属性证据、组合识别和概率质量？

推荐保留原 USG 的 Bernoulli 边、关系化消息和概率平均；以新的条件概率方案作为一个可精确回放基线的接入方式。CGE 组合图作为独立结构实验，不再列为全部 USG 版本的前置条件。

上一版有四处研究判断需要修正：

| 上一版建议 | 本轮修订 | 依据 |
| --- | --- | --- |
| 先证明 CGE-Core 有效，再允许做 USG | CGE-Core 是可选对照；外部知识图可以单独验证 | 原 USG 的问题是外部关系不确定性，与 CGE 的标签拓扑并不相同 |
| 确定图无收益就证明随机阶段不值得研究 | 固定图结果决定预算优先级；噪声下的随机机制可另立有限试验 | 不同模型函数和目标之间不存在这样的必要性定理 |
| 新组合读出必须加入主模型 | 先用原 CPF 分数和共享低秩条件读出；额外组合残差单独消融 | 新容量、读出和图结构同时变化会混淆归因 |
| seen 属性–对象共现边等同于确定的任务骨架 | “组合由两个原语构成”是定义；seen 共现拓扑是训练数据先验 | 保留共现边可能强化 seen 偏置，需比较 seen/unseen 结果 |

## 2. CGE 的主要 idea 与公式

这里用属性符号 $a$ 表示原论文中的 state $s$；图像记为 $x$，对象为 $o$，组合标签为 $y=(a,o)$。以下邻接矩阵改记为 $A_G$，避免与后文的图 Laplacian 混淆。

### 2.1 原语和组合类别共同构图

节点集合是三个带类型的集合的并集：

$$
V=V_a\cup V_o\cup V_y.
$$

对协议中的每个组合 $y=(a,o)$，加入属性–组合、对象–组合和属性–对象边，并保留自环。原论文使用固定、无向、无权邻接；组合节点由两个原语词向量的平均初始化。训练图包含 seen 与其协议允许的 unseen 组合类别名称，不使用 unseen 图像监督。

### 2.2 图输出直接作为组合分类器

原论文的传播规则可以写为：

$$
H^{(\ell+1)}
=
\sigma\left(D^{-1}A_G H^{(\ell)}\Theta^{(\ell)}\right),
\qquad H^{(0)}=E.
$$

$D$ 是含自环邻接的度矩阵；$\sigma$ 在这里是 ReLU。最终组合节点表示与图像特征具有相同维度：

$$
g_\theta(a,o)=H_y^{(L)},
\qquad
f(x,a,o)=F_\psi(x)^\top g_\theta(a,o).
$$

它把“类别语义”变成实际分类器权重，图像编码器和图网络共同学习。

### 2.3 只在 seen 组合上计算分类损失

$$
\mathcal L_{\mathrm{CGE}}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\log
\frac{\exp f(x_i,a_i,o_i)}
{\sum_{(a,o)\in\mathcal Y_s}\exp f(x_i,a,o)}.
$$

unseen 输出没有直接的图像分类监督；共享原语、有限跳消息路径和共享参数使其分类器随 seen 训练共同变化。不能把这一过程描述成“每个 unseen 节点都有独立标签梯度”。

### 2.4 关键消融告诉我们什么

下面是 CVPR 2021 终稿 Table 3 的 MIT-States 验证结果，指标单位为百分数：

| 结构 | AUC | Best HM |
| --- | ---: | ---: |
| 直接使用组合词向量 | 5.9 | 19.4 |
| 属性/对象连接组合，组合节点无自环 | 7.6 | 18.6 |
| 上述结构加组合自环 | 8.1 | 22.7 |
| 再加入属性–对象边，完整 CGE | 8.6 | 23.3 |
| 完整 CGE 加 WordNet 对象层级 | 7.9 | 22.0 |

两层普通 GCN 在该深度消融中最好；加入更深图网络或外部层级并不自动提高性能。这支持浅层、保留自身信息、逐类评估外部关系的设计。

该证据来自 CGE 自己的模型和数据版本，尚不能证明 CPF 上新增同样结构也会有效。方法与消融来源：[CGE CVPR 终稿](https://openaccess.thecvf.com/content/CVPR2021/papers/Naeem_Learning_Graph_Embeddings_for_Compositional_Zero-Shot_Learning_CVPR_2021_paper.pdf)，[作者论文页面](https://arxiv.org/abs/2102.01987)。

## 3. 后续讨论给 USG 增加了哪些约束

### 3.1 文本增强与正确知识归属是两个问题

UT 静态 v2 四控制、三个训练 seed 的平均差值如下，单位为百分点：

| 比较 | CW AUC | CW HM | OW AUC | OW HM |
| --- | ---: | ---: | ---: | ---: |
| C_text − A_unit | +3.067 | +2.176 | +4.131 | +3.943 |
| C_text − C_shuffle | +0.379 | −0.727 | +1.980 | +0.104 |

这使 C_text 成为值得保留的强参照，但正确归属的 CW 增量仍未达到原定 +0.5 pp。OW 的较强信号值得研究，不能事后拿 OW 替代原 CW gate。

在 OW、bias=0 的逐图分析中，C_text 相对 A_unit 的 pair 准确率平均下降 0.664 pp；Boat Shoes 的对象准确率下降约 11.287 pp。文本的 AUC 增益、固定工作点收益和对象错误结构必须分别报告。

### 3.2 已有分支的弱效应提示应检查“知识怎么进入预测”

名称锚点机制诊断显示：同图同属性跨对象的残差变化，仅占平均残差约 1.56%；正确描述分支的平均空槽权重也很小。已有接口没有充分展示对象条件或拒识的实际贡献。

SG-EKR seed 0、三轮的正确支持描述分支，CW AUC 比名称分支低 0.222 pp、比错配分支低 0.208 pp。这是该实现和预算的结果，不是所有条件知识或随机图的结论。

### 3.3 十轮复核已经有结果：延长训练没有挽救当前正确描述分支

共享值对齐 EKR-v2 的三轮主候选选中 epoch 0；当时训练 loss 仍下降，尚不能据此判定整个方向失败。10 月 8 日随后完成了“四组从第 3 轮续到总计第 10 轮”的探索性复核。最新本地结果如下：

| seed 0 分支 | 选中 epoch | CW AUC（%） | 相对共同底座（pp） |
| --- | ---: | ---: | ---: |
| 名称 + 值对齐 | 6 | 43.4762 | +0.3530 |
| 正确描述 + 值对齐 | 0 | 43.1232 | 0.0000 |
| 错配描述 + 值对齐 | 0 | 43.1232 | 0.0000 |
| 正确描述 + 固定值 | 2 | 43.2073 | +0.0841 |

正确对齐描述相对名称对照低 0.3530 pp、相对 fixed 低 0.0841 pp；新旧 proceed 均为 false。正确对齐描述的第 10 轮比共同底座低 0.8967 pp，loss 仍下降。它说明当前训练损失改善没有转成 VAL 组合收益，不能再把这一实现的缺点主要归因于“只有三轮”。

[本地独立审阅记录](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_budget10_review_20261008/independent_audit.json)验证了缓存分数、原生指标和选模规则；归档不含权重，本轮未重新进行 GPU 权重推理。十轮仍不自动代表收敛，单 seed 也不否定所有 GloVe 知识接口。当前实现应降优先级；新 USG 应以新的接入假设和固定预算重新验证，而不是继续无限延长旧 router。

### 3.4 当前 B/F 逻辑方案不能与 USG 试验混为一轮

最新 B/F 手册已把当前执行范围收敛到 CPF + LOGICZSL 三项逻辑损失：A_ref/B、E_ref/F 两组成对比较，保留原 CPF 推理，无新增可训练 head、无随机图。包内状态是本地 CPU 和真实 GloVe 检查通过，真实图像 CUDA 冒烟与完整 B/F 训练尚未执行。

它与 USG 有两点联系：重建的 OI/OE/AE 资产可供未来版本审核参考；逻辑监督提供“知识只进入训练约束”的独立接口对照。但当前 B/F 的新逻辑知识与 F 的旧文本/校准缓存不是同一份资产，不能把 B/F 指标直接当成 USG 的知识内容归因。

因此本文提供后续 USG 设计，不自动改变“先完成 B/F”的执行顺序，也不向 B/F 同时加入随机图、MV 或条件 head。未来若比较逻辑约束与图条件预测，必须另建相同知识版本、同预算、同推理规则的实验包。

本节依据：[完整实验记录](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/截至20261007_CPF_KCC_语义归因与EKR完整实验记录.md)、[机制诊断](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/name_ekr_mechanism_review_20261007/名称锚点EKR_机制诊断审阅与下一步_20261007.md)、[SG-EKR 重算决定](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/sgekr_review_20261008/recomputed_decision.json)、[十轮重算决定](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_budget10_review_20261008/recomputed_decision.json)、[十轮轨迹](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_budget10_review_20261008/trajectory_summary.json)、[B/F 方法重建手册](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md)。

## 4. 先把 USG 的贡献拆清楚

原 USG-CPF 定义为 Uncertain Semantic Graph Conditional Probability Framework。候选关系来自 OI、OE、AE；同一语义关系的正反向消息共享一个 Bernoulli 变量。

| 要回答的问题 | 必需证据 | 最容易混入的解释 |
| --- | --- | --- |
| 外部知识内容有用吗 | 正确关系/描述优于同结构名称与多个错配控制 | 更丰富输入或类别代码 |
| 图拓扑有用吗 | 图优于能使用相同描述的集合聚合及重连控制 | 普通描述加权 |
| 任务适配边权有用吗 | learned mean 优于固定权重与 relation-only 控制 | 每类关系的整体缩放 |
| 随机结构有用吗 | 采样模型优于固定 EdgeDrop；同权重 MC 优于均值图/单图 | 普通随机正则或集成 |
| 特定条件机制有用吗 | 同知识普通融合、去对象条件和输入干预 | 一般低秩适配容量 |
| 能解释知识正确性吗 | 独立人工审核与边检测指标 | 对 seen 分布的任务效用 |

这些是贡献的验证关系，不要求每项实验都涨点。结果支持哪一层，就保留哪一层主张。

## 5. 一个需要优先排查的结构问题：星形图可能等价于描述加权

原 Lite 在 OE/AE 星形关系上做一层聚合。若描述节点仅连接自身所属原语、消息输入被冻结、度数固定，则其消息为：

$$
m_i(G)
=
\frac{1}{d_i^0}
\sum_{j\in\mathcal N(i)}b_{ij}Tz_j^{(0)}.
$$

把均值图代入后：

$$
m_i(\boldsymbol\pi)
=
\frac{1}{d_i^0}
\sum_{j\in\mathcal N(i)}\pi_{ij}Tz_j^{(0)}.
$$

这与使用相同变换、相同归一化的“描述槽加权求和”完全相同。后续再加同一个 GELU/LayerNorm，并不会凭空增加跨原语关系信息。反向边若在同一层仍读取初始节点，也不会立即影响原语的本层输入。

因此，首轮必须保留能使用同一描述的加权集合基线。若实际实现可代数化为同一函数，应直接确认等价，避免把两个相同计算流程反复训练后包装成结构增益。

OI 共享父类、共享描述节点或真正的第二轮消息可能引入跨原语关联。但这会改变传播路径，也可能造成过平滑。应先明确哪些节点确实共享，再验证两轮消息的增量。

这一分析是对当前 USG 接口的推导；CGE 的显式组合图并不能自动消除它。

## 6. 推荐的最小接入：保留强基线，让图知识进入属性条件分布

建议为新的探索版本单独登记 USG-Cond-pilot。它与原 USG-Lite 的区别是：首轮只训练图和条件 head，冻结 CPF 特征与原分数；图更新作为新增条件证据，不立即替换 CPF 内部注意力原型。

这样可直接复用 10 月 8 日的[知识条件概率方案](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/kcc_probability_plan_20261008/知识条件概率建模_下一步稳妥实践方案_20261008.md)，同时检验随机外部关系。原 Lite 的“图替换原语表并重算注意力”保留为后续独立接入消融。

### 6.1 统一记号与候选空间

| 符号 | 含义 |
| --- | --- |
| $B(x,o,a)$ | 冻结 C_text 的原始三分支融合分数 |
| $u(x)$ | 冻结的 300 维属性视觉特征 |
| $\mathcal A,\mathcal O$ | 当前完整属性、对象词表 |
| $\mathcal C_{\mathrm{all}}$ | 完整对象与属性笛卡尔积 |
| $\mathcal Y_s$ | 训练 seen 组合，仅用于训练身份及评价 bias |
| $G_s$ | 第 $s$ 个图样本 |
| $\delta_s(x,o,a)$ | 图知识进入条件 head 后的分数证据 |
| $T$ | 分布建模温度；首轮建议固定 0.05 |
| $S$ | 图样本数，与模型训练 seed 数不同 |

公式中统一写 $P(o,a)$；数据和 evaluator 的 pair 仍按原契约存为 $(a,o)$。转换只能发生在明确的索引 gather 中。

### 6.2 从完整原分数定义参考分布

$$
P_B(o,a\mid x)
=
\frac{\exp\bigl(B(x,o,a)/T\bigr)}
{Z_B(x)}.
$$

$$
Z_B(x)
=
\sum_{o'\in\mathcal O}
\sum_{a'\in\mathcal A}
\exp\bigl(B(x,o',a')/T\bigr).
$$

对象边缘与属性条件分别是：

$$
P_B(o\mid x)
=
\sum_{a\in\mathcal A}P_B(o,a\mid x).
$$

$$
P_B(a\mid o,x)
=
\frac{\exp\bigl(B(x,o,a)/T\bigr)}
{\sum_{a'\in\mathcal A}\exp\bigl(B(x,o,a')/T\bigr)}.
$$

这个对象边缘由完整 CPF 分数诱导，一般不同于原对象辅助头的 softmax。它也不是新增模块的创新；作用是保住原组合分支和评分接口。

### 6.3 每张图只改变属性条件

$$
P_s(a\mid o,x)
=
\operatorname{softmax}_{a}
\left(
\frac{B(x,o,a)+\delta_s(x,o,a)}{T}
\right).
$$

$$
P_s(o,a\mid x)
=
P_B(o\mid x)P_s(a\mid o,x).
$$

每个图样本都保持同一个对象边缘；知识通过图像和候选对象交互，重新分配对象内部的属性概率。

该首轮模型不会训练对象边缘本身。它仍可能改变联合 MAP 组合的对象标签，因为每个对象下最高属性概率可以变化。“对象边缘不变”不等于“pair 预测中的对象一定不变”。对象边缘 argmax 与 pair argmax 投影出的对象准确率应分别统计。

## 7. 建议的图编码器与条件证据

### 7.1 固定自身路径，学习外部关系保留概率

首轮节点包括属性、对象及其描述，OI 父类保留在资产中，之后独立加入。每个原语保留名称表示和确定性自身路径。

$$
q_\phi(G\mid\mathcal K)
=
\prod_{e\in E_{\mathrm{ext}}}
\operatorname{Bern}(b_e;\pi_e).
$$

$$
\pi_e
=
\sigma\left(
\operatorname{logit}(\pi_e^0)+h_\phi(k_e)
\right).
$$

$k_e$ 包含端点语义、差值/乘积、关系类型和 OOV 信息；不包含 pair 的 seen/VAL/TEST 身份或合成污染标记。$D_s$ 通过训练目标影响学得的参数，MLP 的输入仍是静态知识特征。

最后一层零初始化，使初始分布等于人为先验。$\pi_e^0=0.7$ 是旧方案的起点，不是知识正确率；先固定一个值，在确认阶段才做有限敏感性试验。

### 7.2 浅层关系消息

固定候选度：

$$
d_{i,r}^0
=
\max\left(
1,\sum_{e:j\to i,\ \mathrm{type}(e)=r}w_e
\right).
$$

$$
m_{i,r}^{(s)}
=
\frac{1}{d_{i,r}^0}
\sum_{e:j\to i,\ \mathrm{type}(e)=r}
b_e^{(s)}w_eT_rz_j^{(0)}.
$$

$$
h_i^{(s)}
=
\operatorname{LN}
\left(
z_i^{(0)}
+
\operatorname{GELU}
\left(\sum_rm_{i,r}^{(s)}\right)
\right).
$$

首轮使用低秩关系变换，原 GloVe 冻结。OI/OE/AE 分开变换；不把上位类关系、描述关联和属性–对象组成关系都当成同一种“应接近”的边。

### 7.3 有图像和对象依赖的轻量 head

建议的 rank 为 $k=32$。名称向量记为 $n_o$；图表示经共享投影得到对象和属性语义：

$$
q(x,o)
=
\operatorname{LN}\bigl(U_vu(x)+U_on_o\bigr).
$$

$$
t_o^{(s)}=P_oh_o^{(s)},
\qquad
t_a^{(s)}=P_ah_a^{(s)}.
$$

$$
r_{o,a}^{(s)}
=
\operatorname{LN}
\left(t_o^{(s)}\odot t_a^{(s)}\right).
$$

$$
\delta_s(x,o,a)
=
c\,\tanh\left(
\frac{w^\top\bigl(q(x,o)\odot r_{o,a}^{(s)}\bigr)}
{\sqrt{k}}
\right).
$$

$q,t_o,t_a,r,w$ 均为 $k$ 维；$c=0.10$ 是建议的分数幅度上界。输出向量 $w$ 零初始化，其他投影正常初始化，故初始 $\delta_s=0$。

不要同时把输出向量和上游全部投影置零。第一步主要更新 $w$，后续梯度再进入图和后验；应检查这一正常的梯度时序。

这是一个候选条件 head。它与无图普通融合共享同一概率读出；低秩交互本身不能作为图创新的证据。

## 8. 公式重点：精确回放与 MC 的评分单位

### 8.1 单图分数的正确读出

定义每个对象下的 logsumexp：

$$
L_{B,o}
=
\operatorname{LSE}_{a}
\left(B(x,o,a)/T\right).
$$

$$
L_{s,o}
=
\operatorname{LSE}_{a}
\left((B(x,o,a)+\delta_s(x,o,a))/T\right).
$$

则给原 evaluator 的单图分数可以写成：

$$
s_s(x,o,a)
=
B(x,o,a)+\delta_s(x,o,a)
-
T\bigl(L_{s,o}-L_{B,o}\bigr).
$$

它满足：

$$
s_s(x,o,a)
=
T\log P_s(o,a\mid x)+T\log Z_B(x).
$$

最后一项仅依赖图像，因而不改变同一图像的类别排序。更重要的是，$\delta_s=0$ 时有 $s_s=B$；代码关闭分支时直接返回原 $B$，可获得精确回放。

### 8.2 平均概率，而非平均分数

$$
\overline P(o,a\mid x)
=
\frac{1}{S}\sum_{s=1}^{S}P_s(o,a\mid x).
$$

保持原分数尺度的 MC 读出为：

$$
s_{\mathrm{MC}}(x,o,a)
=
T\log\left(
\frac{1}{S}
\sum_{s=1}^{S}
\exp\bigl(s_s(x,o,a)/T\bigr)
\right).
$$

实现使用 logsumexp。由于每张图共享同一个 $Z_B(x)$：

$$
s_{\mathrm{MC}}
=
T\log\overline P+T\log Z_B(x).
$$

这样概率平均与原 cosine 分数单位对齐。直接把 $\log\overline P$ 交给原 evaluator 而保持旧 bias 网格，会改变 bias 的有效尺度；该变化不能归给图收益。额外校准温度另行在 VAL 固定，原生 AUC 的输入分数尺度保持一致。

### 8.3 CW 掩码与图平均通常不交换

本版本统一顺序：

1. 各图都在完整属性×对象网格上定义 $P_s$。
2. 在该网格上平均得到 $\overline P$。
3. 原 evaluator 应用自己的 CW/OW 候选掩码。
4. 如需 CW 的归一化概率指标，再限制并归一化 $\overline P$。

对候选集合 $\mathcal C$：

$$
\overline P^{\mathcal C}(y\mid x)
=
\frac{\mathbf 1[y\in\mathcal C]\overline P(y\mid x)}
{\sum_{y'\in\mathcal C}\overline P(y'\mid x)}.
$$

若每张图在 $\mathcal C$ 上的概率质量不同，先逐图限制、归一化再等权平均，会改变混合模型。这是原归档方案与新完整网格定义需要明确区分的接口；不能混用后仍声称只是更换图模块。

公式中的 $\mathcal C$ 必须来自当前 evaluator 的真实 mask。本地 UT CW 分数表有 116 列，但原生允许集合为 98 类，不能根据导出列数猜候选规则。

## 9. 训练目标、采样与固定预算

### 9.1 条件似然监督

首轮用 TRAIN 的真实对象监督完整属性词表。图混合目标为：

$$
\mathcal L_{\mathrm{data}}
=
-\frac{1}{B_{\mathrm{batch}}}
\sum_{i=1}^{B_{\mathrm{batch}}}
\log\left(
\frac{1}{S_{\mathrm{train}}}
\sum_{s=1}^{S_{\mathrm{train}}}
P_s(a_i\mid o_i,x_i)
\right).
$$

这是有限样本 MC 混合似然；有限 $S$ 的随机估计带来优化误差，不能说它等于精确期望目标。

固定对象边缘下，联合 NLL 还包含 $-\log P_B(o_i\mid x_i)$，但它对新模块是常数。不宣称首轮训练了对象分布。

本目标与原 Lite 的 seen-pair CE 不同，所有新对照必须采用相同条件目标。完整属性 CE 也可能改变 seen/unseen 的竞争，需同时报告标准组合指标与条件属性诊断。

### 9.2 KL 的归一化和解释

$$
\mathcal L_{\mathrm{KL}}
=
\frac{1}{|\mathcal R|}
\sum_{r\in\mathcal R}
\frac{1}{|E_r|}
\sum_{e\in E_r}
\left[
\pi_e\log\frac{\pi_e}{\pi_e^0}
+
(1-\pi_e)
\log\frac{1-\pi_e}{1-\pi_e^0}
\right].
$$

没有边的关系类型不进入该平均。主目标：

$$
\mathcal L
=
\mathcal L_{\mathrm{data}}
+
\lambda_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
$$

关系平衡平均不是整张图分布 KL 的原始总和；该目标也不是自动成立的 Bayesian ELBO。建议将其描述为 KL 正则化的结构分布学习。

### 9.3 训练和推理的随机边

$$
\widetilde b_e
=
\sigma\left(
\frac{\operatorname{logit}(\pi_e)+\log u_e-\log(1-u_e)}
{\tau_g}
\right),
\qquad u_e\sim\operatorname{Uniform}(0,1).
$$

hard-forward/soft-backward 训练，测试采用 Bernoulli 硬采样。同一关系的正反向消息共享 $b_e$。straight-through 梯度有偏；有限温度下 soft Concrete 的均值不能直接认作 $\pi_e$。

### 9.4 建议的 pilot 预算

| 项目 | 建议设置 |
| --- | --- |
| 共同锚点 | UT v2 C_text seed 0 的既定最佳权重 |
| 冻结输入 | 一次缓存 TRAIN/VAL 的 $u(x)$ 与完整网格 $B$ |
| 新模块预算 | 每个身份固定 10 epoch，保存 3/5/10 轮轨迹 |
| 优化 | AdamW，LR 0.001，weight decay 0.0001 |
| batch / rank | 256 / 32，按实际显存预先调整 |
| 温度 / 幅度 | $T=0.05$，$c=0.10$ |
| 图层 | 一层主版本；两层仅在传播假设明确时比较 |
| 采样 | 训练 $S=2$，推理比较 1/5/10 |
| KL | 初始 0.01；首轮固定，不逐轮搜索 |
| 图温度 | 主版本固定 1.0；退火另立消融 |
| 模型选择 | 各身份 epoch 0–10 采用同一 VAL CW AUC 规则 |

这些是探索初值。先测单 head 和采样的实际成本；旧 EKR 的 ViT 训练时间不适用于缓存 head。若决定阶段化暖机，总更新次数、学习率计划和对照暖机时间都应在新 VAL 前固定。

已有 EKR-v2 十轮复核未通过，但它不成为设计 USG 的必要门槛，两者检验不同接入假设。执行上仍尊重当前先完成 B/F 的顺序；USG 首轮复用缓存和审计工具，不同时扩大多套全模型训练。

## 10. 实验矩阵：按问题分包实施

### 10.1 第一包：确定性知识与结构

| 身份 | 新模块 | 核心作用 |
| --- | --- | --- |
| U_base | 冻结 C_text，分支关闭 | 强参照与精确回放 |
| U_name | 同模块、同参数，描述槽填自身名称 | 容量与额外训练对照 |
| U_set | 同描述的共享聚合/普通融合 | 图是否超越文本聚合 |
| U_graph | OE/AE 硬图、关系化消息、同条件 head | 确定关系候选 |
| U_perm1/2 | 同图与 head，两份固定知识归属错配 | 正确内容归因 |

首轮共五个新训练身份。若 U_graph 与 U_set 在实际计算上相同，用等价性核查取代重复训练，并把图拓扑问题留到 OI 共享关系或两层消息版本。

当前每个原语有三条描述，名称控制可直接复制名称填槽。若 MIT/C-GQA 的数量或 OOV mask 不一致，控制必须匹配这些可用性信息；不能让某一方法独占更好的文本或掩码。

### 10.2 第二包：边分布与随机机制

| 身份 | 训练图 | 推理 | 回答的问题 |
| --- | --- | --- | --- |
| U_mean | 学得 $\pi_e$ 的软图 | 一次均值图 | 边权学习 |
| U_drop | 固定先验 Bernoulli | 同样数量 MC | 随机正则/集成 |
| U_sample | 学得 $\pi_e$ 的 Bernoulli | MC | 主要随机候选 |
| U_rel | 每类关系只有一个保留率 | 同样数量 MC | 端点语义是否必要 |

再从同一个 U_sample checkpoint 评估均值图、单图、5 图、10 图。这区分“随机训练”与“测试平均”；单独训练的 U_mean 不能代替同权重均值图控制。

U_mean 失败并不数学上否定 U_sample。若仍有明确噪声机制假设，可只开展一次预先登记的 U_drop/U_sample 小预算比较；没有可重复内容或机制信号时降低该实现优先级。

### 10.3 CGE-Core 作为第三个、独立的结构实验

采用两因素对照：

| | 外部知识关闭 | 外部知识开启 |
| --- | --- | --- |
| 无组合核心图 | 匹配无图 head | 外部图版本 |
| 有组合核心图 | Core-only | Core + 外部图 |

自环和组合–原语归属边固定；属性–对象共现边、seen-only 拓扑单独比较。建议首轮只用 TRAIN 组合构图，通过共享 composer 为全部候选读出；不能把“没有 TRAIN 共现边”解释成不可行组合。

如果按原 CGE 在全部 CW 候选上构图，应明确候选词表与拓扑假设，并单独命名。OW 下禁止把验证/测试出现过的 pair 列表当成外部可行性知识；全笛卡尔积也无需物化为图节点。

Core 的增量、外部关系增量和二者交互分别报告。低秩 pair composer 使用同结构无图控制，不把它的容量收益归给 Core。

## 11. 针对已观察问题增加机制诊断

### 11.1 对象条件是否真的被利用

记录同图同属性在不同候选对象下的证据变化：

$$
V_{\mathrm{obj}}
=
\mathbb E_{x,a,o}
\left[
\left|
\delta(x,o,a)-\mathbb E_{o'}\delta(x,o',a)
\right|
\right].
$$

同时报告去对象条件重训、固定对象上下文错配、条件属性准确率，以及对名称/错配模型的成对纠正与破坏。

$V_{\mathrm{obj}}>0$ 只说明依赖对象，不说明依赖有益；收益仍由匹配对照检验。

### 11.2 避免把无效常数当作条件知识

对给定 $(x,o)$，若所有属性共享相同增量 $k(x,o)$，softmax 会完全消去它：

$$
\operatorname{softmax}_a(z_a+k)
=
\operatorname{softmax}_a(z_a).
$$

因此应监测对象内属性证据的中心化幅度：

$$
\delta^\circ(x,o,a)
=
\delta(x,o,a)-\mathbb E_{a'}\delta(x,o,a').
$$

如果残差范数很大，但 $\delta^\circ$ 很小，新 head 可能只学了在条件读出中被抵消的对象偏置。原均值消息和 LayerNorm 也应按同一标准分析。

### 11.3 Boat Shoes 问题的正确边界

C_text 锚点的分支关闭只能返回 C_text，不能返回 A_unit。冻结对象边缘版本也无法撤销已存在于该边缘中的对象偏差。

先区分两种目标：新增条件知识相对强 C_text 的增量，以及相对名称模型的文本伤害修复。后者需单独设置 A_unit 锚点与相同条件知识对照，并持续报告强 C_text 参照。

若 GT-object 属性改善而组合没有改善，检查对象边缘与候选竞争；后续可单独比较一次属性到对象回写或对象映射微调。不能把条件属性准确率自动算作组合方法成功。

## 12. 图不确定性、知识污染与校准

### 12.1 全局边分布与图像条件是不同层次

首轮 $\pi_e$ 在数据集内共享；条件 head 的 $q(x,o)$ 才依赖图像。它不会为每张图像单独选择一张后验图。

若最终需要图像相关的边概率，可扩展为 $\pi_e(x,o)$，但缓存、参数规模和拒识监督都会变化，应另立 image-conditioned 版本并比较 static-posterior 对照。不能把当前全局边分布称为已实现的视觉可靠性判断。

### 12.2 干净和污染输入都要报告绝对性能

先固定 0/10%/30% 的推理期端点污染及缺边图，各方法共享文件和 hash。保持已选模型、bias、校准温度不变；再比较识别与概率指标。

人工换端点可能仍形成合理关系，因此统称“合成归属污染”。只有人工审核子集才能标为确证错误边。保留分布只能删选候选边，不能恢复从未提供的边。

比较时既报干净性能、污染后绝对性能，也报下降幅度；较低起点、较小跌幅不足以证明鲁棒。

### 12.3 结构分歧

在同一完整候选分布下：

$$
U_{\mathrm{struct}}(x)
=
H\bigl(\overline P(\cdot\mid x)\bigr)
-
\frac1S\sum_sH\bigl(P_s(\cdot\mid x)\bigr).
$$

它表示该结构采样模型的预测分歧，不自动等于真实 epistemic uncertainty。小负值应检查数值精度；图样本之间几乎无差异时，不能宣称有效表达了结构不确定性。

报告固定工作点 NLL、Brier、ECE、risk–coverage/AURC；与预测熵和最大概率控制比较。全局后验、不同噪声样本和不同训练 seed 的离散性分别记录。

### 12.4 边概率的解释

$\pi_e$ 首先是“任务适配保留概率”。语义正确但暂时无用的边可能被删，错误但与 seen 标签相关的边可能被保留。

若要提出“知识可靠性学习”，使用独立的 held-out 人工审核边集，并检验 $1-\pi_e$ 的错误边 AUROC/AUPRC；不能把同一批审核结果既用于开发 prior，又用于最终审核指标。

## 13. MV 能量公式：保留推导，后移实验

设无向语义边每条只计一次；给定模型参数后，确定表示 $\bar Z$ 不依赖此次 Bernoulli 图样本。令：

$$
d_e=\|\bar z_i-\bar z_j\|_2^2,
\qquad
\mathcal E_G=\sum_eb_ew_ed_e.
$$

若边独立，则：

$$
\mathbb E_G[\mathcal E_G]
=
\sum_e\pi_ew_ed_e.
$$

$$
\operatorname{Var}_G(\mathcal E_G)
=
\sum_e\pi_e(1-\pi_e)w_e^2d_e^2.
$$

若 $\bar Z$ 是均值图的确定输出，公式仍可用于对假想独立采样的能量求矩；对模型参数求总梯度时，还要保留 $d_e$ 对参数的依赖。

若将 $Z(G)$ 代入距离，$d_e$ 与边样本相关，上述闭式一般不再成立。若图有相关边，方差还包含协方差项。

能量除以边数后的严格方差为：

$$
\operatorname{Var}\left(\frac{\mathcal E_G}{|E|}\right)
=
\frac{\operatorname{Var}(\mathcal E_G)}{|E|^2}.
$$

旧方案的“方差总和除以边数”可以作为规模归一化惩罚，但应以正则项命名，不能称为平均能量的方差。

此外，OI 是有方向的层级语义，描述也并不要求与原语原始向量相等。若做平滑，先在关系变换后的空间定义适合该关系的距离；不要一律拉近所有节点。主 pilot 关闭 MV，机制明确后比较 mean、mean+variance。

## 14. 如何吸收后续七篇论文讨论

以下是把[七篇论文笔记](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/七篇CZSL论文_关键Idea与当前实验借鉴_20261008.md)转成 USG 的候选改进，均需单独消融：

| 已讨论机制 | 对 USG 的候选用途 | 先验证什么 |
| --- | --- | --- |
| SPA 式局部语义保持 | 约束学习图投影后的语义漂移 | 当前 GloVe 邻域是否视觉合理；正确/错配知识是否同向受益 |
| Duplex 式视觉原型 | TRAIN 视觉证据为文本/关系效用提供辅助诊断 | 使用 held-out head 图像；不能当严格未见组合测试 |
| H²EM 式难负样本 | 同对象难属性、同属性难对象 | 相对普通 hard-negative 的知识增量 |
| PRPC 式双向纠错 | 条件属性之后一次对象更新 | 错误对象注入的纠正率和原正确预测破坏率 |
| FlowComposer 式共享组合 | 低秩读出共享原语，避免自由 pair 参数表 | 同容量无图读出是否已解释收益 |
| DIFFCZSL 式训练教师 | 给视觉/语义投影提供训练期对齐信号 | 正确条件、错配条件和普通视觉教师比较 |

首轮仅采用共享组合和条件读出所需的轻量结构。局部保持、视觉支持、双向更新与教师监督应依据真实失败诊断逐项选择，不同时启用。最新十轮结果中值对齐有明显语义漂移却无正确内容增量，因此局部保持值得作为后续诊断候选，但漂移与性能下降同时出现不构成因果证明。

视觉 support 的标签监督不能进入部署时的图采样输入。如果需要严格的 unseen-composition 验证，必须重新排除对应组合对 backbone 的训练影响；在已训练 backbone 的 TRAIN 缓存上分折，只是机制诊断。

## 15. 评价、预算决策与可发表的贡献边界

### 15.1 三类前提分别解释

实现正确性是必需前提：关闭分支回放、图像与词表配对、全候选归一化、有限值、梯度和缓存身份不能失败后继续评价。

预算门槛是投入规则：先固定一个完整 pilot 预算，避免根据三轮结果或单个好 epoch 无限追加。既有 false gate 原样保留；新 USG 使用新 protocol。

方法有效性由匹配差值决定：低于预定 threshold 可以报告为弱结果，不能写成旧规则通过；超过 threshold 也不是统计显著性证明。

### 15.2 建议的新 USG pilot 筛选

沿用新条件概率方案的量级作为预算起点，在首个新 VAL 结果前冻结：

| 维度 | 建议量级 | 解释 |
| --- | --- | --- |
| seed 0 增量 | 相对强 C_text CW AUC 至少 +0.20 pp | 决定是否确认 |
| 内容归因 | 相对名称与两份错配均值至少 +0.10 pp | 同时完整报告每个错配差值 |
| 保护项 | CW HM、OW AUC 至少不低于 −0.5 pp；OW Umax 不低于 −2 pp | 限定取舍 |
| 三 seed 确认 | 同设置报告成对差、均值与标准差 | 三个 seed 不自动证明显著 |
| 不确定性主张 | 对 learned mean、固定 EdgeDrop 与同权重推理有重复支持 | 识别、污染、校准分开判断 |

如果主研究目标改为知识污染或概率质量，应在实验前另定主指标与保护项，不在 CW 不利后临时改名。采样收益可以来自概率质量而非 top-1；应按实际贡献表述。

### 15.3 根据结果收缩模型

- 正确知识与错配一样好：记录容量/适配收益，暂不主张知识归属贡献。
- 图与集合聚合一样好：采用更简单实现，或把后续问题限定到真实跨原语关系。
- 采样与 EdgeDrop 一样好：随机正则/集成解释仍足够，边效用网络尚无增量。
- MC 与均值图一样好：优先均值图部署，保留训练期随机性的单独结论。
- Core 无收益但外部图有效：保留外部图，Core 不成为阻碍。
- 短训无收益、仍未充分优化：只在有明确理由时另立一次固定预算复核。
- 全指标无信号：降低当前实现优先级，保留负结果及可复用资产。

### 15.4 最近邻方法必须正面区分

Co-CGE 已使用组合可行性影响图权重与分类目标；CVGAE 已学习原语图的变分嵌入与组合可行性；Learning Discrete Structures 已学习离散结构；DropEdge 已随机删边。因此图、可学习边、Bernoulli、KL 和 MC 的一般组合不足以建立新颖性。

可能的研究贡献应落在：**外部 OI/OE/AE 关系的任务适配分布如何进入可回放的组合条件预测，以及它在正确知识、结构污染、概率质量上的独立增量。** CGE-Core 或冻结对象边缘只是设计选择；是否构成贡献取决于最近邻对照与实验。

来源：[Co-CGE](https://arxiv.org/abs/2105.01017)、[CVGAE](https://arxiv.org/abs/2204.11848)、[Learning Discrete Structures](https://proceedings.mlr.press/v97/franceschi19a.html)、[DropEdge 官方实现](https://github.com/DropEdge/DropEdge)。

## 16. 下一步具体交付顺序

1. 冻结新的方案身份、锚点、知识版本与候选契约；复用已审核 UT v2 原文，不改旧结果。
2. 核查原 USG 的 OE/AE 一层图是否退化为集合加权，明确要验证的传播结构。
3. 实现完整网格条件概率读出和 MC 分数尺度，完成真实 CPF/evaluator 回放。
4. 一次缓存 UT 的冻结特征和完整分数，完成确定性五身份固定预算 pilot。
5. 依据信号进入有限的 learned mean、EdgeDrop、learned Bernoulli 比较；同权重评估 mean/1/5/10。
6. 内容或机制有支持后确认不同模型 seed；加入预先固定的污染与校准分析。
7. 另做 CGE-Core 两因素结构对照；只有它有增量时才纳入主方法。
8. 冻结有效版本后检查 MIT 视觉词义与关系，再做独立第二数据集；C-GQA 最后扩展。
9. 原 Lite 的注意力原型替换、对象分布更新、MV 和逻辑损失按真实瓶颈逐项加入。

在 C-GQA 上，组合概率与 object-level logsumexp 分块计算；归一化必须覆盖完整属性维度，不能对每个 pair chunk 独立 softmax。缓存 head 不复制 backbone 前向；原 Lite 如需图相关注意力则应重新计算对应下游流程，不能拿不含图作用的缓存冒充同一模型。

MIT 的旧描述有词义错误，例如 bent 的意志坚定义与视觉弯曲不一致。审核使用词表含义和 TRAIN 证据，不能依据 VAL/TEST 错误挑句子。C-GQA 版本已更新，官方仓库也有说明，原论文旧数据数字与当前本地 split 不直接相减。[CGE 官方仓库](https://github.com/ExplainableML/czsl)

## 17. 公式与排版检查范围

本版使用标准 Markdown 数学分隔符，块公式的起止标记各占独立一行。长式拆成多个短块；表格只保留短符号，条件竖线用 $\mid$，集合基数和范数使用对应 LaTeX 命令。

本轮检查包括：所有数学表达式的 KaTeX 解析、代码围栏/数学分隔符、表格列数、本地来源链接，以及单图/MC 概率恒等式的合成数值核对。KaTeX 可解析表示公式语法成立；概率恒等式检查表示所列数学读出成立，两者都不代表模型性能已得到验证。

数值检查覆盖 36 组温度、词表维度和分数幅度设置，每组 5 张图；核对单图与 MC 联合分布、对象边缘、零分支回放和对象内常数消去，并枚举独立 Bernoulli 图验证能量均值/方差。概率相关最大绝对误差小于 $3\times10^{-14}$。这不是实际 head 梯度检查，也不是原生 evaluator/GPU 模型回放。

公式预览使用标准 KaTeX 页面，可对照编辑器中实际显示。已在无头 Edge 中检查桌面宽度：41 个块公式和所有表格均无横向溢出，字体加载正常；窄屏对长公式和宽表格提供局部横向滚动。本轮未验证 Codex 原生 Markdown 渲染器。若某个 Markdown 查看器不支持数学扩展，应使用本次提供的渲染预览；源文件保留标准数学语法。

## 18. 本次直接采用的材料

| 材料 | 用于本文的部分 |
| --- | --- |
| [USG-CPF 原方案](C:/Users/llenovo/Desktop/20261005方案归档/USG-CPF随机语义图实验设计方案.md) | 节点、边后验、采样、原型接口、MC、MV 与消融 |
| [CPF/KCC 完整记录](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/截至20261007_CPF_KCC_语义归因与EKR完整实验记录.md) | 三 seed 归因、bias=0、Boat Shoes、名称锚点 |
| [名称锚点机制诊断](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/name_ekr_mechanism_review_20261007/名称锚点EKR_机制诊断审阅与下一步_20261007.md) | 对象条件弱效应、空槽与输入干预边界 |
| [EKR-v2 独立决定](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_review_20261008/recomputed_decision.json) | 三轮预算下的真实筛选结果 |
| [SG-EKR 独立决定](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/sgekr_review_20261008/recomputed_decision.json) | 支持筛选未获增量 |
| [十轮预算回溯](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_budget10_20261008/Epoch回溯审查与v2十轮续训方案_20261008.md) | 不过度推断短训失败，独立预算复核 |
| [十轮复核实际决定](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_budget10_review_20261008/recomputed_decision.json) | 正确描述仍选 epoch 0；当前实现降优先级 |
| [十轮轨迹](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ekr_v2_budget10_review_20261008/trajectory_summary.json) | loss/VAL 脱钩、预算结果解释 |
| [知识条件概率方案](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/kcc_probability_plan_20261008/知识条件概率建模_下一步稳妥实践方案_20261008.md) | 强 C_text、完整网格、固定对象边缘、精确读出 |
| [七篇论文笔记](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/七篇CZSL论文_关键Idea与当前实验借鉴_20261008.md) | 局部保持、视觉证据、共享组合、双向纠错、训练教师 |
| [CPF B/F 方法重建](C:/Users/llenovo/Documents/ChatGPT/CZSL/output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md) | 当前执行边界、知识版本、逻辑接口与 USG 的隔离 |

CGE 的具体公式与消融以 CVPR 2021 终稿为准。arXiv v2 与终稿的 C-GQA 数字存在差异，本版未混合这些版本的主结果表。

