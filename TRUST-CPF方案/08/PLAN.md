本页保留 v6 原研究安排。第一批 State 与第二批 Stage 均已完成，见 [08 分析](analysis/README.md) 与 [09 分析](../09/analysis/README.md)。原始计划的后续分支按实际结果解释。

# TRUST-CPF 公式修订与后续实验方案 v6

日期：2026-10-25  
依据：第五版诊断 PDF，以及弱基底、强基底、响应条件化、Compact、Align、Competition 10 epoch、R5 10 epoch 的已归档结果。  
当前平台：UT-Zappos、CLIP ViT-L/14 缓存、工程内强 CPF 基底、96 条关系、seed 3351／3352／3353。

**下一步主线确定为：U-NLL 配对效用监督 → 多教师联合拟合 → Uniform 完成训练 → 在同一冻结模型上提取响应、分配关系并识别。**

保留 TRUST 的效用监督、响应条件化和关系分配三个研究点。主要修改训练与部署的对应方式，以及多教师拟合的正则化尺度。Relative 保留为联合拟合中的对照；Competition 目标、大型效用网络和误差惩罚暂不进入主模型。

本轮首先完成一批复用现有权重的冻结实验，随后完成一批教师阶段匹配实验。两批都使用完整三 seed 队列，不拆成逐项审批的小实验。

## 1. 全部结果把问题指向了哪里

### 1.1 已经确定的实验事实

以下 AUC 均按工程的 0–100 尺度显示。各轮训练协议不同，跨轮数字用于追踪证据；方法增益使用同轮配对对照。

| 实验阶段 | 主要结果 | 对本方案的影响 |
| --- | --- | --- |
| 弱基底 v1 | Robust 9.901／8.172，Capacity 13.902／11.240（CW／OW） | 后续以强 CPF 为共同基底 |
| 强基底前置比较 | Base 34.438／25.140；Knowledge 35.056／25.740；Shuffle 35.307／25.915 | 增加关系分支有收益，但收益不能全部归于正确知识附件 |
| 强基底完整 TRUST | CPF 34.512／25.184；Robust 35.201／26.149；Capacity 35.805／26.400；Dense 35.646／26.288 | 整体扩展有效，效用分配仍需要超过同结构对照 |
| 强基底机制与噪声 | Robust 相对 Drop 为 +0.213／+0.241；相对 Mean 为 −0.029／+0.079；相对知识打乱为 +0.168／+0.604 | 真实关系有部分正向信号；原不确定性项和噪声鲁棒性没有稳定增量 |
| 响应条件化 MLP | 34.916／25.958，低于原 Robust | 加入响应后扩大网络没有解决迁移问题 |
| Compact | 34.922／25.931；Q5 排序相关均值 0.293，打乱效用标签为 −0.022 | 紧凑预测器已能学到部分效用信息，继续保留 |
| Align | 相对 Uniform 为 −0.088／+0.008；3352 的冻结分配在三个 bank 平均提高 +0.158／+0.211，适配后 CW 优势消失 | 分配时点与随后改变的权重需要分开研究 |
| Competition 10 epoch | Uniform 35.934／26.532；Competition 35.950／26.493；U-NLL 35.729／26.316 | 10 epoch 提供了更合适的基线；Competition 未形成稳定最终优势 |
| R5 10 epoch | Relative 三 seed 均选择 κ=0，精确等于 Uniform；Absolute 35.729／26.316 | 类型内相对拟合没有单独解决正式部署问题 |

“知识打乱”和“效用标签打乱”是不同对照，后续继续分别记录。

### 1.2 最有价值的正向证据来自效用测量与 Q5 决策

Competition 的独立 Q5 评价图像上，按测量效用分配关系，U-NLL／Balanced-NLL／Competition 分别提高实际 AUC **2.773／2.789／2.688 点**。按预测效用分配时，对应增益为 **1.282／1.300／−0.059 点**。

这支持两个判断：

1. 关系分配确实有可利用的价值，U-NLL 标签能够指导实际组合识别。
2. 当前损失主要发生在“效用预测与跨状态使用”环节。直接把监督改成 Competition 没有提高这部分能力。

R5 又补充了关键证据：Relative 的 Q5 AUC 增益为 **1.003 点**，低于 Absolute 的 **1.282 点**，逐 seed 差值为 **−0.060／−0.093／−0.683**。两者在 reference 的 18 个非零强度配置，真实分配均低于对应五个类型内置换的均值。

因此，下一轮同时研究教师覆盖和使用状态；不再把“相对标签形式”当成主要突破口。

### 1.3 训练轮数已经不是当前首要问题

Competition／R5 的正式适配已完成 10 epoch；Uniform 最佳 epoch 为 **2／2／6**，有效非均匀组的最佳 epoch 也在训练中段。继续延长当前适配轨迹，不如直接利用这些已训练好的权重研究效用分配。

原 R5 报告提出的“固定 κ=1，再做非均匀适配”，调整为本方案的**固定 κ=1、冻结权重比较**。这样直接检验关系对应；统一训练预算已经由已有 Uniform 10 epoch 轨迹提供。

## 2. 对第五版 PDF 的采用与修改

PDF 成稿早于 R5 结果；本方案据最新数据更新其优先级。

| PDF 建议 | 本次决定 | 理由 |
| --- | --- | --- |
| 跨 seed、多教师联合拟合 | 进入第一批主实验 | 几乎不增加训练成本，直接扩展教师状态覆盖 |
| 绝对／相对拟合 2×2 | 已完成本地拟合复算；GPU 保留 Independent-Absolute、Pooled-Absolute、Pooled-Relative 三个预测器 | R5 已完成相对拟合的独立教师检验，避免再重复完整适配 |
| Uniform 后冻结分配 | 作为主要部署流程 | 响应提取和关系使用共享同一权重，固定模型效用的定义更直接 |
| 教师阶段匹配、重测标签 | 第二批整体实施 | 用任务内部的对应训练阶段学习“训练完成后如何用关系” |
| 两次独立响应测量 | 第一批作为重复测量诊断 | 主读数先保留旧单次探针定义，单独量化探针波动 |
| 经验误差收缩 | 保留公式，放到后续附加比较 | 收缩改变幅度，无法修正错误排序；当前先解决方向与迁移 |
| 大量阈值与逐级验证关口 | 改为两批完整队列和统一比较表 | 依据配对收益、跨 seed 一致性和计算成本作整体判断 |

第一批直接复用旧教师和 Uniform 权重；第二批只更新教师状态及其对应监督，正式识别器继续共享同一批 Uniform 权重。由此分别看到“部署顺序修改”和“教师阶段匹配”的贡献。

## 3. 本次新增的本地数据复算

### 3.1 已执行内容

已读取 R5 归档中三个 seed 的 15 份教师响应与 U-NLL 标签，完成以下 CPU 拟合：

- Independent：每个 seed 使用其 Q1–Q3 三个教师。
- Pooled：合并三个 seed 的 Q1–Q3，共九个教师状态。
- 两种教师配置分别拟合 Absolute 和 Relative。
- 每个配置用三个 Q4 的平均归一化误差选择一个共同正则参数。
- Q5 不参与拟合和选参；本次保留三／九教师预测器直接用于 Q5 和 reference，不做五折再拟合。

这次计算得到的是预测指标和一阶效用代理；尚未新增识别预测或训练。

### 3.2 结果与解释

一阶代理收益为 Q5 测量标签与概率改变量的内积，具体公式见第 4.2 节。表中均值以三个 Q5 教师为单位。

| 配置 | 选中 λ | Q5 整体 Spearman | 类型中心化 R² | 一阶代理收益均值 |
| --- | --- | --- | --- | --- |
| Independent-Absolute | 0.03333 | 0.1987 | 0.0103 | 0.03806 |
| Independent-Relative | 0.03333 | 0.1743 | 0.0109 | 0.03505 |
| Pooled-Absolute | 3.33333 | 0.1950 | 0.0252 | 0.03851 |
| Pooled-Relative | 3.33333 | 0.2493 | 0.0260 | 0.03989 |

三个 seed 的代理收益：

| 配置 | 3351 | 3352 | 3353 |
| --- | --- | --- | --- |
| Independent-Absolute | 0.04867 | 0.05085 | 0.01468 |
| Pooled-Absolute | 0.04071 | 0.02647 | 0.04836 |
| Pooled-Relative | 0.04421 | 0.02657 | 0.04889 |

联合拟合改善了 3353，并降低了 3351／3352 的代理收益，整体变得更均衡。Pooled-Absolute 相对 Independent-Absolute 的均值只增加 **0.00045**；Pooled-Relative 增加 **0.00183**。这是值得实际评分的信号，尚不能据此称为识别提升。

类型中心化 Spearman 的均值依次为 **0.2703／0.2600／0.2338／0.2665**，表明“整体相关提高”和“类型内排序提高”并不相同。原始单位中心化 MSE 由 Independent-Absolute 的 **1.17267×10⁻⁴** 降至 Pooled-Absolute 的 **1.16015×10⁻⁴**，约降低 **1.07%**。

联合模型的 Q5→reference 类型内排序相关均值仍接近零：Absolute 为 **−0.0124**，Relative 为 **−0.0083**。两个状态的真实效用也可能不同，因此排序变化本身不等于预测错误；最终判断要看当前状态的实际分配收益。

同时计算了两个简单读出：

- 九教师的同一条边历史效用均值：Q5 代理收益为 **−0.02229／+0.00235／−0.00684**。
- 直接使用支持集 NLL 响应均值：**+0.00024／−0.01641／+0.01268**。

它们没有表现出一致代理优势，仍保留为第一批的便宜参照，用于判断学习映射是否比直接使用输入更有价值。

这里的 Independent 使用三个 Q4 共同选择 λ，与旧 R5 每 seed 独立选择 α 的流程不同。因此本表和旧 R5 数值分别标注，不把差异全部归于新损失。不同教师配置各有自己的归一化参数，Q4 的归一化损失用于各自选参，跨配置比较使用原始单位误差和实际识别结果。

## 4. 修订后的完整公式链条

### 4.1 模型、图分布和风险定义

记当前工程强 CPF 及关系分支的全部识别参数为 $`\theta`$，候选组合为 $`c=(a,o)`$，关系掩码为 $`b\in\{0,1\}^{96}`$。关系消息、残差三元评分和 CPF 参数化沿用现有实现；本轮修改效用学习和部署流程。

对指定候选集合 $`\mathcal C`$，先计算每张图的候选概率：


```math
p_\theta^{\mathcal C}(c\mid x,b)
=
\frac{\exp s_\theta(x,c;b)}
{\sum_{c'\in\mathcal C}\exp s_\theta(x,c';b)}.
\tag{1}
```


三个类型的关系数分别为 $`N_t=12,36,48`$；采样分布和期望预算为：


```math
q_\pi(b)=\prod_{e=1}^{96}\pi_e^{b_e}(1-\pi_e)^{1-b_e},
\qquad
\sum_{e\in E_t}\pi_e=B_t=\frac{N_t}{2}.
\tag{2}
```


因此类型预算为 **6／18／24**。这是独立 Bernoulli 采样下的**期望边数约束**，每张实际采样图的边数可以变化。

折内教师 $`j`$ 的伪未见测量图像集记为 $`D_j^{u,m}`$，其标签属于留出的组合。沿用 Competition 的 U-NLL：


```math
R_j(\theta,b)
=
-\frac{1}{|D_j^{u,m}|}
\sum_{(x_i,y_i)\in D_j^{u,m}}
\log p_\theta^{\mathcal C_j}(y_i\mid x_i,b).
\tag{3}
```


本轮折内风险候选集合为全部 83 个 TRAIN 组合；伪未见测量图像按图像等权。风险评价仍保存 seen 图像，用于 AUC、HM 和已见性能诊断。原有测量池、评价池及教师训练图像划分继续沿用。

### 4.2 配对效用及其严格含义

固定教师状态 $`\theta_j`$，背景分布取 $`\rho_e=0.5`$。第 $`e`$ 条边的效用定义为：


```math
u_{j,e}
=
\mathbb E_{b_{-e}\sim q_{\rho,-e}}
\left[
R_j(\theta_j,b_{-e},0)
-
R_j(\theta_j,b_{-e},1)
\right].
\tag{4}
```


正值表示开边降低该教师、该风险集合下的 NLL。实现时同一背景、同一批图像、同一候选集配对计算：


```math
\widehat u_{j,e}
=
\frac{1}{K_j}\sum_{k=1}^{K_j}
\left[
R_j(\theta_j,b_{-e}^{(k)},0)
-
R_j(\theta_j,b_{-e}^{(k)},1)
\right].
\tag{5}
```


Q1–Q4 取 $`K_j=8`$，Q5 取 16。原标签已使用这种配对形式，继续保留。

令单图期望风险为 $`F_j(\theta,\pi)=\mathbb E_{b\sim q_\pi}R_j(\theta,b)`$。对某一坐标展开：


```math
\begin{aligned}
F_j(\theta,\pi)
={}&(1-\pi_e)\mathbb E_{b_{-e}}R_j(\theta,b_{-e},0)\\[4pt]
&+\pi_e\mathbb E_{b_{-e}}R_j(\theta,b_{-e},1).
\end{aligned}
\tag{6}
```


其余坐标固定时，右侧关于 $`\pi_e`$ 为仿射函数，因此：


```math
\left.
\frac{\partial F_j(\theta_j,\pi)}{\partial\pi_e}
\right|_{\pi=\rho}
=-u_{j,e}.
\tag{7}
```


围绕均匀概率作局部展开，得到效用分配的依据：


```math
F_j(\theta_j,\rho+\Delta)
=
F_j(\theta_j,\rho)-u_j^\top\Delta
+O(\|\Delta\|_2^2).
\tag{8}
```


本报告的“一阶代理收益”具体为：


```math
G_j^{\mathrm{lin}}(\pi)
=
\widehat u_j^\top(\pi-\rho).
\tag{9}
```


其单位是 NLL 的一阶变化量，既不是 AUC 点，也不是已经实际测得的风险下降。实际混合 NLL 和 AUC 在冻结识别中另行计算。

### 4.3 八维模型响应：保持信息定义，明确状态对应

支持集 $`D_j^s`$ 来自该教师的合法训练图像，每个组合最多抽取两张图，组合等权、组合内图像等权；每条边使用四个共享背景。对某支持图像和背景定义：


```math
\delta_i^{(k,e)}
=
\log p_{\theta_j}^{\mathcal C_j}(y_i\mid x_i,b_{+e}^{(k)})
-
\log p_{\theta_j}^{\mathcal C_j}(y_i\mid x_i,b_{-e}^{(k)}).
\tag{10}
```


设开关边的候选评分差为 $`d_{i,c}^{(k,e)}`$，其候选中心化结果为：


```math
\widetilde d_{i,c}^{(k,e)}
=
d_{i,c}^{(k,e)}
-\frac{1}{|\mathcal C_j|}
\sum_{c'\in\mathcal C_j}d_{i,c'}^{(k,e)}.
\tag{11}
```


对于全部支持图像集合 $`A`$ 或该边目标原语对应的支持子集 $`T_e`$，将权重在集合内部归一化，定义：


```math
\phi(S)
=
\begin{bmatrix}
\mathbb E_S[\delta]\\[4pt]
\sqrt{\mathbb E_S[(\delta-\mathbb E_S\delta)^2]}\\[4pt]
\sqrt{\mathbb E_S\!\left[
|\mathcal C_j|^{-1}\sum_c\widetilde d_c^2
\right]}\\[4pt]
\mathbb E_S[H(p_{\mathrm{on}})-H(p_{\mathrm{off}})]
\end{bmatrix},
\qquad
h_{j,e}=
\begin{bmatrix}
\phi(A)\\[4pt]
\phi(T_e)
\end{bmatrix}\in\mathbb R^8.
\tag{12}
```


$`\mathbb E_S`$ 同时包含支持权重和四个背景的等权平均；熵为 $`H(p)=-\sum_c p_c\log p_c`$。这对应代码的 all／target 八个输入，符号方向与旧实现一致。

训练样本必须明确是：


```math
(h_{j,e},\widehat u_{j,e})
=
\left(
h_e(\theta_j,D_j^s),
\widehat u_e(\theta_j,D_j^{u,m})
\right).
\tag{13}
```


两者使用同一个教师权重。正式部署也在同一个冻结权重上提取响应和使用策略。

第一批主实验保留一个响应 bank，与已有教师输入一致；另计算第二个独立 bank 诊断探针波动。若使用重复平均，则在线性模型下：


```math
\overline\mu_e
=
\frac1R\sum_{r=1}^R f_\beta(h_e^{(r)})
=
f_\beta\!\left(\frac1R\sum_{r=1}^R h_e^{(r)}\right).
\tag{14}
```


应先平均预测，再求一次概率分配。非线性策略映射一般不满足“平均概率等于平均预测的概率”。第二个 bank 的结果单列，不与第一批主结果混在一起解释。

### 4.4 按教师平均的联合岭回归

设拟合教师数为 $`J`$。由拟合教师计算输入均值 $`c_k`$、标准差 $`s_k`$，以及类型标签标准差 $`a_t`$。八维输入共享同一标准化规则，三个类型有独立系数块：


```math
x_{j,e}
=
\mathop{\mathrm{onehot}}\nolimits(t(e))
\otimes
\begin{bmatrix}
1\\[4pt]
(h_{j,e}-c)\oslash s
\end{bmatrix},
\qquad
\mu_{j,e}=a_{t(e)}x_{j,e}^{\top}\beta.
\tag{15}
```


$`\beta`$ 共 27 维。$`X_{jt}`$ 表示教师 $`j`$、类型 $`t`$ 的设计矩阵，$`\widehat u_{jt}`$ 是对应标签向量。

具体归一化使用拟合教师全部边，标准差取总体标准差，与现有实现一致：


```math
\begin{aligned}
c_k&=\frac1{96J}\sum_{j,e}h_{j,e,k},\\[4pt]
s_k&=\max\!\left\{
\sqrt{\frac1{96J}\sum_{j,e}(h_{j,e,k}-c_k)^2},\,10^{-6}
\right\},\\[4pt]
m_t&=\frac1{JN_t}\sum_j\sum_{e\in E_t}\widehat u_{j,e},\\[4pt]
a_t&=\max\!\left\{
\sqrt{\frac1{JN_t}\sum_j\sum_{e\in E_t}
(\widehat u_{j,e}-m_t)^2},\,10^{-12}
\right\}.
\end{aligned}
\tag{15a}
```


Absolute 的标签只除以 $`a_t`$，不减去 $`m_t`$；类型均值由截距表达。Relative 才在每个教师、每个类型内部作投影。Absolute 的拟合目标为：


```math
\mathcal L_{\mathrm{abs}}(\beta)
=
\frac1J\sum_{j=1}^{J}\sum_{t=1}^{3}
\left\|
X_{jt}\beta-\frac{\widehat u_{jt}}{a_t}
\right\|_2^2
+
\lambda\|\beta\|_2^2.
\tag{16}
```


对 $`\beta`$ 求导并令其为零，得到：


```math
\left(
\frac1J\sum_{j,t}X_{jt}^{\top}X_{jt}
+\lambda I
\right)\widehat\beta
=
\frac1J\sum_{j,t}X_{jt}^{\top}
\frac{\widehat u_{jt}}{a_t}.
\tag{17}
```


$`\lambda\gt 0`$ 保证矩阵正定。实现使用线性方程求解，不显式求逆。

若复用旧代码的“未平均误差和＋α 正则”求解器，两者准确对应为：


```math
\alpha=J\lambda.
\tag{18}
```


本批统一 λ 网格取旧三教师 α 网格除以 3，即 $`\{0.1,1,10,100\}/3`$。九教师模型对应 α 网格为 $`\{0.3,3,30,300\}`$。因此本次选出的 Independent α 为 **0.1**，Pooled α 为 **30**。

每个配置共同选取一个 λ。令 $`\mu_{s4,t}^{(\lambda)}`$ 为 seed $`s`$ 的 Q4 预测，$`a_{s,t}`$ 为对应模型仅由拟合数据估计的标签尺度，则：


```math
E_4(\lambda)=
\frac1{3}\sum_{s\in\{3351,3352,3353\}}
\frac1{3}\sum_{t=1}^{3}
\frac{\left\|
P_t(\mu_{s4,t}^{(\lambda)}-\widehat u_{s4,t})/a_{s,t}
\right\|_2^2}{N_t}.
\tag{18a}
```


取使 $`E_4`$ 最小的 λ，并列取较大 λ。Pooled 三个 seed 共享同一尺度；Independent 各自使用其三教师的尺度。该选参误差按类型等权，拟合目标仍沿用式 (16) 的逐边权重。

第一批不增加类型内 $`1/N_t`$ 权重，保留原来 12／36／48 条边的行权重。每个教师配置的 Absolute／Relative 共享归一化参数；归一化只从 Q1–Q3 拟合数据计算。

为比较两个拟合设置真正学到的映射，同时输出原始特征单位下的截距和斜率：


```math
\begin{aligned}
w_{tk}&=\frac{a_t\beta_{tk}}{s_k},\\[4pt]
b_t&=a_t\left(\beta_{t0}-\sum_{k=1}^{8}\frac{\beta_{tk}c_k}{s_k}\right),\\[4pt]
\mu_e&=b_{t(e)}+w_{t(e)}^\top h_e.
\end{aligned}
\tag{19}
```


联合拟合增加教师数时，输入和标签尺度也会变化；原始单位系数可以把这种变化与实际映射变化联系起来。

### 4.5 Relative 的定位与公式

类型内预算固定，因此效用的类型共同偏移不改变分配：


```math
u_t^\top\pi_t
=
(P_tu_t)^\top\pi_t+\overline u_t B_t,
\qquad
P_t=I-\frac{\mathbf1\mathbf1^\top}{N_t}.
\tag{20}
```


其中 $`\overline u_t=N_t^{-1}\sum_{e\in E_t}u_e`$ 是当前效用向量的类型均值。这说明分配只需要类型内相对效用，但并不推出“相对回归一定比绝对回归更容易预测”。R5 已实际检验了后一个问题。

Relative 对照定义为：


```math
\mathcal L_{\mathrm{rel}}(\beta)
=
\frac1J\sum_{j,t}
\left\|
P_t\left(
X_{jt}\beta-\frac{\widehat u_{jt}}{a_t}
\right)
\right\|_2^2
+\lambda\|\beta\|_2^2.
\tag{21}
```


由 $`P_t^\top=P_t`$、$`P_t^2=P_t`$，正规方程为：


```math
\left(
\frac1J\sum_{j,t}X_{jt}^{\top}P_tX_{jt}
+\lambda I
\right)\widehat\beta_{\mathrm{rel}}
=
\frac1J\sum_{j,t}X_{jt}^{\top}P_t
\frac{\widehat u_{jt}}{a_t}.
\tag{22}
```


三个截距列经投影后为零，岭回归最优截距也为零，有效斜率为 24 个。对任意残差向量 $`v`$：


```math
\|P_tv\|_2^2
=
\frac{1}{2N_t}\sum_{e,e'\in E_t}(v_e-v_{e'})^2.
\tag{23}
```


因此无需显式构造所有边对。该形式保留为 Pooled-Relative 对照，检验它是否与教师覆盖形成交互。

### 4.6 分配公式：保留半密度，明确标准化与预算求解

对一个待识别的固定状态，先计算类型中心化预测：


```math
z_t=P_t\mu_t,\qquad
v_t^2=\frac{\|z_t\|_2^2}{N_t},\qquad
r_t=
\begin{cases}
z_t/v_t,&v_t\gt 0,\\[4pt]
0,&v_t=0.
\end{cases}
\tag{24}
```


主实验取 κ=1，求解：


```math
\pi^\star
=
\arg\max_{\substack{0\leq\pi_e\leq1\\[4pt]
\sum_{e\in E_t}\pi_e=N_t/2}}
\left\{
\kappa\sum_e r_e\pi_e
-\sum_e
\mathop{\mathrm{KL}}\nolimits\!\left[
\mathop{\mathrm{Bern}}\nolimits(\pi_e)\,\|\,\mathop{\mathrm{Bern}}\nolimits(1/2)
\right]
\right\}.
\tag{25}
```


KL 对单个坐标的导数为 $`\log(\pi_e/(1-\pi_e))`$。加入类型预算乘子后，一阶条件给出：


```math
\pi_e^\star=\sigma(\kappa r_e-\gamma_{t(e)}),
\qquad
\sum_{e\in E_t}\sigma(\kappa r_e-\gamma_t)=N_t/2.
\tag{26}
```


目标严格凹，有限分数下最优解在开区间内。类型概率和关于 $`\gamma_t`$ 连续、严格递减，从 $`N_t`$ 降至 0，因此每个类型有唯一乘子，可用二分求解。类型分数均值为零并不意味着 $`\gamma_t=0`$。

这一公式优化的是类型标准化效用的正则化一阶代理；实际原始单位代理收益仍按式 (9) 计算。主批固定 κ=1，使每个 seed 都实际检验非均匀关系分配；κ=0 对应独立列出的 Uniform。

### 4.7 冻结部署：把响应和使用状态对齐

Uniform 训练保持原 10 epoch 预算。从 epoch 0–10 中，按既定 VAL CW native AUC 选择共同权重，并列取较早 epoch：


```math
e_s^\star
=
\operatorname*{arg\,max}_{e\in\{0,\ldots,10\}}
A_{\mathrm{VAL,CW}}\!\left(\theta_{s,e}^{U},\rho\right),
\qquad
\theta_s^\star=\theta_{s,e_s^\star}^{U}.
\tag{27}
```


本轮直接复用归档的 runs/uniform/best.pt，三个 seed 对应 epoch 2／2／6。当前归档保存 best／last 和完整曲线，不要求恢复未保存的其他中间权重。

接下来：


```math
\begin{aligned}
H_s^\star&=\mathop{\mathrm{Response}}\nolimits(\theta_s^\star,D_s^{\mathrm{TRAIN}}),\\[4pt]
\mu_s^\star&=f_{\widehat\beta}(H_s^\star),\\[4pt]
\pi_s^\star&=\mathop{\mathrm{Policy}}\nolimits(\mu_s^\star).
\end{aligned}
\tag{28}
```


之后固定 $`\theta_s^\star`$。每个候选集合分别归一化各图概率，再平均 16 张图：


```math
\begin{aligned}
\overline p_{\theta_s^\star,\pi_s^\star}^{\mathcal C}(c\mid x)
&=
\frac1{16}\sum_{m=1}^{16}
p_{\theta_s^\star}^{\mathcal C}(c\mid x,b^{(m)}),\\[4pt]
b_e^{(m)}&=\mathbf1\{U_{me}\lt \pi_{s,e}^\star\}.
\end{aligned}
\tag{29}
```


各方法共享同一组均匀随机数 $`U_{me}`$，减少配对比较的抽图波动。CW／OW 分别使用各自候选集合。

这一步消除了预测之后再适配造成的参数变化。折内教师和正式模型之间仍有组合范围、训练样本量和阶段差异，第二批专门研究其中的阶段因素。

## 5. 第一批：联合拟合与冻结部署，一次运行完整比较

### 5.1 批次目标和方法组

目标是直接回答：**多教师效用预测，在已训练好的强 CPF＋关系分支上，能否给出优于均匀图与概率置换的关系分配？**

| 方法 | 效用来源 | 主作用 |
| --- | --- | --- |
| Uniform | 所有概率为 0.5 | 共同识别权重下的基线 |
| Independent-Absolute | 每 seed 三教师，三个 Q4 共同选 λ | 单独教师配置的监督对照 |
| Pooled-Absolute | 九教师联合拟合 | 主候选方案 |
| Pooled-Relative | 九教师联合、类型内相对拟合 | 教师覆盖与 R5 的交互对照 |
| 匹配概率置换 | 分别置换上述三个预测器的概率 | 检验关系对应的实际贡献 |

前三种预测策略固定 κ=1。每个预测器保存五个类型内概率置换，seed 为 7101–7105；它们属于同一控制族，不作为五个新方法挑选最优者。

两种简单读出——历史边均值、直接支持 NLL——只在主部署状态追加评分。它们共享同一冻结识别器，成本为额外推理。

### 5.2 一个主状态、一个旧状态参照

**主状态为 Uniform 的共同最佳权重 $`\theta_s^\star`$。** 所有策略使用重新测得的 $`H_s^\star`$。

同时在旧 reference/last.pt 上运行同样的主要策略，形成一张“预测器 × 使用状态”比较表。该参照用来判断教师联合拟合本身是否有效，以及它的收益是否依赖识别状态。

对 Pooled-Absolute 额外完成一个紧凑的响应交叉表：

| 使用的权重 | 输入 reference 响应 | 输入 Uniform-best 响应 |
| --- | --- | --- |
| reference | 正常配对 | 状态错配 |
| Uniform-best | 状态错配 | 正常配对，主结果 |

这里固定预测器和 κ，只改变响应来自哪个权重。同步报告概率位移和节点覆盖，判断差异来自当前响应信息还是不同分配幅度。Q5→reference 排序变化单独记录，不直接用作“迁移失败率”。

### 5.3 所有 seed 都执行的固定流程

1. 导入 Competition／R5 的教师、响应、标签、reference 与 Uniform-best。使用现有 CLIP 缓存和同一关系库。
2. 按第 4.4–4.5 节拟合；Q1–Q3 用于拟合，三个 Q4 共同选 λ。保存固定三／九教师模型，直接用于后续所有状态。
3. 在 reference 和 Uniform-best 上分别提取当前 TRAIN 响应。第一响应 bank 为主；第二 bank 只用于探针重复性分析。
4. 在 Q5 的独立评价图像上，对新增预测器重算实际 16 图混合 NLL 和 AUC。沿用八个 Q5 评价 bank。
5. 对正式 VAL／TEST，完成主状态、reference 参照、匹配置换及简单读出，使用 1701／2701／3701 三个 16 图 bank。
6. 输出配对指标、逐 seed 结果和计算时间。原始预测保存后，可统一重算 native／exact AUC、HM、NLL、ECE。

本批**新增识别训练为 0，新增教师训练为 0**。需要 GPU 的工作是提取响应、Q5 决策和冻结推理。主 bank 固定为 1701，另两个 bank 反映采图变化。

不把 Q5 并入最终拟合，是为了让“Q5 使用的预测器”和“正式部署使用的预测器”完全相同。这不是永远只用三折；完整拟合范围的变化放在方法确定后的统一重训中处理。

### 5.4 指标集中为三层

| 层次 | 必须汇总的指标 | 对应问题 |
| --- | --- | --- |
| 最终识别 | CW／OW native AUC、exact AUC、VAL 工作点 HM；对 Uniform、五置换均值的配对差值 | 分配是否提高实际识别 |
| 效用监督 | Q5 原始单位中心化 MSE、R²、类型内 Spearman；一阶代理、实际混合 NLL 和实际 AUC | 拟合改善是否传递到决策 |
| 模型状态 | 正确／错配响应的识别差值；重复响应预测方差；类型概率标准差、KL、节点覆盖 | 收益是否需要当前模型信息 |

AUC 按 0–100 汇报，NLL 使用自然对数；配对差值按 seed 计算，再报告三 seed 均值、标准差和逐 seed 方向。图 bank 和置换重复分别汇总，不扩充训练 seed 数。

概率置换保持类型内概率多重集合和预算，但可能改变每个节点接收的关系。因此同步保存每节点的期望入边数：


```math
d_v^\pi=\sum_{e:\mathop{\mathrm{target}}\nolimits(e)=v}\pi_e.
\tag{30}
```


若后续论文需要把附件语义与节点覆盖进一步区分，再在同一目标节点内置换 OE／AE；OI 每目标只有一条边时，这类置换没有作用。本批不新增独立训练来处理这项诊断。

## 6. 第二批：教师阶段匹配，完成方法形态

第一批研究旧监督能否直接迁移到冻结使用。第二批将“训练完成后分配”的流程同步放进折内任务，在同一批正式 Uniform 权重上比较。

### 6.1 教师训练与状态选择

现有 15 个教师已经包含折内 G0 训练和 10 epoch Uniform 关系训练。以各自 teachers/j/last.pt 为起点，再执行与正式 reference→Uniform 适配相同的 **10 epoch Uniform 阶段**：

- 冻结折内 CPF，只优化原关系分支；
- 使用原 train_ids、原训练组合和原训练损失；
- AdamW、batch 64、基础学习率 3×10⁻⁴、weight decay 10⁻⁴；
- 沿用原一 epoch warmup 与 10 epoch cosine 计划，梯度裁剪 1；
- 保存 epoch 0–10 的指标，以及 best、last 权重。

增加这段训练是为了匹配正式流程的训练阶段与选模方式，不把它解释为“旧教师训练不足”的既定事实。

每折从**原风险池中尚未用于测量或评价的图像**，固定抽取 seen／unseen 各 256 张作为教师停止选择集。抽样 seed 固定为 9401＋fold，并在三个训练 seed 间共享图像 ID。该集合未进入折内 G0 和关系训练。

已检查旧划分余量：五折剩余 seen 图像至少 2992 张，unseen 至少 779 张，因此能够直接抽取这两个 256 图像集合，不需要更换原训练数据。

停止集按“inner 组合为 seen、query 组合为 unseen”计算 83 候选 AUC，从 epoch 0–10 选择共同最佳状态，并列取较早 epoch。query 标签在该批用于停止选择；用于效用测量、Q5 策略评价的图像保持独立。该协议评价的是图像独立的折内决策，不把 Q5 解释为未参与研究选择的新组合确认集。

### 6.2 标签与响应在新状态一起重测

对选中的教师 $`\theta_j^\star`$：


```math
\begin{aligned}
H_j^\star&=\mathop{\mathrm{Response}}\nolimits(\theta_j^\star,D_j^s),\\[4pt]
\widehat u_{j,e}^\star
&=
\frac1{K_j}\sum_k
\left[
R_j(\theta_j^\star,b_{-e}^{(k)},0)
-
R_j(\theta_j^\star,b_{-e}^{(k)},1)
\right].
\end{aligned}
\tag{31}
```


支持图像、测量图像和背景 seed 沿用原值，使前后差异主要对应教师状态。Q1–Q4 为 8 背景，Q5 为 16 背景；仅学习 U-NLL，继续保存 seen 风险用于评价。仍由三个 seed 的 Q1–Q3 九个教师联合拟合，三个 Q4 共同选 λ，Q5 用于决策评价；正式部署使用同一九教师模型。

旧测量标签保留为旧阶段对照。新状态的标签通过新的 on/off 前向计算获得，不能复用旧值。

### 6.3 最终比较收敛到五行

在三个完全相同的正式 Uniform-best 权重上，比较：

| 组别 | 内容 |
| --- | --- |
| Uniform | 半密度均匀采样 |
| 旧教师预测器 | 第一批固定的联合效用模型，输入当前响应 |
| 阶段匹配预测器 | 使用新阶段教师的响应与标签拟合 |
| 匹配置换 | 阶段匹配预测器的五次类型内概率置换 |
| 错配响应 | 固定阶段匹配预测器，输入 reference 状态响应 |

该批新增 **15 条教师适配轨迹**，每条 10 epoch；正式识别器新增训练仍为 0。教师完成后集中拟合和统一推理，避免为每个策略重新训练一套识别器。

Absolute 为默认主目标。Pooled-Relative 只在第一批已经表现出清楚的 VAL 与 Q5 决策优势时替换它；选择依据写入配置后再执行第二批。已有 TEST 用于完整开发记录，最终确认采用新组合划分或第二数据集。

## 7. 经验误差收缩：保留为可解释扩展

第五版的收缩公式可以保留，但应解决“幅度过大”而不是承担“修正排序”的任务。

对每个外层留出的组合折，用其余组合折拟合和选择 λ，再预测该折三个 seed 的效用。相同组合折的三个 seed 一起留出；标准化和内层选参均在训练折中完成。

记原始 NLL 单位下的类型中心化残差为：


```math
e_{jt}=P_t(\mu_{jt}^{\mathrm{out}}-\widehat u_{jt}),
\qquad
\widehat\epsilon_t^2
=
\frac1{J_{\mathrm{out}}}\sum_j
\frac{\|e_{jt}\|_2^2}{N_t}.
\tag{32}
```


保留标签测量噪声在其中，把它解释为预测测量效用时的经验误差。正式需要独立 Q5 审计时，仅用 Q1–Q4 的嵌套外折估计，Q5 保持在外。

把式 (24) 的标准化分数替换为：


```math
\widetilde r_t
=
\frac{z_t}{\sqrt{v_t^2+\widehat\epsilon_t^2}}
=
g_t r_t,
\qquad
g_t=\frac{v_t}{\sqrt{v_t^2+\widehat\epsilon_t^2}}.
\tag{33}
```


分母为零时定义为零。误差大、类型内预测差异小时，策略靠近均匀图。收缩必须作用在标准化之后的幅度，或者直接使用上述分母；若先乘一个类型常数再重新标准化，收缩会被抵消。

该策略还有可直接推导的位移界：


```math
\|\pi_t^\star-\tfrac12\mathbf1\|_2
\leq
\frac{\kappa}{4}\|\widetilde r_t\|_2
=
\frac{\kappa\sqrt{N_t}}4\,g_t.
\tag{34}
```


证明：令 $`\Delta=\pi_t^\star-\tfrac12\mathbf1`$。预算给出 $`\mathbf1^\top\Delta=0`$。将式 (26) 的 logit 条件与 $`\Delta`$ 作内积，乘子项消失；又因为 logit 的导数 $`1/[p(1-p)]\geq4`$，有 $`4\|\Delta\|_2^2\leq\kappa\widetilde r_t^\top\Delta`$，再用 Cauchy–Schwarz 即得式 (34)。

这个界控制概率位移，不是 AUC 或风险改善保证。因此后续启用收缩时，只加一张冻结比较表：标准分配、经验误差收缩、按相近概率位移调节的普通全局 κ、匹配置换。只有优于简单缩小 κ，才把误差校准作为单独研究贡献。

## 8. 下一轮结果怎样决定论文方向

论文主问题可以保留为：

> **能否利用训练内组合留出任务测得的关系效用，根据当前识别模型的响应，在给定关系预算下改善未见组合识别？**

其创新主体仍是效用监督与跨任务决策迁移；将分配移到训练结束后，并不会移除这两个核心。论文需由实际收益确定重点：

| 下一轮结果 | 对应研究结论与后续动作 |
| --- | --- |
| 联合拟合提高 Q5，且正式冻结识别超过 Uniform 与置换 | 以多教师效用迁移为主体，完成阶段匹配及跨数据确认 |
| 当前响应优于错配响应，阶段匹配进一步提高识别 | 以模型状态条件化效用为主体，报告两项配对增量 |
| Q5 有收益，正式部署仍无收益 | 集中解释教师任务到完整 CZSL 的迁移差异；不继续叠加更复杂损失、网络和惩罚 |
| 真实分配与置换持平，Uniform／Capacity 更强 | 关系分支或训练带来的收益占主导；效用选择转为机制发现，论文主方法重新取舍 |

当前证据最适合继续推进前两项的实验，而不是提前认定最终主结论。

方法和固定配置确定后，确认阶段整体完成三件事：用新的训练 seed 4351／4352／4353 重训整套教师与正式模型；在新的组合划分或第二 CZSL 数据集上复现；以相同 10 epoch 预算比较 CPF、Capacity、Uniform、TRUST 及一个匹配置换控制族。联合预测器若由三 seed 教师共同训练，识别表中的三个 seed 应说明为“共享效用模型下的三个识别重复”。

原 1800 行噪声实验和旧 ablation 继续作为已有证据。新主方法在干净识别中形成收益后，再统一补做其知识内容、噪声和计算成本比较；当前两批不重复铺开旧全队列。

## 9. 实现对应与交付依据

### 9.1 后续代码修改范围

| 现有模块 | 需要修改的内容 |
| --- | --- |
| compact.py／r5.py | 按教师平均目标；α=Jλ；联合教师输入；保存原始单位系数 |
| competition_training.py | 九教师拟合、共同 Q4 选参；教师新增 Uniform 阶段；新状态 on/off 测量 |
| response.py | 明确传入权重与支持图像 ID；可重复响应 bank；保持八维定义 |
| align.py | 沿用预算二分；默认固定 κ=1；统一记录类型概率与乘子 |
| workflow／report | 先复用 Uniform-best，再提取响应和冻结评分；合并三 seed 的教师与识别结果 |
| model.py／识别指标 | 本批沿用现有数学实现，保持关系评分、候选归一化和 AUC 口径 |

每份效用模型保存系数、输入／标签尺度、特征顺序、关系类型和 edge ID；每份预测保存权重来源、图 bank 与候选顺序。这些字段用于把同一公式、同一权重和同一结果对应起来。

本次交付是公式与实验计划，以及用于支撑判断的本地拟合复算；尚未编写或运行上述新服务器实验。

### 9.2 来源与新增分析

- [第五版诊断 PDF](references/revision_v6/source_diagnosis_v5.pdf)：重点参考其效用、拟合、状态与实验设计章节。
- [弱基底完整分析](../v1/analysis/server-results-20261022/README.md)。
- [强基底前置比较](../05/references/history/conditional_v2_analysis.md.txt)。
- [强基底完整 TRUST](../02/analysis/README.md)。
- [响应条件化结果](../03/analysis/README.md)。
- [Compact 结果](../04/analysis/README.md)。
- [Align 结果](../05/analysis/README.md)。
- [Competition 10 epoch 结果](../06/analysis/README.md)。
- [R5 10 epoch 结果](../07/analysis/README.md)。
- [本次联合拟合复算](references/revision_v6/pooled_diagnostic.json)及[复算脚本](references/revision_v6/pooled_diagnostic.py)。
- [公式数值核验](references/revision_v6/formula_checks.json)：Bernoulli 偏导、教师平均正则、原始单位系数、重复响应平均、相对投影、预算求解和收缩界均通过。

本方案已检查公式解析、Markdown 表格及渲染布局。当前最直接的一批工作是：**复用三个 Uniform-best，实际评分联合预测器与匹配置换；随后在相同正式权重上检验教师阶段匹配。**


