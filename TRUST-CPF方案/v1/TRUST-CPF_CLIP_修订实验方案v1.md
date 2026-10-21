# TRUST-CPF / CLIP：修订实验方案 v1

版本：2026-10-21。第一轮数据集：UT-Zappos natural split。实现入口：`run.py`。本版以CLIP替换原方案的ViT-B/16＋GloVe，保持“配对关系效用 → 可迁移预测 → 风险与预算选图”的研究主线。

## 1. 修改内容与研究对象

| 模块 | 原公式方案 | 本版定义 |
|---|---|---|
| 编码器 | ViT-B/16视觉＋GloVe300文本 | 冻结CLIP ViT-L/14，图文维度768 |
| 视觉路径 | CPF DDM的文本/对象引导视觉 | CLIP末层补丁，由属性/对象文本分别查询 |
| 概率基底 | CPF原生三分数转换 | 对象查询概率＋对象上下文条件属性概率 |
| 知识表示 | 词均值描述 | CLIP原生文本编码，名称和描述独立保留 |
| 类型消息 | 300×300完整矩阵 | $`W_\rho=I+L_\rho R_\rho`$，768维、秩16 |
| 属性图残差 | 三元交互 | 图像对象上下文×对象图原型×属性图原型，秩64 |
| 视觉证据对齐 | 额外学习图文投影 | 直接使用CLIP统一空间 |
| 效用输入 | 1725维 | 6157维，MLP为6157→256→128→2 |
| 训练方式 | 分阶段教师和效用 | 保留五折，CLIP冻结，训练共享适配器和图头 |

本版模型名为TRUST-CPF/CLIP；其中CPF表示对象概率与对象条件属性概率的分解。识别主对照使用同一个CLIP概率基底R0。GloVe版的历史CPF成绩另作背景表，CLIP换底座的影响由R0单独衡量。

研究假设：关系在组合隔离Query上的预测效用可由语义与Inner视觉证据预测；由这种效用选出的关系优于普通随机删边和仅分类监督的自由边概率；风险项改善知识扰动下的表现。

## 2. 数据、知识与任务

UT标签轴为16属性、12对象，张量顺序`[batch, object, attribute]`，dataset组合元组仍是`(attribute, object)`。训练83组合，图像22998/3214/2914分别对应TRAIN/VAL/TEST。

源知识沿用本机`reviewed-v2`：12条对象父类关系、36条OE、48条AE，共96条候选关系。源文件随包放在`assets/knowledge/`。名称不作为可删边；每条描述与其所属原语构成一个独立干预单位。父类名称作为OI消息源。

官方候选定义：

**式（1）**

```math
\begin{aligned}
\mathcal C_{\mathrm{val}}^{\mathrm{CW}}&=\mathcal C_s\cup\mathcal C_{\mathrm{val}},\\
\mathcal C_{\mathrm{test}}^{\mathrm{CW}}&=\mathcal C_s\cup\mathcal C_{\mathrm{test}},\\
\mathcal C^{\mathrm{OW}}&=\mathcal A\times\mathcal O.
\end{aligned}
```

五折任务划分：

**式（2）**

```math
\begin{aligned}
\mathcal C_s&=C_0\sqcup\bigsqcup_{j=1}^{5}Q_j,\\
I_j&=\mathcal C_s\setminus Q_j.
\end{aligned}
```

分块种子2026，各Query原语在自己的Inner中均出现。仅出现于单个组合的原语对应组合放入公共Inner集合$`C_0`$。每折教师从相同CLIP缓存和适配器初始化独立训练；$`Q_1`$–$`Q_3`$拟合效用，$`Q_4`$选择效用epoch和结构参数，$`Q_5`$作机制审计。

本版reviewed-v2描述沿用先前TRAIN审图修订。知识固定后，教师的组合图像监督按式(2)隔离。检验知识构建独立性时，将`--knowledge`指向既有legacy-v1的三份字典，使用独立cache/work重复队列。

## 3. 冻结CLIP和视觉缓存

CLIP采用ViT-L/14、224×224输入、视觉宽度1024、共同投影维度768。对最终视觉序列的CLS和全部256个补丁施加同一个`ln_post`和`visual.proj`：

**式（3）**

```math
\begin{aligned}
g&=\mathcal N(P_v^\top\mathop{\mathrm{LN}}\nolimits_{\mathrm{post}}(H_0)),\\
p_\ell&=\mathcal N(P_v^\top\mathop{\mathrm{LN}}\nolimits_{\mathrm{post}}(H_\ell)),\\
\ell&=1,\ldots,256.
\end{aligned}
```

名称文本为属性`a photo of footwear made of {attribute}`、对象`a photo of {object}`、组合`a photo of {object} made of {attribute}`。属性/对象英文别名读取`assets/names.json`；全组合文本按对象外层、属性内层排列。单位文本向量记为$`e_a,e_o,e_{oa}\in\mathbb R^{768}`$。

类别查询：

**式（4）**

```math
\begin{aligned}
\alpha_\ell(e)&=\mathop{\mathrm{softmax}}\nolimits_{\ell}(e^\top p_\ell/0.07),\\
f(e)&=\mathcal N\left(\sum_\ell\alpha_\ell(e)p_\ell\right).
\end{aligned}
```

缓存$`g`$、$`f_a=f(e_a)`$、$`f_o=f(e_o)`$，FP32编码、FP16落盘、FP32读取。训练中对这些输入施加共享适配器：

**式（5）**

```math
\begin{aligned}
A_\theta(z)&=\mathcal N\bigl(z+0.1U\mathop{\mathrm{GELU}}\nolimits(Dz)\bigr),\\
D\in\mathbb R^{8\times768},\ U\in\mathbb R^{768\times8},\ U_0&=0.
\end{aligned}
```

后文带波浪线的$`\widetilde g,\widetilde f_a,\widetilde f_o`$是适配器输出。CLIP权重、知识文本和组合文本均冻结。

## 4. CLIP条件概率基底

对象概率读取对象查询视觉；条件分支先引入候选对象上下文：

**式（6）**

```math
\begin{aligned}
z_o^0(x)&=\widetilde f_o^\top e_o/T_0,\\
c_{xo}&=\mathcal N(\widetilde g+\beta\widetilde f_o),\\
T_0&=0.07,\ \beta=0.5.
\end{aligned}
```

**式（7）**

```math
U_{oa}^0(x)=\left(c_{xo}^\top e_{oa}+\widetilde f_a^\top e_a\right)/T_0.
```

式(7)的组合文本和图像上下文均随候选对象改变。它提供可运行的CLIP条件基底，不依赖旧CPF投影或旧checkpoint。

## 5. 类型图消息与三元图残差

原语锚点$`h_c`$分别取$`e_a,e_o`$；OI源文本为父类，OE/AE源文本为`a photo of footwear with {description}`，均通过同一个CLIP文本编码器编码。对$`e=(u\to c,\rho)`$：

**式（8）**

```math
\begin{aligned}
d_e&=h_u-h_c,\\
m_e&=(I+L_\rho R_\rho)d_e,\\
&L_\rho\in\mathbb R^{768\times16},\ R_\rho\in\mathbb R^{16\times768}.
\end{aligned}
```

初始化$`L_\rho`$为标准差0.01的正态矩阵，$`R_\rho=0`$。定义固定候选度数$`n_\rho(c)`$：

**式（9）**

```math
\begin{aligned}
Z_c(G)&=\mathcal N\left(h_c+\eta\sum_\rho\frac{1}{\max(1,n_\rho(c))}
\sum_{e\in N_\rho(c)}b_em_e\right),\\
\eta&=0.1.
\end{aligned}
```

固定度数使边开/关只改变该边的消息。实现保存28个目标原语及每边源向量；四个父概念在多条OI边上共享同一冻结文本表示。

**式（10）**

```math
\begin{aligned}
q_{xo}&=\mathcal N(P_vc_{xo}),\\
q_o(G)&=\mathcal N(P_oZ_o(G)),\\
q_a(G)&=\mathcal N(P_aZ_a(G)),\\
T(x,o,a;G)&=\sum_{k=1}^{64}q_{xo,k}q_{o,k}(G)q_{a,k}(G),\\
\delta z_o&=\widetilde g^\top[Z_o(G)-h_o]/T_g,\\
\delta U_{oa}&=[T(x,o,a;G)-T(x,o,a;\varnothing)]/T_g.
\end{aligned}
```

$`P_v,P_o,P_a`$均为64×768。最终$`z_o=z_o^0+\alpha_g\delta z_o`$、$`U_{oa}=U_{oa}^0+\alpha_g\delta U_{oa}`$，$`\alpha_g=T_g=0.1`$。空图严格退化为R0。

**式（11）**

```math
\begin{aligned}
\ell_o&=\log\mathop{\mathrm{softmax}}\nolimits_o(z),\\
\ell_{a|o}&=\log\mathop{\mathrm{softmax}}\nolimits_a(U_{o,:}),\\
\ell_{oa}&=\ell_o+\ell_{a|o}.
\end{aligned}
```

任意候选$`\mathcal C`$内的概率是$`\mathop{\mathrm{softmax}}\nolimits_{c\in\mathcal C}\ell_c`$。每张图先在同一候选集归一化，最后按概率平均。

## 6. 教师训练和效用测量

教师目标在Inner候选条件化，同时保留对象/条件属性辅助损失：

**式（12）**

```math
\begin{aligned}
\mathcal L_I&=-\ell_{o^*a^*}+\mathop{\mathrm{LSE}}\nolimits_{(a,o)\in I_j}\ell_{oa}\\
&\quad -0.4\ell_{o^*}-0.6\ell_{a^*|o^*}+10^{-3}\mathcal R_W,\\
\mathcal R_W&=768^{-2}\sum_\rho\|L_\rho R_\rho\|_F^2.
\end{aligned}
```

每batch采一张独立Bernoulli(0.5)参考图。教师训练10epoch，只更新适配器、消息低秩矩阵和三线性投影。

每任务从本折Query取8组合，每组合8图；组合内平均后组合等权。Query归一化候选固定为全部TRAIN组合$`\mathcal C_s`$。固定教师、Query和其他边，定义：

**式（13）**

```math
\begin{aligned}
\Delta_e(G_{-e})&=\ell_Q(G_{-e},0)-\ell_Q(G_{-e},1),\\
y_e&=\frac1M\sum_{m=1}^M\Delta_{e,m},\\
\nu_e&=\frac{1}{M(M-1)}\sum_m(\Delta_{e,m}-y_e)^2.
\end{aligned}
```

拟合$`M=4`$，审计$`M=16`$。同一配对共享背景图与Query。每类型每任务随机测8条边，包含概率$`\kappa_e=b_\rho/|E_\rho|`$；拟合三折各512任务，开发/审计各128任务。

期望Query损失$`F(\pi)=\mathbb E_G\ell_Q(G)`$满足：

**式（14）**

```math
\frac{\partial F}{\partial\pi_e}
=\mathbb E_{G_{-e}}[\ell_Q(G_e^+)-\ell_Q(G_e^-)]=-U_e.
```

式(14)把配对效用直接连接到结构优化方向。测量保持教师冻结，效用标签以停止梯度张量保存。

## 7. 6157维可迁移效用输入

语义输入为$`[h_u;h_c;h_u\odot h_c;|h_u-h_c|;\mathop{\mathrm{onehot}}\nolimits(\rho);s_{\mathrm{src}}]`$，共$`4\times768+3+4=3079`$维。来源四项为manual、WordNet、TRAIN修订、OOV；CLIP BPE编码的OOV记录为0。

视觉输入从Inner的冻结全局CLIP向量生成。每原语最多抽16图，先按组合计算均值，再按对象/属性上下文等权。视觉输入为：

**式（15）**

```math
\begin{aligned}
h_e^I&=[r_c;r_u;h_u;h_c;r_c^\top h_u;r_c^\top h_c;\\
&\log(1+n_c);\log(1+k_c);m_c;m_u],
\end{aligned}
```

共$`4\times768+6=3078`$维。OI的$`r_u`$为父类成员Inner摘要的单位均值；描述节点无独立图像，取零向量并标记。总输入6157维，只含语义和Inner证据。

MLP预测均值$`m_\phi`$与任务条件方差$`v_\phi=\mathop{\mathrm{softplus}}\nolimits(v_{\mathrm{raw}})+10^{-4}`$。先以拟合折的类型标准差$`s_\rho`$缩放$`y`$和$`\nu`$，方差按$`s_\rho^2`$还原。观测目标：

**式（16）**

```math
\begin{aligned}
L_e&=\frac12\left[\log(v_\phi+\nu_e)
+\frac{(y_e-m_\phi)^2}{v_\phi+\nu_e}\right],\\
\widehat L&=\frac1{|E|}\sum_{e\in B_t}\frac{L_e}{\kappa_e}.
\end{aligned}
```

三模型独立初始化，开发折NLL选epoch。测量方差$`\nu`$进入观测NLL；部署时任务方差和模型分歧分别计算。

## 8. 跨支持摘要与风险结构

对16份TRAIN支持摘要、3个效用模型，先得每摘要均值$`\bar m_r`$、平均条件方差$`\bar v_r`$、模型均值分歧$`e_r`$。由全方差公式：

**式（17）**

```math
\begin{aligned}
\widehat\mu&=\frac1R\sum_r\bar m_r,\\
\widehat\sigma_{\mathrm{task}}^2&=\frac1R\sum_r\bar v_r+\mathop{\mathrm{Var}}\nolimits_r(\bar m_r),\\
\widehat v_{\mathrm{ens}}&=\frac1R\sum_re_r.
\end{aligned}
```

**式（18）**

```math
R_e=\widehat\mu_e-\lambda_{\mathrm{var}}\widehat\sigma_{\mathrm{task},e}
-\lambda_{\mathrm{ens}}\sqrt{\widehat v_{\mathrm{ens},e}}.
```

结构目标及解：

**式（19）**

```math
\begin{aligned}
&\min_\pi-\sum_e\pi_eR_e\\
+\tau\sum_e\mathop{\mathrm{KL}}\nolimits(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\|\mathop{\mathrm{Bern}}\nolimits(0.5)),
\quad\sum_{e\in E_\rho}\pi_e&\le B_\rho,
\end{aligned}
```

**式（20）**

```math
\begin{aligned}
\pi_e&=\mathop{\mathrm{sigmoid}}\nolimits((R_e-\gamma_{\rho(e)})/\tau),\\
\gamma_\rho&\ge0,\\
\gamma_\rho\left(\sum_{e\in E_\rho}\pi_e-B_\rho\right)&=0.
\end{aligned}
```

二阶导为$`\tau/[\pi_e(1-\pi_e)]>0`$。预算超限时对$`\gamma`$二分60次。默认预算率0.5、$`\tau=0.5s_*`$，$`s_*`$为三拟合折效用标准差。

Q4按实际图识别NLL顺序选择：风险系数9组，预算率0.25/0.5/0.75，温度系数0.25/0.5/1。Q5只报告机制审计；参数固定后用五折标签按已选epoch拟合最终效用模型，全部TRAIN支持摘要产生最终结构。

## 9. 最终训练与七组对照

全TRAIN参考教师训练10epoch，随后图组共享该初始化、各适配3epoch。R0和容量组分别训练自身10epoch无图参考，再进行相同3epoch续训。

| 组 | 结构/作用 |
|---|---|
| r0 | 无图CLIP条件基底；只学习共享适配器 |
| capacity | 同维度低秩名称变换和三元视觉头，检查额外容量 |
| dense | 所有关系打开 |
| drop | 各类型常数概率，期望边数等于robust |
| free | 每边自由logit，以分类损失的score-function梯度、KL与软预算训练 |
| mean | 配对效用均值，关闭任务波动和集成分歧惩罚 |
| robust | 异方差效用集成＋风险与精确预算 |

free组目标的结构梯度使用$`(L-\mathrm{EMA}(L))(b_e-\pi_e)`$；预算为类型平均概率超限的平方惩罚，实际边数随结果保存。mean和robust使用同一类型预算上限。对成本分析另报告实际期望边数。

shuffled在独立work完成所有阶段，固定打乱来源对应，保留名称与关系类型。它与训练后换知识的敏感性实验分开。

AdamW：适配器$`10^{-4}`$，图头$`3\times10^{-4}`$，weight decay $`10^{-4}`$；按更新步数进行一epoch线性warmup和余弦衰减。效用MLP学习率$`10^{-3}`$，batch256，最多50epoch。正式seed为3351/3352/3353。

成本表累加五折教师、配对测量、效用拟合、全TRAIN参考和适配开销；主识别表的各组适配预算统一为3epoch。配对测量文件保存`wall_seconds/head_forwards`，效用选择记录保存`model_seconds`，教师/适配的逐epoch时间保存在`history.json`。

## 10. 推理与核心评价

**式（21）**

```math
\begin{aligned}
\log\bar p_{\mathcal C}(c|x)&=\mathop{\mathrm{LSE}}\nolimits_{k=1}^{K}\log p_{\mathcal C}(c|x,G^{(k)})-\log K,\\
K&=16.
\end{aligned}
```

同一图种子1701为全部评价batch生成相同的16张图。视觉只编码一次；图头执行K次。mean-logit和mean-graph作为readout配置可单独重复实验。

主表按VAL CW `AUC_native`首次严格最大选epoch（包括适配epoch0），同权重评价CW/OW。另报全临界事件`AUC_exact`。实际工作点bias由VAL exact-HM最大选择，温度由VAL NLL在41个对数均匀点[0.1,10]中选择；TEST直接应用。

| 科学问题 | 必须检查的指标 | 结果位置 |
|---|---|---|
| 效用是否可迁移 | Q5 Spearman、RMSE、balanced direction accuracy、AUROC | `utility/audit.json` |
| 方差是否可用 | 90%预测区间覆盖率、可靠方向覆盖比例 | 同上 |
| 是否只记住边 | 类型常数/边查表对照、未测边子集、纯语义预测器 | 同上；`configs/unmeasured.json`、`semantic.json` |
| 图是否改善组合泛化 | CW/OW AUC_native、AUC_exact、oracle-HM、固定VAL bias的Seen/Unseen/HM | `runs/*/metrics.json` |
| 条件属性是否改善 | GT-object属性准确率/NLL、三类错误数量 | 同上 |
| 风险是否增加鲁棒性 | OI/OE/AE错配和删除曲线、RAUC | `noise_*.json`、`robustness_summary.json` |
| 是否靠更多关系 | 各类型期望边数、风险项和预算乘子 | `structure.json`及各组结果 |
| 概率是否更稳定 | NLL、ECE、跨图预测方差 | 各组结果 |
| 是否跨seed重复 | 三seed配对差值、均值和样本标准差 | `paired_comparisons.json` |

可靠方向子集定义为$`|y|>1.96\sqrt\nu`$；同时报告全部测量的方向指标。90%效用预测区间采用$`m\pm1.645\sqrt{v_{\mathrm{task}}+v_{\mathrm{ens}}+\nu}`$。

RAUC为噪声率0/0.1/0.2/0.3/0.5上AUC的梯形积分除以0.5。每类型5个扰动seed。主噪声协议为训练后知识变化，效用组只用原TRAIN支持摘要重估新结构；所有组沿用干净VAL工作点。

## 11. 执行顺序和实现状态

`features → graph → teachers → measure → fit → audit → tune → fit --fit-all → reference → structure → adapt → report`。

对应源码、逐式说明与核心摘录见`TRUST-CPF公式及代码v1.md`。完整服务器命令见`服务器实践命令.md`。实验包提供UT自然划分的可执行流程、源码、知识、配置和测试；真实UT准确率由该服务器队列生成。本地验证区分数学/合成流程测试与正式训练结果，不按AUC阈值设置自动停跑分支。
