# TRUST-CPF A/B/C：部署效用与情境迁移的公式及实现

日期：2026-10-26。对应代码：`trustcpf_abc_v1_10ep`。

本版落实 v8 的 A、B、C 三条机制，并将共同识别基底的适配训练设为完整 10 epoch。实验仍然围绕 TRUST 的“关系干预效用监督 → 跨教师预测 → 预算内分配关系概率 → 强 CPF 上部署”展开。

## 1. 一批实验回答三个问题

| 路线 | 改动 | 要回答的问题 |
|---|---|---|
| A：幅度校准 | 用训练教师的效用尺度归一化预测，再选择共同幅度 | 原策略是否把微弱的预测差异放大了？ |
| B：混合效用 | 用与 16 图概率混合一致的单槽干预效用替换单图标签 | 监督目标与部署读出的一致性是否改善分配？ |
| C：情境效用 | 基于冻结 CPF 物体预测，建立两个情境并预测条件效用 | 同一条关系是否应随图像情境改变采样概率？ |

三条路线都使用每个 seed 的同一份 Uniform best 权重。A 在 Q4 选择的幅度只用于 A；B、C 固定强度 1。队列中的 Q4 屏障是为了完成后统一汇总，不构成 B/C 对 A 结果的研究依赖。

本批比较策略与读出机制。每个 seed 新训练一次分类器，共三次 × 10 epoch。27 参数和 51 参数的效用预测器用闭式岭回归求解，没有迭代训练 epoch。

## 2. 数据、教师和共同 10 epoch 轨迹

使用已有 UT-Zappos50K、CLIP ViT-L/14 特征缓存、强 CPF reference、96 条知识关系和 Stage 的 15 个教师。关系分为 OI 12 条、OE 36 条、AE 48 条。每类期望预算分别为 6、18、24。

seed 为 3351、3352、3353。三个 seed 的 Q1–Q3 共九个教师用于拟合；Q4 用于 A 的幅度选择和开发诊断；Q5 用于保留组合上的独立图像机制评估，沿用 Stage 已完成的 STOP 选模。Stage 的 STOP 使用查询组合的其他图像选择教师权重。本批 Q5 evaluate 与 measure、STOP 图像分开；新的完整教师划分与 seed 确认属于后续方法确认批次。所有教师查询标签来自原 TRAIN 的伪未见组合。正式 VAL 负责识别权重选择及校准，正式 TEST 用于最终结果。

原 Stage 的 Q4 evaluate 为空。本版从原 Q4 的 seen/unseen 图像池中，排除 train、measure、STOP 和已有 evaluate 后，分别固定抽取 256 张。随机种子为 26012604。Q5 沿用原 measure/evaluate 划分。所有划分均落盘。

令冻结 CPF 主干为 $`f_\phi`$，待适配的关系模块为 $`\theta`$。每个 batch 采样一次全局关系图：


```math
b_e\sim\mathop{\mathrm{Bernoulli}}\nolimits(1/2),\qquad e=1,\ldots,96.
```


沿用既有 CPF/TRUST 训练损失。记 TRAIN 候选上的组合 NLL 为 $`\mathcal L_{\rm pair}`$，物体 NLL 为 $`\mathcal L_o`$，真值物体条件下的属性 NLL 为 $`\mathcal L_{a\mid o}`$：


```math
\mathcal L_{\rm train}
=\frac{1.5}{1.075}
\left[\mathcal L_{\rm pair}
+0.075\left(\mathcal L_o+\mathcal L_{a\mid o}\right)\right]
+10^{-3}\mathcal R_{\rm graph}(\theta).
```


使用 AdamW，学习率 $`3\times10^{-4}`$，weight decay $`10^{-4}`$，batch size 64，梯度范数裁剪 1，第一 epoch warmup、之后 cosine。设每 epoch 有 $`L`$ 个 batch、总步数 $`T=10L`$、从 0 开始的步数为 $`t`$：


```math
\mathop{\mathrm{lr}}\nolimits_t
=3\times10^{-4}
\min\left(\frac{t+1}{L},1\right)
\frac{1+\cos\left(\pi\frac{\max(0,t-L)}{T-L}\right)}{2}.
```


保存 epoch 0 和完整 1–10 epoch 的 VAL 曲线。选择 VAL CW native AUC 最高的 epoch，相同时取较早 epoch。后续全部 A/B/C 策略使用这一份冻结权重。新训练器在 `last.pt` 同时保存当前模型、best 模型、优化器及 CPU/CUDA 随机数状态，使中断恢复完整继承已选出的 best。

入口：`abc_train.uniform`。CPF 分类器、图传播、原评估公式保持与 Stage 相同。新轨迹使用缓存 VAL 基底的评估实现；本轮三个策略比较共享这一新轨迹，不要求它与旧 State/Stage 的训练轨迹逐位相同。

## 3. 单图和 16 图部署风险

固定分类器和图像 $`x_i`$。在当前候选集合 $`\mathcal C`$ 内，每张图先做候选归一化：


```math
p_{i,b}(c)
=\frac{\exp z_\theta(x_i,b,c)}
{\sum_{c'\in\mathcal C}\exp z_\theta(x_i,b,c')}.
```


部署时采样 $`M=16`$ 张图，然后混合概率：


```math
\bar p_i(c;B)=\frac1M\sum_{m=1}^{M}p_{i,b^{(m)}}(c),
\qquad B=(b^{(1)},\ldots,b^{(M)}).
```


CW 和 OW 分别在各自候选集合内完成上面的归一化。对伪未见查询图像集合 $`\mathcal U`$，效用所对应的风险为：


```math
R_M(\pi)
=-\frac1{|\mathcal U|}\sum_{i\in\mathcal U}
\mathbb E_{B\sim P_\pi^{\otimes M}}
\log\bar p_i(y_i;B),
\quad
P_\pi(b)=\prod_e\pi_e^{b_e}(1-\pi_e)^{1-b_e}.
```


效用是在基准概率 $`\pi_0=\tfrac12\mathbf1`$ 上的负梯度。对于 $`M=1`$：


```math
u_e^{(1)}
=\frac1{|\mathcal U|}\sum_{i\in\mathcal U}
\mathbb E_{b_{-e}}
\left[\log p_{i,(b_{-e},1)}(y_i)
-\log p_{i,(b_{-e},0)}(y_i)\right].
```


现有 Stage 文件已保存每个 edge、每个背景块、每张查询图像的开/关真值 log probability。A、C 直接复用这些统计量；B 在这些统计量旁新增独立辅助图。

## 4. B：16 图效用的推导与测量

对 Bernoulli 变量，固定其余变量后有：


```math
\frac{\partial}{\partial\pi_e}\mathbb E_{b\sim P_\pi}F(b)
=\mathbb E_{b_{-e}}\left[F(b_{-e},1)-F(b_{-e},0)\right].
```


在 $`M`$ 张图上，对 $`\pi_e`$ 求导会得到 $`M`$ 个图槽的贡献；它们同分布，故可固定第一槽干预并乘 $`M`$。令其余 $`M-1`$ 张独立辅助图的真值概率和为：


```math
S_{i,M-1}=\sum_{m=2}^{M}p_{i,b^{(m)}}(y_i).
```


于是精确的一阶目标为：


```math
\begin{aligned}
u_e^{(M)}
&=-\left.\frac{\partial R_M}{\partial\pi_e}\right|_{\pi_0}\\[4pt]
&=\frac{M}{|\mathcal U|}\sum_{i\in\mathcal U}
\mathbb E\left[
\log\bigl(S_{i,M-1}+p^+_{i,e}\bigr)
-\log\bigl(S_{i,M-1}+p^-_{i,e}\bigr)
\right].
\end{aligned}
```


开/关两项使用相同的其余边与辅助图，辅助图独立于被干预的第一图槽。混合分母 $`M`$ 在 log 差中抵消，外面的导数系数 $`M`$ 保留。$`M=1`$ 时回到原单图效用。

若原开/关日志分别为 $`\ell^-_{i,e},\ell^+_{i,e}`$，辅助图日志为 $`a_{i,m}`$，使用稳定计算：


```math
\begin{aligned}
A_i&=\mathop{\mathrm{LSE}}\nolimits_{m=2}^{M}a_{i,m},\\[4pt]
\Delta_{i,e}^{(M)}
&=M\left[
\mathop{\mathrm{logaddexp}}\nolimits(A_i,\ell^+_{i,e})
-\mathop{\mathrm{logaddexp}}\nolimits(A_i,\ell^-_{i,e})
\right].
\end{aligned}
```


每个原测量块补充 15 张辅助图，同一块内所有边可共用这组图。Q1–Q4 各 8 块，Q5 16 块；三 seed 共 144 块，新增 2160 个辅助图配置。先在伪未见图像上平均，再对块取平均。保存块方差除以块数作为 Monte Carlo 均值方差估计。

实现：`abc.mixture_effect`、`abc_measure.teacher`。B 使用原八维响应输入、原归一化和相同岭惩罚，只替换标签为 $`u^{(16)}`$；部署仍使用每类预测标准差归一化、强度 1。

## 5. 27 参数全局效用回归

每条边的输入 $`h_{j,e}\in\mathbb R^8`$ 为教师支持集的干预响应。分别对“全部支持图像”和“目标原语支持图像”计算四项：真值 log probability 增量均值、增量标准差、候选中心化 logit 变化 RMS、预测熵变化均值。

支持集先按组合均衡取样；四个背景图共用既有 seed。响应只从教师 train/support 图像得到。标签来自教师的 query/measure 图像。

从 Stage 保存的训练归一化读取 $`\bar h,s_h,s_t`$。其中 $`s_t`$ 为九个拟合教师在关系类型 $`t`$ 上的单图标签标准差。对边类型 $`t(e)`$ 构造分块设计：


```math
\phi(h,e)
=\mathop{\mathrm{block}}\nolimits_{t(e)}
\left[1,\frac{h-\bar h}{s_h}\right]
\in\mathbb R^{27}.
```


对单图或混合标签分别求解：


```math
\widehat\beta
=\arg\min_\beta
\left\{
\frac1J\sum_{j=1}^{J}\sum_e
\left[\phi(h_{j,e},e)^\top\beta
-\frac{u_{j,e}}{s_{t(e)}}\right]^2
+\lambda\|\beta\|_2^2
\right\},
\quad J=9,\quad\lambda=\frac{10}{3}.
```


```math
\widehat\beta
=(X^\top X+30I)^{-1}X^\top\widetilde u,
\qquad
\widehat u_e=s_{t(e)}\phi(h_e,e)^\top\widehat\beta.
```


代码使用 `numpy.linalg.solve`，不显式计算矩阵逆。截距也参与惩罚。Matched 用原单图标签，B 用混合标签，二者共享 $`\bar h,s_h,s_t`$ 和 $`\lambda`$。拟合时验证 Matched 系数能复现 Stage 已保存的系数。

## 6. A：固定训练尺度与幅度

原 Matched 在每个部署教师/识别器上，用该类型预测的标准差归一化。A 改用拟合教师标签的类型内中心化 RMS：


```math
\sigma_{t,\rm train}
=\sqrt{\frac1{J|E_t|}
\sum_{j=1}^{J}\sum_{e\in E_t}
\left(u^{(1)}_{j,e}-\overline u^{(1)}_{j,t}\right)^2}.
```


```math
r_e^A=\frac{\widehat u_e-\overline{\widehat u}_{t(e)}}
{\max(\sigma_{t(e),\rm train},10^{-12})}.
```


对任意类型内得分 $`r`$，求解带等式预算的熵正则分配：


```math
\begin{aligned}
\pi^*(r)=\arg\max_{0\lt \pi_e\lt 1}\quad
&\sum_e r_e\pi_e
-\sum_e D_{\rm KL}\!\left(\mathop{\mathrm{Ber}}\nolimits(\pi_e)\Vert\mathop{\mathrm{Ber}}\nolimits(1/2)\right)\\[4pt]
\text{s.t.}\quad &\sum_{e\in E_t}\pi_e=|E_t|/2,\quad t=0,1,2.
\end{aligned}
```


KKT 驻点满足 $`r_e-\mathop{\mathrm{logit}}\nolimits(\pi_e)-\gamma_t=0`$，因此：


```math
\pi_e^*(r)=\mathop{\mathrm{sigmoid}}\nolimits(r_e-\gamma_{t(e)}).
```


左侧类型概率和对 $`\gamma_t`$ 严格递减，用二分法求唯一解。A 再与 Uniform 插值：


```math
\pi_e^A(\eta)=\frac12+\eta\left(\pi_e^*(r^A)-\frac12\right),
\qquad\eta\in\{0,0.25,0.5,1\}.
```


插值保留每类半预算。选择三个 seed、八个图库的 Q4 独立 evaluate exact AUC 均值最高的共同 $`\eta`$，并列取较小值。选择结果写入 `utility/selection.json`。A 的尺度来自拟合教师；Q4 只选择四档幅度。

Matched、B、C 采用各自预测向量的类型内标准差作为得分分母；C 在每个情境内分别处理。它们的正式强度固定为 1。Q4/Q5 同时记录 0、0.25、0.5、1 的插值曲线，用于理解分配方向和幅度。

## 7. C：冻结物体路由、条件响应与条件标签

只用 CLIP 物体文本锚点，将 12 个物体划分成各六个的两组。枚举包含物体索引 0 的六元组，最小化两组内部欧氏平方距离总和。共 $`\binom{11}{5}=462`$ 种划分；锚点先单位化，数值并列按索引顺序选择。


```math
(G_0,G_1)
=\arg\min_{|G_0|=|G_1|=6}
\sum_{g=0}^{1}\sum_{o\in G_g}\|v_o-\overline v_{G_g}\|_2^2.
```


给图像分配情境时，用冻结 CPF 主干的物体概率，而非真值物体：


```math
g(x)=\arg\max_{g\in\{0,1\}}
\sum_{o\in G_g}\mathop{\mathrm{softmax}}\nolimits(z^{\rm CPF}_o(x))_o.
```


并列归入组 0。支持响应、查询测量和最终部署使用完全相同的路由规则。`groups.json` 保存具体对象名称和划分。

对支持图像权重 $`w_i`$，在情境 $`g`$ 内重归一化：


```math
w_{i\mid g}
=\frac{w_i\mathbf1\{g(x_i)=g\}}
{\sum_k w_k\mathbf1\{g(x_k)=g\}}.
```


在全局八维响应的两个四维块上分别这样计算，得到 $`h_{j,e,g}`$。某个四维块没有该情境支持图像时，使用该边对应的全局四维块，其情境差为零，并记录 `empty_blocks`。

对于伪未见 measure 图像，令 $`n_{j,g}`$ 为路由到情境 $`g`$ 的图像数，$`n_j`$ 为总数：


```math
\omega_{j,g}=\frac{n_{j,g}}{n_j},
\qquad
u_{j,e,g}^{(1)}
=\mathbb E\!\left[\Delta_{i,e}^{(1)}\mid g(x_i)=g,\ i\in\mathcal U_j\right].
```


```math
u_{j,e}^{(1)}=\sum_g\omega_{j,g}u_{j,e,g}^{(1)}.
```


代码用同一组逐图像干预统计计算条件与全局标签，检查上式数值一致。查询情境为空时，其权重为零。

定义保留零点的情境差分归一化：


```math
\begin{aligned}
d_k^2&=\frac1{J|E|}\sum_{j,g,e}
\omega_{j,g}(h_{j,e,g,k}-h_{j,e,k})^2,\\[4pt]
\delta h_{j,e,g,k}&=
\frac{h_{j,e,g,k}-h_{j,e,k}}{\max(d_k,10^{-6})}.
\end{aligned}
```


差分不减均值，因此零差分始终表示“与全局响应相同”。51 参数设计为三个类型块，每块包含截距、八维全局项和八维差分项：


```math
\psi_{j,e,g}
=\mathop{\mathrm{block}}\nolimits_{t(e)}
\left[1,\frac{h_{j,e}-\bar h}{s_h},\delta h_{j,e,g}\right]
\in\mathbb R^{51}.
```


```math
\widehat\theta
=\arg\min_\theta
\left\{\frac1J\sum_{j,g,e}\omega_{j,g}
\left(\psi_{j,e,g}^\top\theta-
\frac{u_{j,e,g}^{(1)}}{s_{t(e)}}\right)^2
+\lambda\|\theta\|_2^2\right\}.
```


```math
\widehat\theta
=(Z^\top WZ+30I)^{-1}Z^\top W\widetilde u,
\qquad
\widehat u_{e,g}=s_{t(e)}\psi_{e,g}^\top\widehat\theta.
```


每个教师的两情境权重和为 1。若将情境差分项系数限制为零，由加权平方误差分解，上式对全局系数的目标与 27 参数全局目标只差一个常数，因而严格包含全局模型。若所有情境响应恰好等于全局响应，岭回归也会将差分系数自动解为零。

部署时每个情境各自求满足半预算的概率 $`\pi_{e,g}`$。固定一组公共随机数 $`U_{m,e}`$，图像 $`x`$ 使用：


```math
b_e^{(m)}(x)=\mathbf1\{U_{m,e}\lt \pi_{e,g(x)}\},
\qquad U_{m,e}\sim\mathop{\mathrm{Uniform}}\nolimits(0,1).
```


公共随机数让两个情境及不同策略配对比较。每张图像仍只接受 16 张图的评分，代码按情境分组处理该组图像，不为每张图像重复计算两套完整预测。

## 8. 对照、曲线和统计单位

正式识别包含 Uniform、Matched、A、B、C、情境打乱。Matched/A/B/C 各加五个类型内关系置换，合计 26 种策略。情境模型的置换在两个情境中使用同一边排列，保留该边的跨情境向量。

情境打乱模型固定选择九个拟合教师中的五个，交换其两个情境的输入响应，标签、权重和全局输入保留。所有边共用该教师的交换，不逐边随机拆散。归一化与 C 相同。这样可以比较条件输入和条件效用对应关系的贡献。

Q4/Q5 另有三种“测量效用直接选图”诊断，以及 response averaging 的 readout-only、refit 两种对照。测量选图用有限块得到的效用，不称为真实最优策略。

对于任一策略，定义单图或混合风险的一阶改善近似：


```math
G_M^{\rm lin}(\pi)
=\sum_e u_e^{(M)}(\pi_e-1/2).
```


情境策略则为：


```math
G_{M,\rm ctx}^{\rm lin}(\pi)
=\sum_g\omega_g\sum_e u_{e,g}^{(M)}(\pi_{e,g}-1/2).
```


在同一 measure 图像上，用独立图库估计实际改善 $`R_M(\pi_0)-R_M(\pi)`$，记录其与上述近似之差。另在独立 evaluate 图像上记录泛化后的 NLL 和 exact AUC。两类结果分文件保存，方向近似与图像迁移不混为一个误差。

| 输出 | 核心观察 |
|---|---|
| 正式 CW/OW AUC native、exact；VAL bias 下的 HM | 同 seed、同图库，相对 Uniform 与 Matched 的增益 |
| 真实策略减五次关系置换均值 | 关系对应是否贡献于分配，而不仅是概率离散度 |
| C 减情境打乱；条件标签方差与覆盖 | 情境输入是否解释稳定的条件效用差异 |
| 单图与 16 图标签的秩相关、符号变化；B 的 Q5 预测误差 | 部署读出是否改变了有用关系的排序 |
| 幅度—NLL—AUC 曲线；线性近似偏差 | 有效方向在怎样的幅度上转化为实际收益 |
| epoch 0–10、选中 epoch、耗时与峰值显存 | 适配是否充分，以及新增机制的运行成本 |

正式图库为 1701、2701、3701；Q4/Q5 图库为 8201–8208。先在每个 seed 内平均图库，再给出三个 seed 的均值、样本标准差和逐 seed 配对值。三 seed 共享九教师拟合出的效用模型，因此它们代表三个识别轨迹上的条件结果，图库数不当作额外训练重复。

正式校准逐策略在 VAL 确定 bias 和 temperature，然后原样用于 TEST。AUC 仍由完整 seen–unseen 扫描曲线计算；固定 VAL bias 下的 HM 单独保存。

## 9. 代码与文件对应

| 模块 | 实现 |
|---|---|
| `abc_train.py` | Uniform 10 epoch、epoch 0 选模、优化器和 RNG 恢复 |
| `abc.py` | 有限混合效用、27/51 参数回归、幅度和预算策略 |
| `abc_measure.py` | 原开关统计复用、辅助图测量、条件响应和条件标签 |
| `abc_formal.py` | 两情境路由、16 图概率混合、CW/OW 评分 |
| `abc_eval.py` | Q4/Q5 曲线、正式策略评价、同概率策略去重 |
| `abc_workflow.py` | Stage 导入、Q4 图像划分、训练/拟合/选择顺序 |
| `abc_report.py` | 配对差值、seed 内图库平均、完整性汇总 |
| `scripts/run_queue.py` | 三 GPU 队列与共享拟合屏障 |
| `scripts/pack_results.py`、`upload.ps1`、`download.ps1` | 固定路径传输、SHA256、打包进度、自动解压 |

完整代码以相邻 ZIP 和源码目录为准。下一节直接摘录对应函数，便于核对公式与实际实现。


## 10. 核心实现摘录

以下代码直接从交付模块抽取。模块内的 NumPy、PyTorch 和既有 CPF 工具导入见源码；没有另写一套演示实现。

### 10.1 有限混合效用与条件标签

来源：`trustcpf/abc.py`。

```python
def mixture_effect(off,on,aux):
    """[E,N] paired log probabilities; [M-1,N] independent auxiliary logs."""
    off,on,aux=[v.double() for v in (off,on,aux)]
    m=len(aux)+1
    if m==1:return on-off
    a=aux.logsumexp(0)
    return m*(torch.logaddexp(a[None],on)-torch.logaddexp(a[None],off))

def conditional_targets(effects,routes,sample_seen):
    """effects: K x E x N; supervision uses pseudo-unseen images only."""
    keep=~sample_seen.bool();n=int(keep.sum());weights=[];targets=[];blocks=[]
    for g in range(2):
        use=keep&(routes==g);weights.append(float(use.sum())/n)
        b=effects[:,:,use].mean(-1) if use.any() else effects.new_zeros(effects.shape[:2])
        blocks.append(b);targets.append(b.mean(0))
    global_blocks=effects[:,:,keep].mean(-1)
    targets=torch.stack(targets);blocks=torch.stack(blocks)
    torch.testing.assert_close((targets*targets.new_tensor(weights)[:,None]).sum(0),global_blocks.mean(0),atol=1e-11,rtol=1e-9)
    return dict(target=global_blocks.mean(0),blocks=global_blocks,
                context=targets,context_blocks=blocks,weights=np.array(weights))
```

### 10.2 情境模型的设计与闭式求解

来源：`trustcpf/abc.py`。

```python
def context_design(h,hg,types,norm,delta_scale):
    """G x E x 51; zero context difference stays zero."""
    h=np.asarray(h,dtype=np.float64);hg=np.asarray(hg,dtype=np.float64)
    global_x=(h-norm['center'])/norm['spread']
    diff=(hg-h[None])/delta_scale
    raw=np.concatenate([np.ones((*diff.shape[:2],1)),np.broadcast_to(global_x,diff.shape),diff],-1)
    return np.concatenate([raw*(types==t)[None,:,None] for t in range(3)],-1)

def fit_context(xs,xgs,ys,weights,types,norm,alpha,swap=()):
    delta=np.stack([g-x[None] for x,g in zip(xs,xgs)])
    scale=np.sqrt(np.einsum('jg,jgek->k',np.stack(weights),delta**2)/(len(xs)*len(types)))
    scale=np.maximum(scale,1e-6)
    design=[];labels=[];mass=[]
    for j,(x,g,y,w) in enumerate(zip(xs,xgs,ys,weights)):
        z=context_design(x,g,types,norm,scale)
        if j in swap:
            # Only conditional inputs are swapped. Global input is group-invariant.
            z=z[::-1].copy()
        design.append(z.reshape(-1,51))
        labels.append((y/norm['scales'][types]).reshape(-1))
        mass.append(np.repeat(w,len(types)))
    z=np.concatenate(design);y=np.concatenate(labels);w=np.concatenate(mass)
    beta=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(51),z.T@(w*y))
    return dict(norm,delta_scale=scale,beta=beta,alpha=float(alpha),dimension=51,swapped_teachers=list(swap))
```

### 10.3 固定尺度、幅度与等式预算

来源：`trustcpf/abc.py`。

```python
def fixed_scale(ys,types):
    return np.array([max(np.sqrt(np.mean([np.mean((y[types==t]-y[types==t].mean())**2) for y in ys])),1e-12) for t in range(3)])

def score_policy(mu,types,scale=None,eta=1.):
    mu=np.atleast_2d(np.asarray(mu,dtype=np.float64));out=[]
    for row in mu:
        if scale is None:p,_=align.policy(row,types,1.)
        else:
            p=torch.empty(len(types),dtype=torch.double)
            for t in range(3):
                ix=types==t;r=(row[ix]-row[ix].mean())/scale[t]
                lo,hi=float(r.min()-50),float(r.max()+50)
                for _ in range(100):
                    mid=(lo+hi)/2;q=1/(1+np.exp(np.clip(mid-r,-700,700)))
                    if q.sum()>len(q)/2:lo=mid
                    else:hi=mid
                p[ix]=torch.from_numpy(1/(1+np.exp(np.clip((lo+hi)/2-r,-700,700))))
        out.append(.5+float(eta)*(p-.5))
    return torch.stack(out)
```

### 10.4 冻结物体路由与分组评分

来源：`trustcpf/abc.py`。

```python
def route(obj_logits,groups):
    p=obj_logits.softmax(-1)
    return torch.stack([p[:,g].sum(-1) for g in groups],-1).argmax(-1)
```

### 10.5 每张图像只使用所属情境的图

来源：`trustcpf/abc_formal.py`。

```python
@torch.no_grad()
def routed_stack(model,base,masks,routes):
    if len(masks)==1:
        return torch.stack([model.score(base,mask)[0] for mask in masks[0]])
    n=len(routes);result=None
    for g in range(len(masks)):
        ix=torch.where(routes==g)[0]
        if not len(ix):continue
        part={k:v[ix] for k,v in base.items()}
        scores=torch.stack([model.score(part,mask)[0] for mask in masks[g]])
        if result is None:result=scores.new_empty((len(masks[g]),n,model.no,model.na))
        result[:,ix]=scores
    return result
```

### 10.6 完整 10 epoch 与恢复

来源：`trustcpf/abc_train.py`。

```python
def uniform(args,until_epoch=None):
    work,cfg,manifest,text,bank=align.context(args);cfg=dict(cfg,variant='uniform',readout='probability')
    if cfg['adapt_epochs']!=10:raise ValueError('This experiment uses 10 complete epochs')
    out=work/'runs/uniform';out.mkdir(parents=True,exist_ok=True)
    cp=work/'reference/last.pt';model=align.restore(text,bank,cfg,args.device,cp);model.backbone.requires_grad_(False)
    ids=engine.row_ids(args.cache,manifest['pairs']['train'],manifest['labels'])
    opt=engine.optimizer(model,cfg);loader=engine.batches(args.cache,'train',cfg,ids,True)
    candidates=torch.tensor(pair_ids(manifest['labels'],manifest['pairs']['train']),device=args.device)
    pi=torch.full((len(bank['types']),),.5,device=args.device)
    prepared=state_eval.prepare_phase(model,args.cache,'val',cfg)
    def validate():
        model.eval()
        return state_eval.evaluate_cached(model,args.cache,cfg,manifest,pi,'val',prepared=prepared)
    if (out/'last.pt').exists():
        st=load_torch(out/'last.pt')
        assert st['initial_sha256']==sha(cp) and st['cfg']==cfg
        model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        torch.set_rng_state(st['torch_rng']);np.random.set_state(st['numpy_rng'])
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state_all(st['cuda_rng'])
    else:
        v=validate()
        st=dict(epoch=0,model=copy.deepcopy(model.state_dict()),best_model=copy.deepcopy(model.state_dict()),
            best_epoch=0,best_metrics=v,optimizer=opt.state_dict(),history=[dict(epoch=0,val=v)],
            pi=pi.cpu(),cfg=cfg,initial_sha256=sha(cp),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st)
    export(out,st)
    limit=10 if until_epoch is None else min(10,until_epoch)
    total=max(1,10*len(loader));warmup=max(1,len(loader))
    for epoch in range(st['epoch'],limit):
        model.train();losses=[];tic=time.perf_counter()
        for bi,data in enumerate(loader):
            step=epoch*len(loader)+bi
            factor=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total-warmup)))
            for g in opt.param_groups:g['lr']=cfg['graph_lr']*factor
            g,fa,fo,a,o=[v.to(args.device) for v in data];mask=torch.bernoulli(pi)
            joint,lo,lc=model(g,fa,fo,mask)
            loss=task_loss(joint,lo,lc,a,o,candidates)+cfg['lambda_w']*model.graph.regularizer()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_([p for gg in opt.param_groups for p in gg['params']],cfg['gradient_clip'])
            opt.step();losses.append(float(loss.detach()))
        v=validate()
        if v['cw']['AUC_native']>st['best_metrics']['cw']['AUC_native']:
            st.update(best_epoch=epoch+1,best_metrics=v,best_model=copy.deepcopy(model.state_dict()))
        st['history'].append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val=v,
            learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-tic))
        st.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st);export(out,st)
        print('UNIFORM_EPOCH',args.seed,epoch+1,'loss',st['history'][-1]['loss'],'val_auc',v['cw']['AUC_native'],'best_epoch',st['best_epoch'],flush=True)
    if st['epoch']==10:
        original=load_torch(cp)['model']
        for key,value in original.items():
            if key.startswith('backbone.'):
                assert torch.equal(value,st['model'][key].detach().cpu())
                assert torch.equal(value,st['best_model'][key].detach().cpu())
        chosen=max(st['history'],key=lambda r:(r['val']['cw']['AUC_native'],-r['epoch']))['epoch']
        assert chosen==st['best_epoch'] and len(st['history'])==11
        write_json(out/'training_contract.json',dict(epochs_completed=10,selected_epoch=chosen,
            best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),initial_sha256=sha(cp),
            backbone_identical=True,selection='VAL CW native AUC, earliest maximum among epoch 0..10',
            training='new Uniform trajectory from reference; common frozen weights for A/B/C'))
```
