A/B/C 第一批已于 2026-10-27 全部完成，见 [完整分析](analysis/README.md)。本页保留实验前 v8 公式及计划；末尾完整方法确认属于下一步研究安排。

# TRUST-CPF：部署效用与情境迁移

## 公式修订与后续实验方案 v8

日期：2026-10-26。平台：UT-Zappos natural split、CLIP ViT-L/14 缓存、工程内 CPF 强基底、96 条关系。依据：第七版 PDF，以及截至 Stage 第二批的全部已归档实验分析与实现。

**下一步保留 TRUST 的“组合留出效用监督 → 效用预测 → 关系预算分配”主线，集中完成一批冻结识别实验：A 检验动作幅度，B 将监督改为实际 16 图混合效用，C 学习对象情境下的关系效用。B、C 是主要研究变量，A 是统一的动作诊断。**

第一批直接复用 Stage 的 15 个教师和 State 的三份 Uniform-best，新增效用测量、低维拟合和冻结评分。现有教师已完成追加 10 epoch，正式 Uniform 已完成 10 epoch；本批不再重复相同训练。第二批将选定机制组合并完成新的完整流水线及第二数据集验证。

本文给出可以直接据此编写代码的公式、数据接口、默认配置、比较组与结果解释。新的 B/C 实验尚未执行；文末数值检验验证的是公式和实现约定。

## 1. 历史结果怎样约束下一步

历史表使用各轮原报告口径，跨轮绝对成绩用于追踪研究进展；方法增量在同轮协议内比较。AUC 使用 0–100 尺度，增量单位为百分点。

| 已完成阶段 | 关键事实 | 本次采用的研究决策 |
| --- | --- | --- |
| 弱基底 TRUST v1 | Robust 9.901／8.172，Capacity 13.902／11.240，效用预测未超过简单对照 | 保留已经建立的强 CPF 底座 |
| 静态知识 v2 | Knowledge 相对 Base 为 +0.617／+0.600，Shuffle 均值更高 | 必须评价正确关系对应带来的收益 |
| 强基底完整 TRUST | Robust 35.201／26.149，Capacity 35.805／26.400，Dense 35.646／26.288 | 效用策略以同结构 Uniform 为直接参照，最终统一补齐容量基线 |
| 响应条件化 MLP | 34.916／25.958，未超过原 Robust | 优先增加可用信息，保持小型预测器 |
| Compact | 效用排序相关均值 0.293，标签打乱为 −0.022；正式收益仍弱 | 继续保留效用监督和紧凑拟合 |
| Align | 正式相对 Uniform 约 −0.088／+0.008，部分冻结收益在适配后消失 | 冻结同一识别器比较效用机制 |
| Competition 10 epoch | 正式与 Uniform 接近；Q5 测量 Competition 有收益，预测 Competition 约 −0.059 | 不再次直接用竞争损失替换全部效用目标 |
| R5 | Relative 正式均选择零强度；Q5 收益低于 Absolute | 保留 Absolute 拟合，直接检验目标与情境信息 |
| State | 联合拟合的 Q5 收益由 +1.056 升至 +1.339；正式为 −0.112／−0.008 | 多教师联合拟合作为共同基础 |
| Stage | Matched-Single 正式为 −0.001／+0.002；Q5 为 +1.463 | 阶段匹配已完成，研究重心转向效用迁移的内容 |

强基底阶段的噪声、知识打乱与机制拆解已经提供补充证据：真实关系在部分比较中有作用，原不确定性扣分没有稳定增量。本轮主模型继续使用期望效用，不重新加入风险惩罚、对抗扰动或更大的效用网络。

### 1.1 最新结果提供了三个明确的改进理由

Stage 正式结果先在每个 seed 内平均三个 bank，再对三个 seed 汇总：

| 方法 | TEST CW | TEST OW | 相对 Uniform CW／OW | Q5 相对 Uniform |
| --- | --- | --- | --- | --- |
| Uniform | 35.904 | 26.461 | 0／0 | 0 |
| Old-Single | 35.792 | 26.453 | −0.112／−0.008 | +1.336 |
| Matched-Single | 35.903 | 26.463 | −0.001／+0.002 | +1.463 |
| Old-Average | 35.926 | 26.513 | +0.022／+0.052 | +1.077 |
| Matched-Average | 35.936 | 26.483 | +0.032／+0.023 | +1.208 |

正式列为 native AUC，Q5 为 exact AUC，二者使用不同教师和数据。Matched-Single 在 Q5 相对自身五次关系置换均值提高 **1.319**，正式 CW 相对置换仅 **0.000047**。效用在留出教师上具有关系对应价值，正式迁移仍是核心问题。

第一，单图 U-NLL 与 16 图概率混合是两个不同的风险目标，需要直接检验目标匹配。此前单图、混合收益多数同向，说明目标差异的实际重要程度仍要测量，不能仅凭公式不同认定它是主要原因。

第二，平均响应提高了预测 Spearman 和一阶代理，却使两套预测器的 Q5 AUC 都下降约 0.26 点。类型标准化、动作幅度和实际混合风险需要一起观察。

第三，Stage 已更新教师状态，正式重复响应的排序相关仍约 0.367；当前状态响应也未稳定超过 reference 响应。继续匹配阶段或重复同类探针，对正式组合迁移的帮助已经有限。下一步需要显式表示“哪类输入上，这条关系有用”。

## 2. 对第七版 PDF 的采用与细化

| PDF 的建议 | 本版决定及具体细化 |
| --- | --- |
| A 小步诊断与固定训练尺度 | 采用；四点曲线作为统一诊断，A 的唯一部署幅度由 Q4 实际决策选择 |
| B 有限 16 图边际效用 | 作为主实验；保留精确系数 16，只干预一个图槽位，并明确辅助图独立性 |
| C 少量情境与 51 参数映射 | 采用二情境；对象分组、路由、空支持块和标准化都给出确定实现 |
| 当前模型是 C 的特例 | 保留；用加权平方损失恒等式验证，情境差分采用不减均值的 RMS 缩放 |
| 情境打乱对照 | 固定 51 参数；部分教师交换两个情境输入，标签与权重保持绑定，避免所有教师统一换名 |
| 实际决策参与选择 | A 用 Q4 exact AUC 选幅度；B/C 主比较固定现有正则和强度，Q4 记录完整风险与 AUC |
| 条件测量与条件预测收益 | 测量策略只用于 Q4/Q5 独立图像；正式 VAL/TEST 使用从 TRAIN 学到的预测策略 |
| 五个验证关口 | 合并为一个自动队列和一套结果表，整批执行后统一判断 |

**本批 C 的具体含义是对象情境条件化。** 它是对组合竞争情境的一个低维、可观测近似，并未显式建模完整候选集合的竞争结构。若它有效，再扩展更细的候选竞争表示；首批直接检验这两个固定情境。

## 3. 共同底座与符号

关系集合为 $`E=E_{\mathrm{OI}}\cup E_{\mathrm{OE}}\cup E_{\mathrm{AE}}`$，数量为 12／36／48；每张图独立采样关系掩码 $`B\in\{0,1\}^{96}`$：


```math
q_\pi(B)=\prod_{e\in E}\pi_e^{B_e}(1-\pi_e)^{1-B_e},
\qquad \rho_e=\frac12 .
```


每类型的约束是期望预算：


```math
\mathcal B=
\left\{\pi\in(0,1)^{96}:
\sum_{e\in E_t}\pi_e=\frac{n_t}{2},\
t\in\{\mathrm{OI},\mathrm{OE},\mathrm{AE}\}\right\}.
```


因此预算为 6／18／24，单张图的实际边数随 Bernoulli 采样变化。沿用这个采样分布，才能继续使用下文的精确边际公式。

固定识别器参数 $`\theta`$。工程中物体与条件属性的联合分数对应：


```math
q_\theta(a,o\mid x,B)
=p_\theta(o\mid x,B)\,p_\theta(a\mid o,x,B).
```


先在完整候选集合 $`\mathcal C`$ 上归一化，再做图间概率平均：


```math
p_{\theta,\mathcal C}(y\mid x,B)
=\frac{q_\theta(y\mid x,B)}
{\sum_{c\in\mathcal C}q_\theta(c\mid x,B)},
\qquad
\bar p_M(y\mid x)
=\frac1M\sum_{m=1}^{M}p_{\theta,\mathcal C}(y\mid x,B_m).
```


全篇固定 $`M=16`$。测量与 Q4/Q5 使用原 TRAIN 候选表的 83 个组合；正式 CW／OW 使用现有各自候选集，OW 为 16×12 个组合。情境路由不会裁剪候选集合。

教师索引 $`j`$ 表示一个 seed 与一个组合留出折。Q1–Q3 跨三个 seed 构成 $`J=9`$ 个拟合教师；三个 Q4 用于开发决策；三个 Q5 用于留出教师评价。所有公式中的风险样本 $`U_j`$ 为 measure 中的伪未见图像，当前每教师 256 张。evaluate 中另有独立的 seen／unseen 图像用于完整 AUC。

## 4. 从当前效用到实际 16 图效用

### 4.1 当前单图效用的准确含义

对固定候选集、固定模型和固定图像分布，定义：


```math
R_1(\pi)=
\mathbb E_{(x,y)}\mathbb E_{B\sim q_\pi}
\left[-\log p_{\theta,\mathcal C}(y\mid x,B)\right].
```


令 $`L(B)=\mathbb E_{(x,y)}[-\log p_{\theta,\mathcal C}(y\mid x,B)]`$，将某条边的 Bernoulli 期望写开：


```math
R_1(\pi)=
(1-\pi_e)\mathbb E_{B_{-e}}L(B_{-e},0)
+\pi_e\mathbb E_{B_{-e}}L(B_{-e},1).
```


于是：


```math
u_e^{(1)}(\pi)
:=-\frac{\partial R_1(\pi)}{\partial\pi_e}
=\mathbb E_{(x,y),B_{-e}}
\left[\log p^+_e(y\mid x)-\log p^-_e(y\mid x)\right].
```


当前标签测量的是 $`u_e^{(1)}(\rho)`$。它是随机单图风险的局部负梯度，不能直接称为 16 图混合风险的梯度。

### 4.2 修订后的有限混合效用

把风险改为实际读出：


```math
R_M(\pi)=
\mathbb E_{(x,y)}
\mathbb E_{B_1,\ldots,B_M\stackrel{\mathrm{iid}}{\sim}q_\pi}
\left[
-\log\left(\frac1M\sum_{m=1}^M
p_{\theta,\mathcal C}(y\mid x,B_m)\right)
\right].
```


只在槽位 1 干预关系 $`e`$。另外 $`M-1`$ 张图独立于被干预图，开关两次使用同一组辅助图。记：


```math
S_{M-1}=\sum_{m=2}^{M}p_{\theta,\mathcal C}(y\mid x,B_m),
```


```math
\begin{aligned}
p_e^+&=p_{\theta,\mathcal C}(y\mid x,B_{1,-e},B_{1,e}=1),\\[4pt]
p_e^-&=p_{\theta,\mathcal C}(y\mid x,B_{1,-e},B_{1,e}=0).
\end{aligned}
```


其中 $`p_e^+`$ 对应 1，$`p_e^-`$ 对应 0。下式期望包含样本、被干预图的其余关系，以及另外 15 张辅助图：


```math
\boxed{
u_e^{(M)}(\pi)
:=-\frac{\partial R_M(\pi)}{\partial\pi_e}
=M\,\mathbb E
\left[\log(S_{M-1}+p_e^+)-\log(S_{M-1}+p_e^-)\right].
}
```


**推导。** 图空间有限，对 $`M`$ 个图分布的乘积求导，将导数分成 $`M`$ 项；第 $`m`$ 项为该槽位关闭与打开关系后的混合损失差。各槽位独立同分布，每项期望相同；混合中的 $`1/M`$ 在两项对数之差中抵消，得到上式的系数 $`M`$。

当 $`M=1`$ 时 $`S_0=0`$，公式恢复单图效用。一次同时改变全部 16 张图中的同一关系，会得到另一个有限干预量，不是这里的偏导数。

该导数针对采图分布的期望风险。给定一组固定均匀随机数后，阈值采图得到的有限样本分数对概率是阶梯函数；不能对该阶梯函数做普通反向传播，替代上面的期望导数。

### 4.3 与现有保存文件逐项对应的估计式

第 $`b`$ 个背景块保存每条关系、每张图像的开关真实类 log-probability：


```math
\ell_{j,e,b,i}^{\pm}=\log p_{j,e,b,i}^{\pm}.
```


新增 15 张辅助图的真实类 log-probability，先计算：


```math
A_{j,b,i}=\mathop{\mathrm{LSE}}\nolimits_{m=2,\ldots,16}
\ell_{j,b,m,i},
```


```math
d^{(16)}_{j,e,b,i}
=16\left[
\mathop{\mathrm{logaddexp}}\nolimits(A_{j,b,i},\ell^+_{j,e,b,i})
-\mathop{\mathrm{logaddexp}}\nolimits(A_{j,b,i},\ell^-_{j,e,b,i})
\right].
```


使用 float64 聚合，不先将极小概率取出再求和。最终：


```math
\widehat u^{(16)}_{j,e}
=\frac1{K_jN_j}
\sum_{b=1}^{K_j}\sum_{i\in U_j}d^{(16)}_{j,e,b,i}.
```


Q1–Q4 沿用 $`K_j=8`$，Q5 沿用 16。辅助图使用独立随机数流；同一块的辅助图由全部边共享。不同边共享辅助图会产生相关估计误差，但每条边的估计仍满足上面的期望定义。

背景块均值与标准误保存为：


```math
d_{j,e,b}^{(M)}=\frac1{N_j}\sum_{i\in U_j}d^{(M)}_{j,e,b,i},
\qquad
\widehat{\mathrm{SE}}^2_{j,e}
=\frac{\sum_b(d_{j,e,b}^{(M)}-\widehat u_{j,e}^{(M)})^2}
{K_j(K_j-1)}.
```


这是给定测量图像后的背景采样误差。独立图像上的决策收益另由 evaluate 集合计算。

### 4.4 B 的拟合只改变监督标签

沿用当前全局八维响应 $`h_{j,e}`$，输入中心 $`c`$、尺度 $`s`$ 和拟合标签尺度 $`a_t`$ 均从现有 Matched 拟合记录读取。输入设计为：


```math
z_{j,e}=\mathop{\mathrm{onehot}}\nolimits(t(e))
\otimes
\begin{bmatrix}1\\(h_{j,e}-c)\oslash s\end{bmatrix}
\in\mathbb R^{27}.
```


对 $`M\in\{1,16\}`$：


```math
\widehat\beta_M=
\arg\min_\beta
\left[
\frac1J\sum_{j=1}^J\sum_{e\in E}
\left(z_{j,e}^{\top}\beta-
\frac{\widehat u^{(M)}_{j,e}}{a_{t(e)}}\right)^2
+\lambda\|\beta\|_2^2
\right],
```


```math
\left(\sum_jZ_j^\top Z_j+J\lambda I\right)\widehat\beta_M
=\sum_jZ_j^\top\left(\widehat u_j^{(M)}\oslash a\right),
\qquad
\widehat\mu^{(M)}_{j,e}=a_{t(e)}z_{j,e}^{\top}\widehat\beta_M.
```


固定 $`\lambda=10/3`$、$`J\lambda=30`$，截距继续参与惩罚，使用线性方程求解。损失对教师取平均，对边求和，与当前实现一致。B 不重新选择正则，也不更换输入尺度，主比较直接反映监督目标从 $`u^{(1)}`$ 到 $`u^{(16)}`$ 的变化。

## 5. A：保留效用幅度，直接测量动作范围

### 5.1 当前单位方差策略与问题位置

对类型 $`t`$ 定义中心化矩阵：


```math
P_t=I-\frac1{n_t}\mathbf1\mathbf1^\top,\qquad
z_t=P_t\widehat\mu_t,\qquad
s_t^{\mathrm{deploy}}=\sqrt{\frac{\|z_t\|_2^2}{n_t}}.
```


当前分数为 $`r_t^{\mathrm{std}}=z_t/s_t^{\mathrm{deploy}}`$；若分母为零则令分数为零。对正数 $`c_0`$：


```math
r_t^{\mathrm{std}}(c_0\widehat\mu_t+b\mathbf1)
=r_t^{\mathrm{std}}(\widehat\mu_t).
```


因此，效用差异整体缩小不会减弱动作。其非零点处的微分为：


```math
\mathrm d r_t^{\mathrm{std}}
=\frac1{s_t^{\mathrm{deploy}}}
\left(I-\frac{z_tz_t^\top}{\|z_t\|_2^2}\right)
P_t\,\mathrm d\widehat\mu_t.
```


这给出 Stage 中“原始预测更稳定、最终概率仍波动”的一个明确放大环节。

### 5.2 只从拟合教师确定动作尺度

区分两种尺度：$`a_t`$ 用于拟合坐标，$`a_{t,0}`$ 用于动作。新的动作尺度为：


```math
a_{t,0}^2=
\frac1J\sum_{j=1}^J
\frac{\|P_t\widehat u^{(1)}_{j,t}\|_2^2}{n_t},
\qquad
r_t^{\mathrm{amp}}
=\frac{P_t\widehat\mu_t}{\max(a_{t,0},10^{-12})}.
```


A 首批沿用 Matched-Single 的单图标签、预测器与响应，只替换动作分数。类型中预测的边间差异越小，动作自然越接近 Uniform。未来若将该读出用于 $`u^{(16)}`$，相应 $`a_{t,0}`$ 由其自身 Q1–Q3 标签计算并固定。

### 5.3 预算求解与四点曲线

对任一已确定分数 $`r`$，使用原来的熵正则策略：


```math
\pi^\star(r)=
\arg\max_{\pi\in\mathcal B}
\left[
\kappa r^\top\pi-
\sum_e D_{\mathrm{KL}}
\big(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\,\|\,\mathop{\mathrm{Bern}}\nolimits(1/2)\big)
\right].
```


拉格朗日一阶条件给出：


```math
\pi_e^\star=\sigma(\kappa r_e-\gamma_t),
\qquad
\sum_{e\in E_t}\sigma(\kappa r_e-\gamma_t)=n_t/2.
```


右侧求和关于 $`\gamma_t`$ 连续严格递减，极限分别为 $`n_t`$ 和 0，因此每类型存在唯一解。实现用 float64 二分求根。这里固定 $`\kappa=1`$。

从 Uniform 到同一策略做线段插值：


```math
\pi_e(\eta)=\frac12+\eta\left(\pi_e^\star-\frac12\right),
\qquad \eta\in\{0,\tfrac14,\tfrac12,1\}.
```


对每个类型：


```math
\sum_{e\in E_t}\pi_e(\eta)
=\frac{n_t}{2}
+\eta\left(\sum_{e\in E_t}\pi_e^\star-\frac{n_t}{2}\right)
=\frac{n_t}{2}.
```


小步风险改善的预测量应使用与风险对应的效用：


```math
G_M^{\mathrm{lin}}(\eta)
=\widehat u^{(M)\top}\big(\pi(\eta)-\rho\big),
\qquad
G_M^{\mathrm{act}}(\eta)
=\widehat R_M(\rho)-\widehat R_M(\pi(\eta)).
```


局部展开：


```math
R_M(\rho+\delta)
=R_M(\rho)-u^{(M)\top}\delta
+\frac12\delta^\top H_M(\rho)\delta
+o(\|\delta\|^2).
```


逐点记录单图 NLL、16 图混合 NLL、exact AUC、对应一阶量与差值。共同随机数用于配对评分；四个有限步长避免极小概率变化在有限 bank 中完全不翻边。

A 的正式版本使用固定训练尺度，并在三个 Q4、八个 bank 的平均 exact AUC 上，从四个 $`\eta`$ 中选一个共享值；并列取较小值。选中 0 就原样报告 A 等于 Uniform。主表同时保留 Matched 的固定非零策略，关系对应问题由 B/C 继续检验。

Matched 的原标准化读出也计算同样曲线，以区分“改尺度”和“减小同一动作”的作用。B/C 主比较保留原标准化、$`\kappa=1`$，各自幅度曲线放在 Q4/Q5 诊断表，首批不按 TEST 曲线改变主版本。

## 6. C：对象情境条件化的关系效用

### 6.1 固定两个情境，不新增路由网络

从现有图库的 12 个对象文本锚点出发，单位化为 $`e_o`$。将对象分成大小各为 6 的两组：


```math
(\mathcal O_1,\mathcal O_2)
=\arg\min_{\substack{
\mathcal O_1\cup\mathcal O_2=\mathcal O\\[4pt]
\mathcal O_1\cap\mathcal O_2=\varnothing\\[4pt]
|\mathcal O_1|=|\mathcal O_2|=6}}
\sum_{g=1}^{2}\sum_{o\in\mathcal O_g}
\|e_o-\bar e_g\|_2^2,
\qquad
\bar e_g=\frac16\sum_{o\in\mathcal O_g}e_o.
```


令对象索引 0 必须位于第一组，枚举其余 11 个对象中选 5 个的 462 种划分；目标相同按索引字典序确定。使用单位化输入、普通欧氏均值中心，不在中心求均值后再次单位化。

已用本地保存锚点计算得到：

| 情境 | 固定对象组 |
| --- | --- |
| 1 | Boots.Ankle；Boots.Knee.High；Boots.Mid-Calf；Shoes.Boat.Shoes；Shoes.Oxfords；Shoes.Sneakers.and.Athletic.Shoes |
| 2 | Sandals；Shoes.Clogs.and.Mules；Shoes.Flats；Shoes.Heels；Shoes.Loafers；Slippers |

分组来自已有文本几何，是预先确定的研究变量。组内对象数相等并不保证图像路由人数相等；各教师的路由比例在结果中记录。

路由使用冻结 CPF backbone 的物体概率，直接读取 `model.base(...)[obj]` 的 softmax，不加入关系残差：


```math
p_\theta^0(o\mid x)=\mathop{\mathrm{softmax}}\nolimits(\ell_\theta^0(x))_o,
\qquad
g_\theta(x)=
\arg\max_{g\in\{1,2\}}
\sum_{o\in\mathcal O_g}p_\theta^0(o\mid x).
```


并列取情境 1。分组在所有教师、seed、数据分区中相同；物体概率来自各自对应的冻结 backbone。每张图像的路由在干预前计算一次，关系开关、图采样和评价 bias 都不会改变路由。部署只需要输入图像和已保存模型。

### 6.2 条件标签及全局标签的分解

用相同路由划分伪未见 measure 图像：


```math
U_{j,g}=\{i\in U_j:g_{\theta_j}(x_i)=g\},
\qquad
\omega_{j,g}=\frac{|U_{j,g}|}{|U_j|}.
```


首批 C 使用单图效用：


```math
\widehat u^{(1)}_{j,e,g}
=\frac1{K_j|U_{j,g}|}
\sum_{b=1}^{K_j}\sum_{i\in U_{j,g}}
\left(\ell^+_{j,e,b,i}-\ell^-_{j,e,b,i}\right).
```


空组权重为零，不生成该组监督行。对非空组严格有：


```math
\widehat u^{(1)}_{j,e}
=\sum_g\omega_{j,g}\widehat u^{(1)}_{j,e,g}.
```


已有逐图像 log-probability 足以重算这些标签。条件效用方差：


```math
V_{j,e}^{\mathrm{ctx}}
=\sum_g\omega_{j,g}
\left(\widehat u^{(1)}_{j,e,g}-\widehat u^{(1)}_{j,e}\right)^2
```


用于描述全局平均隐藏了多少差异。另保存两半图像、两半背景的条件差分重复性与符号变化。初始分析不将一次符号翻转直接解释为可迁移规律，核心证据是独立 evaluate 图像上的实际决策增益。

### 6.3 八维条件响应的完整定义

支持图像仍来自教师 Inner，正式模型来自 TRAIN。沿用每组合最多两张图、组合等权、四个配对背景；使用已经保存的第一次响应支持 ID、权重和随机数种子。

对支持图像 $`i`$ 和响应背景 $`b`$，先得到候选组合 log-probability $`\ell^\pm_{b,i,c}`$，定义：


```math
\Delta_{b,i}=\ell^+_{b,i,y_i}-\ell^-_{b,i,y_i},
\qquad
D_{b,i,c}=
(\ell^+_{b,i,c}-\ell^-_{b,i,c})
-\frac1{|\mathcal C|}\sum_{c'}(\ell^+_{b,i,c'}-\ell^-_{b,i,c'}).
```


中心化后的 log-probability 差与中心化后的候选联合 logit 差相同，因为 softmax 归一化只减去样本内常数。


```math
\Delta H_{b,i}
=-\sum_c p^+_{b,i,c}\ell^+_{b,i,c}
+\sum_c p^-_{b,i,c}\ell^-_{b,i,c}.
```


令 $`w_i`$ 为旧支持权重。对于任意非空支持子集 $`A`$，置：


```math
\widetilde w_i^A=
\frac{w_i\mathbf1\{i\in A\}}
{\sum_k w_k\mathbf1\{k\in A\}},
\qquad
\mathbb E_A[v]=
\frac1{K_h}\sum_{b=1}^{K_h}\sum_i\widetilde w_i^A v_{b,i},
\quad K_h=4.
```


四维统计为：


```math
F_e(A)=
\begin{bmatrix}
\mathbb E_A[\Delta]\\[4pt]
\sqrt{\mathbb E_A[(\Delta-\mathbb E_A[\Delta])^2]}\\[4pt]
\sqrt{\mathbb E_A[|\mathcal C|^{-1}\sum_cD_c^2]}\\[4pt]
\mathbb E_A[\Delta H]
\end{bmatrix}.
```


设 $`S_j`$ 为全部支持图像，$`T_{j,e}`$ 为真值原语等于该关系目标原语的支持图像；$`S_{j,g}`$ 为按预测路由分组的支持图像。全局与条件响应为：


```math
h_{j,e}=
\begin{bmatrix}F_e(S_j)\\F_e(T_{j,e})\end{bmatrix},
\qquad
h_{j,e,g}=
\begin{bmatrix}
F_e(S_{j,g})\\[4pt]
F_e(T_{j,e}\cap S_{j,g})
\end{bmatrix}.
```


条件子集为空时，该四维块使用对应全局四维块；因此该块的原始条件差为零。对象路由可能使某个目标原语在某组没有支持图像，这条定义使所有 96 条关系都具有确定输入。记录空块比例，便于解释情境信息覆盖。

重新前向一次即可同时积累全局、两个条件响应及其二阶矩；不需要为每个组分别执行同一批开关前向。旧八维汇总不能恢复所有条件方差和 RMS，新代码需从逐图像统计重新聚合。

### 6.4 51 参数共享模型与零差分约定

全局输入沿用旧中心和尺度。条件差只按拟合教师的 RMS 缩放，不减去均值：


```math
\widetilde h_{j,e}=(h_{j,e}-c)\oslash s,
\qquad
\Delta h_{j,e,g}=h_{j,e,g}-h_{j,e},
```


```math
(s_{\Delta,k})^2=
\frac1{J|E|}
\sum_{j=1}^{J}\sum_e\sum_g
\omega_{j,g}(\Delta h_{j,e,g,k})^2,
\qquad
\Delta\widetilde h_{j,e,g,k}
=\frac{\Delta h_{j,e,g,k}}{\max(s_{\Delta,k},10^{-6})}.
```


这样，“没有条件信息”的原始零差分缩放后仍为零。尺度只从 Q1–Q3 得到，并在 Q4/Q5 和正式部署固定。

条件映射为：


```math
\boxed{
\widehat\mu_{j,e,g}
=a_{t(e)}
\left[
b_{t(e)}
+w_{t(e)}^\top\widetilde h_{j,e}
+v_{t(e)}^\top\Delta\widetilde h_{j,e,g}
\right].
}
```


每类型 $`1+8+8=17`$ 个系数，共 51 个。所有情境共享 $`b_t,w_t,v_t`$；这里不增加情境 ID 截距。一项对同类型所有边都相同的情境偏置，在固定类型预算下只给目标增加常数，不能改变关系的相对分配。

将上述标准化输入写为 $`x_{j,e,g}\in\mathbb R^{51}`$，拟合：


```math
\widehat\beta_C=
\arg\min_\beta
\left[
\frac1J\sum_j\sum_g\omega_{j,g}
\sum_e
\left(x_{j,e,g}^\top\beta-
\frac{\widehat u^{(1)}_{j,e,g}}{a_{t(e)}}\right)^2
+\lambda\|\beta\|_2^2
\right].
```


```math
\left(\sum_{j,g}\omega_{j,g}X_{j,g}^\top X_{j,g}
+30I\right)\widehat\beta_C
=\sum_{j,g}\omega_{j,g}X_{j,g}^\top
\left(\widehat u^{(1)}_{j,g}\oslash a\right).
```


同一教师的组权重之和为 1，教师数仍是 9。使用原拟合标签尺度和 $`\lambda=10/3`$，51 个系数均参与相同岭惩罚。

**全局模型是严格特例。** 固定 $`v_t=0`$，对任意标量预测 $`f`$：


```math
\sum_g\omega_g(f-y_g)^2
=\left(f-\sum_g\omega_gy_g\right)^2
+\sum_g\omega_g
\left(y_g-\sum_{g'}\omega_{g'}y_{g'}\right)^2.
```


最后一项与预测器参数无关；结合全局标签的加权分解，相同标准化、尺度和正则下，限制模型的最优解等于现有 27 参数 Matched。新增模型的变化明确来自条件响应项。

### 6.5 情境策略、概率归一化与实际计算量

对每个 $`g`$ 独立求类型预算，首批使用原单位方差分数、$`\kappa=1`$：


```math
\pi_{\cdot,g}=\pi^\star(r^{\mathrm{std}}(\widehat\mu_{\cdot,g})),
\qquad
B_{g,m,e}=\mathbf1\{U_{m,e}\lt \pi_{e,g}\}.
```


同一个 bank 的 $`U_{m,e}`$ 在所有方法、两个情境和对应置换之间共享，以形成配对比较。每个情境内部的 16 张图仍按独立均匀数生成。


```math
\bar p_C(y\mid x)
=\frac1{16}\sum_{m=1}^{16}
p_{\theta,\mathcal C}(y\mid x,B_{g_\theta(x),m}).
```


每张图均在完整 $`\mathcal C`$ 上归一化，因此：


```math
\sum_{y\in\mathcal C}\bar p_C(y\mid x)=1.
```


先将图像按路由分组，每张图像只计算所属组的 16 图预测；可预计算两套图原型。额外成本为第二套原型、路由和分组管理，不应将全部图像在两套图上各算一遍再挑结果。

对于条件风险：


```math
R_M^{\mathrm{ctx}}=\sum_g\omega_gR_{M,g}(\pi_g),
\qquad
-\frac{\partial R_M^{\mathrm{ctx}}}{\partial\pi_{e,g}}
=\omega_g u_{e,g}^{(M)}.
```


独立情境预算下，若局部目标和 KL 正则都按 $`\omega_g`$ 加权，非空情境中的该权重相消，得到上述逐情境预算求解。不要只将效用乘以组频率而保持 KL 不变，否则会额外压缩低频组动作。

## 7. 对照设计与实际选择规则

### 7.1 第一批主表：五条方法线

| 方法 | 监督与输入 | 分配规则 | 本批作用 |
| --- | --- | --- | --- |
| Uniform | 无效用拟合 | 全部 0.5 | 同权重、同图数基线 |
| Matched | 当前 $`u^{(1)}`$，全局八维，27 参数 | 原标准化，固定强度 1 | 复用 Stage 主方法 |
| A：Amplitude | 完全复用 Matched 预测 | 固定训练尺度，Q4 选择一个共享 $`\eta`$ | 检验幅度 |
| B：Mix16 | $`u^{(16)}`$，相同八维，27 参数 | 原标准化，固定强度 1 | 检验部署目标 |
| C：Context | $`u^{(1)}_{e,g}`$，全局＋条件差，51 参数 | 逐组原标准化，固定强度 1 | 检验情境信息 |

对 Matched、A、B、C，各配五次类型内概率置换，置换 seed 为 7101–7105。置换保持每类型概率多重集、期望边数及同一方法选择的幅度。C 在两个情境使用同一条边索引置换，保留各边跨情境概率向量的对应结构，只改变它们落在哪条关系上。

再加入一个 C-ShuffleContext，检验相同模型容量下的情境对应。固定 9 个拟合教师的顺序，用随机种子 20261026 无放回选 5 个教师交换两组条件差分输入；其余 4 个保持。交换对同一教师的全部边一致。标签、标签权重、全局输入、输入尺度、正则保持原值，重新拟合 51 参数模型；评价时使用真实路由。

这一设计打破“条件输入—条件标签”的对应，同时保留全局效用监督和模型容量。两个组人数可能不同，因此不把它描述为严格保持加权输入边际分布的实验。部分教师交换可以避免全体统一换名被共享模型吸收。

共 **26 个正式策略键**：Uniform 1 个，四条非均匀方法及其五次置换共 24 个，情境打乱 1 个。概率完全相同的键共享计算结果，但保留各自实验标签。

### 7.2 Q4/Q5 的测量策略与响应平均控制

Q4/Q5 另外评价三种测量效用策略：全局单图、全局 Mix16、条件单图。它们由 measure 生成，在独立 evaluate 图像评分，并配关系置换。主要比较：

- 测量 Mix16 对测量单图：读出目标改变后是否提高混合风险与 AUC。
- 测量条件对测量全局：情境差异本身是否具有决策价值。
- 预测对相应测量策略：预测环节保留了多少可用效用。

测量策略只是任务内信息参照，不是严格性能上界，也不用于生成正式 TEST 策略。

平均响应只增加一个轻量控制：用已保存的两次响应均值重新拟合 27 参数单图映射，训练和部署都用均值；固定原输入坐标、标签尺度和正则。与旧“单次输入拟合、平均输入部署”在 Q4/Q5 对照，并保存类型预算与五次关系置换结果。该控制不扩为新的教师训练路线。

### 7.3 参数在什么位置确定

| 内容 | 确定方式 |
| --- | --- |
| 教师与正式识别器 | 直接读取现有 Stage best 与 State Uniform-best |
| 对象分组 | 已保存文本锚点的固定 6／6 划分 |
| 主响应 | 第一次测量的支持 ID、权重和四个背景 |
| 标签目标 | Matched/A/C 用单图；B 用 16 图 |
| 正则 | 所有主比较固定 $`\lambda=10/3`$、$`\alpha=30`$ |
| B/C 分配强度 | 固定 1；幅度曲线作为 Q4/Q5 诊断 |
| A 的 $`\eta`$ | 三个 Q4×八 bank 平均 exact AUC 最大；并列取小 |
| 正式权重选择 | 沿用已有 Uniform-best，最佳 epoch 为 2／2／6 |
| 校准 | VAL 确定每个 world 的 bias／温度，原样用于 TEST |

Q4/Q5 的八个 bank 沿用 8201–8208，正式沿用 1701／2701／3701。主 bank1701 与历史并列；本轮主要归纳采用每 seed 的三 bank 平均，全部逐 seed 数据一并报告。

## 8. 一个完整队列完成第一批

### 8.1 数据与模型来源

输入根目录：

`output/stage_results/server_results_20261025_183634/trustcpf_stage_v1_10ep/work`

该归档已经核验完整：26 个阶段、15 个教师、972 行正式指标、744 行 Q5 决策、144 个测量背景块。当前分析复用这些既有验证，不重复启动训练。

实现接口对应如下：

| 新操作 | 现有接口／文件 | 本次增加 |
| --- | --- | --- |
| 混合效用 | `matched_train.measure_unll`；`measurements/fold*/block*.pt` | 辅助图真实类 log-probability 与 Mix16 标签 |
| 对象路由 | `model.base` 中的 `obj`；图库对象 anchors | 固定对象组及每图像路由 |
| 条件响应 | `response.response_statistics`、`state_compute.extract_response` | 分组权重聚合、条件二阶矩、空块定义 |
| 条件拟合 | `compact.fit_ridge`、`matched.fit_global` | 51 维加权设计、共享尺度、情境打乱 |
| 冻结评分 | `matched_compute`、`state_eval` | 同一模型按路由选择策略，完整候选归一化 |
| 汇总 | 现有 summary／paired／decision 表 | target、context、amplitude、成本字段 |

已抽查保存格式：单块 `on/off` 为 `[96,512,4]`，最后维第 0 项就是归一化后的真实类 log-probability；`sample_seen` 标记 256 seen／256 unseen。新的标签只聚合 unseen 图像，保留 seen 图像用于风险和 AUC 诊断。

### 8.2 队列安排

队列一次启动，依次完成三个计算段，中间共享依赖由程序处理：

1. **各 seed 测量。** 导入原输入，生成固定路由；对 15 个教师补充辅助图，重算 Mix16／条件标签；对 15 个教师和三份正式模型提取条件响应。全局响应同步复算，与旧输入对应。
2. **联合拟合与策略生成。** 汇总九教师，拟合 B、C、情境打乱和平均响应控制；生成 Q4 曲线、确定 A 幅度；一次性写出所有 Q5 和正式策略。
3. **各 seed 评分与报告。** 完成 Q5 实际决策、三 bank 正式 CW／OW、置换配对和资源记录，统一导出主表。

现用三卡按 seed3351／3352／3353 对应物理 GPU1／2／3；显卡编号只影响调度。每个 worker 处理自己的模型，27／51 维联合拟合使用 CPU float64 即可。

### 8.3 训练与计算预算

本批复用既有 10 epoch 训练轨迹，新实验的学习环节是闭式岭回归，没有额外 epoch。对当前识别器继续追加训练会同时改变效用作用对象，因此放到后续完整方法确认。

Mix16 辅助图总数：


```math
\sum_jK_j=3(4\times8+16)=144,
\qquad
144\times15=2160.
```


旧 on/off 对应 $`2\times96\times144=27648`$ 次全测量集图前向。新增辅助图在次数上为其 7.8125%；实际本批只新增这 2160 次，旧开关统计直接复用。每次前向仍需按图像 batch 计算，不能据此直接承诺相同比例的墙钟时间。

条件响应每个模型使用四个背景，开关次数为 $`2\times96\times4=768`$；18 个模型共 13824 次全支持集图前向。两组响应在同一次前向中聚合，图像支持规模与测量集不同。

正式表的完整规模为：


```math
26\text{ 策略}\times3\text{ seed}\times3\text{ bank}
\times2\text{ split}\times2\text{ world}
=936\text{ 指标行}.
```


其中 Uniform、Matched 及其置换可直接复用验证一致的既有预测；A 选中零强度时再复用 Uniform。Q4/Q5 曲线和测量参照单独输出，不把它们变成额外训练方法。

### 8.4 保存内容

| 文件类别 | 必须保存的内容 |
| --- | --- |
| 来源与路由 | checkpoint、候选顺序、支持／measure／evaluate ID、对象分组、各分区路由人数 |
| 效用 | 原始 on/off 引用、辅助图掩码及逐图像 log-probability、单图／Mix16／条件块均值与 SE |
| 响应 | 全局及条件八维输入、支持权重、背景掩码、空块计数、差分 RMS 尺度 |
| 预测器 | 27／51 维参数、标准化、标签尺度、训练教师列表、固定正则、情境交换表 |
| 策略 | 原始效用、分数、预算解、动作幅度、置换索引、每个 bank 的实际图掩码 |
| 决策 | Q4/Q5 每点单图与混合 NLL、exact AUC、线性代理、预测与测量策略的配对差 |
| 正式结果 | VAL/TEST 逐图像概率、路由、CW/OW native／exact AUC、HM、校准 NLL／ECE |
| 成本 | 测量前向数、路由／原型／评分时间、每图像实际图数、CPU 拟合与 GPU 内存口径 |

本次是方案交付，尚未修改发布包和服务器脚本。后续实现按本节接口组织成一个新目录，保持旧实验可复查。

## 9. 怎样从结果决定主线

主表只回答三个问题：效用是否改善实际风险、正确关系／情境对应是否有价值、这种价值是否进入正式未见组合识别。

配对汇总方式为：


```math
\overline\Delta_s
=\frac13\sum_{b\in\{1701,2701,3701\}}
\left(A_{s,b}^{\mathrm{method}}-A_{s,b}^{\mathrm{control}}\right),
\qquad
\overline\Delta=\frac13\sum_s\overline\Delta_s.
```


置换对照先在同 seed、同 bank 内平均五次，再形成上述配对量。三个 seed 共享一个效用预测器；图 bank、置换和情境都不增加独立训练重复数。

| 结果模式 | 研究解释 | 下一步安排 |
| --- | --- | --- |
| A 小步比大步好，真实分配仍超过置换 | 原方向可用，动作需要校准 | 固定该读出，配合 B/C 主机制 |
| A 只在接近 Uniform 时改善 | 动作收缩修复原策略的下降 | A 作为读出设置，贡献判断仍看 B/C |
| B 改善混合 NLL，并提高正式 AUC 与关系对应收益 | 部署目标匹配进入最终识别 | 优先锁定简洁的 Mix16 主方法 |
| B 改善 NLL，AUC 接近原值 | 主要改变概率质量 | 按概率质量报告，不把它计作 AUC 收益 |
| 条件测量好于全局测量，C 也超过全局／情境打乱／关系置换 | 条件信息可测、可学、可迁移 | 以条件效用为主创新，再合并 Mix16 |
| 条件测量有效，条件预测弱 | 条件任务价值存在，当前响应不足 | 下一轮只改条件响应表示与任务覆盖 |
| B/C 仍只在 Q5 改善，正式接近置换 | 当前迁移信息仍未连接到正式新组合 | 重构组合任务覆盖或表示，结束同配方小范围调参 |

不再设置多个数值门槛决定是否运行后续小任务。第一批全部执行，按同预算配对增量、各 seed 方向、native／exact 一致性和计算成本整体决定投入。

## 10. 第二批：选定机制后的完整方法确认

### 10.1 把有增量的机制组成一条主方法

若 B、C 都有作用，使用同一条件分组，将条件标签改为：


```math
\widehat u^{(16)}_{j,e,g}
=\frac1{K_j|U_{j,g}|}
\sum_b\sum_{i\in U_{j,g}}
d^{(16)}_{j,e,b,i}.
```


对应的 **Context-Mix16** 仍使用 51 参数模型和逐组预算，不额外扩网络。辅助图与逐图像统计已经在第一批保存，合并标签可直接重算。固定训练尺度是否进入最终方法，由第一批动作曲线决定。

后续完整比较采用 Uniform、B、C、Context-Mix16 与必要的关系／情境置换，使用相同训练轨迹。若只有一条机制产生正式收益，就以该机制为主方法，其余作为已完成的机制比较。

### 10.2 新的完整流水线与基线

锁定规则后，用新的 TRAIN 组合留出划分与三个尚未使用的完整流水线 seed，从教师、效用拟合到正式识别重新执行。默认候选 seed 为 4351／4352／4353，实现时先与历史 run manifest 核对。

每个完整重复以该 seed 训练正式识别器，并以该 seed、该 seed+100、该 seed+200 形成三个教师初始化，各执行 Q1–Q5。于是每个重复包含 15 个教师，其中 9 个用于联合拟合；三个完整重复共 45 个教师、三份独立效用预测器和三份正式识别器。各重复内部共享对照底座，重复之间独立拟合，不将一个共享预测器下的三个识别器误记为三个完整方法重复。

教师的 Query 组合保留给效用测量和评价。保留原流水线的 CPF 预训练与初始教师训练协议；教师阶段匹配部分固定追加 10 epoch，并使用预定最后一轮，不再用 Query STOP 选教师。正式 Uniform 适配仍为 10 epoch，以 VAL CW 选择 epoch0–10。所有实验组共用这套权重。

统一重跑 CPF、Capacity、Dense、Uniform 和选定 TRUST。重点记录训练参数量、关系附加结构、期望边数及推理图数。Capacity 为无关系附件的容量对照，Dense 为全关系策略，二者预算含义与 Uniform 不同，在表中明确列出；它们不能用来替代同结构、同期望预算的 Uniform／置换比较。

第二数据集确定为 **MIT-States**，复用公开 CZSL 划分和 CW／OW 协议；该数据集的训练配置及开放世界评价入口可在 [CGE／CompCos 官方仓库](https://github.com/ExplainableML/czsl) 查到。仍使用 CLIP ViT-L/14、相同效用公式和每图像 16 图，按其原语重新生成关系库，类型预算保持各类型关系数的 50%。

对象数增多后，保持两个情境及平衡二分目标，用确定性的平衡二均值替代 UT-Zappos 的 462 种穷举：最远文本向量对初始化；按两个中心的距离差分配为人数相差最多 1 的两组；交替更新中心与分组，记录最终分组。其余方法和选择规则沿用锁定版本。MIT-States 的缓存和完整训练进入第二批交付，第一批直接使用已有 UT-Zappos 数据。

### 10.3 论文贡献的收敛方式

保留的研究对象是**组合识别中的可迁移关系效用**：

1. 在 TRAIN 组合留出任务中，用配对干预产生关系价值监督。
2. 将监督定义到实际概率混合读出，并研究效用随可观测情境的变化。
3. 用小型共享预测器和固定预算策略，将这种价值迁移到未见组合识别。

单独使用 Bernoulli、岭回归、KL 或条件路由不是本文拟建立的核心贡献。主张应落在效用的定义、学习和迁移，以及正确关系对应的终端收益。现有 Q5 结果已经支持继续保留效用监督；正式提升由下一批实验回答。

## 11. 证据范围、数学核验与来源

### 11.1 现有证据的使用口径

State／Stage 正式模型完全相同；Stage 新增教师阶段匹配和响应读出。Stage Q5 的 measure、STOP、evaluate 图像互不重叠，但 STOP 使用同一 Query 组合的其他图像标签，所以本批复用 Q5 属于同组合的独立图像决策评价。新的组合迁移确认放在第二批完整流水线。

本方案没有从正式 TEST 标签构造效用、路由或候选情境，也没有将历史 TEST 反复观察视为新的独立验证。历史结果用于选择研究问题；锁定方案后的新划分和第二数据集用于完整确认。

### 11.2 本次已经完成的公式核验

使用小型可枚举概率模型，独立检验了 $`M=1,2,16`$ 的混合风险梯度；对四种可能图的计数分布做精确求和，与中心差分比较。另检验条件岭回归的全局特例、加权方差恒等式、预算约束及标准化 Jacobian。

| 核验项 | 最大绝对误差 |
| --- | --- |
| 有限混合干预公式与风险梯度 | $`1.58\times10^{-11}`$ |
| 条件回归关闭情境项与全局岭解 | $`8.33\times10^{-17}`$ |
| 加权平方损失分解 | $`1.42\times10^{-14}`$ |
| 三类预算及四点插值 | $`3.55\times10^{-15}`$ |
| 类型标准化 Jacobian | $`2.06\times10^{-11}`$ |

同时检验了 log-probability 约 −1000 时的 logsumexp／logaddexp 数值路径。以上结果验证公式和数值约定，不是新的识别实验成绩。

文档的 54 个独立公式块与 80 处行内公式已通过 KaTeX 解析；浏览器预览检查了关键推导、条件模型和实验表格，未发现公式或表格横向溢出。来源文件链接已逐项核对。

### 11.3 主要来源

- [第七版机制分析 PDF](references/source_diagnosis_v7.pdf)：重点参考物理页 9–16 的 A/B/C 方案与页 20 的研究问题。
- [Stage 第二批完整分析](../09/analysis/README.md)：正式识别、Q5、教师曲线、重复响应与复算入口。
- [State 第一批完整分析](../08/analysis/README.md)：联合拟合、冻结部署与原有对照。
- [此前 v6 公式与方案](../08/PLAN.md)：截至 R5 的历史证据及前两批安排。
- [弱基底完整分析](../v1/analysis/server-results-20261022/README.md)。
- [静态知识 v2 完整分析](../05/references/history/conditional_v2_analysis.md.txt)。
- [强基底完整 TRUST 分析](../02/analysis/README.md)。
- [模型响应条件化分析](../03/analysis/README.md)。
- [Compact 完整分析](../04/analysis/README.md)。
- [Align 完整分析](../05/analysis/README.md)。
- [Competition 10 epoch 完整分析](../06/analysis/README.md)。
- [R5 10 epoch 完整分析](../07/analysis/README.md)。

本版将后续工作收敛为**一批机制实验、一批完整方法确认**。第一批优先判断实际混合目标与对象情境信息能否把 Q5 的效用价值迁移到正式 CZSL；第二批围绕有效机制形成最终论文方法。
