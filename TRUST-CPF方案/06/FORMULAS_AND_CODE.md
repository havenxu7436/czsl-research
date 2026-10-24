# TRUST-CPF Competition v1（10 epoch）：公式、实现与核心指标

本实现承接 Align 结果分析。问题集中为：保持强 CPF、响应输入、效用预测器和部署方式一致时，关系的组合竞争收益是否比 NLL 收益更适合作为可迁移监督。

## 1. 统一实验与图像协议

五组为 Uniform-0.5、U-NLL、Balanced、Competition、Permuted-Competition，训练 seed 为 3351、3352、3353。每个 seed 共用一个已有全 TRAIN reference，以及五个新建的双组教师。候选集为全部 TRAIN 组合。

第 j 折 Inner 组合记为 S，留出组合为 U。每个 Inner 组合按固定随机数打乱图像，留出 `ceil(0.2*n)` 张，至少留一张且至少保留一张训练。只有一张图的 Inner 组合全用于训练。G0 10 epoch 和冻结 G0 后的关系头 10 epoch 都只读取剩余训练行；伪未见组合图像全部留作风险池。

每组风险池无放回抽最多 256 张，各组内部图像等权。第五折先把每组池一分为二，再分别抽最多 256 张，形成互不重叠的测量和评价图像。三种监督共用这些图像。与旧 Compact 的组合等权协议区别在清单中明确记录。

```python
initial = strong.fold_backbone(..., train_ids=split['train_ids'])
engine.train_model(..., initial=initial, train_ids=split['train_ids'])
```

实现位置：`competition.split_risk`、`competition_training.teacher`。训练行 ID 同时写入两个阶段的 checkpoint。响应探针只从这些训练图像中按组合选最多两张，使用 4 个背景；最终 reference 继续复用已验证身份的原响应。

## 2. 组合竞争面积

设 z 为限制到 TRAIN 候选集后的对数概率，统一给 U 候选加偏置 beta。对真实类属于 S 的图像 i 和属于 U 的图像 k，定义阈值及组内正确标记：


```math
\begin{aligned}
a_i&=z_{i,y_i}-\max_{c\in U}z_{ic}, &
b_k&=\max_{c\in S}z_{kc}-z_{k,y_k},\\[4pt]
g_i^S&=\mathbf1\{\arg\max_{c\in S}z_{ic}=y_i\}, &
g_k^U&=\mathbf1\{\arg\max_{c\in U}z_{kc}=y_k\}.
\end{aligned}
```


组内判别正确时，已见样本在 beta 小于 a 的一侧正确，未见样本在 beta 大于 b 的一侧正确。因此，二者存在共同正确区间当且仅当 a 大于 b。将准确率曲线按全部临界偏置积分，可把每次未见准确率的上跳贡献展开为：


```math
\begin{aligned}
A_{\rm exact}(z)&=\frac{1}{n_Sn_U}\sum_{i\in D_S}\sum_{k\in D_U}
g_i^Sg_k^U H(a_i-b_k),\\[4pt]
H(v)&=\mathbf1\{v\gt 0\}+\tfrac12\mathbf1\{v=0\}.
\end{aligned}
```


相同阈值时取半个梯形；组内并列采用原 `exact_curve` 的第一个最大值规则。公式的范围为 0–1，报告 AUC 乘以 100。测试已覆盖普通分数、并列阈值、组内并列和单候选分组，与原事件积分一致。

连续标签使用组内真实类 margin：


```math
\begin{aligned}
m_i^S&=z_{i,y_i}-\max_{c\in S\setminus\{y_i\}}z_{ic},\\[4pt]
m_k^U&=z_{k,y_k}-\max_{c\in U\setminus\{y_k\}}z_{kc}.
\end{aligned}
```


组内只有一个候选时 margin 取正无穷，sigmoid 门为 1。第一版温度固定 T=1 nat：


```math
\begin{aligned}
\widetilde A(z)&=\frac1{n_Sn_U}\sum_{i,k}
\sigma(m_i^S/T)\sigma(m_k^U/T)\sigma((a_i-b_k)/T),\\[4pt]
R_{\rm cmp}(z)&=1-\widetilde A(z).
\end{aligned}
```


向 U 全体候选加常数 d 时，a 和 b 同时减去 d，组内 margin 不变，因此竞争风险对统一未见偏置严格不变。它对整体分数缩放仍敏感，所以审计同时输出平滑面积、真实 exact/native AUC、组内门值、margin 绝对尺度和阈值标准差。

核心实现对全部图像对分块求和，没有抽样近似：

```python
for start in range(0, len(s), chunk):
    d = s[start:start+chunk, 2, None] - u[None, :, 2]
    cross = torch.sigmoid(d / temperature)
    total += (gs[start:start+chunk, None] * gu[None, :] * cross).sum()
return total / (len(s) * len(u))
```

实现位置：`competition.sufficient`、`area`、`risks`。

## 3. 三种效用由同一前向产生

同一批图像、同一教师、同一采样图上的三种风险为：


```math
\begin{aligned}
R_U&=-\frac1{n_U}\sum_{k\in D_U}\log P(y_k\mid x_k,b),\\[4pt]
R_{\rm bal}&=-\frac1{2n_S}\sum_{i\in D_S}\log P(y_i\mid x_i,b)
-\frac1{2n_U}\sum_{k\in D_U}\log P(y_k\mid x_k,b),\\[4pt]
R_{\rm cmp}&=1-\widetilde A.
\end{aligned}
```


背景图每条边独立以 0.5 采样。对边 e 固定其余边，并分别强制关闭、开启：


```math
\begin{aligned}
d_{je\ell}^{q}&=R_q(\theta_j,b_{-e}^{\ell},0)-R_q(\theta_j,b_{-e}^{\ell},1),\\[4pt]
\bar y_{je}^{q}&=\frac1K\sum_{\ell=1}^K d_{je\ell}^{q},\\[4pt]
\widehat v_{je}^{q}&=\frac1{K(K-1)}\sum_{\ell=1}^K(d_{je\ell}^{q}-\bar y_{je}^{q})^2.
\end{aligned}
```


正效用代表开边降低风险，Competition 的正效用等于提高平滑面积。Q1–Q4 用 8 个背景，Q5 用 16 个背景；每个背景测全部 96 条关系，三个目标共享同一次开关分数。

该配对量也是 Bernoulli 期望风险对采边概率的负导数：


```math
-\frac{\partial}{\partial\pi_e}\mathbb E_{b\sim\prod_f\mathrm{Bern}(\pi_f)}R_q(b)
=\mathbb E_{b_{-e}}\big[R_q(b_{-e},0)-R_q(b_{-e},1)\big].
```


因而标签直接对应局部概率更新的收益。背景方差单独保存，不加入均值奖励。`measurements/fold*/block*.pt` 保存每张图像的真实类对数概率、组内 margin、跨组阈值和硬判别门，这四项足以精确复算三个风险。无需在每个开关上重复存储全部候选分数。

```python
diff = torch.stack([
    risks(off, sample_seen, temperature) - risks(on, sample_seen, temperature)
    for off, on in zip(off_statistics, on_statistics)
])
target = blocks.mean(0)
se2 = blocks.var(0, ddof=1) / len(blocks)
```

实现位置：`competition_training.measure`。未添加噪声实验或新的独立训练支线。

## 4. 八维响应、27 参数拟合与 Q4 选择

输入沿用原响应：开边后的真实类对数概率变化均值和标准差、去公共分数偏移后的 RMS、预测熵变化；分别在全部支持图像与关系目标对应图像上计算，共八维。

每个类型拥有一个截距和八个系数，共 27 参数。输入均值和标准差仅由当前拟合折估计；每类型标签尺度为拟合标签标准差，数值下限分别为输入 1e-6、标签 1e-12。目标 q 的岭回归为：


```math
\widehat\beta^{q}=(Z^\top Z+\alpha I)^{-1}Z^\top\widetilde y^{q},
\qquad
\widetilde y_e^{q}=\bar y_e^{q}/a_{t(e)}^{q}.
```


实现使用 `numpy.linalg.solve`，没有显式求逆；截距也参与正则。Q1–Q3 拟合，Q4 从 alpha={0.1,1,10,100} 选择。选择标准为类型内中心化、尺度标准化误差：


```math
\mathcal E_4^{q}(\alpha)=\frac13\sum_t\frac1{N_t}
\left\|P_t\frac{\widehat\mu_{4,t,\alpha}^{q}-\bar y_{4,t}^{q}}{a_t^{q}}\right\|_2^2,
\qquad P_t=I-\frac{\mathbf1\mathbf1^\top}{N_t}.
```


并列取更大 alpha。训练仍为绝对岭回归，中心化只用于 Q4 选择。三目标独立选 alpha，共用相同候选和选择规则。审计模型只拟合 Q1–Q3；最终部署按选定 alpha 合并 Q1–Q5 重拟合，并重新计算输入和标签尺度。

```python
model = compact.fit_ridge(xs[:3], ys[:3], types, alpha)
pred4 = compact.predict(model, xs[3], types)
error = centered_type_error(pred4, ys[3], types, model['scales'])
```

实现位置：`competition_training.fit`。Q5 排序及最终 reference 排序同时保存，用于识别情境迁移变化。

## 5. 概率分配与正式识别

先对预测效用按关系类型标准化为 r。每类型保持半密度，求解：


```math
\begin{aligned}
\max_{\pi}\quad&\kappa\sum_e r_e\pi_e-
\sum_e D_{\rm KL}\big(\mathrm{Bern}(\pi_e)\Vert\mathrm{Bern}(1/2)\big),\\[4pt]
\text{s.t.}\quad&\sum_{e\in E_t}\pi_e=N_t/2.
\end{aligned}
```


对概率求导并引入类型乘子，得到唯一解：


```math
\pi_e=\sigma(\kappa r_e-\gamma_{t(e)}),
\qquad
\sum_{e\in E_t}\sigma(\kappa r_e-\gamma_t)=N_t/2.
```


float64 二分求乘子，沿用预算误差检查。常数效用或零强度直接得到 0.5。96 条边按 OI/OE/AE 的期望数为 6/18/24，共 48 条。

三个目标分别在冻结 reference 上按 VAL CW native AUC 从 kappa={0,0.5,1,2} 选择，平分时取较小值。Competition 另保存五个预固定类型内置换种子的相同扫描；正式 Permuted 使用第一个种子 7101 置换 Competition 已选概率，保证概率集合、预算和熵完全一致。置换各自的最优扫描结果也保留。

五组从相同 reference 开始适配 10 epoch，CLIP 和 CPF 主干冻结，仅更新原关系分支；按 VAL CW native AUC 从 epoch0–10 选权重。选中零强度的组复用本批 Uniform 权重与预测，并标记复用来源。每组同时报告 epoch0 和最佳权重。

五组都采用首个 epoch 预热、按完整 10 epoch 计算的余弦学习率衰减。训练从 reference 重新开始，完整运行十轮；`last.pt` 保存 epoch10，`best.pt` 保存 epoch0–10 的 VAL 最佳权重。`training_curves.csv` 汇总全部逐轮损失、当前 VAL AUC 和学习率，用于比较适配过程。

正式评价使用三个独立 16 图 bank：1701 为主结果及选模 bank，2701/3701 用于检查图随机性。CW/OW 使用同一所选权重；TEST 的偏置和温度来自对应 VAL。分类器与评价器实现沿用上一轮。

## 6. Q5 实际决策与核心指标

在第五折测量图像上，分别根据审计模型预测效用和直接实测效用构造 kappa=1 的策略；各自附类型内概率置换，与 Uniform 同时评价。三个目标共用测量和独立评价图像。共 13 种诊断策略，每个使用 8 个独立 bank、每 bank 16 图；这些策略只评估已训练教师，不新增识别训练。

单图风险先逐图计算再平均；混合风险先平均类别概率再计算：


```math
\overline P(c\mid x)=\frac1{16}\sum_{m=1}^{16}P(c\mid x,b_m),
\qquad z_c^{\rm mix}=\log\overline P(c\mid x).
```


所有策略共享相同底层均匀随机数，以形成配对图采样。每个 bank 保存单图充分统计量、混合全候选对数概率及掩码，能够复算风险和真实 AUC。实测策略是诊断参照，不参与正式 TEST 的选模。

| 核心问题 | 直接读取的指标 |
| --- | --- |
| Competition 是否提高最终识别 | CW/OW native 与 exact AUC、VAL 偏置下 HM；相对 Uniform/Balanced/Permuted 的逐 seed 差 |
| 收益是否存在于冻结状态 | epoch0 与最佳权重结果、适配增量、所选 epoch |
| 学到的效用是否真正用于组合竞争 | Q5 实际平滑面积与 exact/native AUC 收益；预测／实测／置换对比 |
| 改善是否只是分数尺度变化 | 平滑面积与真实 AUC 并报、组内门值和 margin 尺度 |
| 预测和迁移哪一步限制收益 | 分类型 Spearman、RMSE、Q5 到 reference 的排序相关；实测与预测决策差距 |
| 部署是否形成有效分配 | 所选强度、概率标准差、类型预算、覆盖和原型位移 |
| 成本 | 教师／测量／扫描／适配耗时、推理时延、峰值张量显存 |

主要结论按三个训练 seed 的配对结果判断。图 bank 的标准误只描述固定教师、固定图像下的图采样变化。代码保留所有原始行，汇总表对相同 bank 跨 seed 计算均值和样本标准差。

## 7. 本地验证范围

交付验证覆盖：竞争硬面积与原 exact AUC 一致、统一未见偏置不变、分块求和一致、Bernoulli 负风险导数等于开关效用、图像划分互斥、训练与响应行限制、三 seed 五组完整队列、所有测量目标复算、冻结 CPF 权重、VAL 校准继承、断点复用和输出身份。

真实数据的全量教师训练和识别结果由服务器命令产生。验证日志和实际归档接口检查保存在 `validation/`。
