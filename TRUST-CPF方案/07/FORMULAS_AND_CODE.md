# TRUST-CPF R5 v1：实验方案、公式与核心实现

## 1. 本轮研究问题与固定设置

本轮检验：**在固定关系类型预算下，将效用预测从绝对值拟合改为教师内、类型内相对差异拟合，能否改善关系分配及最终 CZSL 识别。**

基底沿用本轮 Competition 10ep 的强 CPF；CLIP 图像／文本缓存、96 条关系、reference 权重、15 个教师、风险划分、逐背景效用及八维模型响应全部复用。识别适配仍为 10 epoch，三 seed 为 3351／3352／3353。

| 正式组 | 代码名 | 效用与分配 |
| --- | --- | --- |
| Uniform-0.5 | `uniform` | 每条关系概率 0.5 |
| Absolute-U-NLL | `absolute` | 原绝对岭回归，复现上一轮 U-NLL |
| Relative-U-NLL | `relative` | R5 相对岭回归 |
| Permuted-Absolute | `permuted_absolute` | 类型内置换 Absolute 已选中的概率 |
| Permuted-Relative | `permuted_relative` | 类型内置换 Relative 已选中的概率 |

正式实验共 15 个结果。已有 Uniform、U-NLL 在同一 reference、概率、随机种子、10 epoch 学习率与评价协议下复用。其余组从相同 reference 独立开始适配；如果选中的概率等于 Uniform，则保存匹配结果复用记录。

## 2. 效用监督：沿用 U-NLL 测量

对第 j 折教师 θ_j，风险图像中的伪未见集合为 Q_j，候选类包括该任务的全部 TRAIN 组合。定义：


```math
R_j(b)=-\frac{1}{|Q_j|}\sum_{i\in Q_j}
\log P_{\theta_j}(y_i\mid x_i,b).
```


背景图的各边独立以 0.5 概率抽样。对于同一背景的其余边，分别令边 e 关闭和开启：


```math
d_{jke}=R_j(b_{k,-e},0)-R_j(b_{k,-e},1),
\qquad
\widehat u_{je}=\frac{1}{K_j}\sum_{k=1}^{K_j}d_{jke}.
```


正效用表示该关系降低风险。因为 Bernoulli 期望对单个概率是仿射函数，上述总体效用满足：


```math
u_{je}=-\left.\frac{\partial}{\partial\pi_e}
\mathbb E_{b\sim\mathop{\mathrm{Bernoulli}}\nolimits(\pi)}R_j(b)
\right|_{\pi=\frac12\mathbf 1}.
```


本轮直接读取 `measurements/foldj.pt` 中 `targets` 对应 `unll` 的列。Q1–Q4 每折 K=8，Q5 的 K=16。背景块、效用均值及标准误保持上一轮定义，标准误不加入本轮回归权重或风险扣分。

## 3. 输入标准化与绝对对照

每条边的输入 x_je 为八维训练图像响应：全支持／目标支持的效用均值、效用标准差、中心化评分 RMS、熵变化。输入只使用对应教师合法的训练支持图像，沿用每组合最多2张、4个响应背景。

审计拟合使用 Q1–Q3；其输入均值 c、标准差 s，以及类型效用尺度 a_t，都仅由这三折估计：


```math
z_{je}=\left[1;\frac{x_{je}-c}{s}\right],
\qquad
a_t=\max\left\{\mathop{\mathrm{std}}\nolimits_{j\in\{1,2,3\},e\in E_t}
(\widehat u_{je}),10^{-12}\right\}.
```


输入 s 的各维下限为 10⁻⁶。将 z 放入该边类型对应的9列，其余18列为零，得到 27 维设计行。记同一教师、同一类型的设计矩阵为 Z_jt，标签向量为 y_jt。

Absolute 沿用原目标，包含三个带正则的截距：


```math
L_{\mathrm{abs}}(\beta)=
\sum_{j,t}\left\|Z_{jt}\beta-y_{jt}/a_t\right\|_2^2
+\alpha\|\beta\|_2^2.
```


Absolute 与 Relative 使用相同的 c、s、a_t 和求和损失。每类边数分别为 OI 12、OE 36、AE 48，保留原每边权重，不另做等类型或等教师损失平均。

## 4. R5 相对目标与闭式解

在类型 t 的 n_t 条关系上定义中心化投影：


```math
P_t=I_{n_t}-\frac{1}{n_t}\mathbf 1\mathbf 1^\top,
\qquad P_t^\top=P_t,\quad P_t^2=P_t,\quad P_t\mathbf 1=0.
```


R5 对每个教师、每个类型独立投影设计矩阵和标签：


```math
L_{\mathrm{rel}}(\beta)=
\sum_{j,t}\left\|P_t\left(Z_{jt}\beta-y_{jt}/a_t\right)\right\|_2^2
+\alpha\|\beta\|_2^2.
```


利用 P_t 的对称性和幂等性，梯度为：


```math
\nabla_\beta L_{\mathrm{rel}}
=2\left(\sum_{j,t}Z_{jt}^\top P_tZ_{jt}+\alpha I\right)\beta
-2\sum_{j,t}Z_{jt}^\top P_t(y_{jt}/a_t).
```


α>0 时系数矩阵正定，解唯一。实现求解线性方程：


```math
\left(\sum_{j,t}Z_{jt}^\top P_tZ_{jt}+\alpha I\right)\widehat\beta
=\sum_{j,t}Z_{jt}^\top P_t(y_{jt}/a_t).
```


等价地，对于残差向量 r，有：


```math
\|P_tr\|_2^2=\frac{1}{n_t}\sum_{e\lt f}(r_e-r_f)^2.
```


因此该目标拟合同一教师、同一类型内的关系差异。共同截距在投影后为零，岭正则使其最优系数为零：文件仍保存27个参数，其中24个斜率系数有效，三个截距为零。共同偏移不影响投影残差的性质是在共用固定标准化尺度下成立的。

固定预算解释了该修改与决策的关系：


```math
u_t^\top\pi_t=(P_tu_t)^\top\pi_t+\bar u_t B_t,
\qquad \mathbf 1^\top\pi_t=B_t.
```


第二项与分配方式无关。R5 将拟合误差集中到第一项；部署时仅减类型均值本来就不改变等式预算策略，本轮修改发生在回归训练阶段。

核心实现直接来自 `trustcpf/r5.py`：

```python
def fit_relative(xs, ys, types, alpha, normalization):
    """Min sum_jt ||P_t(Z_jt beta-y_jt/a_t)||^2 + alpha ||beta||^2.

    Center each teacher/type independently, including the design rows. Keep the
    original unaveraged squared-error sum; do not change type or teacher weights.
    The three intercept columns vanish, and their ridge optimum is exactly zero.
    """
    if alpha <= 0:
        raise ValueError('Ridge alpha must be positive')
    types = np.asarray(types, dtype=np.int64)
    center, spread, scales = [np.asarray(normalization[k], dtype=np.float64)
                              for k in ('center', 'spread', 'scales')]
    zz, yy = [], []
    for x, y in zip(xs, ys):
        z = compact.design(np.asarray(x, dtype=np.float64), types, center, spread)
        q = np.asarray(y, dtype=np.float64) / scales[types]
        for t in range(3):
            keep = types == t
            zz.append(z[keep] - z[keep].mean(0))
            yy.append(q[keep] - q[keep].mean())
    z, y = np.concatenate(zz), np.concatenate(yy)
    beta = np.linalg.solve(z.T @ z + float(alpha)*np.eye(z.shape[1]), z.T @ y)
    return dict(beta=beta, center=center, spread=spread, scales=scales, alpha=float(alpha),
                dimension=int(z.shape[1]), fit_folds=len(xs), intercepts_penalized=True,
                objective='within_teacher_type_centered_ridge', effective_coefficients=24)

```

## 5. 正则选择、审计与最终重拟合

两种拟合分别扫描 α∈{0.1,1,10,100}，都用 Q4 的类型内中心化标准化误差选择；并列选择较大 α：


```math
E_4(\alpha)=\frac13\sum_{t=0}^{2}\frac1{n_t}
\left\|P_t\frac{\widehat u_{4,t}(\alpha)-y_{4,t}}{a_t}\right\|_2^2.
```


审计预测器只拟合 Q1–Q3，Q5 用于独立决策评价。最终部署分别使用各自已选 α，在 Q1–Q5 上重拟合；此时两方法共同重算五折 c、s、a_t，再对全 TRAIN reference 响应预测效用。

`utility/absolute_reproduction.json` 核对 Absolute 的 α、Q5预测、reference预测与上一轮 U-NLL 一致。真实数据上的三 seed 和全部四个强度的 Absolute 概率均已在本机复算通过。

## 6. 概率分配与公平置换

每个类型内将预测效用中心化并除以该类预测标准差。标准差不超过10⁻¹²时令标准化响应为零：


```math
r_e=\frac{\widehat u_e-\overline{\widehat u}_t}
{\mathop{\mathrm{std}}\nolimits_{f\in E_t}(\widehat u_f)},
\qquad
\pi_e=\sigma(\kappa r_e-\gamma_t),
\qquad
\sum_{e\in E_t}\pi_e=|E_t|/2.
```


γ_t 用二分求解。类型期望边数为6／18／24，合计48。κ=0 精确恢复 Uniform。每种方法分别在同一冻结 reference 上扫描 κ∈{0,0.5,1,2}，以 VAL CW native AUC 选择，并列取较小 κ。

两个置换组分别使用各自真实组已选中的 κ 和概率集合，只在同类型内重新对应关系。正式置换种子固定7101；7101–7105 的冻结扫描作为诊断保存，置换组不独立挑选 κ。

两种方法、4个强度、真实对应及5个置换、3个seed，共144条扫描记录。κ=0 的相同结果自动复用；Absolute 的真实对应扫描复用上一轮。

## 7. Q5 实际决策审计

固定 Q5 教师；测量图像与评价图像独立，每侧各256张伪已见、256张伪未见。预测器使用合法训练响应，监督来自测量侧 U-NLL。评价侧保存七种策略：Uniform、Absolute、Relative、两个匹配置换、Measured、Permuted-Measured。最后两组使用测量效用，作为有标签诊断参照。

全部策略在固定 κ=1 下评价，共用每个 bank 的底层均匀随机数。图 seed 为8201–8208，每个bank采样16图。保存单图平均指标和概率混合指标：


```math
\bar P(y\mid x)=\frac1{16}\sum_{m=1}^{16}P(y\mid x,b^{(m)}),
\qquad
\log\bar P=\mathop{\mathrm{logsumexp}}\nolimits_m(\log P_m)-\log16.
```


同时记录 U-NLL、Balanced-NLL、平滑 Competition 风险和 exact／native AUC；回归标签仅用 U-NLL。收益定义为 Uniform 风险减当前风险，AUC收益定义为当前AUC减Uniform AUC。每个bank保留逐图充分统计和混合logprob，共168行策略评价，能够重新计算指标。

五个已存在的诊断策略按相同掩码复用上一轮结果，Relative及其置换按当前概率重新评价。相同掩码直接复用同一输出。MCSE描述固定教师、图像条件下的图采样变化，跨seed差异单独汇总。

## 8. 最终识别适配与核心指标

各组共享 reference、batch64、AdamW、graph_lr=3×10⁻⁴、weight_decay=10⁻⁴、梯度裁剪1.0及10epoch学习率计划。CPF backbone冻结，适配关系及组合残差参数。第一个epoch warmup，随后余弦衰减；逐轮保存loss、学习率和VAL曲线。

使用epoch0–10中VAL CW native AUC最高的权重，平分保留较早权重。保存 epoch0／最佳权重在1701、2701、3701三图bank上的VAL／TEST、CW／OW预测，每次16图概率混合。偏置、温度仅由VAL选择并传递到TEST。

| 要回答的问题 | 指标与比较 |
| --- | --- |
| 相对拟合是否改善效用预测 | Q5类型内Spearman、中心化RMSE／R²，Relative对Absolute |
| 预测是否改善真实决策 | Q5实际U-NLL和AUC收益，真实对应对匹配置换 |
| 是否改善最终CZSL | 主bank1701的TEST CW／OW AUC、VAL工作点HM；Relative对Uniform、Absolute及Permuted-Relative |
| 是否依赖图采样 | 2701、3701补充bank配对差值 |
| 适配阶段产生何种变化 | epoch0到最佳权重变化、epoch0–10完整曲线 |
| 审计到reference的排序如何变化 | 固定三折模型换输入、固定reference输入换五折模型的类型内相关 |

主要比较全部配对到相同seed。相同概率的复用结果保留零差值并明确标记，不增加独立实验数量。

## 9. 文件与执行流程

入口为 `run.py`，实现分为 `r5.py`（公式）、`r5_training.py`（拟合／审计／扫描／适配）和 `r5_workflow.py`（复用、断点、汇总）。分类器、评价器、CLIP缓存和原训练循环保持原实现。

每seed9阶段，合计27阶段：import、fit、decision、scan，以及5个adapt阶段。完整产物为15个正式结果、360行识别记录、165行训练曲线、144行扫描及168行Q5决策。图像缓存继续放在上一轮目录；本轮工作目录独立。

本地验证包含数学恒等式、三个截距消除、类型预算、置换集合一致、CRLF／LF源码比较，以及三seed完整小规模队列、10epoch完成、结果复用、源目录不变、阶段续跑与指标输出。35项测试中34项通过；1项CLIP重新提取测试因本轮复用已有特征而跳过。真实数据CPU核验覆盖三seed的Absolute回归、α及四个强度的概率复现，并核对Relative系数有限。正式UT识别训练由服务器命令执行。

运行命令见同目录 `SERVER_COMMANDS.md`。上传ZIP与结果tar.gz均以SHA256校验，结果通过JSON描述文件确定路径并由Python按UTF-8解压。
