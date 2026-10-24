# TRUST 在强 CPF 基底上的完整实验方案

版本：Strong v1。日期：2026-10-23。

## 1. 本轮只回答一个研究问题

**当分类基底恢复到已验证的强 CPF 水平后，TRUST 的可迁移关系效用能否预测关系对未见组合的帮助，并据此选择出优于普通删边和自由学习边概率的结构？**

研究主体继续是 TRUST。基底采用本机历史 G0，即 CLIP ViT-L/14＋条件概率分解＋静态 CP32，历史三 seed TEST CW/OW AUC 均值约为 34.51/25.18。这里“CPF”指已有工程的条件概率基底；本轮复用该实际权重与缓存。

与弱基底 v1 相比，改变的是基础分类器和相应任务损失。五折组合隔离、配对效用、三成员异方差集成、类型预算、随机图平均、七组主对照、知识内容实验、噪声与效用消融均继续使用。Compare-first 视觉新方法不进入本轮。

## 2. 强基底与 TRUST 的连接

设冻结全局特征为 $`g`$，原词与组合文本向量为 $`e_a,e_o,e_{ao}`$。G0 的完整分数为：


```math
\begin{aligned}
s_o^0&=20g^\top e_o+(g^\top B_o)(e_o^\top C_o)^\top,\\[4pt]
c_{ao}^0&=20g^\top e_{ao}+(g^\top B_c)(e_{ao}^\top C_c)^\top
+3g^\top W_c[(A_u e_a)\odot(A_v e_o)].
\end{aligned}
```


低秩头 rank8，CP rank32。主实验加载对应 seed 的 G0，冻结这些基底参数，使无关系图时始终恢复同一个强基底。没有删除静态 CP，也没有用随机初始化的弱分类头替换它。

TRUST 保留物体条件局部上下文：


```math
v_o(x)=\mathcal N\bigl(\mathcal N(g)+0.5\mathcal N(f_o(x))\bigr).
```


$`f_o`$ 是原 CLIP 缓存中的物体查询局部特征；只为 TRUST 残差使用。G0 本身仍按原代码读取全局 $`g`$。无需重新提取图像特征。

对 96 条 OI/OE/AE 关系，类型消息与节点原型为：


```math
\begin{aligned}
m_e&=(I+L_{t(e)}R_{t(e)})(h_e-e_{v(e)}),\\[4pt]
z_v(G)&=\mathcal N\left(e_v+0.1\sum_{e:v(e)=v}\frac{b_e}{d_{v,t(e)}}m_e\right).
\end{aligned}
```


每种关系类型的消息秩为 16。定义 v1 的归一化三元交互，秩为 64：


```math
T(x,o,a;z)=\sum_{k=1}^{64}
\mathcal N(P_vv_o(x))_k\mathcal N(P_oz_o)_k\mathcal N(P_az_a)_k.
```


完整模型为：


```math
\begin{aligned}
s_o&=s_o^0+\frac\alpha{t_g}g^\top(z_o-e_o),\\[4pt]
c_{ao}&=c_{ao}^0+\frac\alpha{t_g}\left[T(x,o,a;z)-T(x,o,a;e)\right],\\[4pt]
J_{ao}&=\log\mathop{\mathrm{softmax}}\nolimits_o(s)_o+
\log\mathop{\mathrm{softmax}}\nolimits_a(c_{\cdot o})_a.
\end{aligned}
```


保持 v1 的固定 $`\alpha=0.1,t_g=0.1`$，没有引入 v2 的可学习有符号 alpha。所有边关闭时直接使用原型锚点，使两个残差精确为零。三个实际 G0 checkpoint 的该性质已逐元素验证。

原 v1 的弱视觉 Adapter 由完整冻结 G0 代替。训练参数为类型消息 $`L,R`$ 与三元投影 $`P_v,P_o,P_a`$，共 221,184 个；capacity 使用相同参数集合构造不依赖关系附件的容量分支。r0 是已训练好的固定 G0，残差对输出为零。

任务损失与强 G0 保持一致：


```math
\begin{aligned}
\ell_{\rm pair}&=-J_{a_i o_i}+\log\sum_{c\in\mathcal C_{\rm train}}\exp J_c,\\[4pt]
\ell_o&=-\log p(o_i\mid x_i),\qquad
\ell_a=-\log p(a_i\mid o_i,x_i),\\[4pt]
\mathcal L&=\frac{1.5}{1.075}\mathbb E[\ell_{\rm pair}+0.075(\ell_o+\ell_a)]
+\lambda_W\mathcal R_W,\\[4pt]
\mathcal R_W&=\frac1{768^2}\sum_{t=0}^{2}\|L_tR_t\|_F^2,\qquad\lambda_W=0.001.
\end{aligned}
```


## 3. 教师与效用测量

**全量 G0 用于正式主实验，五折教师使用各自 Inner 上重新训练的同构 G0。** 全量 G0 已使用完整 TRAIN，直接作为五折教师会让 Query 组合进入基底训练；因此程序为每折执行：

1. 新初始化 G0 架构，在该折 Inner 上训练 10 epochs，完整任务损失使用 Inner 组合候选；使用最后一个 epoch。
2. 冻结该折 G0，在同一 Inner 上训练 TRUST 消息和三元残差 10 epochs，每 batch 使用 Bernoulli(0.5) 参考图。
3. 在留出的 Query 组合上测量配对关系增益。支持摘要仅使用该折 Inner 图像。

五折角色保持 Q1–Q3 拟合、Q4 开发、Q5 独立审计。拟合折各 512 个 episode，开发和审计各 128 个；每个 episode 最多 8 个 Query 组合、每组合 8 张图；每类型采 8 条边，共 24 条。普通配对使用 4 个背景图，Q5 使用 16 个背景图。

对同一背景图，边关闭和开启共享所有其他关系及 Query 样本：


```math
\begin{aligned}
d_{e,k}&=R_Q(G_{-e}^{(k)})-R_Q(G_{+e}^{(k)}),\\[4pt]
y_e&=\frac1K\sum_kd_{e,k},\qquad
\nu_e=\frac{\mathop{\mathrm{Var}}\nolimits_{k}(d_{e,k})}{K}.
\end{aligned}
```


$`R_Q`$ 是 Query 组合等权、组合内图像等权的候选 NLL，候选包含全部 TRAIN 组合。$`y_e\gt 0`$ 表示开启该关系降低 Query 风险。每个真实/打乱知识流程产生 1,792 个 episode、43,008 条效用观测和 417,792 次关系头前向；原始配对差也保存。

效用输入保持 v1：语义、Inner 视觉摘要、类型与来源特征。三成员网络输出均值与正方差；每类型尺度仅从拟合标签计算。带测量噪声的目标为：


```math
\mathcal L_u=\frac{|B|}{|E|}\frac1{|B|}\sum_{e\in B}
\frac1{q_e}\frac12\left[\log(\sigma_e^2+\nu_e)
+\frac{(\mu_e-y_e)^2}{\sigma_e^2+\nu_e}\right].
```


$`q_e`$ 为分层采边包含概率。类型标准化时，目标与噪声分别除以尺度及尺度平方，预测后恢复原单位。Q4 选择各成员训练 epoch；Q5 审计在最终五折标签合并拟合之前完成。

部署时使用 16 次 TRAIN 支持摘要，合并任务方差、摘要方差和集成方差。关系奖励与结构为：


```math
\begin{aligned}
r_e&=\mu_e-\lambda_{\rm var}\sqrt{v_e}-\lambda_{\rm ens}\sqrt{u_e},\\[4pt]
\pi_e&=\mathop{\mathrm{sigmoid}}\nolimits\left(\mathop{\mathrm{logit}}\nolimits(0.5)+\frac{r_e-\gamma_{t(e)}}\tau\right),\\[4pt]
\gamma_t&\ge0,\qquad\sum_{e:t(e)=t}\pi_e\le b|E_t|.
\end{aligned}
```


通过非负乘子二分求解类型上界预算。Q4 顺序选择风险系数、预算比例和温度尺度；使用原 v1 的固定候选集合。mean 把两项风险系数置零，robust 使用所选风险系数。drop 在每个类型内使用与 robust 相同的平均边概率，实际采图是独立 Bernoulli。

## 4. 完整实验范围

| 实验 | 每 seed 内容 | 三 seed 合计 |
| --- | --- | --- |
| 主对照 | r0、capacity、dense、drop、free、mean、robust | 21 组最终结果 |
| 知识内容 | shuffled 独立完成五折教师、测量、拟合、调参、参考与适配 | 3 组最终结果 |
| 效用机制 | semantic、unmeasured；分别重拟合与 Q5 审计 | 6 份消融审计 |
| 审计内对照 | 类型常数、边查表、OI/OE/AE、可靠方向子集 | 写入每份审计 |
| 读出诊断 | robust 同一最优权重改为 mean-graph / mean-logit，分别在 VAL 校准 | 6 组固定权重结果 |
| 噪声 | dense/drop/mean/robust × 5 扰动 seed × 3 类型 × 5 强度 × 2 方式 | 1,800 条评价 |

噪声 seed 为 7101–7105，强度为 0、0.1、0.2、0.3、0.5，方式为 source mismatch 与 deletion。每个条件同时评价 CW/OW。固定干净 VAL 的 bias/温度；噪声阶段只改变知识附件和预测结构，不重新拟合效用、不训练分类器。零噪声行直接引用干净结果，保证曲线起点一致。

semantic 与 unmeasured 复用对应主流程五折测量；unmeasured 按类型用 `floor(0.2×边数)` 留出，共 18/96 条边，排除其效用拟合和开发选择记录，再单独报告 Q5。真实知识与 shuffled 使用各自测量，不能交叉复用效用标签。

概率读出沿用 v1：每张图先在对应候选集归一化，再平均 16 张图的候选概率。CW/OW 候选归一化分别计算，避免直接截取 OW 混合结果冒充 CW 混合结果。所有 batch 使用固定图种子 1701。

## 5. 训练与产物

主流程和 shuffled 的全 TRAIN 参考教师均从对应强 G0 出发，冻结基底，训练 TRUST 10 epochs。七组主实验适配 3 epochs，包括适配 epoch0 的 VAL 选模。r0/capacity 分别使用自己的参考训练，capacity 不继承关系参考教师的已学消息参数。free 保留分类风险的 REINFORCE、KL 与预算惩罚。

| 项目 | 设置 |
| --- | --- |
| 正式 seed / GPU | 3351→1，3352→2，3353→3 |
| 折内 G0 | 10 epochs，batch64；头 LR1e-3，CP 因子 LR3e-4，余弦至 1% |
| TRUST 教师/参考 | 10 epochs，batch64，消息与三元头 LR3e-4 |
| TRUST 适配 | 3 epochs，LR3e-4；一 epoch warmup 后余弦下降 |
| 优化器 | AdamW，weight decay1e-4，梯度裁剪1 |
| 效用网络 | 3 成员，最多50 epochs，LR1e-3，batch256；Q4选 epoch |
| 选模 | VAL CW native AUC 首次严格最大；同一权重用于 CW/OW |
| 校准 | VAL exact HM 选 bias；41 点温度网格选 NLL；TEST 原样应用 |
| 训练恢复 | 最近完整 epoch；效用测量按折复用；效用拟合未完成则按固定 seed 重做该阶段 |
| 噪声恢复 | 已保存条件直接跳过，继续剩余条件 |

每个 seed 共 38 个调度阶段，三卡总计 114 个阶段；各卡内部自动串行，无需手动拆任务。正式时长以阶段日志记录为准，效用配对和 1,800 条噪声评价是主要开销。

最终 `status.json` 同时检查 114 个阶段、30 组模型/读出结果、12 份效用审计和 1,800 条噪声记录。`summary.csv`、`aggregates.json`、`paired_comparisons.json`、`robustness_summary.json`、`utility_audits.json` 给出总表。主模型和读出诊断都保存 VAL/TEST 概率、候选轴、真值、完整混合联合概率与样本顺序。

核心判断使用同一批完整结果：robust 相对 drop/free/mean 的配对 CW/OW AUC 与固定工作点 HM；Q5 的效用相关性和方向准确率是否超过类型常数/边查表；噪声 RAUC 是否改善；真实知识是否胜过 shuffled。三个 seed 的结果和关系数量一起报告。

## 6. 已完成的本地验证

已检查三个实际 G0 checkpoint 的无图输出逐元素一致性、强基底冻结、折内组合隔离、任务损失梯度、配对效用符号及方差、预算 KKT、异方差梯度、概率平均、评价复算、完整队列与恢复。测试日志和合成完整队列状态随包交付。

本机没有正式 UT 特征缓存；服务器免密 SSH 认证未通过。因此本次交付是可运行完整实验的代码包和命令，正式 UT 指标由服务器队列产生。合成全流程测试没有被记作正式实验结果。
