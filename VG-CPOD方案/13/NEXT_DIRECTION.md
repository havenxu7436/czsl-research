# L组之后的拟议方向：可训练视觉中的“先比较，再汇聚”

**状态：研究提案，尚未实现训练代码、尚未启动训练。** 本轮已完成的是13/L组的静态知识残差对照；下文R0/R1/R2为未来实验，不能计入本轮九组完成数。

研究目的：判断视觉任务梯度、局部读取与比较顺序分别能带来什么收益。它属于VG-CPOD后续视觉判别方向，不预设返回TRUST；其结果也不能直接证明可迁移关系效用预测有效。如果另行继续TRUST，应独立验证正确关系增益和同预算效用选择，本轮不混入该机制。

[完整L组数据分析](analysis/README.md) · [已完成的九组结果](results/README.md)


### 5.1 问题、定位与三组比较

**研究问题：两个属性是否应先在同一个图像位置进行比较，再汇聚其对同一物体的判别证据？** 常规非线性局部汇聚允许每个属性分别寻找高响应位置；这些位置可以不同，也可能由共同响应推动。先比较逐位置差值，可以把任务梯度直接分配到区分属性的证据上。

当前预测只包含全局 logits，没有 patch 响应，所以它不能直接证明“共同局部响应”已经是当前错误的成因。下一轮将这一定义明确的机制放进同参数对照中验证。目标同时包含减少已见搭配误选、改善多候选未见内部排序和提升联合 AUC。

这条主线延续此前 VG-CPOD 的局部判别方向，并把规模收敛到 UT-Zappos 的九次训练：

| 模型 | 视觉能力与条件读出 | 本轮回答的问题 |
| --- | --- | --- |
| R0：Global | 可训练视觉 Adapter，条件头读取 CLS | 让现有全局评分器获得视觉任务梯度能改善多少 |
| R1：Pool-first | 同一 Adapter 与评分器；逐属性先汇聚 patch 响应 | 普通局部证据学习相对全局读取的收益 |
| R2：Compare-first | 同一 Adapter 与评分器；同位置先比较，再汇聚并还原属性 logits | 比较与汇聚顺序本身是否有价值 |

R2−R1 是核心机制比较；R1−R0 说明局部读取带来的变化。三组的可训练参数集合、数量、优化器步数和选模规则相同。历史 G0 作为已有参考，无须新增一组复训。

条件属性学习已在 [CANet（CVPR 2023）](https://arxiv.org/abs/2305.17940) 中研究；[CPF（ICCV 2025）](https://arxiv.org/abs/2507.17377) 使用条件概率框架和条件视觉细化；[FILIP](https://arxiv.org/abs/2111.07783) 已使用视觉—文本 token 级晚交互。因此，Adapter、局部 token 或条件概率分解本身都不能作为创新结论。本方案的待验证贡献集中在“同位置成对比较后再边缘化”的算子及其对未见组合判别的作用。

本地 G0/CPF＋CP 是当前工程中的条件概率评分实现，不能直接称为原 CPF 完整模型的复现；原论文的视觉细化等组件与本地基座不同。

### 5.2 共同视觉模型与评分器

使用同一份 OpenAI CLIP ViT-L/14、224×224 输入。视觉 Transformer 隐藏维度 1,024，最后层投影维度 768，patch 数为 256。冻结预训练主权重，在最后四个视觉 block 的 MLP 旁加入 bottleneck64 的并行 Adapter。以列向量记每个 token，在注意力残差之后的 $`h`$ 上计算：


```math
\begin{aligned}
\widetilde h&=\operatorname{LN}_2(h),\\[4pt]
h'&=h+\operatorname{MLP}(\widetilde h)
 +0.1\,U_b\operatorname{GELU}(D_b\widetilde h),\\[4pt]
D_b&\in\mathbb R^{64\times1024},\qquad U_b\in\mathbb R^{1024\times64}.
\end{aligned}
```


Adapter 不带 bias，$`D_b`$ 以标准差 0.01 的高斯分布初始化，$`U_b=0`$。零输出初始化使初始视觉前向等于原 CLIP。模型在最后层对 CLS 和 patch 使用 `ln_post`、视觉投影与 L2 归一化，得到 $`g,p_n\in\mathbb R^{768}`$。官方默认前向只输出 CLS；下一轮要显式返回 patch，并将这条路径接入梯度。[CLIP 官方模型实现](https://github.com/openai/CLIP/blob/main/clip/model.py)

各组从同 seed 的 G0 载入头参数。冻结文本向量和 $`A_u,A_v`$，训练 $`B_o,C_o,B_c,C_c,W_c`$ 以及视觉 Adapter。与 v2 相比，物体头也参与更新，以便改进约 70% 的独立物体识别。

共同条件评分函数对任意视觉向量 $`v`$ 定义为：


```math
\begin{aligned}
u_a&=A_u e_a,\qquad v_o=A_v e_o,\\[4pt]
R(v,a,o)&=\kappa v^\top e_{ao}
 +(v^\top B_c)(e_{ao}^\top C_c)^\top
 +\rho\kappa v^\top W_c(u_a\odot v_o).
\end{aligned}
```


保持 CP rank32、低秩头 rank8、$`\kappa=20`$、$`\rho=0.15`$。R0 使用 $`U_{ao}=R(g,a,o)`$。R1/R2 使用相同的局部响应与物体条件位置权重：


```math
r_{nao}=R(p_n,a,o),\qquad
w_{no}=\frac{\exp(\kappa p_n^\top e_o)}{\sum_{k=1}^{N}\exp(\kappa p_k^\top e_o)}.
```


$`w_{no}`$ 对同一物体的全部属性相同，随视觉 Adapter 训练而变化。训练和推理都对所有候选物体计算，实际推理不提供真实物体标签。

同一参数初始化不意味着三组的初始预测相同：R0 延续全局读出，R1/R2 从一开始就使用各自的局部读出。应保存 epoch0 的 VAL 结果说明读出转换的起点；正式选模仍统一使用 epoch1—10。

### 5.3 R1 与 R2 的精确定义

**R1：先分别汇聚，再比较。**


```math
\ell_{ao}=\log\sum_n w_{no}\exp(r_{nao}),\qquad
U^{\rm pool}_{ao}=\ell_{ao}-\frac1A\sum_b\ell_{bo}.
```


减去属性均值不改变 softmax，只使两组都采用零和 logits。

**R2：先逐位置比较，再汇聚。** 对属性对 $`(a,b)`$：


```math
\begin{aligned}
\delta_{n,ab,o}&=r_{nao}-r_{nbo},\\[4pt]
Z^{+}_{ab,o}&=\sum_n w_{no}\exp(\delta_{n,ab,o}/2),\\[4pt]
Z^{-}_{ab,o}&=\sum_n w_{no}\exp(-\delta_{n,ab,o}/2),\\[4pt]
m_{ab,o}&=\log Z^{+}_{ab,o}-\log Z^{-}_{ab,o}.
\end{aligned}
```


这是两个方向对称的局部能量配分函数之对数比。若定义二元概率 $`q=Z^+/(Z^++Z^-)`$，则 $`m=\log[q/(1-q)]`$。系数 $`1/2`$ 保证单 patch 时 $`m_{ab,o}=r_{1ao}-r_{1bo}`$。

成对 log-odds 一般不满足传递一致性，需要投影成同一组多类 logits。UT 只有 16 个属性，直接使用全部 120 个无序属性对，不引入稀疏图超参数：


```math
U_o=\underset{\mathbf1^\top U=0}{\operatorname{argmin}}
\frac12\sum_{a<b}\left[(U_a-U_b)-m_{ab,o}\right]^2.
```


由于 $`m_{ba,o}=-m_{ab,o}`$，第 $`a`$ 个分量的一阶条件为


```math
A U_a-\sum_bU_b-\sum_bm_{ab,o}=0.
```


代入零和约束得到唯一解：


```math
\boxed{U^{\rm compare}_{ao}=\frac1A\sum_bm_{ab,o}}.
```


完整图在零和子空间上的 Hessian 是 $`A I`$，因此解唯一。计算只需成对 margin 的行均值，无须求逆、图网络或额外训练损失。将 R1 的 $`m_{ab}=\ell_a-\ell_b`$ 代入同一投影，恰好恢复 $`\ell_a-\overline\ell`$，所以 R1/R2 的比较确实落在局部边证据的计算方式上。

### 5.4 算子能解决什么，不能由公式直接推出什么

固定位置权重，给某物体全部属性在位置 $`n`$ 加上共同响应 $`h_{no}`$，R2 的差值保持不变：


```math
r'_{nao}=r_{nao}+h_{no}
\quad\Longrightarrow\quad
\delta'_{n,ab,o}=\delta_{n,ab,o}
\quad\Longrightarrow\quad m'_{ab,o}=m_{ab,o}.
```


分别做非线性汇聚通常不具备这个性质。例如两属性在三个位置的响应为 $`(4,0,0)`$、$`(0,3,0)`$，位置权重均匀，给两者都加 $`(0,5,0)`$：

| 方法 | 添加共同项前的 a−b 分数 | 添加共同项后 |
| --- | --- | --- |
| R1：分别汇聚后相减 | +0.9411 | −2.6825 |
| R2：先比较再汇聚 | +0.4274 | +0.4274 |

这个例子说明两种算子不同，尚不代表真实图像上的性能。实际图像变化还可能改变 $`w`$，完整模型需要端到端评价。

尤其要结合本轮错误模式澄清：**R2 不会自动去掉已见搭配偏好。** 如果加入属性特有、位置无关的偏置 $`b_{ao}`$，有


```math
\begin{aligned}
r'_{nao}&=r_{nao}+b_{ao},\\[4pt]
m'_{ab,o}&=m_{ab,o}+b_{ao}-b_{bo},\\[4pt]
U'_{ao}&=U_{ao}+b_{ao}-\frac1A\sum_b b_{bo}.
\end{aligned}
```


类别偏置会被保留下来。因此，新机制的目标是学习足以区分属性的图像证据，而不是在代数上保证 seen/unseen 去偏。若只提高完整候选下的准确率，却不改善多候选未见排序，应继续解释为偏好变化。

其梯度具有明确的局部作用。令


```math
\pi^{\pm}_{n,ab,o}=\frac{w_{no}\exp(\pm\delta_{n,ab,o}/2)}{Z^{\pm}_{ab,o}},\qquad
\left.\frac{\partial m_{ab,o}}{\partial\delta_{n,ab,o}}\right|_{w}
=\frac{\pi^+_{n,ab,o}+\pi^-_{n,ab,o}}{2}.
```


这个偏导非负，对位置求和为 1；支持与反对该属性的差异位置都可以接收梯度。实际训练同时通过 $`w`$ 反传。单 patch 时，该机制退化为普通属性响应减去均值；多 patch 才能产生新的汇聚行为。

上述反对称性、共同项不变性、类别偏置传递、单位置极限、完整图正规方程和梯度恒等式均已用 float64 构造数组核验，最大误差小于 10⁻¹⁵。数学定义已核验，实际识别表现由下一轮九组训练评价。

### 5.5 概率与训练目标

物体分支与条件分支共同形成联合分数：


```math
\begin{aligned}
S_o(x)&=\kappa g^\top e_o+(g^\top B_o)(e_o^\top C_o)^\top,\\[4pt]
p(o\mid x)&=\operatorname{softmax}_o S_o(x),\\[4pt]
p(a\mid o,x)&=\operatorname{softmax}_a U_{ao}(x),\\[4pt]
J_{ao}(x)&=\log p(o\mid x)+\log p(a\mid o,x).
\end{aligned}
```


继续使用 G0 的训练目标。对训练真值 $`y`$ 与训练组合集合 $`\mathcal C_{\rm tr}`$，令


```math
\begin{aligned}
\ell_{\rm pair}&=-J_y+\log\sum_{c\in\mathcal C_{\rm tr}}\exp J_c,\\[4pt]
\ell_{\rm support}&=-\log\sum_{c\in\mathcal C_{\rm tr}}\exp J_c.
\end{aligned}
```


因为 $`\ell_{\rm cond}+\ell_{\rm obj}=-J_y=\ell_{\rm pair}+\ell_{\rm support}`$，原损失可严格改写为：


```math
\begin{aligned}
\mathcal L
&=\frac{1.5}{1+\beta}\mathbb E\left[\ell_{\rm pair}+\beta(\ell_{\rm cond}+\ell_{\rm obj})\right]\\[4pt]
&=1.5\,\mathbb E\ell_{\rm pair}+\frac{1.5\beta}{1+\beta}\mathbb E\ell_{\rm support}\\[4pt]
&=1.5\,\mathbb E\ell_{\rm pair}+\frac9{86}\mathbb E\ell_{\rm support},\qquad\beta=0.075.
\end{aligned}
```


支持集项会把概率质量向训练组合集中，这是目标函数的明确性质；本轮数据尚不能将全部已见搭配误选归因于这一项。三组保持同一个目标，直接观察视觉算子能否带来更好的组合判别。

## 6. 一套九次训练的执行规格与结果判断

### 6.1 固定训练规格

| 项目 | 下一轮统一设置 |
| --- | --- |
| 数据 | UT-Zappos 原有 split、样本和标签映射 |
| 初始化 | 对应 seed 的 G0 头参数；同一 CLIP 权重与文本向量；相同 Adapter 初始化 |
| 输入 | CLIP 官方 224 预处理；三组统一使用同一 resize、中心裁剪和归一化 |
| 视觉训练 | 最后 4 个 block 的并行 Adapter；其余 CLIP 主权重冻结 |
| 训练头 | Bo、Co、Bc、Cc、Wc；Au、Av 和文本向量固定 |
| 参数量 | Adapter 524,288＋共同头 49,152＝573,440，三组相同 |
| 训练长度 | 每组固定 10 epochs，有效 batch64 |
| 显存安排 | microbatch4、累积 16 次；最后不足一组的累积按实际样本数归一化 |
| 学习率 | 视觉 Adapter 5e-5；Bo/Co/Bc/Cc 2e-4；Wc 6e-5 |
| 优化器 | AdamW，betas=(0.9,0.999)，eps=1e-8，weight decay=1e-4，梯度裁剪 1.0 |
| 调度 | 按优化器步做余弦下降，首步为初始学习率，末步为初始学习率的 1%；三组使用相同步数 |
| 数值 | AMP FP16＋GradScaler；softmax、logsumexp、最终损失在 FP32 中计算；梯度裁剪在 unscale 后执行 |
| 选模 | epoch1—10 中 VAL CW native AUC 的首次最大值；保存 epoch0 VAL 作为起点记录 |
| TEST | 选定同一权重评价 CW/OW，分别使用相应 VAL bias 与温度 |
| 保存 | best/last、完整 epoch 历史、逐图像联合/条件/物体 logits、样本路径和标签、耗时与峰值显存 |

参数量计算来自四个无 bias 的 1024→64→1024 Adapter，共 $`4\times2\times1024\times64=524288`$；四个 768×8 头矩阵加一个 768×32 的 CP 读出，共 $`49152`$。

局部分数张量为 `[B,O,A,N]`。R2 的完整有序差值张量在 microbatch4 下有 $`4\times12\times16\times16\times256=3145728`$ 个元素，单个 FP32 张量约 12 MiB；反向图还会产生中间量，可按物体分块以降低峰值。这个估算不包含视觉 Transformer 激活。

现有 `g/fa/fo` 缓存不能承担新 Adapter 的在线梯度。可继续复用图像路径、文本向量、G0 权重、数据 split 和评价程序；下一轮训练需要读取原图并运行视觉前向。`run.py` 的 v2 全局缓存入口尚不是这套新方案的训练实现。

### 6.2 三张卡直接覆盖完整比较

| 物理 GPU | seed | 同卡依次运行 |
| --- | --- | --- |
| 1 | 3351 | R0 → R1 → R2 |
| 2 | 3352 | R0 → R1 → R2 |
| 3 | 3353 | R0 → R1 → R2 |

共九个完整任务，每个模型都有三个配对 seed。当前一轮集中在 UT-Zappos；跨数据集扩展在这套比较形成结论后单独安排。下一轮实际耗时需要按在线 CLIP 测量，不能照搬 v2 的几十秒缓存训练时间。

### 6.3 一张主结果表回答三个问题

| 问题 | 保留的核心指标 | 对结果的解释 |
| --- | --- | --- |
| 联合识别是否提高 | CW/OW native AUC、exact AUC、VAL bias 下 TEST HM；逐 seed 配对差值与均值±标准差 | 关注 R2−R1，再看 R1−R0 与历史 G0 的关系 |
| 是否学到更好的属性判别 | unseen 全 16 属性准确率、组合宏平均；多候选未见内部准确率；错选已见搭配的纠正/新增数量 | 分开识别“减少已见偏好”和“改善未见之间排序” |
| 是否改善跨物体竞争及实用性 | 独立物体准确率、CW 未见分区联合准确率；OW HM/NLL/ECE；训练耗时、峰值显存 | 避免只在给定真实物体的诊断任务上得到收益 |

这些指标从同一组预测文件计算，不增加训练任务。局部机制还可从同一 best 权重导出真实错误样本的响应图，并记录成对 margin 到一致属性分数的投影残差，帮助解释算子；它们作为结果诊断，不再拆成新的模型队列。

结果解释采用直接的比较关系：R2 若在 R1 之上同时改善联合 AUC 和多候选未见排序，才支持“先比较后汇聚”的机制价值；若 R1/R2 都优于 R0 而彼此接近，主要贡献应归于局部视觉学习；若仅完整候选属性准确率上升、未见内部排序未变，则延续本轮的“组合偏好重排”解释。这样九次训练就能回答主要研究问题。
