# 12 · 动态读出学习率分组与更新尺度对照

**K 组实验，VG-CPOD v1，归档日期 2026-10-21。** 使用冻结 OpenAI CLIP ViT-L/14 与 UT-Zappos natural split。K1/K2 各三个 seed、各十轮，共六次训练、60 个模型 epoch；native/exact 各24份预测，十二项动态／TRAIN均值评估保存96份预测，共 **144份预测文件、24份checkpoint文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [11 J组](../11/README.md) · [08 G组](../08/README.md)

## 1. 实验目的与修订范围

J1 的静态与动态读出已经分开，但选中 checkpoint 中，Wd/Ws 的一阶矩范数比仅约2%–3%，Adam 自适应更新范数比却达到95%–100%。本轮用优化器分组直接控制 Wd 的更新速度，检验是否改善未见组合排序。

| 实验 | Wd相对CP学习率倍率 | Wd学习率 | 训练来源 |
|---|---|---|---|
| J1 | 1 | 3e-4→3e-6 | [11](../11/README.md)已完成结果 |
| K1 | 0.3 | 9e-5→9e-7 | 本轮三个seed |
| K2 | 0.1 | 3e-5→3e-7 | 本轮三个seed |

K组模型前向、概率、损失和147,456个可训练参数均沿用J1。公式上的本轮修订是参数组学习率与AdamW更新；同时把更新量、门控变化、分数变化和最终排序分别记录。模型仍使用缓存上的后编码适配矩阵，CLIP骨干固定。

## 2. 公式与实现

### 2.1 固定 TRAIN 均值与残差门控

令 $`g_b\in\mathbb{R}^{768}`$ 为图像全局特征，$`f^o_{bj}`$ 为物体查询特征，$`R=32`$，TRAIN 图像数 $`N=22998`$：

```math
\begin{aligned}
\mu_j&=\frac{1}{N}\sum_{b\in\mathrm{TRAIN}}f^o_{bj},\\[4pt]
r_{bj}&=f^o_{bj}-\mu_j,\qquad
\widehat r_{bj}=\frac{r_{bj}}{\max(\lVert r_{bj}\rVert_2,10^{-12})}.
\end{aligned}
```

`round_k.py` 从 J 组逐字复制与当前缓存一致的均值文件。首次生成采用 TRAIN 特征 float64 分块累计，随后转成 float32 buffer `gate_fo_mean`，随 checkpoint 保存。所有 seed 共用同一均值。

设 $`B_{\mathrm{vis}}\in\mathbb{R}^{R\times768}`$ 第 $`k`$ 行转置为 $`b_k`$，门控为：

```math
\begin{aligned}
q_{bjk}&=2\sigma\!\left(\frac{b_k^\top\widehat r_{bj}}{0.56}\right)-1,\\[4pt]
h_{bjk}&=\frac{0.25}{2}\left(q_{bjk}-\frac{1}{R}\sum_{\ell=1}^{R}q_{bj\ell}\right),\\[4pt]
\gamma_{bjk}&=1+h_{bjk},\qquad
\frac{1}{R}\sum_k\gamma_{bjk}=1,\quad 0.75\le\gamma_{bjk}\le1.25.
\end{aligned}
```

### 2.2 静态与动态独立读出

设 $`A_u,A_v\in\mathbb{R}^{R\times768}`$，$`W_s,W_d\in\mathbb{R}^{768\times R}`$。代码中 `Wc` 对应 $`W_s`$，`Wd` 对应 $`W_d`$：

```math
\begin{aligned}
u_i&=A_ue_i^a,\qquad v_j=A_ve_j^o,\qquad a_{ij}=u_i\odot v_j,\\[4pt]
\Delta U^s_{bji}&=\rho\kappa g_b^\top W_s a_{ij},\\[4pt]
\Delta U^d_{bji}&=\rho\kappa g_b^\top W_d(a_{ij}\odot h_{bj}),\\[4pt]
U_{bji}&=U^{\mathrm{base}}_{bji}+\Delta U^s_{bji}+\Delta U^d_{bji}.
\end{aligned}
```

两组与 J1 使用相同的前向公式。类别算子及单个分数对读出的梯度为：

```math
\begin{aligned}
M_{bj}&=\left[W_s\mathrm{diag}(v_j)
+W_d\mathrm{diag}(v_j\odot h_{bj})\right]A_u,
\qquad\mathrm{rank}(M_{bj})\le R,\\[4pt]
\nabla_{W_s}\Delta U_{bji}&=\rho\kappa g_ba_{ij}^{\top},\\[4pt]
\nabla_{W_d}\Delta U_{bji}&=\rho\kappa g_b(a_{ij}\odot h_{bj})^{\top}.
\end{aligned}
```

两路共享 $`A_u`$，算子秩上界为 32。首步任务梯度进入两个读出矩阵，后续经非零读出传播到属性／物体因子及门控。K1/K2 改变 Wd 的优化器学习率，梯度公式由上述前向直接计算。

### 2.3 联合概率与损失

沿用 CLIP 特征上的物体分支 $`S_b`$ 和组合适配器基底 $`U^{\mathrm{base}}`$：

```math
s_{bji}=\log\mathrm{softmax}(S_b)_j
+\log\mathrm{softmax}(U_{bj:})_i.
```

设 $`\mathcal{S}`$ 为 TRAIN 组合集合，批量大小为 $`B`$，样本真实标签为 $`(i_b,j_b)`$：

```math
\begin{aligned}
Z_b&=\sum_{(i,j)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],\\[4pt]
\mathcal{L}_{\mathrm{support}}&=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_a+\mathcal{L}_o&=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}&=\frac{1.5}{1+0.075}
\left[\mathcal{L}_{\mathrm{pair}}+0.075(\mathcal{L}_a+\mathcal{L}_o)\right]\\[4pt]
&=\frac{3}{2}\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}&=\mathcal{L}_{\mathrm{task}}
+10^{-3}\frac{\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_R\rVert_F^2}{R^2}.
\end{aligned}
```

$`\mathcal{L}_a`$ 为真实物体条件下的属性负对数似然，$`\mathcal{L}_o`$ 为物体负对数似然。`V1` 配置开启残差门控；primitive/cent 项为零。

### 2.4 动态学习率和 AdamW 更新

令总更新数 $`T=10\lceil N/64\rceil=3600`$，更新编号 $`t=1,\ldots,T`$。代码日程索引为 `step=t-1`：

```math
\begin{aligned}
\alpha_t&=10^{-5}+\frac{10^{-3}-10^{-5}}{2}
\left[1+\cos\!\left(\frac{\pi(t-1)}{T-1}\right)\right],\\[4pt]
\alpha_{\mathrm{adapter},t}&=\alpha_t,\qquad
\alpha_{\mathrm{CP},t}=0.3\alpha_t,\\[4pt]
\alpha_{s,t}&=\alpha_{\mathrm{CP},t},\qquad
\alpha_{d,t}=m_d\alpha_{\mathrm{CP},t},\qquad
m_d\in\{0.3,0.1\}.
\end{aligned}
```

`make_optimizer` 建立三个参数组，每个可训练参数属于一个组：

| 参数组 | 参数 | lr_scale |
|---|---|---|
| adapter | Bo、Co、Bc、Cc | 1 |
| cp | Au、Av、Bvis、Wc | 0.3 |
| dynamic_cp | Wd | K1 为 0.09，K2 为 0.03 |

对全模型梯度按原实现做全局范数裁剪。设 $`\mathbf{G}_t^{\mathrm{raw}}`$ 为所有参数梯度的拼接，参数 $`p`$ 的原始梯度为 $`G_{p,t}^{\mathrm{raw}}`$：

```math
c_t=\min\!\left(1,\frac{1}{\lVert\mathbf{G}_t^{\mathrm{raw}}\rVert_2+10^{-12}}\right),
\qquad G_{p,t}=c_tG_{p,t}^{\mathrm{raw}}.
```

各参数按所属组的学习率 $`\alpha_{p,t}`$ 进行 AdamW 更新。以下矩状态对应同一个参数 $`p`$，初始化 $`m_0=v_0=0`$：

```math
\begin{aligned}
m_t&=0.9m_{t-1}+0.1G_{p,t},\\[4pt]
v_t&=0.999v_{t-1}+0.001G_{p,t}^{\odot2},\\[4pt]
\widehat m_t&=\frac{m_t}{1-0.9^t},\qquad
\widehat v_t=\frac{v_t}{1-0.999^t},\\[4pt]
D_t&=\frac{\widehat m_t}{\sqrt{\widehat v_t}+10^{-8}},\\[4pt]
\theta_t&=(1-\alpha_{p,t}10^{-4})\theta_{t-1}-\alpha_{p,t}D_t.
\end{aligned}
```

开方、平方与除法逐元素执行。组学习率同时缩放自适应更新和 AdamW 的解耦衰减，weight decay 系数继续为 1e-4。未配置新倍率的历史模型沿用原参数组和参数顺序。

`readout_diagnostics.py` 从选中与末轮 checkpoint 的 AdamW 状态重建 $`\lVert m_t\rVert_F`$、$`\lVert v_t\rVert_F`$、$`\lVert D_t\rVert_F`$、$`\alpha_{p,t}\lVert D_t\rVert_F`$，以及动态／静态比值。最后一项对应该 checkpoint 最后一步的自适应更新范数。矩阵范数来自更新后的参数，AdamW 衰减单独执行。

### 2.5 TRAIN 均值门控与幅值诊断

在 native/exact 各自选中的权重下，对全部 TRAIN 图像统计门控均值，再评估：

```math
\begin{aligned}
\bar h_j&=\frac{1}{N}\sum_{b\in\mathrm{TRAIN}}\gamma_{bj}-\mathbf{1},\\[4pt]
U^{\mathrm{mean}}_{bji}&=U^{\mathrm{base}}_{bji}
+\rho\kappa g_b^\top W_s a_{ij}
+\rho\kappa g_b^\top W_d(a_{ij}\odot\bar h_j).
\end{aligned}
```

dynamic/mean 分别重算条件 softmax、联合概率和 CW/OW 曲线，并各自在 VAL 上选择 bias，用对应 bias 评价 TEST。每项均值评估优化步数为零。

训练日志保留两路修正的原始和属性中心化 RMS。令 $`A,O`$ 为属性数和物体数，对 $`z\in\{s,d\}`$：

```math
\begin{aligned}
\widetilde{\Delta U}^{z}_{bji}&=\Delta U^{z}_{bji}
-\frac{1}{A}\sum_{i'=1}^{A}\Delta U^{z}_{bji'},\\[4pt]
\mathrm{RMS}_{z,\mathrm{centered}}&=
\sqrt{\frac{1}{NOA}\sum_{b,j,i}(\widetilde{\Delta U}^{z}_{bji})^2}.
\end{aligned}
```

每个 epoch 的统计来自各 batch 前向时的权重，均方值按图像数加权汇总后再开平方。`history.json` 同时记录三组学习率。


## 3. 具体代码修订与验证

| 文件 | 本轮具体改动 | 作用 |
|---|---|---|
| [engine.py](code/vgcpod_v1_roundK/vgcpod/engine.py) | `make_optimizer`新增`dynamic_cp`组；日程按`lr_scale`更新 | 仅Wd使用独立倍率 |
| [readout_diagnostics.py](code/vgcpod_v1_roundK/readout_diagnostics.py) | 新增checkpoint到参数名的优化器状态映射、偏差修正与更新范数 | 核验学习率在实际更新中的作用 |
| [K1/K2配置](configs/round_k/) | 新增`dynamic_readout_lr_multiplier=0.3/0.1` | 预先固定两档 |
| [round_k.py](code/vgcpod_v1_roundK/round_k.py) | 每卡K1→K2；复用J组TRAIN均值；十二项均值评估；优化器与排序汇总 | 六训练完整队列 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundK/ranking_diagnostics.py) | 增加K1/K2−J1、K1−K2等配对 | 衡量预测变化 |
| [tmux脚本](code/vgcpod_v1_roundK/scripts/start_round_k_tmux.sh) | 三卡seed队列、日志、退出码、续训 | GPU1/2/3分别对应3351/3352/3353 |
| [K组测试](code/vgcpod_v1_roundK/tests/test_round_k.py) | 参数分组、Adam更新、历史兼容、配置与流程验证 | 发布前检查 |

相对11的完整变更见[源码补丁](code/source_changes.patch)。训练核心文件中只有engine改变；model.py与mean_gate_eval.py保留J版原字节。[服务器源码身份](results/server_source_identity.json)记录原包与服务器文件的核对。

原发布 **61项测试通过**；CPU合成CLIP缓存完成六训练、十二评估；重复启动跳过已完成任务。旧J1配置在新旧训练器中的模型参数、AdamW和Torch RNG逐元素相同。证据见[发布验证](results/package_validation/K/validation/result.json)与[测试摘要](results/package_validation/K/validation/unit_test_summary.json)。本次仓库归档复算正式结果，不启动服务器训练。

## 4. 超参数选取策略与固定设置

1. **以J1作直接对照。** 根据其Adam状态选取0.3、0.1两档Wd倍率，J1的倍率1作为已有参照。
2. **固定其余训练要素。** 输入为归一化TRAIN残差；τγ=0.56、η=0.25、ρ=0.15、β=0.075、rank32、adapter rank8。
3. **保留初始化和数据顺序。** Wc/Wd从零开始，对应seed沿用同一因子和后编码适配器初始化；每次10轮、3600次更新。
4. **分别检查设置、更新、输出和指标。** 从checkpoint重建Adam更新，结合门控方差与dynamic−mean分数RMS分析。
5. **固定VAL规则。** native按VAL CW native AUC首次严格最大选权重；exact独立保存。各权重及均值条件分别用自己的VAL bias评价TEST。

| 固定项 | 实际设置 |
|---|---|
| CLIP／缓存 | 冻结OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.56 |
| epochs／batch／eval batch／workers | 10 / 64 / 64 / 4 |
| 后编码适配矩阵LR | 1e-3→1e-5 |
| Au、Av、Bvis、Wc的LR | 3e-4→3e-6 |
| Wd的LR | K1：9e-5→9e-7；K2：3e-5→3e-7 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 全局梯度裁剪／λorth／残差ε | 1 / 0.001 / 1e-12 |
| TRAIN查询均值／可训练参数 | [12,768] / 147,456 |

模板见[round_k](configs/round_k/)，服务器实际配置见[runtime](configs/runtime/)。J/K的TRAIN均值文件逐字一致，统计来自22,998张TRAIN图像。当前配置与原始结果按原字节归档；新的视觉判别研究提案放在末节数据分析。

## 5. 完整训练结果

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| K1 | 3351 | 5／5 | 44.3233 | 31.1919 | 33.4992 | 24.3035 |
| K1 | 3352 | 8／6 | 41.6102 | 30.0067 | 35.3735 | 23.9467 |
| K1 | 3353 | 6／6 | 45.0030 | 34.1870 | 34.0870 | 25.4192 |
| K2 | 3351 | 5／5 | 43.9091 | 30.9288 | 33.6030 | 24.5263 |
| K2 | 3352 | 8／6 | 41.5693 | 29.8716 | 35.3351 | 23.9131 |
| K2 | 3353 | 6／6 | 44.8551 | 34.2888 | 33.8095 | 25.0468 |

数值均为native AUC，单位为点；主表使用native选择权重。两组native最佳epoch均为5/8/6，exact均为5/6/6。seed3352的两条选择轨道不同，分别保留权重、指标与预测，详见[结果README](results/README.md)。

## 6. 仓库结构与复现入口

```text
12/
  README.md          # 公式、优化器、代码与超参数，末节数据分析
  code/              # 原K源码、相对J补丁、原分析脚本
  configs/           # 模板、运行配置、资产、参考结果与清单
  results/           # 六训练、双选择、十二评估、日志与原始包
  analysis/          # 分析、图表、复算、公式与网页Markdown核对
```

在仓库根目录恢复运行包，复用服务器现有CLIP缓存与J组结果：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 12 \
  --dest /data/xhw/vgcpod_project/replay_12
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_12/vgcpod_v1_roundK

export VG_K_OUT=/data/xhw/vgcpod_v1_runs/replay_12_K
export VG_K_SESSION=vgcpod_12_K
bash scripts/start_round_k_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --j-out /data/xhw/vgcpod_v1_runs/roundJ_20261021
tmux attach -t vgcpod_12_K
```

Ctrl+b、d脱离tmux；主日志出现`ROUND_K_COMPLETE ... training_jobs 6 mean_gate_jobs 12`且`launcher.exitcode`为0表示队列完成。其余历史对照默认读取原服务器目录，路径见[外部输入](external_inputs.json)，提取命令见[runbook.json](runbook.json)。

在仓库根目录复算结果：

```bash
python VG-CPOD方案/12/analysis/recompute.py
python tools/reproduce.py verify
```

依赖NumPy、PyTorch、Matplotlib。复算输出到`analysis/recomputed/`；正式分析位于`analysis/confirmed/`。materialize按[成员映射](package_members.json)恢复原K包99个文件的字节和布局。

## 7. 数据分析与后续主线

| 模型 | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- |
| D1 | 43.301 | 31.564 | 34.791 | 25.453 |
| G0 | 43.343 | 31.615 | 34.512 | 25.184 |
| I2 | 43.433 | 31.787 | 34.205 | 24.853 |
| J1 | 43.329 | 31.542 | 34.653 | 24.526 |
| K1 | 43.645 | 31.795 | 34.320 | 24.556 |
| K2 | 43.445 | 31.696 | 34.249 | 24.495 |

K1相对J1的TEST CW下降0.333点、OW提高0.031点；K2的CW/OW均下降。K1/K2均低于同β/ρ静态G0。Wd/Ws实际自适应步长比由J1的0.971降至0.307和0.111，动态条件logit RMS由0.648降至0.300和0.111，说明更新尺度已按设计受控，但组合泛化没有整体上移。

进一步逐图复算发现：G0未见组合排序错误的88.3%–92.0%在I2/J1/K1/K2中持续存在；约30.72%的未见图像在给定真实物体后仍排错属性。这一候选内排序诊断避开了把零bias条件属性准确率直接当作视觉能力上限的解释。

**本轮收束冻结缓存头上的Wd学习率与门控强度搜索。下一步研究“物体条件的局部判别证据”：同一patch先比较两个属性，再边缘化位置并形成条件概率。** CLIP视觉Adapter和静态CP作为共同基座。R0全局、R1普通局部、R2新算子，在UT/MIT各三seed，共18次完整训练；主要机制比较为R2−R1。该方向属于后续研究提案，本轮已完成实验为K1/K2。

![更新量、动态输出与OW结果](analysis/confirmed/figures/02_update_scale.png)

[完整数据分析](analysis/README.md)集中给出同seed差值、选权重比较、共同错误、候选排序、近邻文献差异，以及下一步算子推导和正式实验规格。
