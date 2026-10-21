# 11 · 静态与动态CP独立读出

**J 组实验，VG-CPOD v1，归档日期 2026-10-21。** 使用冻结 OpenAI CLIP ViT-L/14 与 UT-Zappos natural split。本阶段新增 J1 三个 seed，共三次训练、30 个模型 epoch；保存 native/exact 各12份预测，六项动态／TRAIN均值评估保存48份预测，共 **72份预测文件、12份checkpoint文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [10 I组](../10/README.md) · [08 G组](../08/README.md)

## 1. 实验目的与修订范围

10 的 I2 使用归一化 TRAIN 残差、τγ=0.56、η=0.25，相对 H1 回收了主要损失，但 TEST CW/OW 仍低于独立训练 G0。I2 的静态与动态 CP 修正共用 Wc。J1 将两类修正的读出矩阵分别学习，继续共享属性／物体因子。

| 方案 | 静态读出 | 动态读出 | 门控输入／τγ／η | 可训练参数 |
|---|---|---|---|---|
| I2：直接对照 | Wc | 共用Wc | 归一化TRAIN残差／0.56／0.25 | 122,880 |
| J1：本轮 | Ws，代码名Wc | 新增Wd | 同I2 | 147,456 |

新增一个768×32矩阵，共24,576个参数，增加20%。两路联合训练，均从零初始化；新增矩阵不消耗随机数，原有因子、门控和适配器保留对应seed的初始化。

## 2. 公式修订：静态与动态分别读出

令 $`g_b\in\mathbb{R}^{768}`$ 为全局图像特征，$`f^o_{bj}`$ 为图像b对物体查询j的冻结视觉特征。TRAIN图像数N=22,998，查询数O=12，CP rank R=32。沿用I2的固定统计与门控：

```math
\begin{aligned}
\mu_j&=\frac{1}{N}\sum_{b\in\mathrm{TRAIN}}f^o_{bj},\qquad
\widehat r_{bj}=\frac{f^o_{bj}-\mu_j}{\max(\lVert f^o_{bj}-\mu_j\rVert_2,10^{-12})},\\[4pt]
q_{bjk}&=2\sigma\!\left(\frac{b_k^\top\widehat r_{bj}}{0.56}\right)-1,\\[4pt]
h_{bjk}&=\frac{0.25}{2}\left(q_{bjk}-\frac{1}{R}\sum_{\ell=1}^{R}q_{bj\ell}\right),\qquad
\gamma_{bjk}=1+h_{bjk}.
\end{aligned}
```

Bvis的第k行转置为 $`b_k`$。TRAIN均值使用全部图像和查询、float64累计，forward使用固定float32 buffer；J/I统计文件逐字一致。模式均值为1，门控在[0.75,1.25]内。

令 $`u_i=A_ue_i^a`$、$`v_j=A_ve_j^o`$、$`a_{ij}=u_i\odot v_j`$。I2可以精确展开，J1据此将读出分开：

```math
\begin{aligned}
\Delta U^{\mathrm{I2}}_{bji}
&=\rho\kappa g_b^\top W_c(a_{ij}\odot\gamma_{bj})\\[4pt]
&=\rho\kappa g_b^\top W_ca_{ij}
+\rho\kappa g_b^\top W_c(a_{ij}\odot h_{bj}),\\[4pt]
\Delta U^s_{bji}&=\rho\kappa g_b^\top W_sa_{ij},\\[4pt]
\Delta U^d_{bji}&=\rho\kappa g_b^\top W_d(a_{ij}\odot h_{bj}),\\[4pt]
U^{\mathrm{J1}}_{bji}&=U^{\mathrm{base}}_{bji}+\Delta U^s_{bji}+\Delta U^d_{bji}.
\end{aligned}
```

$`W_s,W_d\in\mathbb{R}^{768\times32}`$；ρ=0.15，κ=20。代码保留Wc作为Ws，`cp_readout_mode="separate"`启用新分支。$`W_s=W_d=W_c`$精确恢复I2，$`W_d=0`$恢复静态CP修正。

### 2.1 低秩性质与梯度

```math
\begin{aligned}
M_{bj}&=\left[W_s\mathrm{diag}(v_j)
+W_d\mathrm{diag}(v_j\odot h_{bj})\right]A_u,\qquad
\mathrm{rank}(M_{bj})\le R,\\[4pt]
\nabla_{W_s}\Delta U_{bji}&=\rho\kappa g_ba_{ij}^{\top},\\[4pt]
\nabla_{W_d}\Delta U_{bji}&=\rho\kappa g_b(a_{ij}\odot h_{bj})^{\top}.
\end{aligned}
```

共享属性因子Au使算子秩上界继续为32。共享Wc接收的分数梯度为两项之和；J1将两项梯度分配到各自矩阵，再由AdamW更新。首步两路读出均接收任务梯度，后续梯度进入Au/Av/Bvis。

## 3. 概率、目标函数与均值评估

沿用物体分支S、组合基底适配器、β=0.075及原损失：

```math
\begin{aligned}
s_{bji}&=\log\mathrm{softmax}(S_b)_j+
\log\mathrm{softmax}(U^{\mathrm{J1}}_{bj:})_i,\\[4pt]
Z_b&=\sum_{(i,j)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],\qquad
\mathcal{L}_{\mathrm{support}}=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_{\mathrm{task}}&=\frac{3}{2}\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}&=\mathcal{L}_{\mathrm{task}}+
\frac{0.001}{R^2}\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_R\rVert_F^2.
\end{aligned}
```

S为物体logits，$`\mathcal{S}`$为TRAIN组合集合，B为批量大小。内部variant=V1，通过配置启用动态门控；prim/cent项为零，AdamW weight decay独立执行。

native/exact各自选择首次严格最大VAL CW AUC的权重，本轮三seed两种选择均为5/6/6。每套权重使用全部TRAIN图像统计门控均值，均值条件仍使用两个不同读出：

```math
\begin{aligned}
\bar\gamma_{jk}&=\frac{1}{N}\sum_{b\in\mathrm{TRAIN}}\gamma_{bjk},\qquad
\bar h_j=\bar\gamma_j-\mathbf{1},\\[4pt]
U^{\mathrm{mean}}_{bji}&=U^{\mathrm{base}}_{bji}+
\rho\kappa g_b^\top W_sa_{ij}+
\rho\kappa g_b^\top W_d(a_{ij}\odot\bar h_j).
\end{aligned}
```

均值评估复用原前向的静态项，通过Wd计算均值动态项，再重算条件概率。每项评估保存dynamic/train_mean两条件，优化步数为零。各条件分别选择VAL CW/OW native bias并固定用于对应TEST。

## 4. 具体代码修订与验证

| 文件 | 具体修订 | 作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundJ/vgcpod/model.py) | 新增零初始化Wd、separate分支、cp_static/cp_dynamic输出 | 静态／动态读出独立学习 |
| [engine.py](code/vgcpod_v1_roundJ/vgcpod/engine.py) | 记录两路MSE、属性中心化MSE及epoch RMS | 量化训练修正幅值 |
| [mean_gate_eval.py](code/vgcpod_v1_roundJ/mean_gate_eval.py) | 静态项与Wd均值动态项分别计算 | 干预对应J1前向 |
| [J1.json](configs/round_j/J1.json) | I2上新增separate模式及J1编号 | 固定输入与优化设置 |
| [round_j.py](code/vgcpod_v1_roundJ/round_j.py) | 三卡、复用I统计、六评估、读出幅值及配对汇总 | 完整流程 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundJ/ranking_diagnostics.py) | J1−I2/G0/G1等、J1 dynamic−mean | 预测级比较 |
| [tmux脚本](code/vgcpod_v1_roundJ/scripts/start_round_j_tmux.sh) | 会话、Python路径、日志与退出码 | GPU1/2/3各一个seed |
| [J组测试](code/vgcpod_v1_roundJ/tests/test_round_j.py) | 七项新测试 | 恒等式、梯度、双路学习、干预和续训 |

两路矩阵在本轮使用同一个CP参数组，初始LR均为3e-4。详细差异见[相对10的补丁](code/source_changes.patch)，服务器字节核对见[源码身份](results/server_source_identity.json)。核心运行改动为model/engine/mean_gate_eval。

原发布验证55项测试通过；CPU合成缓存完成三训练、六评估，重复启动跳过已完成任务，checkpoint一致。旧I2配置在新旧训练器中的参数、AdamW和Torch RNG逐元素一致。记录见[发布验证](results/package_validation/J/validation/result.json)和[仓库复算核对](analysis/repository_validation.json)。

## 5. 超参数选取策略与实际结果

1. **固定I2输入表示。** 继续使用TRAIN归一化残差、τγ=0.56、η=0.25，直接比较独立读出与共享读出。
2. **固定任务与低秩设置。** β=0.075、ρ=0.15、rank32、adapter rank8及全局g读出。
3. **双路联合训练。** Ws/Wd均零初始化，三seed沿用原因子和适配器初始化。两路同CP学习率3e-4→3e-6。
4. **同seed完整比较。** GPU1/2/3分别训练3351/3352/3353，各10轮、3600次更新。I2/G0/G1为同β/ρ对照，D1作性能参照。
5. **保存双准则和均值条件。** native为主结果，exact独立保存，六项干预分别使用自己的VAL bias。

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| J1 | 3351 | 5／5 | 43.9794 | 31.0402 | 33.8136 | 23.7420 |
| J1 | 3352 | 6／6 | 41.2374 | 29.8925 | 35.9243 | 24.3884 |
| J1 | 3353 | 6／6 | 44.7703 | 33.6924 | 34.2198 | 25.4464 |

| 固定项 | 设置 |
|---|---|
| CLIP／输入缓存 | 冻结OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.56 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／两路CP LR | 1e-3→1e-5 / 3e-4→3e-6，原余弦日程 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |

模板见[round_j](configs/round_j/)，实际运行配置见[runtime](configs/runtime/)。可训练参数147,456；门控均值buffer形状[12,768]。本轮按训练前确定的配置执行，后续学习率建议保存在末节数据分析。

## 6. 仓库结构与复现入口

```text
11/
  README.md          # 公式、代码修订、超参数策略，末节数据分析
  code/              # 原J源码、相对I补丁、原分析脚本
  configs/           # 模板、运行配置、资产、历史参考和清单
  results/           # 三训练、双选择、六评估、日志与原始包
  analysis/          # 完整分析、图表、复算与网页排版核对
```

在仓库根目录恢复原运行包，复用服务器已有CLIP缓存和历史对照：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 11 \
  --dest /data/xhw/vgcpod_project/replay_11
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_11/vgcpod_v1_roundJ

export VG_J_OUT=/data/xhw/vgcpod_v1_runs/replay_11_J
export VG_J_SESSION=vgcpod_11_J
bash scripts/start_round_j_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --i-out /data/xhw/vgcpod_v1_runs/roundI_20261021
tmux attach -t vgcpod_11_J
```

其他A/C/D/E/F/G/H对照默认读取此前服务器目录；具体外部输入见[external_inputs.json](external_inputs.json)，命令见[runbook.json](runbook.json)。主日志出现`ROUND_J_COMPLETE ... training_jobs 3 mean_gate_jobs 6`且退出码为0表示完成。Ctrl+b、d脱离tmux。

在仓库根目录复算保存结果与优化器分析：

```bash
python VG-CPOD方案/11/analysis/recompute.py
python tools/reproduce.py verify
```

依赖NumPy、PyTorch、Matplotlib，输出至`analysis/recomputed/`，原分析位于`analysis/confirmed/`。materialize按[映射](package_members.json)恢复原包91个成员的字节和布局。复算沿用10中已核验的历史分析，并校验原I结果包身份；新J预测与checkpoint重新核查。

## 7. 数据分析与下一轮方向

| 方案 | VAL CW | VAL OW | TEST CW | TEST OW | 未见条件属性 |
| --- | --- | --- | --- | --- | --- |
| D1 | 43.301 | 31.564 | 34.791 | 25.453 | 24.343% |
| G0 | 43.343 | 31.615 | 34.512 | 25.184 | 24.731% |
| G1 | 43.544 | 31.675 | 34.531 | 25.057 | 24.696% |
| I2 | 43.433 | 31.787 | 34.205 | 24.853 | 24.061% |
| J1 | 43.329 | 31.542 | 34.653 | 24.526 | 24.432% |

J1相对I2的TEST CW提高0.448点，OW下降0.328点。seed3352/3353两项均提高；seed3351的OW下降2.445点，是均值下降的主要来源。未见条件属性平均提高0.370个百分点，全部未见候选top-1平均下降0.529个百分点。

TRAIN门控标准差I2/J1为0.0281/0.0269，但TEST未见图像的dynamic−mean中心化条件logit RMS由0.0653增至0.6480，约增大9.9倍。checkpoint中Wd一阶矩范数仅为静态读出的2%–3%，自适应步长范数却达到95%–100%。

下一轮方向是为Wd单独设置学习率，比较9e-5与3e-5两档；其他参数沿用J1，检验两路学习速度的配比。已有J1作为3e-4对照。本阶段已完成结果为J1，下一轮配置与推导见[完整数据分析](analysis/README.md)。

![J1与历史对照的三seed结果](analysis/confirmed/figures/01_results.png)

[数据分析README](analysis/README.md)最后集中记录配对SD、不同seed表现、候选排序、动态幅值、Adam推导及下一轮实验范围。
