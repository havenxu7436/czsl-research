# 10 · 残差幅值保留与门控温度对照

**I 组实验，VG-CPOD v1，归档日期 2026-10-21。** 框架为冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 I1/I2 各三个 seed，共六次训练、60 个模型 epoch；保存 native／exact 选择的 48 份预测，以及十二项动态／TRAIN 均值评估的 96 份预测，共 **144 份预测文件、24 份 checkpoint 文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [09 H组](../09/README.md) · [08 G组](../08/README.md)

## 1. 实验目的与修订范围

09 的 H1 在 TRAIN 去均值后将残差单位化。相对 G1，H1 的 TEST CW／OW AUC 下降 0.902／1.186 点；图像门控标准差约为 G1 的 5.8 倍。I 组通过两项对照检验残差幅值和投影尺度：I1 保留原始残差幅值，I2 保留单位方向并将门控温度从 0.1 调整为 0.56。

| 方案 | gate_input | 门控输入 | τγ | 相对H1变化 |
|---|---|---|---|---|
| H1：已有对照 | train_residual | 归一化TRAIN残差 | 0.1 | 复用09结果 |
| I1：幅值保留 | train_centered | 原始TRAIN残差 | 0.1 | 去掉逐向量L2归一化 |
| I2：温度对照 | train_residual | 归一化TRAIN残差 | 0.56 | 仅调整门控温度 |

两组沿用 β=0.075、ρ=0.15、η=0.25、全局 g 读出和原优化设置，可训练参数均为 **122,880**。六次训练复用 H 组的同一 TRAIN 均值文件；新增输入分支不增加模型参数。

## 2. 公式修订：残差幅值与温度尺度

令 $`f^o_{bj}\in\mathbb{R}^{768}`$ 为图像 b 对物体查询 j 的冻结 CLIP 特征；TRAIN 图像数 N=22,998，查询数 O=12，模式数 R=32。对全部 TRAIN 图像和全部查询计算：

```math
\begin{aligned}
\mu_j&=\frac1N\sum_{b\in\mathrm{TRAIN}}f^o_{bj},
\qquad r_{bj}=f^o_{bj}-\mu_j,\\[4pt]
x^{\mathrm{I1}}_{bj}&=r_{bj},
\qquad\tau^{\mathrm{I1}}_\gamma=0.1,\\[4pt]
x^{\mathrm{I2}}_{bj}&=\frac{r_{bj}}{\max(\lVert r_{bj}\rVert_2,10^{-12})},
\qquad\tau^{\mathrm{I2}}_\gamma=0.56.
\end{aligned}
```

均值计算不使用标签。统计以 float64 累计，forward 使用固定 float32 buffer `gate_fo_mean`；VAL／TEST、三 seed 及两条选权重路径均使用该 TRAIN 均值。I 组统计文件与 H 组逐字一致。

令 $`B_{\mathrm{vis}}\in\mathbb{R}^{R\times768}`$ 的第 k 行转置为 $`b_k`$，对 $`c\in\{\mathrm{I1},\mathrm{I2}\}`$，沿用模式中心化门控：

```math
\begin{aligned}
z^c_{bjk}&=b_k^\top x^c_{bj}/\tau^c_\gamma,\\[4pt]
q^c_{bjk}&=2\sigma(z^c_{bjk})-1,\\[4pt]
\gamma^c_{bjk}&=1+\frac{\eta}{2}\left(q^c_{bjk}-\frac1R\sum_{\ell=1}^R q^c_{bj\ell}\right),
\qquad\eta=0.25,\\[4pt]
\frac1R\sum_k\gamma^c_{bjk}&=1,
\qquad0.75\le\gamma^c_{bjk}\le1.25.
\end{aligned}
```

门控对视觉基的梯度为：

```math
\nabla_{b_\ell}\gamma^c_{bjk}
=\frac{\eta}{\tau^c_\gamma}\left(\delta_{k\ell}-\frac1R\right)
\sigma(z^c_{bj\ell})[1-\sigma(z^c_{bj\ell})]x^c_{bj}.
```

两组残差为零时，门控均为 1。I1 对幅值变化保留响应；I2 对超过 ε 的正比例残差使用相同单位方向。

### 2.1 温度0.56的选取依据

每个查询的中心化二阶矩满足精确恒等式：

```math
\frac1N\sum_b\lVert r_{bj}\rVert_2^2
=\frac1N\sum_b\lVert f^o_{bj}\rVert_2^2-\lVert\mu_j\rVert_2^2.
```

用单位范数 CLIP 缓存特征估算合并残差 RMS：

```math
\begin{aligned}
s_{\mathrm{TRAIN}}&\approx
\sqrt{\frac1O\sum_{j=1}^O(1-\lVert\mu_j\rVert_2^2)}
\approx0.17835,\\[4pt]
\tau^{\mathrm{I2}}_\gamma&\approx\frac{0.1}{s_{\mathrm{TRAIN}}}\approx0.56.
\end{aligned}
```

0.56 是本轮训练前由 TRAIN 代表尺度确定的固定值。若某个残差范数为 0.1/0.56，则同一 Bvis 下，两组 sigmoid 输入精确相等；真实残差范数随图像和查询变化，I1/I2 因而构成代表尺度下的幅值／方向对照。

## 3. CP读出、目标函数与评价

令 $`u_i=A_u e_i^a`$、$`v_j=A_v e_j^o`$，$`W_c\in\mathbb{R}^{768\times32}`$，张量顺序为 `[batch, object, attribute]`。两组沿用全局 CLIP 特征 g 的读出：

```math
\begin{aligned}
U^c_{bji}&=U^{\mathrm{base}}_{bji}
+\rho\kappa g_b^\top W_c(u_i\odot v_j\odot\gamma^c_{bj}),
\qquad\rho=0.15,\ \kappa=20,\\[4pt]
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
\mathcal{L}_{\mathrm{task}}(\beta)&=\frac{3}{2(1+\beta)}
\left[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_{\mathrm{cond}}+\mathcal{L}_{\mathrm{obj}})\right],\\[4pt]
\mathcal{L}_{\mathrm{cond}}+\mathcal{L}_{\mathrm{obj}}
&=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{c}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}
+\frac{0.001}{R^2}\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_R\rVert_F^2.
\end{aligned}
```

β=0.075；pair 为 TRAIN 支持集内重归一化后的交叉熵，support 为 TRAIN 组合概率质量的负对数，具体定义见 [08](../08/README.md)。内部 variant=V1，prim／cent 分支为零。Wc 零初始化，AdamW weight decay 独立执行。

native／exact 各自选择首次最大 VAL CW AUC 的 epoch。每套选中权重在 TRAIN 上计算门控均值：

```math
\begin{aligned}
t_m^*&=\min\{t\in\{1,\ldots,10\}:A_t^m=\max_{1\le s\le10}A_s^m\},
\qquad m\in\{\mathrm{native},\mathrm{exact}\},\\[4pt]
\bar\gamma_{jk}&=\frac1N\sum_{b\in\mathrm{TRAIN}}\gamma_{bjk},
\qquad\gamma^{\mathrm{mean}}_{bjk}=\bar\gamma_{jk}.
\end{aligned}
```

十二项均值评估各保存 dynamic／train_mean 两种条件；优化步数为零。各权重和条件分别从 VAL CW／OW 取得 native bias，并固定用于对应 TEST。I1 的 seed3351 选中第8／10轮，其余 I1 与全部 I2 的两套权重 epoch 相同。

## 4. 具体代码修订与验证

| 文件 | 具体修订 | 实验作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundI/vgcpod/model.py) | 新增 `train_centered` 和原幅值 `fo - mean` 分支 | I1去掉逐向量单位化 |
| [engine.py](code/vgcpod_v1_roundI/vgcpod/engine.py) | 两种残差输入均加载固定TRAIN均值 | 训练构造器支持I1/I2 |
| [gate_features.py](code/vgcpod_v1_roundI/vgcpod/gate_features.py) | 将H1专用提示改为TRAIN中心化通用提示 | 均值加载计算沿用H版本 |
| [I1](configs/round_i/I1.json)／[I2](configs/round_i/I2.json) | I1改输入，I2改温度；分别命名 | 其他训练超参同H1 |
| [round_i.py](code/vgcpod_v1_roundI/round_i.py) | 一次统计／复用均值、三卡各I1→I2、续训、十二项评估 | 六训练及同seed配对汇总 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundI/ranking_diagnostics.py) | 纳入I1/I2−H1/G0/G1、I1−I2及dynamic−mean | 概率质量与排序诊断 |
| [历史参考](configs/references/pre_i_complete.json) | 29项历史native及H1/G0/G1的exact指标 | 使用历史已完成结果 |
| [tmux启动](code/vgcpod_v1_roundI/scripts/start_round_i_tmux.sh)／[I组测试](code/vgcpod_v1_roundI/tests/test_round_i.py) | 三卡队列与四项新增测试 | 累计48项发布测试 |

相对09的计算修订位于 model／engine；gate_features 仅修改提示文本，均值加载计算沿用原实现。mean_gate_eval.py 及其 TRAIN 均值替换路径逐字沿用 H 版本。[完整源码补丁](code/source_changes.patch)和[服务器源码身份](results/server_source_identity.json)保留变化范围与原字节核对。

原发布验证完成48项测试、六项CPU合成训练、十二项均值评估和重复启动复用。测试覆盖 I1 的显式CP／门控、有限差分梯度、代表尺度恒等式、幅值响应、固定buffer与续训。旧H1路径的新旧模型参数、AdamW与Torch RNG逐元素一致。[原验证](results/package_validation/I/validation/result.json)与[仓库复算](analysis/recompute.py)分别保存发布流程和正式数据核查。

## 5. 超参数选取策略与实际结果

1. **从TRAIN几何确定温度。** I2使用τγ=0.56，对应残差RMS约0.17835；该值在本轮训练前确定。
2. **将幅值和尺度分开比较。** I1−H1检验去掉单位化，I2−H1检验温度变化；I1−I2比较代表尺度接近时的幅值／方向处理。
3. **沿用目标和CP设置。** β=0.075、ρ=0.15、η=0.25、rank32、全局g读出，适配器与CP学习率保持原日程。
4. **三seed从原始初始化训练。** GPU1/2/3分别跑3351/3352/3353；每卡先I1、再I2，各10轮、3600次更新。
5. **比较完整训练与推理干预。** I1/I2−G0/G1为独立训练方案比较，dynamic−mean为同一权重的图像变化贡献。
6. **保存双准则选择。** 主表继续使用VAL CW native AUC；exact结果独立保存，对应条件使用自己的VAL bias。

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| I1 | 3351 | 8／10 | 43.6373 | 31.1230 | 33.9122 | 26.1414 |
| I1 | 3352 | 8／8 | 41.7728 | 29.8443 | 35.1903 | 23.6263 |
| I1 | 3353 | 6／6 | 44.8723 | 34.5470 | 33.3681 | 24.5984 |
| I2 | 3351 | 8／8 | 43.7343 | 31.1052 | 33.9697 | 26.1872 |
| I2 | 3352 | 8／8 | 41.6859 | 29.7517 | 35.1811 | 23.7244 |
| I2 | 3353 | 6／6 | 44.8780 | 34.5049 | 33.4632 | 24.6480 |

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结OpenAI ViT-L/14，224px、768维；原特征缓存 |
| ρ／β／η | 0.15 / 0.075 / 0.25 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，原余弦日程 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |

CP优化组为Au/Av/Wc/Bvis，`cp_lr_scale=0.3`；`gate_fo_mean`为固定buffer。模板与服务器实际配置分别保存在[round_i](configs/round_i/)与[runtime](configs/runtime/)。

## 6. 仓库结构与复现入口

```text
10/
  README.md          # 修订说明、公式、超参策略，末节数据分析
  code/              # I原源码、相对H补丁与原分析源码
  configs/           # I1/I2模板、运行配置、资产、参考与清单
  results/           # 六训练、双选择、十二项评估、均值、日志与原归档
  analysis/          # 数据分析、图表、预测复算和归档核对
```

在仓库根目录恢复原运行包，复用服务器已有CLIP缓存及历史对照：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 10 \
  --dest /data/xhw/vgcpod_project/replay_10
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_10/vgcpod_v1_roundI

export VG_I_OUT=/data/xhw/vgcpod_v1_runs/replay_10_I
export VG_I_SESSION=vgcpod_10_I
bash scripts/start_round_i_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020 \
  --f-out /data/xhw/vgcpod_v1_runs/roundF_20261020 \
  --g-out /data/xhw/vgcpod_v1_runs/roundG_20261020 \
  --h-out /data/xhw/vgcpod_v1_runs/roundH_20261020

tmux attach -t vgcpod_10_I
```

脚本先复用／生成一次共享TRAIN均值，再生成I1/I2运行配置并启动三卡队列。Ctrl+b、d脱离tmux。`launcher.exitcode=0`且日志出现`ROUND_I_COMPLETE ... training_jobs 6 mean_gate_jobs 12`表示整轮完成。输入见[external_inputs.json](external_inputs.json)，命令见[runbook.json](runbook.json)。

在仓库根目录复算保存预测的指标、配对差值和图表：

```bash
python VG-CPOD方案/10/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖NumPy、PyTorch、Matplotlib，输出至`analysis/recomputed/`；原分析保存在`analysis/confirmed/`。materialize按[映射](package_members.json)恢复原包85个成员的布局与字节。

## 7. 数据分析与下一轮方向

| 方案 | VAL CW | VAL OW | TEST CW | TEST OW | 未见条件属性 |
| --- | --- | --- | --- | --- | --- |
| D1 | 43.301 | 31.564 | 34.791 | 25.453 | 24.343% |
| G0 | 43.343 | 31.615 | 34.512 | 25.184 | 24.731% |
| G1 | 43.544 | 31.675 | 34.531 | 25.057 | 24.696% |
| H1 | 43.519 | 32.031 | 33.629 | 23.870 | 23.321% |
| I1 | 43.427 | 31.838 | 34.157 | 24.789 | 23.973% |
| I2 | 43.433 | 31.787 | 34.205 | 24.853 | 24.061% |

- **I1/I2收回H1的主要退化。** 相对H1，TEST CW／OW分别回升0.528／0.918和0.576／0.983点，三个seed均改善。
- **两种残差处理结果接近。** I2比I1高0.048／0.064点，TEST未见条件属性预测一致率99.63%–99.84%；exact选择后CW排名翻转。
- **输入尺度回落，图像变化保留。** TRAIN图像门控标准差从H1的0.06636降至I1/I2的0.02748／0.02807，图像方差占比仍接近100%。
- **完整结果仍低于静态G0。** I2−G0的TEST CW／OW为−0.307／−0.330点；I2 dynamic−mean为+0.428／+0.098点。条件属性与未见组合排序仍有差距。
- **下一轮建议J1检验读出共享约束。** 沿用I2的τγ=0.56、η=0.25，分别学习静态／动态CP读出，属性／物体因子共享；三seed各10轮。J1为下一轮计划，本阶段正式结果为I1/I2。

![I组与历史对照及双准则选择](analysis/confirmed/figures/01_results_and_selection.png)

完整配对SD、逐seed差值、动态／均值分析、公式核查及J1的推导见[数据分析README](analysis/README.md)。D1继续作为已有性能参照，G0/G1为同β／ρ对照。
