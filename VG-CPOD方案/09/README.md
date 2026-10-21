# 09 · 训练均值中心化与归一化残差门控

**H 组实验，VG-CPOD v1，归档日期 2026-10-21。** 使用冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 H1 三个 seed、三次训练和 30 个模型 epoch；保存 native／exact 两种选择的 24 份预测，以及六项动态／TRAIN 均值替换评估的 48 份预测，共 **72 份预测文件、12 份 checkpoint 文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [08 G组](../08/README.md) · [07 F组](../07/README.md)

## 1. 实验目的与修订范围

08 的 G1 使用物体查询特征直接驱动门控；选中权重的 TRAIN 门控总方差中，图像变化约占 2%，其余主要为固定物体／模式差异。本轮先统计每个查询在全部 TRAIN 图像上的均值，再把归一化视觉残差输入门控，检验图像变化对属性与组合识别的贡献。

H1 相对 G1 将 `gate_input=local` 改为 `train_residual`，新增 `gate_residual_eps=1e-12` 和固定均值 buffer，实验编号为 H1。β=0.075、ρ=0.15、η=0.25、全局 g 读出、rank32、适配器和 CP 学习率均沿用 G1。三 seed 共用同一 TRAIN 统计资产，模型可训练参数仍为 **122,880**；新增 buffer 为 12×768 个 float32 数。

## 2. 公式修订：TRAIN 查询均值与残差门控

令 $`f^o_{bj}\in\mathbb{R}^{768}`$ 为冻结 CLIP 的物体查询视觉特征，j 遍历 12 个查询，b 遍历 22,998 张 TRAIN 图像。均值对全部图像、全部查询计算，不读取图像标签：

```math
\begin{aligned}
\mu_j&=\frac{1}{N_{\mathrm{train}}}\sum_{b\in\mathrm{TRAIN}}f^o_{bj},\\[4pt]
r_{bj}&=f^o_{bj}-\mu_j,\\[4pt]
\widehat r_{bj}&=\frac{r_{bj}}{\max(\lVert r_{bj}\rVert_2,\epsilon)},
\qquad\epsilon=10^{-12}.
\end{aligned}
```

统计代码以 float64 分块累计，保存 float64 均值及 float32 副本；模型 forward 使用固定 float32 buffer `gate_fo_mean`。VAL／TEST 和两个权重选择轨道使用同一 TRAIN 均值。数学上，TRAIN 原始残差均值为零；逐向量单位化后的方向均值由数据决定。

令模式数 R=32，$`B_{\mathrm{vis}}\in\mathbb{R}^{R\times768}`$，其第 k 行转置为 $`b_k`$。H1 保留原模式中心化：

```math
\begin{aligned}
z_{bjk}&=b_k^\top\widehat r_{bj}/\tau_\gamma,
\qquad \tau_\gamma=0.1,\\[4pt]
q_{bjk}&=2\sigma(z_{bjk})-1,\\[4pt]
\gamma_{bjk}&=1+\frac{\eta}{2}\left(q_{bjk}-\frac1R\sum_{\ell=1}^Rq_{bj\ell}\right),
\qquad\eta=0.25,\\[4pt]
\frac1R\sum_k\gamma_{bjk}&=1,
\qquad 0.75\le\gamma_{bjk}\le1.25.
\end{aligned}
```

残差为零时，单位化结果为零、所有门控模式均为 1。中心化门控的视觉基梯度为：

```math
\nabla_{b_\ell}\gamma_{bjk}
=\frac{\eta}{\tau_\gamma}\left(\delta_{k\ell}-\frac1R\right)
\sigma(z_{bj\ell})[1-\sigma(z_{bj\ell})]\widehat r_{bj}.
```

## 3. CP 读出、训练目标与评价公式

张量轴为 `[batch, object, attribute]`。冻结文本特征定义 $`u_i=A_ue_i^a`$、$`v_j=A_ve_j^o`$，$`W_c\in\mathbb{R}^{768\times R}`$。CP 仍由全局图像特征 g 读出：

```math
\begin{aligned}
U^{\mathrm{H1}}_{bji}&=U^{\mathrm{base}}_{bji}
+0.15\kappa g_b^\top W_c(u_i\odot v_j\odot\gamma_{bj}),\\[4pt]
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
\mathcal{L}_{\mathrm{task}}(0.075)&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{\mathrm{H1}}&=\mathcal{L}_{\mathrm{task}}(0.075)
+\frac{0.001}{R^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_R\right\rVert_F^2.
\end{aligned}
```

κ=20；pair/support 依照 [08 的损失定义](../08/README.md)，support 是 TRAIN 组合概率质量的负对数。内部 variant=V1，prim／cent 分支为零；AdamW weight decay 独立执行。Wc 零初始化，初始 CP 修正为零。固定均值不进入优化器。

native／exact 两种权重分别取首次最大 VAL CW AUC，并对每种权重统计 TRAIN 门控均值：

```math
\begin{aligned}
t_m^*&=\min\{t\in\{1,\ldots,T\}:A_t^m=\max_{1\le s\le T}A_s^m\},
\qquad m\in\{\mathrm{native},\mathrm{exact}\},\\[4pt]
\bar\gamma_{jk}&=\frac1{N_{\mathrm{train}}}\sum_b\gamma_{bjk},
\qquad \gamma^{\mathrm{mean}}_{bjk}=\bar\gamma_{jk}.
\end{aligned}
```

本轮两个选择轨道均选中 5／8／6 epoch，各自保存权重和导出。动态与均值条件各自选取 VAL CW／OW 的 native bias，再固定用于 TEST；均值干预优化步数为零。

## 4. 具体代码修订与验证

| 文件 | 具体改动 | 作用 |
|---|---|---|
| [gate_features.py](code/vgcpod_v1_roundH/vgcpod/gate_features.py) | `prepare_gate_statistics()` 和 `load_gate_mean()` | 分块统计、复用、身份检查与加载 |
| [model.py](code/vgcpod_v1_roundH/vgcpod/model.py) | 固定 `gate_fo_mean` buffer；`train_residual` 分支 | 残差与 L2 单位化后进入原门控 |
| [engine.py](code/vgcpod_v1_roundH/vgcpod/engine.py) | 构建模型加载均值；`evaluate_run()` 从 checkpoint 恢复 | 训练与离线评价共享同一均值 |
| [mean_gate_eval.py](code/vgcpod_v1_roundH/mean_gate_eval.py) | 从 checkpoint 模型状态传入固定均值 | 对 H1 的两套权重做均值替换 |
| [H1.json](configs/round_h/H1.json)／[运行配置](configs/runtime/H1.json) | 模板与服务器生成配置分开保存 | 运行配置补充均值路径及 SHA256 |
| [round_h.py](code/vgcpod_v1_roundH/round_h.py) | 先统计一次，再三卡训练与六项评估 | H1−G0/G1、双选择及逐组合汇总 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundH/ranking_diagnostics.py) | H1 及 exact 历史对照、dynamic−mean | 保存分组排序量与同 seed 差值 |
| [tmux启动](code/vgcpod_v1_roundH/scripts/start_round_h_tmux.sh)／[H组测试](code/vgcpod_v1_roundH/tests/test_round_h.py) | 会话、日志与退出码；五项新增测试 | 累计 44 项发布测试 |

[相对08的源码补丁](code/source_changes.patch)保留完整差异。[源码身份](results/server_source_identity.json)逐文件核对服务器源字节；均值资产、12 份 checkpoint 和全部预测进入本轮结果目录。

原发布验证完成 44 项测试、三项 CPU 合成训练、六项均值评估及重复启动复用。相同 G1 配置在新旧核心上的模型参数、AdamW 状态和 Torch RNG 逐元素一致；H1 续训与固定 buffer 恢复通过核对。[原验证明细](results/package_validation/H/validation/result.json)、[测试日志](results/package_validation/H/validation/unit_tests.txt)、[仓库复算](analysis/recompute.py)。

## 5. 超参数选取策略与实际矩阵

1. **围绕门控输入做单项表示比较。** H1 延续 G1 的 β、ρ、η、CP 读出与优化设置，变化集中于 TRAIN 中心化和残差单位化。
2. **均值来自固定 TRAIN 缓存。** 三 seed 共用 22,998 张图像、12 个查询的均值；float64 累计、float32 buffer，统计身份随配置与 checkpoint 保存。
3. **保持原门控强度。** η=0.25、τγ=0.1、模式中心化保留，门控范围沿用 [0.75,1.25]；ε=1e-12 定义零范数分支。
4. **保留十轮与三 seed 配对。** seed3351/3352/3353 分别在物理 GPU1/2/3，从各自原始初始化训练，每次 3600 次更新。
5. **分开读取三类比较。** H1−G1 测输入变换，H1−G0 测相对静态模型的训练效果，H1 dynamic−mean 测同一权重对图像变化的依赖。
6. **沿用原选权重规则。** native 为主表，exact 独立归档；各条件使用自己的 VAL native bias。

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | 3351 | 5／5 | 43.7328 | 31.6663 | 33.1104 | 24.3121 |
| H1 | 3352 | 8／8 | 41.7932 | 29.9935 | 34.9722 | 22.9596 |
| H1 | 3353 | 6／6 | 45.0315 | 34.4327 | 32.8046 | 24.3395 |

| 固定项目 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 OpenAI ViT-L/14，224px、768维，复用原缓存 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.1 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，原余弦日程 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |

CP 优化组为 Au/Av/Wc/Bvis，`cp_lr_scale=0.3`。`gate_fo_mean` 是固定 buffer，未增加可训练参数。

## 6. 仓库结构与复现入口

```text
09/
  README.md          # 实验修订、公式、超参策略，末节数据分析
  code/              # H原代码、相对G补丁、原分析源码
  configs/           # H1模板、服务器运行配置、资产、参考与清单
  results/           # 三seed、双选择、六项干预、TRAIN均值、日志及原归档
  analysis/          # 数据分析、图表、保存预测复算与归档核对
```

恢复原运行包，复用服务器现有 CLIP 缓存：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 09 \
  --dest /data/xhw/vgcpod_project/replay_09
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_09/vgcpod_v1_roundH

export VG_H_OUT=/data/xhw/vgcpod_v1_runs/replay_09_H
export VG_H_SESSION=vgcpod_09_H
bash scripts/start_round_h_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020 \
  --f-out /data/xhw/vgcpod_v1_runs/roundF_20261020 \
  --g-out /data/xhw/vgcpod_v1_runs/roundG_20261020

tmux attach -t vgcpod_09_H
```

脚本先计算并保存共享 TRAIN 均值，再生成输出目录下的 `configs/H1.json`，随后三个 seed 并行训练及评价。按 Ctrl+b、d 脱离 tmux。`launcher.exitcode=0` 且日志出现 `ROUND_H_COMPLETE ... training_jobs 3 mean_gate_jobs 6` 表示整轮完成。外部输入见 [external_inputs.json](external_inputs.json)，命令见 [runbook.json](runbook.json)。

从仓库保存的预测、checkpoint、均值和缓存元数据复算指标与图表：

```bash
python VG-CPOD方案/09/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`，原分析保存在 `analysis/confirmed/`。materialize 按映射恢复原包 80 个成员的布局与字节。

## 7. 数据分析与下一轮方向

| 三 seed 方案 | VAL CW AUC | VAL OW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- | --- |
| G0 | 43.343 | 31.615 | 34.512 | 25.184 | 24.731% |
| G1 | 43.544 | 31.675 | 34.531 | 25.057 | 24.696% |
| H1 | 43.519 | 32.031 | 33.629 | 23.870 | 23.321% |

- **完整 H1 的 TEST 排序下降。** H1−G1 的 TEST CW／OW AUC 为 −0.902±0.617／−1.186±0.641 点，三个 seed 两项均下降；exact 选权重结果为 −0.708／−0.971 点。
- **图像变化的推理贡献增强。** H1 dynamic−mean 的 TEST CW／OW AUC 为 +2.146±0.486／+1.516±0.630 点，未见条件属性 +1.340 个百分点；该量与独立训练 G0/G1 的总效果分别报告。
- **TRAIN 门控变化几乎全部转向图像维度。** 图像方差占比约 99.9%，图像标准差约为 G1 的 5.8 倍；代表残差 RMS 约 0.17–0.19，单位化对应代表放大倍数约 5.4–5.9。
- **主要下降位于条件属性和未见组合排序。** H1−G1 的 TEST 未见条件属性下降 1.375 个百分点、NLL 增加 0.0545，全部未见候选内真实组合 top-1 下降 1.851 个百分点；物体准确率下降 0.053 个百分点。
- **后续建议 I1/I2 拆分幅值因素。** I1 保留 TRAIN 去均值并去掉残差 L2；I2 保留归一化残差、τγ=0.56，作为 TRAIN 代表尺度对照。两组固定 η=0.25、ρ=0.15、β=0.075，各三个 seed。它们为下一轮方案，本阶段实际结果为已完成 H1。

![同配置静态、原门控与残差门控](analysis/confirmed/figures/01_matched_results.png)

完整逐 seed 差值、动态／均值干预、双选择、残差尺度推导与下一轮计划，见[数据分析 README](analysis/README.md)。
