# 08 · 支持集权重折中与门控训练对照

**G 组实验，VG-CPOD v1，归档日期 2026-10-20。** 使用冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 G0/G1 各三 seed、共六次训练和 60 个模型 epoch；保存 native／exact 两种选择的 48 份评价与预测，以及 G1 六项动态／TRAIN 均值替换评估的 48 份评价与预测，共 **96 份预测导出、24 份 checkpoint 文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [07 F组](../07/README.md) · [06 E组](../06/README.md)

## 1. 实验目的与修订范围

07 的 F1 使用 β=0.05、ρ=0.15。相对 E1 的 β=0.1，未见条件属性提高、组合 AUC 回落。本轮取 β=0.075，检验支持集权重的中间强度，并在同一 β、ρ 下训练静态 G0 与动态 G1，量化视觉门控模块的独立训练收益。

G1 相对 F1 只改变 `task_aux_beta` 与 `experiment_id`；G0 在 G1 配置上将 `gate_input` 从 `local` 改为 `none`，并使用自己的实验编号。全局 g 读出、ρ=0.15、rank32、适配器、CP 学习率及十轮日程沿用 F 组。

训练核心、模型与损失、CLIP 特征提取、均值评估器和 `run.py` 共十份文件与 F 包逐字节一致。新增代码集中于 G0/G1 配置、三卡双任务队列、排序诊断及同 seed 汇总。

## 2. 公式修订：支持集系数改为 9/86

张量轴为 `[batch, object, attribute]`，属性索引为 i、物体索引为 j；TRAIN 组合集合按 $`(i,j)`$ 记为 $`\mathcal{S}`$。当前链式联合分布与损失为：

```math
\begin{aligned}
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
Z_b&=\sum_{(i,j)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],\\[4pt]
\mathcal{L}_{\mathrm{support}}&=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_a+\mathcal{L}_o&=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_a+\mathcal{L}_o)]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

条件 softmax 遍历全部 16 个属性，物体 softmax 遍历全部 12 个物体；完整联合分布在 192 个组合上归一化，Z 是其中 83 个 TRAIN 组合的总概率质量。代入本轮 β：

```math
\begin{aligned}
\mathcal{L}^{\mathrm{E1}}_{\mathrm{task}}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{3}{22}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{\mathrm{G0/G1}}_{\mathrm{task}}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{\mathrm{F1}}_{\mathrm{task}}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac1{14}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

本轮支持集系数为 **0.104651163**，位于 E1 的 0.136363636 与 F1 的 0.071428571 之间；pair 项保持 1.5。反向传播继续使用原 objective 表达式；原始、加权损失及其重构误差日志沿用 F 实现。

## 3. 静态与动态修正、权重选择及诊断公式

令 $`u_i=A_ue_i^a`$、$`v_j=A_ve_j^o`$，r=32、d=768。G1 保留局部门控和模式中心化；G0 关闭视觉门控输入：

```math
\begin{aligned}
H_r&=I_r-\frac{1}{r}\mathbf{1}_r\mathbf{1}_r^\top,\\[4pt]
q_{bj}&=2\sigma(B_{\mathrm{vis}}f^o_{bj}/\tau_\gamma)-\mathbf{1}_r,\\[4pt]
\gamma^{\mathrm{G1}}_{bj}&=\mathbf{1}_r+\frac{\eta}{2}H_rq_{bj},
\qquad\eta=0.25,\quad\tau_\gamma=0.1,\\[4pt]
\gamma^{\mathrm{G0}}_{bj}&=\mathbf{1}_r,\\[4pt]
U^{\mathrm{G1}}_{bji}&=U^{\mathrm{base}}_{bji}+0.15\kappa g_b^\top W_c(u_i\odot v_j\odot\gamma^{\mathrm{G1}}_{bj}),\\[4pt]
U^{\mathrm{G0}}_{bji}&=U^{\mathrm{base}}_{bji}+0.15\kappa g_b^\top W_c(u_i\odot v_j).
\end{aligned}
```

G0 仍训练 CP 分解和适配器，Bvis 冻结；G1 训练视觉门控并沿用其正交项：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{G0}}&=\mathcal{L}_{\mathrm{task}}(0.075),\\[4pt]
\mathcal{L}_{\mathrm{G1}}&=\mathcal{L}_{\mathrm{task}}(0.075)
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_r\right\rVert_F^2.
\end{aligned}
```

两组 prim、cent 分支关闭；AdamW 的 weight decay 独立执行。G0 与 G1 的可训练参数量分别为 98,304 与 122,880。该对照比较视觉门控模块及其配套正交项的训练效果。

两种选权重依据分别保存首次严格最大 VAL CW AUC；G1 的选中权重再统计全部 TRAIN 图像的门控均值：

```math
\begin{aligned}
t_m^*&=\min\{t\in\{1,\ldots,T\}:A_t^m=\max_{1\le s\le T}A_s^m\},
\qquad m\in\{\mathrm{native},\mathrm{exact}\},\\[4pt]
\bar\gamma_{jk}&=\frac{1}{N_{\mathrm{train}}}\sum_b\gamma_{bjk},
\qquad\gamma^{\mathrm{mean}}_{bjk}=\bar\gamma_{jk}.
\end{aligned}
```

本轮 seed3353 的 native／exact 分别选中第 9／6 轮，因此两套结果独立保存、各自按 VAL CW／OW native bias 评价 TEST。

新增诊断从保存的完整 OW 分布读取。令 $`\mathcal{P}=\mathcal{A}\times\mathcal{O}`$、$`\mathcal{C}_h=\mathcal{S}\cup\mathcal{Y}_h`$，$`\mathcal{E}_h=\mathcal{P}\setminus\mathcal{C}_h`$：

```math
\begin{aligned}
M_b^{\mathrm{seen}}&=\sum_{(i,j)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
M_b^{\mathrm{extra}}&=\sum_{(i,j)\in\mathcal{E}_h}\exp(s_{bji}),\\[4pt]
D_b^{\mathrm{extra}}&=\mathbf{1}\!\left[
\max_{(i,j)\in\mathcal{E}_h}s_{bji}
>\max_{(i,j)\in\mathcal{C}_h\setminus\mathcal{S}}s_{bji}\right].
\end{aligned}
```

程序输出概率质量、CW 集合外最高分超越比例、全部未见候选内真实组合 top-1，以及物体／GT-object 条件属性的准确率与 NLL；均按 all/seen/unseen 图像分组。这些诊断描述原预测，不改变模型分数。

## 4. 具体代码修订与验证

| 文件 | 本轮具体改动 | 作用 |
|---|---|---|
| [G1.json](configs/round_g/G1.json) | β=0.075、experiment_id=G1 | 支持集权重中间点 |
| [G0.json](configs/round_g/G0.json) | G1 配置的 gate_input 改为 none | 相同目标下的静态训练对照 |
| [round_g.py](code/vgcpod_v1_roundG/round_g.py) | 每卡 G0→G1 队列、续训、六项均值评估、两组与历史配对汇总 | 六项训练的完整编排 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundG/ranking_diagnostics.py) | `diagnose()`、逐组合统计、同 seed 诊断差值及 SD | 连接属性、概率质量与组合排序 |
| [tmux启动](code/vgcpod_v1_roundG/scripts/start_round_g_tmux.sh) | G 组会话、输出目录、日志和退出码 | 三 GPU 各负责一个 seed |
| [G组测试](code/vgcpod_v1_roundG/tests/test_round_g.py) | 配置范围、静态极限、β=0.075 梯度、排序诊断 | 累计39项发布测试 |
| [pre_g_complete.json](configs/references/pre_g_complete.json) | 20份已有训练运行的指标快照 | E1/F1 与历史同 seed 对照 |

[相对07的源码补丁](code/source_changes.patch)记录实际文件差异。模型、objective、优化器与 CLIP 缓存读取沿用 F 包；[服务器身份核对](results/server_source_identity.json)与[原包76成员映射](package_members.json)保留源字节身份。

[原发布验证](results/package_validation/G/validation/result.json)跑通六项训练、六项评估及重复启动复用。与原 F 训练器使用相同 G0/G1 配置比较，native／exact／last 的模型参数、AdamW 状态及 Torch RNG 逐元素一致。新增诊断在 E1/F1 真实预测上复算 36 行，与独立 F 分析的最大差异小于 1e-8。

## 5. 超参数选取策略与实际矩阵

1. **β 中间点。** 在固定 ρ=0.15 下，已有 β=0.1 与 0.05 的直接对照；本轮取 β=0.075，系数按 1.5β/(1+β) 精确计算。
2. **补同配置模块对照。** G0/G1 使用相同 β、ρ、读出、rank 和优化日程，区别为静态／视觉门控模块及动态分支的正交项。
3. **保持十轮和三 seed。** 两组均从各自原始初始化训练，seed3351/3352/3353 分别在 GPU1/2/3 顺序运行 G0→G1，各3600次更新。
4. **同 seed 配对。** G1−E1/F1 测量目标权重变化，G1−G0 测量门控模块，G1 dynamic−mean 测量已训练权重对图像变化的依赖。
5. **保留原选模规则。** 主结果取首次严格最大 VAL CW native AUC；exact 选中结果独立保存，各条件使用自己的 VAL native bias。

| 实验 | seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| G0 | 3351 | 8／8 | 43.9071 | 31.0988 | 34.1754 | 26.2127 |
| G0 | 3352 | 8／8 | 41.6325 | 29.6697 | 35.2053 | 23.7875 |
| G0 | 3353 | 9／6 | 44.4907 | 34.0762 | 34.1545 | 25.5505 |
| G1 | 3351 | 8／8 | 44.0321 | 31.0817 | 34.2140 | 26.1916 |
| G1 | 3352 | 8／8 | 41.8392 | 29.7950 | 35.1813 | 23.5737 |
| G1 | 3353 | 9／6 | 44.7601 | 34.1473 | 34.1965 | 25.4049 |

主表为 native 选择的 native AUC，面积×100。G0/G1 的 native 选中轮次均为8/8/9，exact 均为8/8/6；两种权重完整指标见[六项训练清单](configs/experiment_inventory.json)。

| 固定项 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 ViT-L/14，224px、768维，同一缓存 |
| ρ／β／η／τγ | 0.15 / **0.075** / 0.25 / 0.1 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |
| eval batch／workers | 64 / 4 |

内部 `variant=V1`；通过 `gate_input=none/local` 区分静态和动态。G0 的 CP 组为 Au/Av/Wc，G1 增加 Bvis；`cp_lr_scale=0.3`，Wc 零初始化。

## 6. 仓库结构与复现入口

```text
08/
  README.md          # 修订、公式、超参策略与末节数据分析
  code/              # 原G代码、相对F补丁、原分析源码
  configs/           # G0/G1、兼容配置、资产、参考与运行清单
  results/           # 六次训练、双选择、均值评估、日志及原归档
  analysis/          # 数据分析、图表、静态等价验证与仓库复算
```

恢复原运行包并复用服务器现有 CLIP 缓存：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 08 \
  --dest /data/xhw/vgcpod_project/replay_08
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_08/vgcpod_v1_roundG

export VG_G_OUT=/data/xhw/vgcpod_v1_runs/replay_08_G
export VG_G_SESSION=vgcpod_08_G
bash scripts/start_round_g_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020 \
  --f-out /data/xhw/vgcpod_v1_runs/roundF_20261020
tmux attach -t vgcpod_08_G
```

每卡顺序完成 G0/G1 及本 seed 的 G1 两项均值评估。按 Ctrl+b、d 脱离 tmux。`launcher.exitcode=0` 且日志出现 `ROUND_G_COMPLETE ... training_jobs 6 mean_gate_jobs 6` 表示整轮完成。外部输入见 [external_inputs.json](external_inputs.json)，命令见 [runbook.json](runbook.json)。

从仓库保存的预测、checkpoint 与缓存元数据复算结果及均值门控的静态等价关系：

```bash
python VG-CPOD方案/08/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`，原分析保存在 `analysis/confirmed/`。materialize 按映射恢复原包布局与字节。

## 7. 数据分析与下一轮方向

| 三 seed 方案 | ρ | β | VAL CW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | 0.125 | 0.1 | 43.301 | 34.791 | 25.453 | 24.343% |
| E1 | 0.15 | 0.1 | 43.554 | 34.427 | 25.180 | 23.374% |
| F1 | 0.15 | 0.05 | 43.505 | 34.030 | 24.351 | 25.189% |
| G0 | 0.15 | 0.075 | 43.343 | 34.512 | 25.184 | 24.731% |
| G1 | 0.15 | 0.075 | 43.544 | 34.531 | 25.057 | 24.696% |

- **β=0.075 回收 F1 的排序损失并保留部分属性收益。** G1−F1 的 TEST CW/OW AUC 为 +0.501±0.278／+0.705±0.029 点；G1−E1 为 +0.103／−0.124 点，未见条件属性 +1.322 个百分点。
- **门控模块的 TEST 排序收益较小。** G1−G0 的 native 选择结果为 CW +0.019±0.037、OW −0.127±0.098 点；相同权重的 exact AUC 为 −0.019／−0.138 点。两组 native 轮次逐 seed 对齐。
- **seed3353 首次出现两种选择分歧。** G0/G1 的 native 选择 epoch9，exact 选择 epoch6；G1 exact 选中结果相对 native 的 TEST CW/OW native AUC 平均低0.193／0.215点。
- **门控方差约98%为固定物体／模式成分。** G1 native 的图像变化占比为2.02%／1.95%／1.83%；dynamic−mean 的 TEST CW/OW AUC 为 +0.135±0.179／−0.037±0.187点。
- **固定均值门控可吸收进静态 CP 物体因子。** 12×768 物体文本矩阵满行秩；六套选中权重的数值验证中，折叠前后的修正 logits 误差小于1e-14。
- **下一轮建议 H1：用相对 TRAIN 均值的局部视觉残差驱动门控。** 固定 β=0.075、ρ=0.15、η=0.25，检验图像相关成分；H1 为后续表示设计，本阶段实际归档为已完成的 G0/G1。

![同配置静态与动态训练对照](analysis/confirmed/figures/02_matched_static_dynamic.png)

完整分组、双选择、逐组合变化、门控方差与静态等价推导，以及 H1 的公式和实践安排，见[数据分析 README](analysis/README.md)。
