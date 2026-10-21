# 05 · 中心化 CP 幅度联动与均值门控评估

**D 组实验与三 seed 补充轮，VG-CPOD v1，归档日期 2026-10-20。** 框架为冻结 CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增归档 6 次训练、60 个模型 epoch、24 份训练后评价与预测、12 份 best/last 权重；另保存 6 个 checkpoint 的动态／TRAIN 均值替换，共 12 个无训练评估条件、48 份评价与预测。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [04 C组](../04/README.md) · [02 A组基座](../02/README.md)

## 1. 实验目的与修订范围

04 已完成 C1/3351、C2/3351。C1 测量静态 ρ=0.125，C2 使用 ρ=0.1 与单位模式均值的视觉调制。本轮新增 D1：在 C2 上只将 ρ 改为 0.125，研究修正幅度与中心化视觉模块的联动。

训练分两次队列完成。首次 D 队列训练 D1/3351，并复用已由原 C 程序完成的 C2/3352、3353；补充队列训练 D1/3352、3353 与 C1/3352。六次新增归档训练分别为 D1 三 seed、C2 两个补 seed、C1 一个补 seed。C1/C2 的 seed3351 继续引用 04，A5/A6 三 seed 引用 02。

**D 的 `vgcpod/*.py` 与 `run.py` 和 C 组逐字节一致。** 中心化公式、概率分解、训练损失和优化器沿用 04；新增代码集中于 D1 配置、同权重均值替换评估、两轮三卡编排及交互汇总。

## 2. 公式修订：将中心化 CP 的 ρ 调为 0.125

记全局图像特征为 $`g_b\in\mathbb{R}^{d}`$，物体查询特征为 $`f^o_{bj}`$，属性／物体文本特征为 $`e_i^a,e_j^o`$。本轮 d=768、r=32。沿用 04：

```math
\begin{aligned}
u_i&=A_u e_i^a,\qquad v_j=A_v e_j^o,\\[4pt]
q_{bj}&=2\sigma\!\left(B_{\mathrm{vis}}f^o_{bj}/\tau_{\gamma}\right)-\mathbf{1}_r,\\[4pt]
H_r&=I_r-\frac{1}{r}\mathbf{1}_r\mathbf{1}_r^{\top},\\[4pt]
\gamma_{bj}&=\mathbf{1}_r+\frac{\eta}{2}H_rq_{bj},
\qquad\eta=0.25,\quad\tau_{\gamma}=0.1.
\end{aligned}
```

每张图像、每个物体查询的模式系数均值为 1，理论范围为 (0.75,1.25)。该约束作用于模式坐标；修正 logits 是模式贡献的加权和，其尺度仍由因子、Wc、读出与 ρ 决定。

```math
\begin{aligned}
\Delta U^{\mathrm{D1}}_{bji}
&=0.125\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\gamma_{bj}\right),\\[4pt]
U_{bji}&=U^{\mathrm{base}}_{bji}+\Delta U_{bji},\\[4pt]
\log p(a_i,o_j\mid x_b)
&=\log\mathrm{softmax}(S_b)_j+\log\mathrm{softmax}(U_{bj:})_i.
\end{aligned}
```

条件 softmax 沿全部属性计算；联合概率在全部属性与物体组合上归一化，张量轴为 `[batch, object, attribute]`。在同一组参数和输入下：

```math
\Delta U^{\mathrm{D1}}=1.25\Delta U^{\mathrm{C2}},
\qquad
\frac{\partial\Delta U^{\mathrm{D1}}}{\partial\theta}
=1.25\frac{\partial\Delta U^{\mathrm{C2}}}{\partial\theta}.
```

这里 θ 是 CP 修正参数。独立训练后，两组权重按各自损失和优化轨迹产生。D1 的 Wc 仍从零初始化，首个前向从零修正开始。

## 3. 新增公式：同 checkpoint 的 TRAIN 均值门控替换

对全部 22,998 张 TRAIN 图像和每个物体查询统计模式均值，图像标签不参与统计：

```math
\bar\gamma_{jk}=\frac{1}{N_{\mathrm{train}}}
\sum_{b=1}^{N_{\mathrm{train}}}\gamma_{bjk},\qquad
\gamma^{\mathrm{mean}}_{bjk}=\bar\gamma_{jk}.
```

均值张量为 `[object, rank]=[12,32]`。先按图像求和，再除以总图像数；最后一个 batch 使用实际图像数。均值替换固定全部模型权重，分别重新归一化条件和联合概率。

```math
\begin{aligned}
\Delta U^{\mathrm{mean}}_{bji}
&=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\bar\gamma_j\right),\\[4pt]
\varepsilon_{bj}&=\gamma_{bj}-\bar\gamma_j,\\[4pt]
\Delta U^{\mathrm{dynamic}}_{bji}-\Delta U^{\mathrm{mean}}_{bji}
&=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\varepsilon_{bj}\right).
\end{aligned}
```

静态部分可写成固定物体模式因子 $`v'_j=v_j\odot\bar\gamma_j`$。两种条件各自在自身 VAL 上选择 CW/OW bias，再固定用于 TEST；评估优化步数为零，输出与原训练文件分开保存。

同时统计每个物体、模式上的 TRAIN 图像方差：

```math
V_{jk}=\frac{1}{N_{\mathrm{train}}}\sum_b\gamma_{bjk}^2-\bar\gamma_{jk}^2,
\qquad
R_{\mathrm{image}}=
\frac{\mathrm{mean}_{j,k}(V_{jk})}
{\mathrm{mean}_{j,k}(V_{jk})+\mathrm{Var}_{j,k}(\bar\gamma_{jk})}.
```

实现使用 float64 累加，方差的浮点负尾截为零。训练日志里的 `gamma_image_variance` 来自训练 batch；本节统计来自选中权重对 TRAIN 全集的固定模型评估。

## 4. 沿用的损失与具体代码修订

β=0.1 的归一化目标及动态正交项保持不变：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{task}}
&=\frac{1.5}{1+\beta}\left[
\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_{a}+\mathcal{L}_{o})\right]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}+
\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{D1}}
&=\mathcal{L}_{\mathrm{task}}
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
\end{aligned}
```

支持项定义为训练组合联合概率质量的负对数，系数为 0.13636。原子属性和伙伴辅助分支关闭。

| 文件 | 本轮具体修订 | 作用 |
|---|---|---|
| [D1.json](configs/round_d/D1.json) | 相对 C2 只改 `rho=0.125` 与 `experiment_id=D1` | 中心化与较高修正强度联用 |
| [mean_gate_eval.py](code/vgcpod_v1_roundD/mean_gate_eval.py) | 新增 TRAIN 均值／方差统计、同权重两条件评估 | 分开测量固定模式与图像相关调制 |
| [round_d.py](code/vgcpod_v1_roundD/round_d.py) | 原 C 源码补 C2 两 seed，D1/3351 单独训练，四权重均值评估及汇总 | 首轮三卡任务 |
| [D组tmux](code/vgcpod_v1_roundD/scripts/start_round_d_tmux.sh) | 捕获当前 Python 路径，日志与退出码 | SSH 脱离后继续运行 |
| [next_round.py](code/vgcpod_next_v1_20261020/next_round.py) | 三卡并行 D1/3352、D1/3353、C1/3352；复用原 D/C 函数 | 补齐 D1 三 seed 与第二组四格 |
| [补充轮tmux](code/vgcpod_next_v1_20261020/start_tmux.sh) | 独立任务状态、日志和退出码 | 区分首轮与补充轮完成状态 |
| [D组数学测试](code/vgcpod_v1_roundD/tests/test_round_d.py) | 四项新测试，累计 30 项 | 比例／梯度、加权均值、固定模式替换与归一化 |

训练核心沿用 [model.py](code/vgcpod_v1_roundD/vgcpod/model.py) 与 [engine.py](code/vgcpod_v1_roundD/vgcpod/engine.py)。本轮实际仍按首次严格最大 VAL CW native AUC 选 best，保存 best/last。相对 04 的[完整补丁](code/source_changes.patch)保留新增配置、评估、测试和两轮调度；双 native/exact 最优权重保存属于后续建议。

## 5. 超参数选取策略与已完成矩阵

1. **从已有两项结果组合 D1。** C1 的静态 ρ=0.125 和 C2 的中心化门控分别已完成，D1 将两者合用。
2. **保持 A5 的训练基座。** β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32 与 10 轮固定，使本轮比较集中于 ρ 与模式调制。
3. **同 seed 配对和四格交互。** D1/C2/A5/A6 均形成三 seed 结果；C1 在 3351、3352 形成两组完整的静态／动态 × 两档 ρ 对照。
4. **均值替换使用 TRAIN 统计。** C2/D1 六个选中 checkpoint 都完成动态与固定均值评估，各用自身 VAL bias。
5. **按既定 VAL 规则选权重。** 主结果保持 native CW 选择；TEST 指标与逐组合表现用于记录方案结果。

| 组 | seed | 最佳 epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | 3352 | 8 | 41.9346 | 29.7615 | 35.4659 | 24.6193 |
| C2 | 3352 | 8 | 41.8360 | 29.7651 | 35.3923 | 24.7416 |
| C2 | 3353 | 6 | 44.5015 | 34.0340 | 33.5573 | 24.3978 |
| D1 | 3351 | 10 | 43.8745 | 30.8834 | 34.7717 | 26.6923 |
| D1 | 3352 | 8 | 41.4983 | 29.7811 | 35.3665 | 24.4136 |
| D1 | 3353 | 9 | 44.5294 | 34.0270 | 34.2359 | 25.2524 |

| 固定项 | 设置 |
|---|---|
| CLIP／特征 | 冻结 ViT-L/14，224px，768 维，同一缓存 |
| D1 ρ／η／τγ | 0.125 / 0.25 / 0.1 |
| β／rank／adapter rank | 0.1 / 32 / 8 |
| 适配器 LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦调度 |
| epochs／batch／更新次数 | 10 / 64 / 3600 |
| κ／τv | 20 / 0.07 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |

原配置字段 `variant=V1`，动态分支由显式 `gate_input=local`、`gate_center_modes=true` 启用。D1 的 Bvis 与 Au/Av/Wc 同属 CP 参数组。

## 6. 仓库结构与复现入口

```text
05/
  README.md          # 修订、公式、超参策略与末节数据分析
  code/              # 原D运行代码、补充轮启动器、分析源码和补丁
  configs/           # D1、兼容配置、资产、参考与运行清单
  results/           # 六次新增训练、六项均值干预、状态和原始归档
  analysis/          # 首轮/三seed分析、图表、仓库复算入口
```

恢复原 D 包、原 C 源码及补充轮启动器，复用服务器已有 C 参考结果与特征缓存：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 05 \
  --dest /data/xhw/vgcpod_project/replay_05
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_05/vgcpod_v1_roundD

export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/roundA_20261019
export VG_B_OUT=/data/xhw/vgcpod_v1_runs/roundB_20261019
export VG_C_ROOT=/data/xhw/vgcpod_project/replay_05/vgcpod_v1_roundC
export VG_C_OUT=/data/xhw/vgcpod_v1_runs/roundC_20261020
export VG_D_OUT=/data/xhw/vgcpod_v1_runs/replay_05_D
export VG_D_SESSION=vgcpod_05_pilot

bash scripts/start_round_d_tmux.sh --gpus 1 2 3
tmux attach -t vgcpod_05_pilot
```

该首轮重跑 D1/3351，复用同源已完成的 C2 两 seed。首轮退出码为 0 后执行补充轮：

```bash
cd /data/xhw/vgcpod_project/replay_05/vgcpod_next_v1_20261020
export VG_NEXT_SESSION=vgcpod_05_confirm
export VG_NEXT_OUT=/data/xhw/vgcpod_v1_runs/replay_05_next

bash start_tmux.sh --gpus 1 2 3 \
  --d-root /data/xhw/vgcpod_project/replay_05/vgcpod_v1_roundD \
  --c-root /data/xhw/vgcpod_project/replay_05/vgcpod_v1_roundC \
  --d-out /data/xhw/vgcpod_v1_runs/replay_05_D \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --b-out /data/xhw/vgcpod_v1_runs/roundB_20261019
tmux attach -t vgcpod_05_confirm
```

GPU1/2 执行 D1/3352、3353 并做均值替换；GPU3 读取同源 C1/3352。`Ctrl+b` 后按 `d` 脱离 tmux。完整命令与外部参考见 [runbook.json](runbook.json)、[external_inputs.json](external_inputs.json)。

从仓库保存的预测复算数据分析：

```bash
python VG-CPOD方案/05/analysis/recompute.py
python VG-CPOD方案/05/analysis/recompute_bias.py
python tools/reproduce.py verify
```

复算使用 NumPy、PyTorch、Matplotlib，读取 01 的缓存元数据、02 的 A5/A6、04 的 C1/C2 首 seed，以及本轮新增运行和均值干预。输出至 `analysis/recomputed/`，数值计算与原分析一致。

## 7. 数据分析与本轮结论

| 三 seed 方案 | VAL CW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- |
| A5 | 43.078 ± 1.289 | 34.437 ± 1.043 | 25.255 ± 0.792 | 23.638% |
| C2 | 43.243 ± 1.339 | 34.329 ± 0.952 | 25.130 ± 0.985 | 23.638% |
| D1 | 43.301 ± 1.595 | 34.791 ± 0.566 | 25.453 ± 1.153 | 24.343% |
| D1_mean | 43.239 ± 1.128 | 34.702 ± 0.698 | 25.466 ± 1.074 | 24.273% |

**D1 三 seed 相对 A5 的平均 TEST CW/OW AUC 提高 0.354／0.197 点。** 配对差值标准差为 0.537／0.544；exact AUC 增量为 +0.273／+0.228。CW 固定 VAL bias HM +0.298，OW HM −0.662。AUC 为面积×100，条件属性指标给定真实物体。

- D1−A5 的三个 seed CW/OW 增量依次为 +0.749/+0.524、−0.257/−0.430、+0.570/+0.499；未见属性正确数变化为 +4、−4、+40。
- D1 相对同权重 TRAIN 均值门控的平均 CW/OW AUC 增量为 +0.090/−0.013；未见属性净正确数为 +3、−1、+2。均值替换后的 D1 相对 A5 仍有 +0.264/+0.211 的平均 AUC 增量。
- 两组四格交互 `(D1-C2)-(C1-A5)` 的 CW 为 +0.488、+0.132，OW 为 +0.283、−0.103。视觉条件化训练中的修正幅度成为后续重点。
- D1 的 OW HM 三个 seed 均低于 A5；改用 exact VAL bias 后配对平均差仍为 −0.866。后续同时报告排序与已见／未见平衡。
- D1/3353 的 native 选中 epoch9，exact 峰值位于 epoch6；未来计划同步保存两种最优权重。下一实验建议 E1 取 ρ=0.15、η=0.25，三 seed 各 10 轮；该建议与本轮已完成代码和结果分开记录。

![D1三seed同seed配对结果](analysis/confirmed/figures/01_paired_results.png)

完整均值、配对差值、均值替换、交互、逐组合变化与训练曲线见[数据分析 README](analysis/README.md)。
