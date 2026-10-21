# 06 · 中心化 CP 强度扩展与双准则选权重

**E 组实验，VG-CPOD v1，归档日期 2026-10-20。** 框架为冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 E1 三 seed、30 个模型 epoch；保存 native／exact 两种选权重的 24 份评价与预测、12 份 checkpoint 文件，以及六项均值替换评估的 48 份评价与预测，共 72 份预测导出。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [05 D组](../05/README.md) · [04 C组](../04/README.md)

## 1. 实验目的与修订范围

05 的 D1 使用中心化局部视觉调制，ρ=0.125。E1 将 ρ 提高到 0.15，与已有 C2／D1 的 0.1／0.125 构成三点比较，测量视觉条件化训练中 CP 修正强度的变化。E1 与 D1 的配置仅 `rho`、`experiment_id` 两个字段不同。

D1/3353 的 native 与 exact 验证峰值出现在不同 epoch，本轮在每次训练中独立保存两种最优权重，并分别导出结果。两套权重各自进行动态／TRAIN 均值替换评估。

**模型和损失源码 `vgcpod/model.py` 与 D 组逐字节一致。** 本轮实现改动集中于训练引擎的双选权重、评估入口、E1 配置和三 GPU 编排。每个 seed 从同一初始化规则重新训练；CLIP 图文双编码器冻结，复用同一份全局、物体查询和文本特征缓存。

## 2. 公式修订：中心化 CP 强度由 0.125 提高至 0.15

记全局图像特征为 $`g_b\in\mathbb{R}^{d}`$，物体查询特征为 $`f^o_{bj}\in\mathbb{R}^{d}`$，属性／物体文本特征为 $`e_i^a,e_j^o`$。本轮 d=768、r=32，张量轴为 `[batch, object, attribute]`。沿用中心化公式：

```math
\begin{aligned}
u_i&=A_u e_i^a,\qquad v_j=A_v e_j^o,\\[4pt]
q_{bj}&=2\sigma\!\left(B_{\mathrm{vis}}f^o_{bj}/\tau_{\gamma}\right)-\mathbf{1}_r,\\[4pt]
H_r&=I_r-\frac{1}{r}\mathbf{1}_r\mathbf{1}_r^{\top},\\[4pt]
\gamma_{bj}&=\mathbf{1}_r+\frac{\eta}{2}H_rq_{bj},
\qquad \eta=0.25,\quad\tau_{\gamma}=0.1.
\end{aligned}
```

因 $`\mathbf{1}_r^{\top}H_r=0`$，每个图像、物体查询的模式系数均值为 1，坐标范围为 (0.75,1.25)。模式系数与因子相乘后形成 CP 修正：

```math
\begin{aligned}
\Delta U^{\mathrm{E1}}_{bji}
&=0.15\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\gamma_{bj}\right),\\[4pt]
U_{bji}&=U^{\mathrm{base}}_{bji}+\Delta U^{\mathrm{E1}}_{bji},\\[4pt]
s_{bji}=\log p(a_i,o_j\mid x_b)
&=\log\mathrm{softmax}(S_b)_j
+\log\mathrm{softmax}(U_{bj:})_i.
\end{aligned}
```

条件 softmax 遍历全部 16 个属性，物体 softmax 遍历全部 12 个物体，联合概率在全部 192 个组合上归一化。基础条件分支和物体分支均保持全局 g 读出；局部特征只生成模式系数。

在同一参数与输入下，E1 的修正及该修正对 CP 参数的梯度为 D1 的 1.2 倍：

```math
\Delta U^{\mathrm{E1}}=1.2\Delta U^{\mathrm{D1}},\qquad
\nabla_{\theta}\Delta U^{\mathrm{E1}}=1.2\nabla_{\theta}\Delta U^{\mathrm{D1}}.
```

独立训练的权重由各自优化轨迹产生。模式均值约束保持坐标均值为 1；最终 logits 的尺度还取决于 Wc、u、v 和读出。给定图像和物体时，修正算子 $`W_c\mathrm{diag}(v_j\odot\gamma_{bj})A_u`$ 的秩至多为 r。ρ 调整改变训练参数化与优化尺度；对任意正 ρ′，将 Wc 乘以 ρ/ρ′ 可保持修正函数不变。

## 3. 新增公式：双准则选权重与同权重均值替换

令 $`A_t^{\mathrm{native}}`$、$`A_t^{\mathrm{exact}}`$ 分别为第 t 轮的 VAL CW AUC，T=10。两种选择均取首次严格最大值，并行维护各自最优状态：

```math
\begin{aligned}
t_{\mathrm{native}}^{*}
&=\min\left\{t\in\{1,\ldots,T\}:
A_t^{\mathrm{native}}=\max_{1\le s\le T}A_s^{\mathrm{native}}\right\},\\[4pt]
t_{\mathrm{exact}}^{*}
&=\min\left\{t\in\{1,\ldots,T\}:
A_t^{\mathrm{exact}}=\max_{1\le s\le T}A_s^{\mathrm{exact}}\right\}.
\end{aligned}
```

native 沿用原有限 bias 扫描，exact 遍历完整临界点曲线。`best_native.pt` 保存主选择，`best.pt` 为其逐字节兼容副本；`best_exact.pt` 保存第二种选择，`last.pt` 保存训练状态和两套最大值。主结果沿用 native 规则，exact 选择单独写入 `exact_selection/`。

对每套选中权重，分别在全部 22,998 张 TRAIN 图像上统计物体／模式均值，标签不参与计算：

```math
\begin{aligned}
\bar\gamma_{jk}&=\frac{1}{N_{\mathrm{train}}}\sum_{b=1}^{N_{\mathrm{train}}}\gamma_{bjk},\\[4pt]
\Delta U^{\mathrm{mean}}_{bji}
&=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\bar\gamma_j\right),\\[4pt]
\Delta U^{\mathrm{dynamic}}_{bji}-\Delta U^{\mathrm{mean}}_{bji}
&=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot(\gamma_{bj}-\bar\gamma_j)\right).
\end{aligned}
```

均值形状 `[12,32]`，按图像总数加权、float64 累加，末 batch 使用实际数量。固定模式可以折入物体因子 $`v'_j=v_j\odot\bar\gamma_j`$。两种条件重新计算归一化概率，各自在自身 VAL 上选 CW／OW native bias，再固定用于 TEST；exact 选择的权重也使用这一校准规则。每项均值替换的优化步数为零。

## 4. 沿用的损失与具体代码修订

设训练组合集合为 $`\mathcal{S}`$，对 TRAIN 样本定义：

```math
\begin{aligned}
Z_b&=\sum_{(j,i)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],
\qquad\mathcal{L}_{\mathrm{support}}=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_{a}+\mathcal{L}_{o}
&=-\frac{1}{B}\sum_b s_{bj_bi_b}
=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

E1 仍使用 β=0.1 的归一化目标与原正交项：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}
\left[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_{a}+\mathcal{L}_{o})\right]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{E1}}
&=\mathcal{L}_{\mathrm{task}}(0.1)
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
\end{aligned}
```

支持项系数为 0.1363636。原子属性和伙伴辅助分支关闭。

| 文件 | 本轮具体修订 | 作用 |
|---|---|---|
| [E1.json](configs/round_e/E1.json) | `rho=0.15`、`experiment_id=E1` | 形成三档修正强度比较 |
| [engine.py](code/vgcpod_v1_roundE/vgcpod/engine.py) | 独立 `best_exact_auc`／`selected_exact`，保存两套 best、兼容副本和 last；扩展指定 checkpoint／输出目录的评价入口 | 分别测量 native／exact 选权重结果 |
| [mean_gate_eval.py](code/vgcpod_v1_roundE/mean_gate_eval.py) | 新增 `--selection native/exact`，记录来源权重与评价器身份 | 对两套选择独立完成动态／均值评估 |
| [round_e.py](code/vgcpod_v1_roundE/round_e.py) | 三卡三 seed，训练后各做两项均值评估；新增 ρ 汇总、选权重差值与同 seed 对比 | 完成 3 次训练和 6 项无训练评估 |
| [tmux启动](code/vgcpod_v1_roundE/scripts/start_round_e_tmux.sh) | 捕获当前 Python，保留日志、任务状态与退出码 | GPU1／2／3 独立运行，支持脱离与续跑 |
| [E组测试](code/vgcpod_v1_roundE/tests/test_round_e.py) | E1 配置差异、独立首次峰值及双输出、双轨续训一致性 | 累计 33 项原发布测试 |

[相对05的源码补丁](code/source_changes.patch)保留本轮实际差异。[原 CPU 流程验证](results/package_validation/E/validation/result.json)完成 3 次合成数据训练、6 项均值评估及重复启动复用；在相同 ρ=0.15 配置下，新旧引擎的参数与 Torch RNG 逐位一致。服务器源码与发布包的核对见[来源记录](results/server_source_identity.json)。

## 5. 超参数选取策略与已完成矩阵

1. **沿相邻强度继续测量。** E1 取 ρ=0.15，与现成 C2=0.1、D1=0.125 比较；本轮只新增这一档强度。
2. **固定 D1 训练基座。** η=0.25、τγ=0.1、β=0.1、rank32、学习率和 10 轮保持原值，分离 ρ 的影响。
3. **三 seed 同 seed 配对。** 3351／3352／3353 分别使用物理 GPU1／2／3，各卡运行一个独立训练任务。
4. **区分配置、选权重与校准。** 主比较使用 native 选择；exact 选择单列。每个选中模型／干预条件用自己的 VAL bias，ρ 表只混合相同选择规则。
5. **后续配置按 VAL 规则确定。** 完成本轮后，E1 的 VAL CW 三 seed 均值最高，下一轮以其为基座测 β；D1 保存为已有 TEST 表现参照。

| seed | native／exact 最佳 epoch | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---|---|---|---|---|
| 3351 | 8／8 | 43.8332 | 30.9922 | 34.2948 | 26.4335 |
| 3352 | 8／8 | 41.8212 | 29.7720 | 35.4295 | 24.4062 |
| 3353 | 6／6 | 45.0071 | 34.5988 | 33.5570 | 24.7015 |

| 固定项 | 设置 |
|---|---|
| CLIP／特征 | 冻结 ViT-L/14，224px，768 维，同一缓存 |
| ρ／η／τγ | 0.15 / 0.25 / 0.1 |
| β／rank／adapter rank | 0.1 / 32 / 8 |
| 适配器 LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦调度 |
| epochs／batch／更新次数 | 10 / 64 / 3600 |
| κ／τv | 20 / 0.07 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |

完整实际配置和两套指标见[实验清单](configs/experiment_inventory.json)。内部 `variant=V1`，动态行为由 `gate_input=local`、`gate_center_modes=true` 显式启用。Wc 从零初始化，Bvis、Au、Av、Wc 同属 CP 参数组。

## 6. 仓库结构与复现入口

```text
06/
  README.md          # 修订、公式、超参策略与末节数据分析
  code/              # 原E运行代码、相对D补丁、原分析源码
  configs/           # E1、兼容配置、资产、参考与运行清单
  results/           # 三seed、双选择、均值干预、状态和原始归档
  analysis/          # 三seed数据分析、图表、仓库复算入口
```

恢复运行包并复用服务器现有冻结 CLIP 缓存：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 06 \
  --dest /data/xhw/vgcpod_project/replay_06
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_06/vgcpod_v1_roundE

export VG_E_OUT=/data/xhw/vgcpod_v1_runs/replay_06_E
export VG_E_SESSION=vgcpod_06_E
bash scripts/start_round_e_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020
tmux attach -t vgcpod_06_E
```

每卡先训练对应 seed，再完成 native／exact 均值评估。`Ctrl+b` 后按 `d` 脱离 tmux。`launcher.exitcode=0` 且三项状态均为 complete 表示整轮完成。引用旧组指标时优先读取对应运行，同包的 14 份标量快照提供汇总参考。训练输入见[外部输入](external_inputs.json)，命令见[runbook.json](runbook.json)。

从仓库保存的预测、权重与缓存元数据复算分析：

```bash
python VG-CPOD方案/06/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`；原始分析保存在 `analysis/confirmed/`。原发布 README／公式说明保留原字节，materialize 恢复完整原包及其校验身份。

## 7. 数据分析与本轮结论

| 三 seed 方案 | ρ | VAL CW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
|---|---|---|---|---|---|
| A5：静态 | 0.1 | 43.078 | 34.437 | 25.255 | 23.638% |
| C2：中心化动态 | 0.1 | 43.243 | 34.329 | 25.130 | 23.638% |
| D1：中心化动态 | 0.125 | 43.301 | 34.791 | 25.453 | 24.343% |
| E1：中心化动态 | 0.15 | 43.554 | 34.427 | 25.180 | 23.374% |
| E1：TRAIN均值替换 | 0.15 | 43.552 | 34.309 | 25.171 | 23.303% |

AUC 为面积×100，条件属性使用 GT-object。**E1 相对 D1 的 TEST CW／OW AUC 配对均值下降 0.364／0.272 点，VAL CW／OW 均值提高 0.253／0.224 点。** TEST 未见条件属性下降 0.970 个百分点，对应三个 seed 净减少 6、3、46 张正确图像。

- native／exact 均选中 epoch8／8／6，模型参数与 12 份对应预测数组一致，本轮两种选权重方式的结果差值为零。
- E1 相对 A5 的 TEST CW／OW AUC 为 −0.010／−0.075 点。继续提高 ρ 到 0.15 的收益主要出现在 VAL，D1 仍是这三档中已有 TEST 排序最高的参照。
- E1 动态−TRAIN 均值替换的 TEST CW／OW AUC 为 +0.118／+0.010 点，配对 SD 为 0.176／0.088；门控 TRAIN 图像变化占比为 1.59%—2.20%。
- TEST 未见图像的已见组合概率质量从 D1 的 73.466% 增至 E1 的 73.698%，未见条件属性 NLL 从 2.3763 增至 2.3872。下一轮直接测量支持集损失权重的作用。
- **下一轮建议 F1：固定 E1 的 ρ=0.15，将 β 从 0.1 调至 0.05，三 seed 各 10 轮。** pair 系数保持 1.5，support 系数从 0.1363636 降至 0.0714286，约为原来的 52.38%。该项属于后续计划，当前归档为已完成 E1。

![三档修正强度的验证与测试表现](analysis/confirmed/figures/01_rho_sweep.png)

完整配对差异、双选择核查、动态／均值对比、逐组合变化、训练诊断和下一轮变量推导见[数据分析 README](analysis/README.md)。
