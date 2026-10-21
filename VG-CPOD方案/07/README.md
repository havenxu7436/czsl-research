# 07 · 支持集权重下调与跨组合排序诊断

**F 组实验，VG-CPOD v1，归档日期 2026-10-20。** 使用冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 F1 三 seed、30 个模型 epoch，保存 native／exact 两种选择的 24 份评价与预测，以及六项动态／TRAIN 均值替换评估的 48 份评价与预测，共 72 份预测导出、12 份 checkpoint 文件。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [06 E组](../06/README.md) · [05 D组](../05/README.md)

## 1. 实验目的与修订范围

06 的 E1 使用 ρ=0.15、β=0.1。其 VAL CW AUC 三 seed 均值最高，TEST 未见条件属性低于 D1，同时未见图像上的 TRAIN 组合概率质量增加。本轮以 E1 为基座，只把 β 下调为 0.05，测量支持集损失权重对属性泛化和组合排序的作用。

F1 与 E1 的配置仅 `task_aux_beta`、`experiment_id` 两个字段不同。ρ=0.15、中心化门控 η=0.25、τγ=0.1、全局 g 读出、rank、学习率和 10 epochs 均沿用 E1。

模型与损失文件 `vgcpod/model.py`、CLIP 特征实现和 `mean_gate_eval.py` 与 E 组逐字节一致。代码新增四个加权损失日志字段、F1 三卡编排、β 对照表及支持集训练诊断；两个选权重轨道继续沿用。

## 2. 公式修订：降低支持集项系数，pair 项保持 1.5

完整联合分布按物体与条件属性分解。记张量轴为 `[batch, object, attribute]`，真实标签为 $`(i_b,j_b)`$，TRAIN 组合集合为 $`\mathcal{S}`$：

```math
\begin{aligned}
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
Z_b&=\sum_{(j,i)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}
&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],\\[4pt]
\mathcal{L}_{\mathrm{support}}&=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_a+\mathcal{L}_o
&=-\frac{1}{B}\sum_b s_{bj_bi_b}
=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

条件 softmax 遍历全部 16 个属性，物体 softmax 遍历全部 12 个物体，联合概率在 192 个组合上归一化；Z 是其中 83 个 TRAIN 组合的总概率质量。

现有归一化训练目标可精确改写为：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}
\left[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_a+\mathcal{L}_o)\right]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}^{\mathrm{E1}}
&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{3}{22}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}^{\mathrm{F1}}
&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{1}{14}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

因此本轮支持集项系数从 0.136363636 降为 0.071428571，比例为 11/21，即原来的 **52.38%**；pair 项系数保持 1.5。这是对已有 objective 的系数调整与恒等分解，反向传播沿用原表达式及运算顺序。

F1 的动态正交项和 AdamW weight decay 沿用原实现：

```math
\mathcal{L}_{\mathrm{F1}}
=\frac32\mathcal{L}_{\mathrm{pair}}+\frac1{14}\mathcal{L}_{\mathrm{support}}
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
```

原子属性与伙伴辅助分支关闭，prim 和 cent 为零。权重衰减由 AdamW 执行。

## 3. 沿用的视觉修正与新增日志公式

令 r=32，d=768，属性／物体文本模式为 $`u_i=A_u e_i^a`$、$`v_j=A_v e_j^o`$。沿用模式中心化和全局读出：

```math
\begin{aligned}
q_{bj}&=2\sigma(B_{\mathrm{vis}}f^o_{bj}/\tau_\gamma)-\mathbf{1}_r,\\[4pt]
H_r&=I_r-\frac{1}{r}\mathbf{1}_r\mathbf{1}_r^\top,\\[4pt]
\gamma_{bj}&=\mathbf{1}_r+\frac{\eta}{2}H_rq_{bj},
\qquad\eta=0.25,\quad\tau_\gamma=0.1,\\[4pt]
U_{bji}&=U^{\mathrm{base}}_{bji}
+0.15\kappa\,g_b^\top W_c(u_i\odot v_j\odot\gamma_{bj}).
\end{aligned}
```

每个图像／物体查询的模式均值为 1。CLIP 图文编码器冻结，全局与物体查询特征从同一缓存读取。

`engine.py` 在 `optimizer.step()` 后，从已脱离计算图的标量记录：

```math
\begin{aligned}
w_{\mathrm{support}}&=\frac{1.5\beta}{1+\beta},\\[4pt]
\ell_{\mathrm{pair,weighted}}&=1.5\mathcal{L}_{\mathrm{pair}},\\[4pt]
\ell_{\mathrm{support,weighted}}&=w_{\mathrm{support}}\mathcal{L}_{\mathrm{support}},\\[4pt]
\epsilon_{\mathrm{task}}&=\left|\mathcal{L}_{\mathrm{task}}
-\ell_{\mathrm{pair,weighted}}-\ell_{\mathrm{support,weighted}}\right|.
\end{aligned}
```

四个字段为 `support_weight`、`pair_weighted`、`support_weighted`、`task_reconstruction_error`。epoch 按每 batch 的实际图像数加权；最后一个字段是平均 batch 绝对误差。汇总表另列 epoch 均值的重构误差，二者分别记录。

## 4. 具体代码修订与验证

| 文件 | 本轮具体改动 | 作用 |
|---|---|---|
| [F1.json](configs/round_f/F1.json) | `task_aux_beta=0.05`、`experiment_id=F1` | 单独测量支持集项系数 |
| [engine.py](code/vgcpod_v1_roundF/vgcpod/engine.py) | 新增 `task_decomposition()` 和四个标量日志字段 | 直接记录原始与加权损失的关系 |
| [round_f.py](code/vgcpod_v1_roundF/round_f.py) | 三卡训练、续训、六项均值评估、β 主表及 F1−E1 等配对汇总 | 输出完整 F 组结果 |
| [pre_f_complete.json](configs/references/pre_f_complete.json) | 17 份已有 A5/A6/C1/C2/D1/E1 指标和选中 epoch TRAIN 快照 | 提供已有对照 |
| [tmux启动](code/vgcpod_v1_roundF/scripts/start_round_f_tmux.sh) | F 组独立会话、输出与日志路径 | GPU1/2/3 分别运行一个 seed |
| [F组测试](code/vgcpod_v1_roundF/tests/test_round_f.py) | 配置差异、损失及梯度恒等式、有限差分、加权日志 | 累计 36 项发布测试 |

[相对06的源码补丁](code/source_changes.patch)保留实际差异。新增代码不改变模型初始化、objective、优化器或学习率的实现。

[原 CPU 完整流程](results/package_validation/F/validation/result.json)完成三项训练、六项均值评估与重复启动复用；将原 E 引擎设为相同 β=0.05 后，native best／last 的模型参数、AdamW 状态与 Torch 随机状态逐元素一致。[服务器源码核对](results/server_source_identity.json)及[原包69成员映射](package_members.json)保留来源身份。

## 5. 超参数选取策略与实际矩阵

1. **从 E1 单独改变 β。** E1 按既定 VAL CW 三 seed 均值规则作为基座，ρ=0.15 固定，β 从 0.1 下调到 0.05。
2. **用精确系数解释调整。** pair 系数固定 1.5，support 系数约减半；由此测量 TRAIN 组合质量与未见属性的变化。
3. **保持相同初始化和训练日程。** 三个 seed 从各自原始初始化开始，各 10 epochs、3600 次更新。
4. **同 seed 配对。** 首要比较 F1−E1，随后列 D1、A5、A6、C2 和已有两个 seed 的 C1；配对使用相同特征缓存。
5. **保留两种选权重和均值评估。** native 主结果、exact 选择单列，各条件使用自己的 VAL native bias 并固定到 TEST。

| seed | native／exact最佳epoch | VAL CW | VAL OW | TEST CW | TEST OW |
| --- | --- | --- | --- | --- | --- |
| 3351 | 8／8 | 43.7359 | 30.9921 | 33.7102 | 25.5140 |
| 3352 | 8／8 | 41.7244 | 29.8303 | 34.9602 | 22.8710 |
| 3353 | 6／6 | 45.0561 | 34.4285 | 33.4192 | 24.6694 |

| 固定项 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 ViT-L/14，224px，768 维，同一缓存 |
| ρ／η／τγ／β | 0.15 / 0.25 / 0.1 / **0.05** |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| 适配器 LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |
| eval batch／workers | 64 / 4 |

CP 参数组为 Bvis、Au、Av、Wc，`cp_lr_scale=0.3`，Wc 从零初始化。内部 `variant=V1`，动态行为由 `gate_input=local` 和 `gate_center_modes=true` 显式启用。实际全字段配置和两套指标见[实验清单](configs/experiment_inventory.json)。

## 6. 仓库结构与复现入口

```text
07/
  README.md          # 修订、公式、超参策略与末节数据分析
  code/              # 原F代码、相对E补丁、原始分析源码
  configs/           # F1、兼容配置、资产、参考与运行清单
  results/           # 三seed、双选择、均值评估、状态及原归档
  analysis/          # 数据分析、机制分解、图表及仓库复算入口
```

恢复本轮训练包并复用服务器已有 CLIP 特征：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 07 \
  --dest /data/xhw/vgcpod_project/replay_07
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_07/vgcpod_v1_roundF

export VG_F_OUT=/data/xhw/vgcpod_v1_runs/replay_07_F
export VG_F_SESSION=vgcpod_07_F
bash scripts/start_round_f_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020
tmux attach -t vgcpod_07_F
```

每卡先训练对应 seed，再运行 native／exact 两项均值评估。按 Ctrl+b、d 脱离 tmux。`launcher.exitcode=0` 且三项状态均为 complete 表示整轮完成。已有对照优先读取对应运行，包内 17 份快照提供同 seed 标量对比。外部输入见 [external_inputs.json](external_inputs.json)，命令见 [runbook.json](runbook.json)。

从仓库预测、权重和缓存元数据复算本次分析与概率因子交换：

```bash
python VG-CPOD方案/07/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`，原分析保存于 `analysis/confirmed/`。运行包的原 README、说明及验证字节分区保留，materialize 恢复原始布局。

## 7. 数据分析与下一轮方向

| 三 seed 方案 | ρ | β | VAL CW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- | --- | --- |
| A5 | 0.1 | 0.1 | 43.078 | 34.437 | 25.255 | 23.638% |
| D1 | 0.125 | 0.1 | 43.301 | 34.791 | 25.453 | 24.343% |
| E1 | 0.15 | 0.1 | 43.554 | 34.427 | 25.180 | 23.374% |
| F1 | 0.15 | 0.05 | 43.505 | 34.030 | 24.351 | 25.189% |
| F1_mean | 0.15 | 0.05 | 43.543 | 34.014 | 24.355 | 24.819% |

AUC 为面积×100，条件属性使用 GT-object。**F1 相对 E1 的 TEST CW／OW AUC 为 −0.397 ± 0.232／−0.829 ± 0.756 点，未见条件属性为 +1.816 ± 0.424 个百分点。** 三个 seed 均呈现属性提高、组合 AUC 下降；E1 和 F1 同 seed 均选中 8／8／6 轮。

- TEST 未见图像上的 TRAIN 组合概率质量从 73.698% 降至 70.770%，条件属性 NLL 从 2.3872 降至 2.2884；调整支持集权重确实改善了属性泛化。
- 未见图像的物体准确率由 63.353% 升至 63.741%；全部未见候选内真实组合 top-1 由 49.921% 降至 48.017%。CW 集合外候选最高分压过 CW 内未见最高分的比例由 23.621% 升至 25.771%。
- 概率因子交换中，E1 物体×F1 条件属性的 OW exact AUC 为 24.548，F1 物体×E1 条件属性为 25.131；原 E1／F1 为 25.261／24.451。排序变化主要随条件分支出现。
- F1 dynamic−TRAIN mean 的 TEST CW／OW AUC 为 +0.016／−0.003 点，配对 SD 为 0.096／0.076；门控 TRAIN 图像变化占比为 1.51%—1.88%。
- F1 的 OW HM 均值为 40.349，相对 E1 的 39.792 提高 0.556 点；三个 seed 的增量为 −0.588／−1.389／+3.645。已见准确率提高、未见准确率下降，工作点向已见侧移动。
- **下一轮建议 G1：ρ=0.15、β=0.075、η=0.25；并补 G0：相同 β 和 ρ，gate_input=none 的静态训练对照。** 各三个 seed、10 epochs，共六次训练。该项为后续计划；本阶段归档的实际训练为 F1、β=0.05。

![F1 相对 E1 的属性与组合排序变化](analysis/confirmed/figures/06_f1_e1_paired.png)

完整配对结果、损失日志核查、逐组合概率质量、概率因子交换、动态／均值分析及下一轮设计见[数据分析 README](analysis/README.md)。
