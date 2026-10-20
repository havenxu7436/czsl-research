# 04 · 静态幅度对照与模式中心化门控

**C 组实验，VG-CPOD v1，归档日期 2026-10-20。** 框架为冻结 CLIP ViT-L/14，数据为 UT-Zappos natural split。C1/C2 各完成 seed3351，另纳入原 B3 的 seed3352；共归档 3 次训练、30 个模型 epoch、12 份 VAL/TEST CW/OW 评价与预测、6 份 best/last 权重。C 队列实际训练 C1/C2，并读取已完成的 B3/3352。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [03 B 组](../03/README.md) · [02 A 组基座](../02/README.md)

## 1. 实验目的与修订范围

03 中 B3 的训练门控均值约为 1.241，接近系数上界 1.25，而跨图像变化尺度为 0.002861。本轮将静态幅度与模式相对分配分别检验：C1 将静态 CP 的 ρ 从 0.1 改为 0.125；C2 保持 ρ=0.1，在模式轴中心化局部响应，使每张图像、每个物体的模式系数均值为 1。B3/3352 用于检查原弱门控增量的重复性。

沿用 A5 的全局读出、基础条件分支、物体分支、概率链式分解、β 归一化损失和 CP 分组学习率。CLIP 图文编码器冻结，训练读取同一份全局、物体查询及文本特征缓存。新增实现集中在模式中心化、对应日志与三 GPU 队列。

## 2. 公式修订：C1 静态幅度对照

记全局图像特征为 $g_b\in\mathbb{R}^{d}$，物体查询特征为 $f^o_{bj}\in\mathbb{R}^{d}$，属性与物体文本特征为 $e_i^a,e_j^o$；本轮 d=768、r=32。定义：

```math
\begin{aligned}
u_i&=A_u e_i^a\in\mathbb{R}^{r},\qquad
v_j=A_v e_j^o\in\mathbb{R}^{r},\qquad W_c\in\mathbb{R}^{d\times r},\\[4pt]
\Delta U_{bji}&=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\gamma_{bj}\right),\\[4pt]
U_{bji}&=U^{\mathrm{base}}_{bji}+\Delta U_{bji},\\[4pt]
\log p(a_i,o_j\mid x_b)&=\log\mathrm{softmax}(S_b)_j
+\log\mathrm{softmax}(U_{bj:})_i.
\end{aligned}
```

S 为物体 logits，条件 softmax 沿全部属性计算；联合概率在属性与物体的笛卡尔积上归一化。实现张量轴为 `[batch, object, attribute]`。给定图像与物体后的修正算子为 $W_c\mathrm{diag}(v_j\odot\gamma_{bj})A_u$，秩至多为 r。

C1 采用全 1 模式系数、ρ=0.125。在相同参数与输入下：

```math
\begin{aligned}
\gamma^{\mathrm{C1}}_{bj}&=\mathbf{1}_{r},\\[4pt]
\Delta U^{\mathrm{C1}}_{bji}
&=0.125\kappa\,g_b^{\top}W_c(u_i\odot v_j)
=1.25\,\Delta U^{\mathrm{A5}}_{bji}.
\end{aligned}
```

因此 C1 测量与恒定 1.25 模式系数等价的静态幅度。C1 从 seed3351 重新训练，Bvis 冻结。上述 1.25 倍等价关系用于相同参数下的修正与该修正的参数梯度；独立训练后的权重由各自优化过程产生。

## 3. 公式修订：C2 模式中心化与训练目标

C2 采用全局 g 读出、局部物体查询特征生成模式系数。中心化沿最后一个模式轴进行：

```math
\begin{aligned}
z_{bj}&=\frac{B_{\mathrm{vis}}f^o_{bj}}{\tau_{\gamma}},\qquad
q_{bj}=2\sigma(z_{bj})-\mathbf{1}_{r},\\[4pt]
\bar q_{bj}&=\frac{1}{r}\sum_{k=1}^{r}q_{bjk},\\[4pt]
\gamma^{\mathrm{C2}}_{bjk}
&=1+\frac{\eta}{2}(q_{bjk}-\bar q_{bj}),\\[4pt]
\eta&=0.25,\qquad\tau_{\gamma}=0.1,\qquad\rho=0.1.
\end{aligned}
```

中心化后的模式响应之和为零，因此：

```math
\begin{aligned}
\frac{1}{r}\sum_k\gamma^{\mathrm{C2}}_{bjk}
&=1+\frac{\eta}{2r}\left(\sum_k q_{bjk}-r\bar q_{bj}\right)=1,\\[4pt]
q_{bjk}-\bar q_{bj}&\in(-2,2),\qquad
\gamma^{\mathrm{C2}}_{bjk}\in(1-\eta,1+\eta)=(0.75,1.25).
\end{aligned}
```

这里固定的是模式系数的算术均值。各模式对修正 logits 的贡献仍由 u、v、Wc 与图像读出共同决定。令 P 为模式中心化矩阵，得到实现的雅可比：

```math
\begin{aligned}
P&=I_r-\frac{\mathbf{1}_{r}\mathbf{1}_{r}^{\top}}{r},\qquad
\gamma^{\mathrm{C2}}=\mathbf{1}_{r}+\frac{\eta}{2}Pq,\\[4pt]
\frac{\partial\gamma^{\mathrm{C2}}}{\partial z}
&=\eta P\,\mathrm{diag}\!\left(\sigma(z)\odot[\mathbf{1}_{r}-\sigma(z)]\right).
\end{aligned}
```

PyTorch 沿上述完整计算图自动求导。Wc 初始化为零，C1/C2 均从零修正开始；C2 随训练更新 Wc、Au、Av 和 Bvis。

训练目标沿用 02。对一张训练图像，记训练组合概率质量为 Zs：

```math
\begin{aligned}
Z_s(x)&=\sum_{(i,j)\in\mathcal{C}^{s}}p(a_i,o_j\mid x),\\[4pt]
\ell_{\mathrm{pair}}&=-\log\frac{p(a,o\mid x)}{Z_s(x)},\qquad
\ell_{\mathrm{support}}=-\log Z_s(x),\\[4pt]
\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}}
&=\ell_{\mathrm{pair}}+\ell_{\mathrm{support}},\\[4pt]
\ell_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}\left[\ell_{\mathrm{pair}}+
\beta(\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}})\right]\\[4pt]
&=1.5\ell_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\ell_{\mathrm{support}},\qquad\beta=0.1.
\end{aligned}
```

取 batch 均值得到训练 task loss，支持损失系数为 0.13636。C1 的目标为 task；C2 加入现有 Bvis 正交项：

```math
\mathcal{L}_{\mathrm{C2}}=\mathcal{L}_{\mathrm{task}}
+\frac{0.001}{r^2}\left\lVert
B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
```

原子属性与伙伴辅助分支关闭，训练采用上述 task 与动态正交项。

## 4. 具体代码修订

| 文件 | 已实施的修改 | 实验作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundC/vgcpod/model.py) | 新增 `gate_center_modes`；按模式轴中心化 q，再乘 η/2 | C2 实现单位模式均值；默认 false 沿用原 B 公式 |
| [engine.py](code/vgcpod_v1_roundC/vgcpod/engine.py) | 新增 `gamma_mode_mean_deviation` | 检查每张图像、每个物体的模式均值 |
| [C1/C2 配置](configs/round_c/) | C1 ρ=0.125；C2 开启中心化、η=0.25、ρ=0.1 | 分开检验幅度与相对模式调制 |
| [round_c.py](code/vgcpod_v1_roundC/round_c.py) | C1/C2 队列，调用原 B 源码运行 B3；续训、跳过完成项、跨组汇总 | GPU1/2/3 分别执行 C1/3351、C2/3351、B3/3352 |
| [tmux 脚本](code/vgcpod_v1_roundC/scripts/start_round_c_tmux.sh) | 传入当前 conda 环境的绝对 Python 路径，保存日志与退出码 | 脱离 SSH 后继续训练 |
| [test_round_c.py](code/vgcpod_v1_roundC/tests/test_round_c.py) | 新增 4 项 C 组检查，累计 26 项 | 幅度等价、中心化、梯度、训练导出与续训 |
| [references](configs/references/) | 加入已完成 A 组与 B 组指标快照 | 汇总同 seed 参考与原 B3 重复结果 |

核心实现直接对应公式：

```python
raw_gamma = 2 * torch.sigmoid((gate_input @ self.Bvis.T) / c['tau_gamma'])
if self.gate_center_modes:
    q = raw_gamma - 1
    gamma = 1 + (self.gate_eta / 2) * (q - q.mean(dim=-1, keepdim=True))
else:
    gamma = raw_gamma if self.gate_eta == 1 else 1 + self.gate_eta * (raw_gamma - 1)
```

新增日志对应每个 epoch 内按 batch 图像数加权的均值误差：

```math
E_{\mathrm{mode}}=
\frac{1}{\sum_t n_t}\sum_t\sum_{b\in\mathcal{B}_t}
\frac{1}{|\mathcal{O}|}\sum_j
\left|\frac{1}{r}\sum_k\gamma_{bjk}-1\right|.
```

沿用 03 的 `gamma_image_variance`：对固定物体、模式沿训练 batch 图像轴求总体方差，再按图像数累计；变化尺度为其平方根。日志从梯度图分离。

[相对03的完整补丁](code/source_changes.patch)保存新增公式、配置与调度差异。[原验证](results/package_validation/validation/result.json)包含 26 项测试、12 种旧配置各四次优化更新逐位一致，以及真实三进程 CPU 合成缓存调度。B3/3352 的训练仍使用[03 原 B 源码](../03/code/README.md)。

## 5. 超参数选取策略与实际实验矩阵

| 组 | 修正读出 | 门控 | ρ / η | seed | 选中 epoch |
|---|---|---|---|---|---|
| [C1](configs/round_c/C1.json) | 全局 g | 静态全 1 | 0.125 / 0 | 3351 | 8 |
| [C2](configs/round_c/C2.json) | 全局 g | 局部响应，模式均值 1 | 0.1 / 0.25 | 3351 | 8 |
| [B3 补 seed](configs/round_b/B3.json) | 全局 g | 原 B 组弱门控 | 0.1 / 0.25 | 3352 | 8 |

1. **采用 A5 的训练基座。** 固定 β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32 与 10 轮，比较集中于门控与修正幅度。
2. **C1 的 ρ 来自上一轮观测。** B3 门控均值约 1.241，对应平均静态尺度约 0.1241；取 0.125 形成清晰的 1.25 倍幅度对照。
3. **C2 的 η 沿用 B3。** 保持 η=0.25、τγ=0.1，以 η/2 缩放中心化响应，保留系数位于 (0.75,1.25) 的范围，并去除模式共同偏移。
4. **补 seed 检查原结论。** B3/3352 与 A5/3352 配对；C1/C2 与 A5、A6、B3 的 seed3351 比较。每个配对均使用相同 seed。
5. **checkpoint 与工作点。** 各训练均由首次严格最大 VAL CW AUC_native 选择 best；同一权重评价四种设置，TEST HM 使用相应 VAL 选定的 bias。

| 固定项目 | 取值 |
|---|---|
| CLIP / 特征 | 冻结 ViT-L/14，224px，768 维，同一缓存 |
| 适配器 LR / CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦调度 |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| epochs / batch / 更新次数 | 10 / 64 / 3600 |
| κ / τv / τγ | 20 / 0.07 / 0.1 |
| AdamW β₁、β₂ / ε / weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪阈值 / 动态正交项权重 | 1.0 / 0.001 |

Bvis 与 Au、Av、Wc 同属 CP 参数组。C1 的 Bvis 冻结，正交项为零。配置中的 `readout_alpha`、伙伴及原子属性辅助分支参数为历史兼容字段，本轮运行由实际启用分支决定。

## 6. 仓库结构与复现命令

```text
04/
  README.md          # 公式、代码修订、超参策略和末节数据分析
  code/              # 原C训练源码、原分析源码、相对03补丁
  configs/           # C1/C2、兼容配置、资产与标量参考
  results/           # C1/C2和B3补seed的完整结果及原归档
  analysis/          # 配对分析、图表和仓库复算入口
```

在仓库根目录恢复原 C 运行包及 B3 需要的原 B 包：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 04 \
  --dest /data/xhw/vgcpod_project/replay_04
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_04/vgcpod_v1_roundC

export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/roundA_20261019
export VG_B_ROOT=/data/xhw/vgcpod_project/replay_04/companion_03/vgcpod_v1_roundB
export VG_B_OUT=/data/xhw/vgcpod_v1_runs/replay_04_B
export VG_C_OUT=/data/xhw/vgcpod_v1_runs/replay_04_C
export VG_C_SESSION=vgcpod_04

bash scripts/start_round_c_tmux.sh \
  --gpus 1 2 3 --cases C1 C2 --seeds 3351 --b3-seed 3352
tmux attach -t vgcpod_04
```

GPU1=C1/3351、GPU2=C2/3351、GPU3=B3/3352。`Ctrl+b` 后按 `d` 脱离 tmux。队列自动续训并跳过已完成项；本次历史队列的 B3/3352 即为读取已完成结果。恢复过程保留运行文件原字节和配置、资产的原布局。

从保存预测复算本轮分析，在仓库根目录执行：

```bash
python VG-CPOD方案/04/analysis/recompute.py
python tools/reproduce.py verify
```

复算需要 NumPy、PyTorch、Matplotlib，读取 01 的缓存元数据、02 的 A5/A6、03 的 B3/3351 与本轮三份新增运行结果，输出至 `analysis/recomputed/`。

## 7. 数据分析与本轮结论

| 组 | VAL CW AUC | VAL OW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
|---|---|---|---|---|---|
| A5 | 43.228 | 30.291 | 34.022 | 26.169 | 24.379% |
| A6 | 34.456 | 21.722 | 30.107 | 24.826 | 24.432% |
| B3 | 43.562 | 30.707 | 34.330 | 26.316 | 24.114% |
| C1 | 43.611 | 30.688 | 34.270 | 26.329 | 24.167% |
| C2 | 43.392 | 30.314 | 34.036 | 26.249 | 24.643% |

上表均为 seed3351；AUC 为面积×100，条件属性准确率给定真实物体并在全部属性中预测。

**C1 几乎复现 B3 的预测行为，C2 实现单位模式均值并得到小幅未见属性增量。A5 保留为三 seed 基座，后续研究转向 C2。**

- C1 与 B3 的 TEST 条件属性 top-1 一致率为 99.863%，2,914 张中仅 4 张不同；无偏组合 top-1 仅 5 张不同。C1−B3 的 TEST CW/OW AUC 为 −0.060/+0.012 点，支持原 B3 主要受共同放大影响的解释。
- B3−A5 的 TEST CW 配对增量在 seed3351/3352 分别为 +0.308/−0.259，OW 为 +0.148/−0.350；两 seed 均少识别对 5 张未见属性图像。后续训练资源转向中心化调制。
- C2 的训练模式均值误差为 4.76×10⁻⁹；跨图像方差为 B3/3351 的 21.6 倍，对应变化尺度由 0.002861 增至 0.013298。
- C2 相对 A5 的 TEST 未见正确图像从 461 增至 466：修正 7 张、退化 2 张；已见正确数均为 745。TEST CW 基本持平，OW native AUC +0.081 点，固定 VAL bias 的 OW HM −0.158 点。
- C2 的批内跨图像方差占总方差约 2.68%。下一阶段补 C2/3352、3353，并比较 C2 与 ρ=0.125 的组合；附带训练均值门控替换评估，量化图像相关调制的贡献。这里记录 C 轮结束时的实验建议。

![本轮同seed结果](analysis/pilot/figures/01_results.png)

完整配对表、逐组合变化、门控统计和复算误差见[数据分析 README](analysis/README.md)。
