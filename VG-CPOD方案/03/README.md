# 03 · 读出解耦与局部弱门控

**B 组实验，VG-CPOD v1，训练日期 2026-10-19，归档日期 2026-10-20。** 框架为冻结 CLIP ViT-L/14。UT-Zappos natural split，B1/B2/B3 各完成 seed3351、10 epochs，共 3 次训练、30 个模型 epoch、12 份 VAL/TEST CW/OW 评价及预测、6 份 best/last 权重。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [02 A 组基座](../02/README.md)

## 1. 实验目的与修订范围

02 确定了 A5 的 β=0.1、CP LR=3e-4。静态 CP 的组合 AUC 有提升，未见条件属性识别仍低于同 β 的 A6。本轮研究局部视觉证据怎样进入 CP：B1 改变修正读出，B2/B3 保留全局读出，由物体查询特征调节 CP 模式，并比较完整与较弱的调制幅度。

CLIP 编码、已有特征缓存、物体 logits、基础条件 logits、概率链式分解，以及 A5 的训练损失和分组学习率继续沿用。新增修改集中在修正读出与门控输入的分离、门控幅度 η、门控变化统计和三 GPU 队列。

## 2. 公式修订：分开修正读出与门控输入

记全局特征为 $`g_b\in\mathbb{R}^{d}`$，物体查询特征为 $`f^o_{bj}\in\mathbb{R}^{d}`$，本轮 $`d=768`$。读出特征定义为：

```math
\begin{aligned}
\mathrm{unit}(z)&=\frac{z}{\max(\lVert z\rVert_2,10^{-12})},\\[4pt]
h_{bj}^{\mathrm{B1}}&=\mathrm{unit}\!\left((1-\alpha)g_b+\alpha f^o_{bj}\right),\quad\alpha=0.5,\\[4pt]
h_{bj}^{\mathrm{B2}}&=h_{bj}^{\mathrm{B3}}=g_b.
\end{aligned}
```

α 表示局部特征的权重。混合特征只进入 CP 修正读出；物体分支和基础条件分支继续读取全局 g。定义文本模式：

```math
\begin{aligned}
u_i&=A_u e_i^a\in\mathbb{R}^{r},\\[4pt]
v_j&=A_v e_j^o\in\mathbb{R}^{r},\\[4pt]
W_c&\in\mathbb{R}^{d\times r},\qquad r=32.
\end{aligned}
```

修正统一为：

```math
\begin{aligned}
\Delta U_{bji}&=\rho\kappa\,h_{bj}^{\top}W_c
\left(u_i\odot v_j\odot\gamma_{bj}\right),\\[4pt]
U_{bji}&=U_{bji}^{\mathrm{base}}+\Delta U_{bji},\\[4pt]
\log p(a_i,o_j\mid x_b)&=\log\mathrm{softmax}(S_b)_j
+\log\mathrm{softmax}(U_{bj:})_i.
\end{aligned}
```

$`S_b`$ 为物体 logits。联合概率在全部物体与属性组合上归一化，张量轴为 `[batch, object, attribute]`。给定门控和物体后，等价算子为 $`W_c\mathrm{diag}(v_j\odot\gamma_{bj})A_u`$，秩至多为 r。Wc 初始为零，三个实验都从相同 seed 的零修正开始。

## 3. 公式修订：以 η 控制局部模式调制幅度

B1 使用全 1 模式系数。B2/B3 从局部特征计算：

```math
\begin{aligned}
\gamma_{bj}^{(\eta)}
&=\mathbf{1}_{r}+\eta\left[
2\sigma\!\left(\frac{B_{\mathrm{vis}}f^o_{bj}}{\tau_{\gamma}}\right)-\mathbf{1}_{r}\right],\\[4pt]
B_{\mathrm{vis}}&\in\mathbb{R}^{r\times d},\\[4pt]
\eta_{\mathrm{B2}}&=1,\qquad \eta_{\mathrm{B3}}=0.25,\qquad \tau_{\gamma}=0.1.
\end{aligned}
```

η=1 恢复原 $`2\sigma(\cdot)`$，各模式系数位于 (0,2)；η=0.25 时位于 (0.75,1.25)；η=0 恢复全 1。在相同参数和输入下，CP 收缩对门控线性，因此：

```math
\Delta U_{\eta=0.25}=0.75\,\Delta U_{\eta=0}+0.25\,\Delta U_{\eta=1}.
```

本轮沿用 02 的归一化目标：

```math
\begin{aligned}
\ell_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}\left[\ell_{\mathrm{pair}}
+\beta(\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}})\right]\\[4pt]
&=1.5\ell_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\ell_{\mathrm{support}},\qquad\beta=0.1.
\end{aligned}
```

支持损失系数保持为 0.13636。B2/B3 的总目标加入现有正交项：

```math
\ell_{\mathrm{B2/B3}}=\ell_{\mathrm{task}}
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
```

B1 的目标为 task。三个实验均关闭伙伴与原子属性辅助分支。

## 4. 具体代码修订

| 文件 | 已实施的修改 | 实验作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundB/vgcpod/model.py) | `readout_feature` 与 `gate_input` 独立配置；加入 `readout_alpha` | 分开改变读出与模式调制，保留原变体默认计算 |
| 同上 | `gate_input` 决定 Bvis 是否训练；`gate_eta` 控制模式系数幅度 | B1 静态，B2/B3 动态；η=1 保留原浮点运算顺序 |
| [engine.py](code/vgcpod_v1_roundB/vgcpod/engine.py) | 增加 `gamma_image_variance`、`gamma_abs_deviation` | 区分图像变化和整体偏离 1 的程度 |
| [round_b.py](code/vgcpod_v1_roundB/round_b.py) | 独立三 GPU 队列、续训、跳过已完成运行；A5/A6 同 seed 配对及配置聚合 | 三组结构实验及结果汇总 |
| [scripts](code/vgcpod_v1_roundB/scripts/) | tmux 内使用当前环境的绝对 Python 路径，保存退出码 | GPU1/2/3 分别运行 B1/B2/B3 |
| [test_round_b.py](code/vgcpod_v1_roundB/tests/test_round_b.py) | 特征干预、η 插值、算子秩、梯度、训练导出与中断续训 | 与原测试合计 22 项检查 |

三组内部 `variant` 均为 V1，实际读出与动态机制由配置记录。B2/B3 的 Bvis 与 Au/Av/Wc 同属 CP 参数组，使用 3e-4→3e-6 的学习率。

核心实现对应：

```python
raw_gamma = 2 * torch.sigmoid((gate_input @ self.Bvis.T) / c['tau_gamma'])
gamma = raw_gamma if self.gate_eta == 1 else 1 + self.gate_eta * (raw_gamma - 1)
modes = (readout @ self.Wc) * gamma * v[None, :, :]
cond = base + rho * kappa * torch.einsum('bor,ir->boi', modes, u)
```

训练日志的新增跨图像方差为：

```math
\begin{aligned}
\mu_{tjk}&=\frac{1}{n_t}\sum_{b\in\mathcal{B}_t}\gamma_{bjk},\\[4pt]
V_{\mathrm{img}}&=\frac{1}{\sum_t n_t}\sum_t n_t\,
\frac{1}{|\mathcal{O}|r}\sum_{j,k}\frac{1}{n_t}
\sum_{b\in\mathcal{B}_t}(\gamma_{bjk}-\mu_{tjk})^2,\\[4pt]
\sigma_{\mathrm{img}}&=\sqrt{V_{\mathrm{img}}}.
\end{aligned}
```

这里 t 遍历选定 epoch 内的训练 batch。方差使用 `unbiased=False`，按图像数累计；分析报告中的跨图像变化尺度是该 epoch 日志的平方根。`gamma_abs_deviation` 同样按图像数累计各系数与 1 的绝对差均值。诊断计算从梯度图分离。

原变体默认配置的四次优化更新与 A 组实现逐位一致，记录见[兼容性检查](results/package_validation/validation/legacy_compatibility.json)。完整差异见[代码补丁](code/source_changes.patch)。

## 5. 超参数选取策略与实际实验矩阵

| 组 | 修正读出 | 门控输入 | η / 系数范围 | 已完成 seed |
| --- | --- | --- | --- | --- |
| [B1](configs/round_b/B1.json) | 归一化的 0.5 全局 + 0.5 物体查询特征 | 关闭 | 全 1 | 3351 |
| [B2](configs/round_b/B2.json) | 全局 g | 物体查询特征 | 1 / (0,2) | 3351 |
| [B3](configs/round_b/B3.json) | 全局 g | 物体查询特征 | 0.25 / (0.75,1.25) | 3351 |

1. **采用 02 已确认的训练基座。** 固定 β=0.1、CP LR=3e-4、适配器 LR=1e-3，结构比较集中于读出与门控。
2. **B1 比较读出。** 取等权混合 α=0.5，与全局静态 A5 比较局部信息进入修正的效果。
3. **B2 比较输入通路。** 保留 A5 全局读出，让物体查询特征只产生模式系数。
4. **B3 比较幅度。** 相对 B2 只将 η 从 1 降为 0.25，保持相同的局部输入、损失和学习率。
5. **运行与选模。** 三组均从 seed3351 初始化训练 10 轮，checkpoint 按首次严格最大 VAL CW AUC 选取；同 seed A5/A6 提供静态 CP 和无 CP 参照。配置比较同时观察 VAL OW 与未见条件属性，TEST 报告迁移表现。

| 固定项目 | 取值 |
|---|---|
| CLIP / 特征 | 冻结 ViT-L/14，224px，768 维，复用现有缓存 |
| 适配器 LR / CP LR | 1e-3→1e-5 / 3e-4→3e-6，共 3600 次更新 |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| epochs / batch | 10 / 64 |
| ρ / κ / τv / τγ | 0.1 / 20 / 0.07 / 0.1 |
| AdamW β₁、β₂ / ε / weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪阈值 / 动态正交项权重 | 1.0 / 0.001 |

## 6. 仓库结构与复现命令

```text
03/
  README.md          # 公式、代码修订、超参策略和末节数据分析
  code/              # 冻结训练源码、原分析源码、相对02补丁
  configs/           # B1/B2/B3、兼容配置、资产、A组参考
  results/           # 3次训练、预测、权重、日志和原始归档
  analysis/          # 同seed分析、图表和仓库复算入口
```

在仓库根目录恢复本轮实际运行包：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 03 \
  --dest /data/xhw/vgcpod_project/replay_03
cd /data/xhw/vgcpod_project/replay_03/vgcpod_v1_roundB
conda activate vgcpod_v1
export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/roundA_20261019
export VG_B_OUT=/data/xhw/vgcpod_v1_runs/replay_03
export VG_B_SESSION=vgcpod_03
bash scripts/start_round_b_tmux.sh --gpus 1 2 3 --cases B1 B2 B3 --seeds 3351
```

配置和原包支持文件恢复后保持原字节，训练直接使用已有 CLIP 缓存。已有本轮结果位于 [results](results/README.md)。从保存预测复算数据分析：

```bash
python VG-CPOD方案/03/analysis/recompute.py
```

## 7. 数据分析与本轮结论

| 组 | VAL CW AUC | VAL OW AUC | TEST CW AUC | TEST OW AUC | TEST 未见条件属性 |
| --- | --- | --- | --- | --- | --- |
| A5 | 43.228 | 30.291 | 34.022 | 26.169 | 24.379% |
| A6 | 34.456 | 21.722 | 30.107 | 24.826 | 24.432% |
| B1 | 42.666 | 29.630 | 33.274 | 25.839 | 24.008% |
| B2 | 44.089 | 31.122 | 33.450 | 24.656 | 22.845% |
| B3 | 43.562 | 30.707 | 34.330 | 26.316 | 24.114% |

本表全部为 seed3351，A5/A6 来自 02；AUC 为面积×100，条件属性准确率给定真实物体后在全部属性中预测。B1/B2/B3 最佳 epoch 分别为 10/5/8。

**A5 继续作为主基座，B3 保留为小幅组合增益参照。** B3 相对 A5 的 VAL CW/OW AUC 提高 0.334 / 0.416 点，TEST CW/OW 提高 **0.308 / 0.148 点**；TEST 未见属性少识别对 5 张，OW 固定 VAL bias 的 HM 变化为 −0.067 点。

- B1 的 VAL/TEST CW/OW AUC 均低于 A5，本轮等权混合读出没有改善组合与属性表现。
- B2 的 VAL CW/OW AUC 提高 0.861 / 0.831 点，TEST OW 降低 1.512 点，未见属性少识别对 29 张。
- B3 在未见图像上修正 7 张 A5 的错误，同时将 12 张原本正确的图像变为错误；Suede–Slippers、Suede–Boots.Mid-Calf、Synthetic–Boots.Mid-Calf 合计少识别对 10 张，其他五个组合各增加 1 张。
- B3 选中 epoch 的训练门控均值为 **1.240767**，接近上界 1.25；跨图像变化尺度为 **0.002861**。B2 对应均值 1.978014、变化尺度 0.004859。两组都明显向共同放大移动。

![本轮同seed结果](analysis/pilot/figures/01_results.png)

这些门控统计支持进一步区分静态修正幅度与局部模式选择。后续方向为静态 1.25 倍修正对照、均值为 1 的弱门控，以及 B3 重复 seed。上述属于后续实验设计；03 保存本轮已经训练的原门控公式。详细门控证据、逐组合变化与下一步公式见[数据分析 README](analysis/README.md)。
