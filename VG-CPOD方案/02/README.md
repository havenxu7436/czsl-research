# 02 · 支持损失重加权与 CP 分组学习率

**A 组实验，VG-CPOD v1，2026-10-19。** 主框架为冻结 CLIP ViT-L/14。UT-Zappos natural split，共完成 12 次训练、120 个模型 epoch、48 份 VAL/TEST CW/OW 评价；A1/A5/A6 各三个 seed，A2/A3/A4 各一个 seed。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [01 原实验](../v1/README.md)

## 1. 实验目的与修订范围

原静态 CP 在 CW 上优于 D0，但 OW 与未见条件属性识别下降。本轮分别降低 CP 学习率、降低训练组合支持损失权重，再检验两项调整的联合效果。

图文编码、缓存、全局静态 CP 前向及概率链式分解沿用 01/V1。新增修订集中在训练目标、优化器学习率分组、诊断输出，以及三 GPU 与 tmux 队列。A6 为相同 β 下的 V0/D0 对照。

## 2. 公式修订：分离组合判别与训练支持压力

全局静态 CP 修正保持为：

```math
u_i=A_u e_i^a,\quad v_j=A_v e_j^o,\quad
\Delta U_{bji}=\rho\kappa\,g_b^\top W_c(u_i\odot v_j),\quad
U_{bji}=U^{\mathrm{base}}_{bji}+\Delta U_{bji}.
```

联合概率在全部属性与物体的笛卡尔积上归一化，训练时的组合交叉熵在训练组合内计算：

```math
\begin{aligned}
p_{ij}(x)&=p(o_j\mid x)\,p(a_i\mid o_j,x),\qquad \sum_j\sum_i p_{ij}(x)=1,\\
Z_s(x)&=\sum_{(i,j)\in\mathcal{C}^{s}}p_{ij}(x),\\
\ell_{\mathrm{pair}}&=-\log\frac{p_{a o}(x)}{Z_s(x)},\qquad
\ell_{\mathrm{support}}=-\log Z_s(x),\\
\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}}&=-\log p_{a o}(x)
=\ell_{\mathrm{pair}}+\ell_{\mathrm{support}}.
\end{aligned}
```

这里 $a,o$ 是该训练样本的真实属性与物体。原目标可精确改写为：

```math
\ell_{\mathrm{old}}
=\ell_{\mathrm{pair}}+0.5\ell_{\mathrm{cond}}+0.5\ell_{\mathrm{obj}}
=1.5\ell_{\mathrm{pair}}+0.5\ell_{\mathrm{support}}.
```

新增 β 归一化目标：

```math
\begin{aligned}
\ell_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}\left[\ell_{\mathrm{pair}}
+\beta(\ell_{\mathrm{cond}}+\ell_{\mathrm{obj}})\right]\\
&=1.5\ell_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\ell_{\mathrm{support}}.
\end{aligned}
```

| β | 组合判别项系数 | 支持损失系数 | 相对原目标 |
|---|---|---|---|
| 0.5 | 1.5 | 0.5 | 恢复原目标 |
| 0.25 | 1.5 | 0.3 | 支持项减少 40% |
| 0.1 | 1.5 | 0.13636 | 支持项减少 72.73% |

归一化保持组合判别项系数为 1.5，使 β 的比较集中于支持损失。上述恒等式对每个样本成立，取 batch 均值后同样成立。代码在 β=0.5 时保留原来的浮点运算顺序。

## 3. 公式修订：CP 使用固定比例的分组学习率

适配器参数组为 `Bo/Co/Bc/Cc`，静态 CP 参数组为 `Au/Av/Wc`。两组使用同一余弦进程：

```math
\begin{aligned}
\eta_t&=\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}
\left[1+\cos\left(\frac{\pi t}{T-1}\right)\right],\quad t=0,\ldots,T-1,\\
\eta_t^{\mathrm{adapter}}&=\eta_t,\qquad
\eta_t^{\mathrm{CP}}=s_{\mathrm{CP}}\eta_t,\qquad
s_{\mathrm{CP}}\in\{1,0.3,0.1\}.
\end{aligned}
```

训练图像 22,998 张，batch64，每轮 360 次更新，10 轮共 $T=3600$。适配器保持 1e-3→1e-5；CP 比较 1e-3→1e-5、3e-4→3e-6、1e-4→1e-6。

AdamW 的矩估计、weight decay 与总梯度裁剪沿用原实现。分组学习率缩放 AdamW 更新步及其解耦权重衰减步；梯度裁剪仍在所有可训练参数上统一计算。每一步都乘 `lr_scale`，从而保持组间比例。

## 4. 具体代码修订

| 文件 | 已实施的修改 | 实验作用 |
|---|---|---|
| [model.py](code/vgcpod_v1_roundA/vgcpod/model.py) | `task_aux_beta` 控制归一化目标；新增 `task/support` | 直接调节支持损失，记录等价分解 |
| [engine.py](code/vgcpod_v1_roundA/vgcpod/engine.py) | `make_optimizer` 分组；`set_learning_rates` 每步保留 `lr_scale` | CP 与适配器使用不同更新尺度 |
| 同上 | 同一 VAL 前向计算 CW/OW；新增支持概率质量、未见条件属性、中心化修正 RMS、实际裁剪步骤比例 | 解释性能变化来自哪里 |
| [run.py](code/vgcpod_v1_roundA/run.py) | `--variant` 省略时采用配置文件值 | A6 按配置正确运行 V0 |
| [round_a.py](code/vgcpod_v1_roundA/round_a.py) | 三卡队列、同一实验续训、完成任务跳过、逐运行汇总 | 运行初始六组及后续补 seed |
| [scripts](code/vgcpod_v1_roundA/scripts/) | tmux 启动、记录退出码、按 GPU 分配独立模型 | 三 GPU 持续运行 |
| [tests](code/vgcpod_v1_roundA/tests/) | 原数学检查与 A 组集成检查共 17 项 | 保存已交付版本的检查记录 |

核心损失实现：

```python
beta = self.cfg.get('task_aux_beta', 0.5)
task = (pair + 0.5 * cond + 0.5 * obj) if beta == 0.5 else (
    1.5 / (1.0 + beta) * (pair + beta * (cond + obj)))
support = -torch.logsumexp(s[:, train_mask], dim=-1).mean()
```

关键调度实现：

```python
for group in optimizer.param_groups:
    group['lr'] = eta * group['lr_scale']
```

新增的中心化修正 RMS 对应：

```math
\Delta U^{c}_{bji}=\Delta U_{bji}-\frac{1}{|\mathcal{A}|}\sum_{k\in\mathcal{A}}\Delta U_{bjk},\qquad
R_{\Delta}=\sqrt{\frac{1}{N|\mathcal{O}||\mathcal{A}|}\sum_{b,j,i}(\Delta U^{c}_{bji})^2}.
```

中心化去除条件 softmax 不敏感的属性公共平移。训练日志先按图像数累计中心化均方，再对 epoch 均方开根号；裁剪比例按实际发生裁剪的更新次数除以总更新次数计算。新诊断不参与模型梯度。

## 5. 超参数选取策略与实际实验矩阵

| 组 | 模型 | CP 学习率起点→终点 | β | 已完成 seed |
| --- | --- | --- | --- | --- |
| [A1](configs/round_a/A1.json) | V1 | 3e-04 → 3e-06 | 0.5 | 3351/3352/3353 |
| [A2](configs/round_a/A2.json) | V1 | 1e-04 → 1e-06 | 0.5 | 3351 |
| [A3](configs/round_a/A3.json) | V1 | 1e-03 → 1e-05 | 0.25 | 3351 |
| [A4](configs/round_a/A4.json) | V1 | 1e-03 → 1e-05 | 0.1 | 3351 |
| [A5](configs/round_a/A5.json) | V1 | 3e-04 → 3e-06 | 0.1 | 3351/3352/3353 |
| [A6](configs/round_a/A6.json) | V0 | 关闭 | 0.1 | 3351/3352/3353 |

1. **初始探索：** seed3351 跑 A1–A6。A1/A2 与原 V1 比较 CP 学习率；A3/A4 与原 V1 比较 β；A5 联合降低 β 与 CP 学习率；A6 提供 β=0.1 的无 CP 对照。
2. **组合效果：** 原 V1、A1、A4、A5 构成学习率倍率 1/0.3 与 β=0.5/0.1 的 2×2 比较。各组从同 seed 的初始化重新训练。
3. **三 seed 确认：** A1/A5/A6 补 seed3352、3353，共新增六次训练。A2/A3/A4 保留单 seed 探索结果，统计时分开呈现。
4. **选模和设置取舍：** checkpoint 取首次严格最大 VAL CW AUC；设置比较结合 VAL CW/OW、未见条件属性及配对 seed 变化。TEST 用于报告本轮迁移表现。

| 固定项目 | 取值 |
|---|---|
| 框架与缓存 | 冻结 CLIP ViT-L/14，224px，768 维，复用既有特征 |
| 适配器 LR / adapter rank | 1e-3→1e-5 / 8 |
| rank / epochs / batch | 32 / 10 / 64 |
| ρ / κ / weight decay | 0.1 / 20 / 1e-4 |
| AdamW β₁、β₂ / ε | 0.9、0.999 / 1e-8 |
| 总梯度裁剪阈值 / τv | 1.0 / 0.07 |
| 本轮视觉门控与伙伴分支 | V1 静态 CP 及 V0 对照均关闭 |

## 6. 仓库结构与服务器命令

```text
VG-CPOD方案/
  v1/                  # 01 原实验
  02/
    README.md           # 本轮公式、代码修订、策略及末节数据分析
    code/README.md      # 训练源码与原版差异
    configs/README.md   # A1–A6、资产及参考指标
    results/README.md   # 12 次训练、48 份评价、24 份权重、预测、原导出包
    analysis/README.md  # 配对差值、类别变化、图表与复算
```

在仓库根目录恢复已运行的 A 组包：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 02 \
  --dest /data/xhw/vgcpod_project/replay_02
cd /data/xhw/vgcpod_project/replay_02/vgcpod_v1_roundA
conda activate vgcpod_v1
export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/replay_02
export VG_A_SESSION=vgcpod_02_initial
bash scripts/start_round_a_tmux.sh --gpus 1 2 3 --seeds 3351
```

初始六组完成后，补齐本轮确认实验：

```bash
export VG_A_SESSION=vgcpod_02_confirm
bash scripts/start_round_a_tmux.sh \
  --gpus 1 2 3 --cases A1 A5 A6 --seeds 3352 3353
```

以上为复现命令。已有正式结果位于 [results](results/README.md)，复算统计使用：

```bash
python VG-CPOD方案/02/analysis/recompute.py
```

## 7. 数据分析与本轮结论

| 组 | VAL CW AUC | VAL OW AUC | TEST CW AUC | TEST OW AUC |
| --- | --- | --- | --- | --- |
| D0 | 34.096 ± 0.700 | 22.146 ± 0.262 | 30.286 ± 0.448 | 24.556 ± 0.280 |
| V1 | 40.099 ± 2.703 | 26.903 ± 2.986 | 32.431 ± 0.374 | 23.421 ± 0.413 |
| A1 | 42.788 ± 1.076 | 30.409 ± 2.618 | 34.634 ± 0.887 | 25.235 ± 1.632 |
| A5 | 43.078 ± 1.289 | 31.228 ± 2.129 | 34.437 ± 1.043 | 25.255 ± 0.792 |
| A6 | 34.881 ± 0.417 | 21.705 ± 0.360 | 30.112 ± 0.051 | 24.784 ± 0.259 |

均值 ± 三 seed 样本标准差；AUC 采用原保存的 `AUC_native`，单位为面积×100。

**保留 A5 作为下一轮主配置：β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32、10 epochs。** A1 保留为 β=0.5 参照，A6 作为同 β 的无 CP 对照。

- A1 相对原 V1 的 TEST CW AUC 平均提高 **2.203 点**，三个 seed 均提高，支持将 CP 学习率从 1e-3 降至 3e-4。
- A5 相对原 V1 的 CW/OW AUC 提高 **2.007 / 1.835 点**，相对 D0 提高 **4.151 / 0.699 点**；上述比较在三个 seed 均提高。
- A5 与 A1 的 TEST AUC 均值接近：CW −0.197、OW +0.021。A5 的 VAL OW 三 seed 均更高，TEST OW 标准差更小，未见条件属性准确率平均高 **6.011 点**。
- A5 相对 A6 的 TEST CW/OW 增量为 **4.325 / 0.471 点**；未见条件属性准确率为 **23.638% / 24.431%**。CP 平均在已见图像上多识别对约 56 张，在未见图像上少约 15 张，说明收益仍主要集中于已见组合。
- A5 的 TEST OW 固定 VAL bias HM 为 **40.638**，D0 为 **42.063**。精确 VAL bias 只将 A5 提高到 **40.764**，下一轮继续观察排序与固定工作点表现。

![三 seed 主结果](analysis/confirmed/figures/01_main.png)

本轮已完成支持损失与 CP 更新尺度的调整。下一轮直接在 A5 上比较混合读出、局部门控及弱局部门控，重点改善未见材质组合的迁移。详细配对结果与逐组合变化见 [数据分析 README](analysis/README.md)。
