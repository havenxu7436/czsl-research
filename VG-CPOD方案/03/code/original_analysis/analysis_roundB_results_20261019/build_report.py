"""Build the B-pilot interpretation and the next three experiments."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA = json.loads((OUT / 'analysis.json').read_text(encoding='utf-8'))
NAME = 'VG-CPOD_v1_B组结果分析与下一轮方向_20261020'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def metric(case, setting):
    return next(r for r in DATA['metrics'] if r['case'] == case and r['setting'] == setting)


def diag(case, phase='test'):
    return next(r for r in DATA['diagnostics'] if r['case'] == case and r['phase'] == phase)


def delta(case, setting):
    return next(r for r in DATA['comparisons'] if r['comparison'] == case + '-A5' and r['setting'] == setting)


cases = ['A1', 'A5', 'A6', 'B1', 'B2', 'B3']
main = table(['组', '最佳 epoch', 'VAL CW AUC', 'VAL OW AUC', 'TEST CW AUC', 'TEST OW AUC', 'TEST 未见条件属性'],
    [[c, metric(c, 'test_cw')['epoch']] + [f"{metric(c, s)['AUC_native']:.3f}" for s in ['val_cw','val_ow','test_cw','test_ow']] +
     [f"{diag(c)['attr_unseen']:.3f}%"] for c in cases])
deltas = table(['相对同 seed A5', 'VAL CW AUC', 'VAL OW AUC', 'TEST CW AUC', 'TEST OW AUC', 'TEST 未见属性', 'TEST OW HM'],
    [[c] + [f"{delta(c, s)['AUC_native']:+.3f}" for s in ['val_cw','val_ow','test_cw','test_ow']] +
     [f"{delta(c,'test_ow')['gt_object_attr_acc_unseen']:+.3f}", f"{delta(c,'test_ow')['HM_at_val_bias']:+.3f}"]
     for c in ['B1','B2','B3']])
transfer = table(['组', '已见条件属性', '未见条件属性', '未见正确图像数', '未见图像的已见概率质量', '物体准确率'],
    [[c, f"{diag(c)['attr_seen']:.3f}%", f"{diag(c)['attr_unseen']:.3f}%", diag(c)['unseen_correct'],
      f"{diag(c)['support_unseen']:.3f}%", f"{diag(c)['object_acc']:.3f}%"] for c in ['A5','A6','B1','B2','B3']])
pair_rows = []
for a, o in [('Suede','Slippers'), ('Suede','Boots.Mid-Calf'), ('Synthetic','Boots.Mid-Calf')]:
    left = next(r for r in DATA['pair_deltas'] if r['comparison']=='B3-A5' and r['phase']=='test' and r['attribute']==a and r['object']==o)
    ref = next(r for r in DATA['pair_deltas'] if r['comparison']=='B3-A6' and r['phase']=='test' and r['attribute']==a and r['object']==o)
    pair_rows.append([a+'–'+o, left['images'], f"{left['right_acc']:.2f}", f"{left['left_acc']:.2f}",
                      f"{ref['right_acc']:.2f}", f"{left['correct_delta']:+d}"])
pairs = table(['未见组合', '图像数', 'A5', 'B3', 'A6', 'B3−A5 正确图像数'], pair_rows)
gate_rows = []
for c in ['B2','B3']:
    r = next(r for r in DATA['gates'] if r['case']==c and r['chosen'])
    gate_rows.append([c, r['epoch'], f"{r['gamma_mean']:.6f}", f"{r['gamma_std']:.6f}",
                      f"{r['gamma_image_std']:.6f}", f"{r['raw_sigmoid_mean']:.4f}", f"{r['effective_scalar_rho']:.6f}"])
gates = table(['组', '选中 epoch', 'γ 均值', 'γ 批内总标准差', '跨图像变化尺度', 'sigmoid 输出均值', 'ρ×γ 均值'], gate_rows)
hm = table(['组', 'TEST CW HM', 'TEST OW HM'],
           [[c, f"{metric(c,'test_cw')['HM_at_val_bias']:.3f}", f"{metric(c,'test_ow')['HM_at_val_bias']:.3f}"]
            for c in ['A5','A6','B1','B2','B3']])

report = r'''# VG-CPOD v1：B 组结果分析与下一轮方向

报告日期：2026-10-20。数据来自 2026-10-19 的 B1/B2/B3 训练，每组均为 seed3351、10 epochs，框架为冻结 CLIP ViT-L/14。

**A5 继续作为主基座，B3 保留为小幅增益参照。下一轮优先并行三个实验：C1 静态 1.25 倍修正、C2 模式均值为 1 的局部弱门控，以及 B3 的 seed3352。**

本轮得到的关键结果是：B3 相对同 seed A5 的 TEST CW/OW AUC 提高 **0.308 / 0.148 点**，未见条件属性却少识别对 **5 张**。B3 的选中 epoch 训练门控均值达到 **1.24077**，接近上界 1.25；跨图像变化尺度约为 **0.00286**。这些数据支持下一步检验：当前收益有多少来自修正幅度，有多少来自图像相关的模式选择。

## 1. 同 seed 的主结果

下表全部使用 seed3351，A1/A5/A6 为已有结果。AUC 为 `AUC_native`，面积乘 100；增量采用百分点。所有模型仍按 VAL CW AUC 选择 checkpoint。

{{MAIN}}

![同 seed 的主结果](analysis_roundB_results_20261019/figures/01_results.png)

相对 A5 的配对增量为：

{{DELTAS}}

- **B1：0.5 混合读出本轮没有带来收益。** VAL CW/OW、TEST CW/OW AUC 均低于 A5，未见属性也下降。下一轮保留全局读出，把变量集中到修正系数。
- **B2：VAL 组合 AUC 提高，未见属性表现变差。** VAL 未见条件属性从 A5 的 20.924% 降到 19.897%；TEST OW AUC 下降 1.512 点，未见条件属性少识别对 29 张。较大的门控幅度没有实现本轮希望得到的属性迁移改善。
- **B3：组合排序小幅改善。** VAL CW/OW AUC 均提高，TEST CW/OW 也提高；TEST 条件属性识别仍略低于 A5。B3 的研究价值在于提供了一个较好的弱调制参照，后续应进一步区分幅度和视觉作用。

## 2. 未见属性识别是否得到改善

条件属性准确率是在给定真实物体后，从全部属性中预测属性。TEST 包含 1,023 张已见组合图像和 1,891 张未见组合图像。

{{TRANSFER}}

B3 与 A5 的已见条件属性正确图像数完全相同；在未见组合上，B3 修正了 7 张 A5 的错误，同时将 12 张原本正确的图像变为错误，净减少 5 张。B3 与 A5 的 TEST 条件属性 top-1 预测一致率为 **98.730%**，说明当前修改对属性决策的整体影响较小。

B3 的 TEST OW 未见组内排序准确率从 A5 的 **50.291% 升到 50.767%**，与 AUC 的小幅改善一致；给定真实物体后的材质判别仍需要增强。

### 2.1 原有重点组合仍有损失

下表均为条件属性准确率：

{{PAIRS}}

这三个组合相对 A5 合计少识别对 10 张；Canvas–Shoes.Flats、Leather–Slippers、Satin–Sandals、Patent.Leather–Shoes.Flats、Synthetic–Boots.Ankle 各多识别对 1 张，最终得到未见总体净减少 5 张。

![B3 相对 A5 的未见组合变化](analysis_roundB_results_20261019/figures/03_unseen_changes.png)

### 2.2 固定工作点的结果基本未变

以下 HM 均使用对应 VAL 选择的 bias：

{{HM}}

B3 相对 A5 的 TEST CW HM 变化为 +0.067，OW HM 为 −0.067，接近原水平。下一轮继续同时记录 AUC、固定 VAL bias 的 HM 和未见条件属性，分别解释组合排序、工作点折中与材质判别。

## 3. 最值得关注的现象：门控接近共同放大

原门控为：

$$
q_{bjk}=2\sigma\!\left(\frac{(B_{\mathrm{vis}}f^o_{bj})_k}{\tau_{\gamma}}\right)-1,
\qquad
\gamma_{bjk}=1+\eta q_{bjk}.
$$

B2 的 η=1，B3 的 η=0.25。下面是各组选中 epoch 的训练批次汇总：

{{GATES}}

“跨图像变化尺度”定义为选中 epoch 中，各训练 batch 对图像轴求方差后，按图像数加权平均，再开平方。它与将模式、物体和图像混合计算的总标准差分开报告。

这些数值说明：

1. **两个动态组都明显向正向放大移动。** B2 的归一化门控偏移为 0.9780，B3 为 0.9631，均接近最大偏移 1；其原始 sigmoid 输出均值分别达到 0.9890、0.9815。
2. **B3 的共同偏移大于跨图像变化。** 平均偏移 0.24077，跨图像变化尺度 0.00286，两者相差约 84 倍。训练中，门控主要表现为提高修正强度，同时保留较小的图像变化。
3. **B2 的优化扰动更大。** 选中 epoch 的梯度裁剪步骤比例为 50.28%，B3 为 29.44%，A5 为 17.78%。下一轮使用 η=0.25 的弱调制范围。

在选中 checkpoint 的 Bvis 中，B2/B3 投影行之间的平均非对角余弦相似度分别为 **0.264 / 0.183**；初始化时投影行正交。这个权重变化与多模式共同提高系数的现象一致。

**由此推断，B3 当前的作用更接近“静态修正放大＋较小的视觉变化”。C1 用于直接测量静态幅度的收益，C2 用于研究移除共同放大后的局部视觉作用。**

![学习曲线与训练门控统计](analysis_roundB_results_20261019/figures/02_learning_and_gate.png)

## 4. 下一轮首先做的三个实验

| 物理 GPU | 实验 | seed | 设置 | 主要问题 |
|---|---|---|---|---|
| GPU1 | C1：静态幅度对照 | 3351 | A5，全局读出，静态 CP，ρ=0.125 | 固定 1.25 倍修正能取得多少收益 |
| GPU2 | C2：均值为 1 的局部弱门控 | 3351 | 全局读出，局部门控，η=0.25，ρ=0.1 | 局部证据能否通过模式间的相对权重改善迁移 |
| GPU3 | B3：重复实验 | 3352 | 当前 B3 配置完全沿用 | 检查当前小幅组合收益在另一个 seed 上的表现 |

三项可并行开展。C1 是最直接的幅度对照，C2 是本轮最值得投入的结构方向，B3 补 seed 提供重复性信息。C1 对照 A5/B3，C2 主要对照同样采用单位平均模式系数的 A5。

### 4.1 C1：用已有配置能力实现静态 1.25 倍修正

原修正为：

$$
\Delta U_{bji}
=\rho\kappa\,g_b^{\top}W_c
\left(u_i\odot v_j\odot\gamma_{bj}\right).
$$

如果所有模式和图像都采用常数 $\gamma_{bj}=c\mathbf{1}$，则在相同参数下严格等价于：

$$
\Delta U_{bji}
=(c\rho)\kappa\,g_b^{\top}W_c(u_i\odot v_j).
$$

因此 C1 采用 A5 的静态 CP，并把 ρ 从 0.1 改为 **0.125**。1.25 来自 B3 预设的上界；B3 选中 epoch 的平均系数对应 ρ≈0.12408，两者接近。C1 保留 A5 的学习率、损失和初始化方式，从头训练。

C1 只需新增配置。它衡量简单幅度调整的实际收益，也为 C2 提供更有针对性的参照。

### 4.2 C2：消除模式系数的共同偏移

代码与现有门控、损失公式一致。本轮优化集中于模式系数的共同偏移：继续使用局部特征计算 $q_{bjk}$，在模式轴上定义：

$$
\bar q_{bj}=\frac{1}{r}\sum_{k=1}^{r}q_{bjk},
\qquad
\gamma_{bjk}^{\mathrm{C2}}
=1+\frac{\eta}{2}\left(q_{bjk}-\bar q_{bj}\right),
\quad\eta=0.25.
$$

该式满足两个直接性质：

$$
\frac{1}{r}\sum_{k=1}^{r}\gamma_{bjk}^{\mathrm{C2}}=1,
\qquad
q_{bjk}\in(-1,1)
\Longrightarrow
\gamma_{bjk}^{\mathrm{C2}}\in(0.75,1.25).
$$

均值恒等式来自 $\sum_k(q_{bjk}-\bar q_{bj})=0$；范围来自 $q_{bjk}-\bar q_{bj}\in(-2,2)$。系数 $\eta/2$ 将中心化后的幅度保持在上述范围内。

当所有模式的 q 相同时，γ 恢复全 1；当局部视觉证据使不同模式产生差异时，部分模式提高、部分降低。该形式把门控的作用集中到模式间的相对分配。修正读出仍为全局 g，ρ 保持 0.1，现有 β 损失和正交项继续沿用。

实现集中在门控计算处：计算 q 后，减去 `q.mean(dim=-1, keepdim=True)`，再乘以 η/2 并加 1；加入一个配置开关区分原 B3 与 C2。C1/C2 的输出使用新的实验目录，现有 B3 用原版本继续补 seed。

### 4.3 本轮超参决定

| 项目 | 下一轮设置 |
|---|---|
| CLIP | 冻结 ViT-L/14，复用当前缓存 |
| 适配器 LR / CP LR | 1e-3 / 3e-4；余弦终点 1e-5 / 3e-6 |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| epochs / batch size | 10 / 64 |
| ρ | C1=0.125；C2/B3=0.1 |
| η / τγ | 动态组 0.25 / 0.1 |
| λorth / weight decay | 动态组 0.001 / 1e-4 |
| 读出 | 保留全局 g |
| 新增变量 | 静态幅度，以及模式均值约束 |

B2 从选中 epoch5 到 epoch10，VAL CW 仅回落 0.052 点；B3 从 epoch8 到 epoch10 回落 0.249 点。当前训练尺度继续保持 10 轮，将新增实验投入到上述两个具体变量。

下一轮以完整 VAL 比较 C1/C2，确定继续投入的配置后补 seed3352、3353，与已有三 seed A5/A6 做配对分析。

## 5. 可以直接执行的 B3 补 seed 命令

C1/C2 的新增实现按上面的设计准备；B3 的 seed3352 可直接用当前服务器代码运行。在服务器执行：

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundB

export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/roundA_20261019
export VG_B_OUT=/data/xhw/vgcpod_v1_runs/roundB_20261019

VG_B_SESSION=vgcpod_B_seed3352 bash scripts/start_round_b_tmux.sh \
  --gpus 3 --cases B3 --seeds 3352

tmux attach -t vgcpod_B_seed3352
```

该实验使用物理 GPU3，输出写入原 B 组目录的新 seed3352 子目录。按 Ctrl+b，松开，再按 d，训练继续。

## 6. 文件与复算入口

B1/B2/B3 均完成 10 轮，队列退出码为 0；运行源码与交付包一致。此次按同 seed 对照复算了八组模型的 32 份预测导出，native/exact AUC 和固定 VAL bias 的 HM 均与保存指标一致。

- [全部指标](analysis_roundB_results_20261019/tables/all_metrics.csv)、[相对 A5/A6 的配对差值](analysis_roundB_results_20261019/tables/comparisons.csv)。
- [已见/未见属性诊断](analysis_roundB_results_20261019/tables/diagnostics.csv)、[条件属性预测的修正与退化图像数](analysis_roundB_results_20261019/tables/conditional_changes.csv)。
- [逐组合表现](analysis_roundB_results_20261019/tables/per_pair.csv)、[逐组合增量](analysis_roundB_results_20261019/tables/per_pair_deltas.csv)。
- [各轮历史](analysis_roundB_results_20261019/tables/history.csv)、[训练门控统计](analysis_roundB_results_20261019/tables/gates.csv)、[选中 checkpoint 的 Bvis 几何统计](analysis_roundB_results_20261019/tables/gate_geometry.csv)。
- [结构化分析](analysis_roundB_results_20261019/analysis.json)、[复算记录](analysis_roundB_results_20261019/audit.json)、[分析脚本](analysis_roundB_results_20261019/analyze_b.py)、[报告生成脚本](analysis_roundB_results_20261019/build_report.py)。
'''
for key, value in {'MAIN':main, 'DELTAS':deltas, 'TRANSFER':transfer, 'PAIRS':pairs, 'GATES':gates, 'HM':hm}.items():
    report = report.replace('{{'+key+'}}', value)
assert '{{' not in report
target = ROOT / (NAME + '.md')
target.write_text(report, encoding='utf-8')
print(target)
print('display equations:', report.count('$$') // 2)
