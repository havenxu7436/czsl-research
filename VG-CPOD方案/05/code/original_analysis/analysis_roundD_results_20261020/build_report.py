"""Write the D-result interpretation and a focused three-GPU next round."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
D = json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
NAME = 'VG-CPOD_v1_D组结果分析与下一轮方向_20261020'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def metric(case,seed,setting):
    return next(r for r in D['metrics'] if r['case']==case and r['seed']==seed and r['setting']==setting)


def delta(comparison,seed,setting):
    return next(r for r in D['comparisons'] if r['comparison']==comparison and r['seed']==seed and r['setting']==setting)


def diag(case,seed=3351,phase='test'):
    return next(r for r in D['diagnostics'] if r['case']==case and r['seed']==seed and r['phase']==phase)


def changes(comparison,seed=3351,group='unseen'):
    return next(r for r in D['similarity'] if r['comparison']==comparison and r['seed']==seed and r['phase']=='test' and r['group']==group)


def paired(comparison,setting):
    return next(r for r in D['paired_means'] if r['comparison']==comparison and r['setting']==setting)


def main():
    seed_table = table(['方案','选中 epoch','VAL CW AUC','VAL OW AUC','TEST CW AUC','TEST OW AUC','未见属性准确率'],
        [[c,metric(c,3351,'val_cw')['selected_epoch']]+
         [f"{metric(c,3351,s)['AUC_native']:.3f}" for s in ['val_cw','val_ow','test_cw','test_ow']]+
         [f"{diag(c)['attr_unseen']:.3f}%"] for c in ['A5','C1','C2','D1']])
    d1_table = table(['配对比较','CW AUC 增量','OW AUC 增量','CW HM 增量','OW HM 增量','未见属性正确数变化'],
        [[comp]+[f"{delta(comp,3351,s)['AUC_native']:+.3f}" for s in ['test_cw','test_ow']]+
         [f"{delta(comp,3351,s)['HM_at_val_bias']:+.3f}" for s in ['test_cw','test_ow']]+
         [f"{changes(comp)['net_correct']:+d}"] for comp in ['D1-A5','D1-C1','D1-C2','D1-A6']])
    c2_table = table(['seed','C2−A5 VAL CW AUC','C2−A5 TEST CW AUC','C2−A5 TEST OW AUC','未见属性正确数变化'],
        [[s]+[f"{delta('C2-A5',s,k)['AUC_native']:+.3f}" for k in ['val_cw','test_cw','test_ow']]+
         [f"{changes('C2-A5',s)['net_correct']:+d}"] for s in [3351,3352,3353]])
    c2_mean = table(['C2−A5 配对指标','三个 seed 的差值均值 ± 样本标准差'],
        [[setting.upper()+' '+label,
          f"{paired('C2-A5',setting)[key+'_mean']:+.3f} ± {paired('C2-A5',setting)[key+'_sd']:.3f}"]
         for setting,key,label in [('val_cw','AUC_native','AUC'),('test_cw','AUC_native','AUC'),
                                   ('test_ow','AUC_native','AUC'),('test_cw','HM_at_val_bias','HM'),
                                   ('test_ow','HM_at_val_bias','HM')]])
    gate_table = table(['checkpoint','动态−均值 TEST CW AUC','动态−均值 TEST OW AUC','未见属性正确数变化'],
        [[c+'/'+str(s)]+[f"{delta(c+'-'+c+'_mean',s,w)['AUC_native']:+.3f}" for w in ['test_cw','test_ow']]+
         [f"{changes(c+'-'+c+'_mean',s)['net_correct']:+d}"]
         for c,s in [('C2',3351),('C2',3352),('C2',3353),('D1',3351)]])
    mean_table = table(['seed3351 条件','TEST CW AUC','TEST OW AUC','未见属性正确数'],
        [[c]+[f"{metric(c,3351,w)['AUC_native']:.3f}" for w in ['test_cw','test_ow']]+
         [diag(c)['unseen_correct']] for c in ['A5','C1','D1_mean','D1']])
    moment_table = table(['checkpoint','跨图像门控标准差','门控总方差中的图像变化比例','模式均值误差'],
        [[r['case']+'/'+str(r['seed']),f"{r['image_variance']**.5:.5f}",
          f"{r['image_fraction']*100:.2f}%",f"{r['mode_mean_error']:.2e}"] for r in D['moments']])
    pair_rows = []
    for a,o in [('Satin','Sandals'),('Faux.Leather','Boots.Knee.High'),('Hair.Calf','Shoes.Heels'),
                ('Suede','Boots.Mid-Calf'),('Synthetic','Boots.Mid-Calf'),('Suede','Slippers')]:
        row = next(r for r in D['pair_deltas'] if r['comparison']=='D1-A5' and r['phase']=='test'
                   and r['attribute']==a and r['object']==o)
        pair_rows.append([a+' / '+o,row['images'],f"{row['correct_delta']:+d}",f"{row['calibrated_pair_delta']:+d}"])
    pair_table = table(['未见组合','图像数','条件属性正确数变化','固定 VAL bias 的 OW 组合正确数变化'],pair_rows)
    exact_table = table(['比较','TEST CW exact AUC 增量','TEST OW exact AUC 增量'],
        [[comp]+[f"{delta(comp,3351,w)['AUC_exact']:+.3f}" for w in ['test_cw','test_ow']]
         for comp in ['D1-A5','D1-C1','D1-C2','D1-D1_mean']])
    interaction = {w:metric('D1',3351,w)['AUC_native']-metric('C2',3351,w)['AUC_native']-
                   metric('C1',3351,w)['AUC_native']+metric('A5',3351,w)['AUC_native']
                   for w in ['test_cw','test_ow']}
    retain = {w:100*delta('D1_mean-A5',3351,w)['AUC_native']/delta('D1-A5',3351,w)['AUC_native']
              for w in ['test_cw','test_ow']}
    text = r'''# VG-CPOD v1：D 组结果分析与下一轮方向

日期：2026-10-20。框架：冻结 OpenAI CLIP ViT-L/14；数据：UT-Zappos。版本保持 v1。

**下一轮优先推进 D1：保持中心化门控和 ρ=0.125，GPU1/2 补 seed3352、3353，GPU3 补 C1/seed3352。** 这样可同时得到 D1 三 seed 结果，以及 seed3352 的静态／动态 × 两档 ρ 完整对照。

本轮 D1/3351 相对 A5 的 TEST CW/OW AUC 分别增加 **0.749／0.524 点**；C2 三 seed 相对 A5 的均值分别为 **−0.109／−0.126 点**。当前主要投入方向是 **CP 修正强度与中心化视觉模块的联动**。

## 1. D1 的增量体现在什么地方

以下均为 seed3351，各自按原 VAL CW AUC_native 规则选择 checkpoint。AUC 为面积乘 100，差值以点计；属性准确率差值以百分点计。

{{SEED_TABLE}}

A5 是 ρ=0.1 的静态 CP；C1 是 ρ=0.125 的静态 CP；C2 是 ρ=0.1 的中心化动态 CP；D1 是 ρ=0.125 的中心化动态 CP。

{{D1_TABLE}}

![D1 与静态及动态对照的 TEST 结果](analysis_roundD_results_20261020/figures/02_d1_results.png)

D1 的 CW AUC、OW AUC、CW 固定 VAL bias 的 HM 都提高，未见条件属性相对 A5 净多识别对 4 张图。与相同 ρ 的 C1 比较，D1 未见条件属性修正 10 张、退化 2 张，净增 8 张。

OW AUC 与固定工作点的 HM 表现不同：D1 的 OW HM 为 **43.116**，A5 为 **43.187**，差 **−0.071 点**。在各自 VAL bias 下，D1 的已见组合准确率从 44.575% 变为 43.402%，未见组合准确率从 41.883% 变为 42.834%；对应已见正确图像少 12 张、未见多 18 张。D1 改善了组合排序曲线，并把该工作点向未见组合移动。

exact AUC 的方向与 native AUC 一致：

{{EXACT_TABLE}}

本轮继续用原 native 规则选权重，同时在报告中保留 exact AUC、固定 VAL bias 的 HM 和条件属性正确数。

## 2. C2 的三个 seed 已给出清楚的方向

{{C2_TABLE}}

{{C2_MEAN}}

这里的标准差来自同 seed 差值。C2 的 VAL CW AUC 三个 seed 都比 A5 高，TEST CW/OW 平均略低；未见条件属性正确数依次为 **+5、0、−5**，平均净变化为零。

![C2 与 A5 的三个 seed 配对结果](analysis_roundD_results_20261020/figures/01_c2_seeds.png)

因此，ρ=0.1 的 C2 作为已完成对照保留。后续新增训练集中在 ρ=0.125 的 D1，研究提高修正强度后中心化模块如何参与组合排序。

三 seed 的另一个诊断是：C2 相对 A6 的 TEST 未见条件属性平均降低 **0.793 个百分点**，但 CW AUC 平均提高 **4.216 点**。当前 CP 的明显作用集中在联合组合分布；未见属性判别仍需通过条件属性指标单独观察。

## 3. TRAIN 均值替换说明了什么

四个 checkpoint 均使用全部 **22,998 张 TRAIN 图像**，对每个物体查询、每个模式统计均值；统计过程忽略标签。动态与均值替换各自在自己的 VAL 上选 CW/OW bias，TEST 使用对应的固定 bias。

{{GATE_TABLE}}

C2 的 dynamic−mean 三 seed 平均 TEST CW AUC 为 **+0.053 ± 0.156**，OW 为 **−0.123 ± 0.170**；对应 exact AUC 均值为 +0.060／−0.085。图像动态调制在 C2 的收益随 seed 改变。

D1 的动态项在当前 seed 上带来额外 **+0.189 CW AUC、+0.033 OW AUC**，未见条件属性净增 3 张。完整指标为：

{{MEAN_TABLE}}

用 TRAIN 均值替换后，D1 相对 A5 的 CW／OW AUC 增量分别仍有 **+0.560／+0.491 点**，保留约 **{{RETAIN_CW}}%／{{RETAIN_OW}}%** 的原增量。这说明大部分增量在固定门控模式下依然存在，动态输入对当前结果提供较小的额外贡献。

![动态门控与 TRAIN 均值替换的同权重对比](analysis_roundD_results_20261020/figures/03_gate_effects.png)

均值替换复用动态训练得到的全部权重。D1−C1 衡量两种训练方案的整体差异，D1−D1_mean 衡量同权重下图像动态输入的预测作用；这两个比较分别保留。

### 3.1 门控公式正确，图像变化占比仍小

对物体查询特征 $f^o_{bj}$，当前门控为：

$$
q_{bjk}=2\sigma\!\left(\frac{(f^o_{bj})^{\top}(B_{\mathrm{vis}})_{k:}^{\top}}{\tau_\gamma}\right)-1,
\qquad
\gamma_{bjk}=1+\frac{\eta}{2}\left(q_{bjk}-\frac{1}{r}\sum_\ell q_{bj\ell}\right).
$$

沿模式维求平均得到 1。实际 TRAIN 全集统计为：

{{MOMENT_TABLE}}

表中的图像变化占比使用总方差分解：先对每个物体／模式统计跨图像方差，再除以“这些方差的均值 + TRAIN 均值张量在物体／模式上的方差”。D1 为 **2.52%**，其余约 97.48% 来自不同物体／模式之间的固定差异。

将门控写成 TRAIN 均值与图像残差：

$$
\begin{aligned}
\gamma_{bj}&=\bar\gamma_j+\varepsilon_{bj},\\
\Delta U_{bji}
&=\rho\kappa\sum_k(g_b^{\top}W_c)_k u_{ik}v_{jk}\bar\gamma_{jk}
+\rho\kappa\sum_k(g_b^{\top}W_c)_k u_{ik}v_{jk}\varepsilon_{bjk}.
\end{aligned}
$$

第一项是固定物体模式系数下的 CP 修正，第二项是图像相关调制。该分解与本轮均值替换代码一致，现有中心化公式与实现可继续使用。

## 4. 为什么下一轮围绕 ρ=0.125

seed3351 已构成以下四格：

| CP 形式 | ρ=0.1 | ρ=0.125 |
|---|---|---|
| 静态 | A5 | C1 |
| 中心化动态 | C2 | D1 |

静态方案将 ρ 提高至 0.125，TEST CW/OW AUC 增加 0.248／0.160 点；中心化动态方案对应增加 0.736／0.443 点。按各自 VAL 规则选中权重后，方案指标的交互差为：

$$
I=(\mathrm{D1}-\mathrm{C2})-(\mathrm{C1}-\mathrm{A5}).
$$

TEST CW 为 **+{{INTERACTION_CW}} 点**，OW 为 **+{{INTERACTION_OW}} 点**；exact AUC 对应为 +0.394／+0.367 点。这是下一轮最值得量化的现象：较高的修正强度与中心化模块合用时，当前 seed 的整体增量更大。

ρ 的调整也改变训练过程。第 10 轮 C2／D1 的梯度裁剪比例分别为 **22.8%／34.2%**；选中权重的修正 RMS 为 **1.232／1.289**。重新训练后，修正并未简单保持 1.25 倍比例。下一轮保持其余优化设置，直接补齐这一方案交互的同 seed 数据。

![训练过程与梯度裁剪比例](analysis_roundD_results_20261020/figures/04_training.png)

D1 选中第 10 轮，第 8 至第 10 轮 VAL CW AUC 从 43.735 提高到 43.875。本轮优先完成当前 10 轮设定的三 seed 结果，训练轮数留作后续独立变量。

## 5. 下一轮首先执行的三项实验

| 物理 GPU | 任务 | seed | 参数与作用 |
|---|---|---|---|
| GPU1 | D1 补 seed | 3352 | ρ=0.125、η=0.25，得到与 A5/C2 同 seed 的增量 |
| GPU2 | D1 补 seed | 3353 | 补齐 D1 三 seed，与 A5/C2 三 seed 配对 |
| GPU3 | C1 静态幅度对照 | 3352 | ρ=0.125、γ=1，使 seed3352 也形成完整四格 |

三项各 10 epochs，总计 30 个模型 epoch，直接并行运行。D1 使用原 D 源目录和原配置，C1 使用原 C 源目录和原配置。输出分别追加到原 D1、C1 的新 seed 子目录。

| 项目 | 下一轮设置 |
|---|---|
| CLIP / 特征缓存 | 冻结 ViT-L/14，复用现有缓存 |
| 读出 | 全局 g |
| ρ | D1/C1 均为 0.125 |
| 门控 | D1 沿用中心化公式，η=0.25、τγ=0.1；C1 为 γ=1 |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| 适配器 LR / CP LR | 1e-3 / 3e-4，余弦终点 1e-5 / 3e-6 |
| epochs / batch size | 10 / 64 |
| 正交项 / weight decay / clip | 动态组 0.001 / 1e-4 / 1 |
| 权重选择 | VAL CW AUC_native 首次严格最大值 |
| 训练后分析 | D1 两个新 seed 各导出动态与 TRAIN 均值替换结果 |

首批汇总直接回答三个问题：

1. D1 相对 A5 的 CW/OW AUC 增量在三个 seed 上分别是多少，配对均值与标准差如何。
2. D1−C1 在 seed3351、3352 上的方向，以及两个完整四格中的交互差。
3. D1 三个 seed 的 dynamic−mean 是否共同改善 CW 排序、OW 排序或未见条件属性。

后续调参顺序沿 **ρ 单轴** 继续，优先增加 **ρ=0.15**，与已有 0.1、0.125 组成相邻三点；η、τγ、rank、学习率和训练轮数沿用当前值。公式研究与参数研究都集中于修正幅度和模式调制之间的作用。

## 6. 公式与代码的具体决定

**当前模型公式无需修正。** 中心化约束、低秩修正、概率归一化、训练损失和均值替换之间一致。

现有损失还满足如下恒等式。令 $\mathcal{L}_{\mathrm{support}}=-\mathbb{E}\log P(\mathcal{S}_{\mathrm{train}}\mid x)$：

$$
\begin{aligned}
\mathcal{L}_{a}+\mathcal{L}_{o}
&=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}},\\
\mathcal{L}_{\mathrm{task}}
&=\frac{1.5}{1+\beta}
\left[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_{a}+\mathcal{L}_{o})\right]\\
&=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
$$

训练记录复算该恒等式的最大误差为 $3.36\times10^{-8}$。β 同时控制向训练组合支持集集中的压力，本轮继续使用已经比较过的 0.1。D1 的 TEST 未见图像上，已见组合概率质量从 A5 的 72.863% 变为 72.621%，条件属性 NLL 从 2.3573 降至 2.3414，与排序改善一致。

下一轮代码工作集中于调度和汇总：新增 D1/3352、3353 与 C1/3352 的三卡任务，继续调用原训练核心；汇总 D1 三 seed 配对结果、两个 seed 的交互差，以及新 D1 权重的均值替换结果。模型核心、损失和优化器保持当前实现。

## 7. 哪些组合贡献了变化

以下是 D1−A5 的 seed3351 TEST 未见组合变化。条件属性使用 GT-object；组合正确数使用各自 VAL 选出的 OW bias。

{{PAIR_TABLE}}

属性层面，Suede / Boots.Mid-Calf 的 +6 与 Suede / Slippers 的 −6 同时存在；组合层面的增量更多分布在 Satin / Sandals、Faux.Leather / Boots.Knee.High 和 Hair.Calf / Shoes.Heels。当前收益集中在组合排序及属性与物体的配合关系。

后续继续保存逐组合的条件属性与校准组合正确数，直接跟踪上述增量在新 seed 上的变化。

## 8. 数据核对与复算入口

D 组退出码为 0，三项训练任务已完成或复用完成结果，四项均值替换任务均成功。新归档中 C2/3352、3353 已存在，D 调度复用它们；D1/3351 为本轮新训练。

此次分析包含 **11 个训练运行、4 个均值替换条件、76 份预测文件**。核对 45 处训练源文件 SHA256，四个动态重评估的 16 份预测数组与原训练导出逐元素一致。固定 VAL bias 的 HM 和条件属性准确率复算一致；exact AUC 最大误差约 0.000049 点，native AUC 最大误差约 0.00864 点。正文表格采用服务器原保存指标。

- [全部指标](analysis_roundD_results_20261020/tables/metrics.csv)、[分组均值](analysis_roundD_results_20261020/tables/means.csv)、[同 seed 差值](analysis_roundD_results_20261020/tables/comparisons.csv)、[配对均值与标准差](analysis_roundD_results_20261020/tables/paired_means.csv)。
- [条件属性与概率分布](analysis_roundD_results_20261020/tables/diagnostics.csv)、[预测修正／退化图像数](analysis_roundD_results_20261020/tables/similarity.csv)、[逐组合增量](analysis_roundD_results_20261020/tables/pair_deltas.csv)。
- [TRAIN 门控统计](analysis_roundD_results_20261020/tables/moments.csv)、[训练历史](analysis_roundD_results_20261020/tables/history.csv)、[门控训练曲线数据](analysis_roundD_results_20261020/tables/gates.csv)。
- [完整分析数据](analysis_roundD_results_20261020/analysis.json)、[复算记录](analysis_roundD_results_20261020/audit.json)、[分析脚本](analysis_roundD_results_20261020/analyze_d.py)、[报告生成脚本](analysis_roundD_results_20261020/build_report.py)。
'''
    replacements = dict(SEED_TABLE=seed_table,D1_TABLE=d1_table,C2_TABLE=c2_table,C2_MEAN=c2_mean,
        GATE_TABLE=gate_table,MEAN_TABLE=mean_table,MOMENT_TABLE=moment_table,PAIR_TABLE=pair_table,
        EXACT_TABLE=exact_table,INTERACTION_CW=f"{interaction['test_cw']:.3f}",
        INTERACTION_OW=f"{interaction['test_ow']:.3f}",RETAIN_CW=f"{retain['test_cw']:.1f}",
        RETAIN_OW=f"{retain['test_ow']:.1f}")
    for key,value in replacements.items():
        text = text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    target = ROOT/(NAME+'.md')
    target.write_text(text,encoding='utf-8')
    print(target)


if __name__=='__main__':
    main()
