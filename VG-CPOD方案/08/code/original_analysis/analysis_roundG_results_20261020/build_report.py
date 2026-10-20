"""Write G findings and the next experiment on centered local visual residuals."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
ARCHIVE=ROOT/'server_results_roundG_20261020'
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
S=json.loads((OUT/'supplementary_audit.json').read_text(encoding='utf-8'))
R=json.loads((ARCHIVE/'roundG_20261020/ranking_diagnostics.json').read_text(encoding='utf-8'))
SEEDS=(3351,3352,3353)
NAME='VG-CPOD_v1_G组结果分析与下一轮方向_20261020'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def metric(case,seed,setting):
    return next(r for r in D['metrics'] if r['case']==case and r['seed']==seed and r['setting']==setting)


def mean(case,setting):
    return next(r for r in D['means'] if r['case']==case and r['setting']==setting)


def delta(comp,seed,setting):
    return next(r for r in D['comparisons'] if r['comparison']==comp and r['seed']==seed and r['setting']==setting)


def paired(comp,setting):
    return next(r for r in D['paired_means'] if r['comparison']==comp and r['setting']==setting)


def diag(case,key):
    values=[r[key] for r in R['runs'] if r['case']==case and r['selection']=='native'
        and r['condition']!='train_mean' and r['phase']=='test' and r['group']=='unseen']
    return sum(values)/len(values)


def net(comp,seed):
    return next(r['net_correct'] for r in D['similarity'] if r['comparison']==comp and r['seed']==seed
                and r['phase']=='test' and r['group']=='unseen')


def main():
    overview=table(['方案','ρ','β','VAL CW AUC','TEST CW AUC','TEST OW AUC','未见条件属性'],
        [[case,rho,beta]+[f"{mean(case,k)['AUC_native_mean']:.3f}" for k in ('val_cw','test_cw','test_ow')]+
         [f"{mean(case,'test_ow')['gt_object_attr_acc_unseen_mean']:.3f}%"]
         for case,rho,beta in [('A5',.1,.1),('D1',.125,.1),('E1',.15,.1),('F1',.15,.05),
                             ('G0',.15,.075),('G1',.15,.075)]])
    paired_table=table(['比较','TEST CW AUC：均值 ± 配对 SD','TEST OW AUC：均值 ± 配对 SD','未见属性增量'],
        [[comp]+[f"{paired(comp,k)['AUC_native_mean']:+.3f} ± {paired(comp,k)['AUC_native_sd']:.3f}" for k in ('test_cw','test_ow')]+
         [f"{paired(comp,'test_ow')['gt_object_attr_acc_unseen_mean']:+.3f} 个百分点"]
         for comp in ('G1-F1','G1-E1','G1-D1','G1-G0')])
    matched=table(['seed','G0/G1 native 选中 epoch','CW AUC 增量','OW AUC 增量','未见属性正确数变化'],
        [[s,str(metric('G0',s,'val_cw')['selected_epoch'])+'/'+str(metric('G1',s,'val_cw')['selected_epoch'])]+
         [f"{delta('G1-G0',s,k)['AUC_native']:+.3f}" for k in ('test_cw','test_ow')]+[f'{net("G1-G0",s):+d}'] for s in SEEDS])
    matched_selection=table(['选权重依据','CW native AUC 增量','OW native AUC 增量','CW exact AUC 增量','OW exact AUC 增量'],
        [[label]+[f"{paired(comp,setting)[key+'_mean']:+.3f}" for key,setting in
            [('AUC_native','test_cw'),('AUC_native','test_ow'),('AUC_exact','test_cw'),('AUC_exact','test_ow')]]
         for label,comp in [('native（8/8/9）','G1-G0'),('exact（8/8/6）','G1_exact-G0_exact')]])
    selection_table=table(['模型/seed3353','选权重依据','epoch','VAL CW native AUC','VAL CW exact AUC','TEST CW native AUC','TEST OW native AUC'],
        [[base,label,metric(case,3353,'val_cw')['selected_epoch'],
          f"{metric(case,3353,'val_cw')['AUC_native']:.4f}",f"{metric(case,3353,'val_cw')['AUC_exact']:.4f}",
          f"{metric(case,3353,'test_cw')['AUC_native']:.4f}",f"{metric(case,3353,'test_ow')['AUC_native']:.4f}"]
         for base,case,label in [('G0','G0','native'),('G0','G0_exact','exact'),('G1','G1','native'),('G1','G1_exact','exact')]])
    diagnostics=table(['TEST 未见图像指标','E1 β=0.1','F1 β=0.05','G0 β=0.075','G1 β=0.075'],
        [[label]+[f'{diag(case,key):.4f}'+unit for case in ('E1','F1','G0','G1')]
         for key,label,unit in [('object_accuracy','物体准确率','%'),('conditional_accuracy','GT-object 条件属性准确率','%'),
           ('conditional_nll','GT-object 条件属性 NLL',''),('seen_support_pct','TRAIN 组合概率质量','%'),
           ('extra_support_pct','CW 集合外组合概率质量','%'),('unseen_rank_ceiling','全部未见候选内真实组合 top-1','%'),
           ('extra_dominates_valid','CW 集合外最高分超越 CW 内未见最高分','%')]])
    gate=table(['seed','图像变化占门控总方差','dynamic−mean CW AUC','dynamic−mean OW AUC','未见属性正确数变化'],
        [[s,f"{100*next(r['image_fraction'] for r in D['moments'] if r['case']=='G1' and r['seed']==s):.2f}%"]+
         [f"{delta('G1-G1_mean',s,k)['AUC_native']:+.3f}" for k in ('test_cw','test_ow')]+[f'{net("G1-G1_mean",s):+d}'] for s in SEEDS])
    text=r'''# VG-CPOD v1：G 组结果分析与下一轮方向

日期：2026-10-20。框架：冻结 OpenAI CLIP ViT-L/14；数据：UT-Zappos。分析本次 `vgcpod_roundG_results_20261020.tar.gz`。

**下一轮优先做 H1：固定 G1 的 β=0.075、ρ=0.15、η=0.25，将门控输入改为“相对 TRAIN 均值的局部视觉残差”，跑三个 seed。** 本轮已经补齐 β 三点比较和同配置静态训练对照，后续计算集中于门控输入的表示方式。

G1 相对 F1 的 TEST CW／OW AUC 提高 **0.501／0.705 点**，属性与排序的折中回到 β=0.1 附近。同配置下，G1 相对静态 G0 的 TEST CW／OW AUC 为 **+0.019／−0.127 点**；连续 exact AUC 为 **−0.019／−0.138 点**。目前视觉门控带来的独立组合排序收益较小，门控总方差中约 98% 来自固定物体／模式差异。

## 1. β=0.075 完成了损失权重的中间点比较

下表为三个 seed 的均值。主表采用各配置原定的 VAL CW native AUC 选权重规则，AUC 为面积乘 100。条件属性使用真实物体标签评价。

{{OVERVIEW}}

{{PAIRED}}

![三档 beta 与两种选权重路径](analysis_roundG_results_20261020/figures/01_beta_and_selection.png)

G1 相对 F1 的 OW AUC 增量在三个 seed 上分别为 **+0.678／+0.703／+0.735 点**，方向一致。G1 相对 E1 的 CW AUC 为 +0.103 点、OW AUC 为 −0.124 点，未见条件属性提高 1.322 个百分点。

G1 的 native 选中 epoch 为 8／8／9，E1/F1 为 8／8／6。按 exact 选择时，三种 β 都选中 8／8／6；此时 G1 相对 F1 的 CW／OW native AUC 仍提高 **0.307／0.490 点**，相对 E1 为 **−0.090／−0.339 点**。两条结果共同说明：β=0.075 回收了 β=0.05 的排序损失，同时保留一部分属性收益。

G1 的 VAL CW native AUC 比 E1 低 0.010 点；D1 继续作为已有 TEST 排序较高的历史参照。H1 使用 G1 作为直接基线，因其已有相同 β、ρ 下的静态与动态训练对照。

## 2. 同配置静态训练对照明确了门控的实际贡献

G0 与 G1 使用同一 β=0.075、ρ=0.15、CLIP、rank 和优化设置；G0 的门控输入为 none，G1 使用 local、η=0.25。两组对应 seed 的 native 选中 epoch 完全对齐。

{{MATCHED}}

G1−G0 的 VAL CW AUC 为 **+0.200 ± 0.072 点**，TEST CW 为 **+0.019 ± 0.037 点**，TEST OW 为 **−0.127 ± 0.098 点**。三个 seed 的 OW AUC 均下降。未见条件属性正确数净变化为 +1／−3／0，平均 −0.035 个百分点。

![同配置动态与静态训练对比](analysis_roundG_results_20261020/figures/02_matched_static_dynamic.png)

下面将“选权重依据”和“评价 AUC 的方式”分别列出，数值均为 G1−G0 的三 seed 平均：

{{MATCHED_SELECTION}}

在相同 native 选中权重上，CW native 与 exact AUC 的小幅差异方向相反；OW 的方向一致。采用 exact 选中权重时，G1 的 CW／OW AUC 也低于 G0。当前门控的验证集收益较清楚，TEST 排序收益较小。

## 3. seed3353 首次出现 native／exact 选择分歧

G0、G1 的 seed3351/3352 均选择第 8 轮；seed3353 的 native 选择第 9 轮，exact 选择第 6 轮。

{{SELECTION}}

![seed3353 的两种 VAL AUC 与选中轮次](analysis_roundG_results_20261020/figures/04_selection_divergence.png)

native 使用既有有限 bias 扫描，exact 使用完整临界 bias 曲线。G1/3353 的第 9 轮 native VAL AUC 比第 6 轮高 0.1306 点，而第 6 轮 exact VAL AUC 比第 9 轮高 0.3103 点；二者因此保存不同权重。

G1 的 exact 选中结果比 native 选中结果平均低 **0.193／0.215 点 TEST CW／OW native AUC**。该差异全部来自第三个 seed。主结果继续沿用既定 native 选择规则，两种选择的结果分别保留、分别分析。

## 4. 属性与跨组合竞争仍构成主要折中

下表在 TEST 的 1,891 张未见组合图像上计算，再取三 seed 均值。“CW 集合外”为该 split 的 CW 候选集之外、OW 候选集之内的组合。

{{DIAGNOSTICS}}

从 F1 到 G1，TRAIN 组合概率质量回升、CW 集合外概率质量下降，全部未见候选内的真实组合 top-1 回升。属性准确率从 25.189% 调整为 24.696%，NLL 由 2.2884 调整为 2.3286。β 中间值呈现了预期的折中。

G0 与 G1 的物体准确率、条件属性 NLL 和支持集质量接近。G1 的 CW 集合外最高分超越比例比 G0 高 0.282 个百分点，未见候选内真实组合 top-1 低 0.194 个百分点。两者的差异集中在少量组合排序上。

G1−G0 三 seed 合计的未见属性正确数变化较大项为：Synthetic／Boots.Mid-Calf +5，Satin／Sandals −3，Suede／Slippers −3，Suede／Boots.Mid-Calf −2。相同图像在三个模型中分别计数。

## 5. 约 98% 的门控方差是静态成分

在全部 22,998 张 TRAIN 图像上，门控的总体方差可分解为物体／模式均值的方差和逐图像变化：

$$
\operatorname{Var}_{b,j,k}(\gamma_{bjk})
=\operatorname{Var}_{j,k}(\bar\gamma_{jk})
+\mathbb{E}_{j,k}\operatorname{Var}_b(\gamma_{bjk}),
\qquad \bar\gamma_{jk}=\mathbb{E}_b[\gamma_{bjk}].
$$

G1 native 选中权重对应的统计与推理替换效果如下：

{{GATE}}

dynamic−TRAIN mean 的三 seed 平均为：

- TEST CW native AUC：**+0.135 ± 0.179 点**；exact AUC 为 +0.055 点。
- TEST OW native AUC：**−0.037 ± 0.187 点**；exact AUC 为 −0.037 点。
- TEST 未见条件属性：**−0.088 个百分点**，正确数变化为 0／−4／−1。

![门控方差分解与均值替换效果](analysis_roundG_results_20261020/figures/03_gate_variance_and_effect.png)

G1 exact 选中权重的动态−均值 TEST CW／OW native AUC 为 **+0.063／−0.068 点**。两条选择路径都将下一轮的重点指向图像相关门控成分。

### 5.1 固定均值门控可由现有 CP 物体因子表达

令物体文本矩阵 $E_o\in\mathbb{R}^{O\times d}$，物体 CP 因子矩阵 $V=E_oA_v^\top\in\mathbb{R}^{O\times r}$。固定均值门控后，模型使用 $V\odot\bar\Gamma$。在当前已知物体词表上，可定义：

$$
\begin{aligned}
V_{\mathrm{mean}}&=V\odot\bar\Gamma,\\
\widetilde A_v^\top&=E_o^+V_{\mathrm{mean}},\\
E_o\widetilde A_v^\top&=E_oE_o^+V_{\mathrm{mean}}=V_{\mathrm{mean}}.
\end{aligned}
$$

这里 $E_o^+$ 为 Moore–Penrose 伪逆，最后一步使用 $E_o$ 满行秩。实际 CLIP 物体文本矩阵形状为 **12×768**、秩为 **12**、条件数约 **9.49**。因此，在该词表上，固定均值门控可以吸收进静态 CP 的物体投影矩阵，同时保持原 rank。

已在 G1 六个 native／exact 选中条件上进行数值验证：保持其他权重不变，使用 19 个确定性生成的全局视觉向量，均值门控与折叠后的静态 CP 修正 logits 最大误差小于 **1×10⁻¹⁴**。该验证确认两者的算子等价关系。

这将 VG-CPOD 的视觉创新目标具体落在 $\gamma_{bjk}-\bar\gamma_{jk}$ 这一逐图像成分。下一轮针对输入表示检验这部分的学习效果。

## 6. 下一轮 H1：以 TRAIN 均值中心化的视觉残差驱动门控

**H1 = G1，改变门控输入，其他训练参数固定。** 先对冻结 CLIP 缓存中的物体查询特征统计 TRAIN 均值，再用每张图像相对该均值的偏移方向驱动门控：

$$
\begin{aligned}
\mu_j&=\frac{1}{N_{\mathrm{train}}}\sum_{b\in\mathrm{TRAIN}}f^o_{bj},\\
r_{bj}&=\frac{f^o_{bj}-\mu_j}{\max(\lVert f^o_{bj}-\mu_j\rVert_2,\varepsilon)},
\qquad \varepsilon=10^{-12},\\
q^{\mathrm{H1}}_{bjk}&=2\sigma\!\left(\frac{r_{bj}^\top(B_{\mathrm{vis}})_{k:}^\top}{0.1}\right)-1,\\
\gamma^{\mathrm{H1}}_{bjk}&=1+\frac{0.25}{2}\left(q^{\mathrm{H1}}_{bjk}
-\frac{1}{r}\sum_{\ell=1}^{r}q^{\mathrm{H1}}_{bj\ell}\right).
\end{aligned}
$$

每个物体查询的均值覆盖所有 TRAIN 图像，不按真实物体标签分组；统计矩阵为 12×768，同一份统计用于三个 seed、VAL 和 TEST。均值中心化在归一化之前成立，归一化后的向量突出相对均值的方向。CLIP 特征缓存继续复用。

H1 的模式均值继续为 1，门控仍处于原 η=0.25 对应的 [0.75,1.25] 范围。CP 的全局读出、属性／物体因子与修正强度沿用 G1。task 损失继续为：

$$
\mathcal{L}_{\mathrm{task}}(0.075)
=\frac{3}{2}\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}}.
$$

当前概率分解、损失恒等式、门控中心化和评价实现核查一致。H1 是针对本轮静态成分占主导这一现象提出的新表示设计，目标是测量突出图像偏移后，视觉动态能否带来独立的组合排序收益。

## 7. H1 的具体实验安排

| 物理 GPU | 实验 | seed | epochs |
|---|---|---|---|
| 1 | H1：TRAIN 均值中心化视觉残差门控 | 3351 | 10 |
| 2 | H1：TRAIN 均值中心化视觉残差门控 | 3352 | 10 |
| 3 | H1：TRAIN 均值中心化视觉残差门控 | 3353 | 10 |

共 **3 次新训练、30 个模型 epoch**，从对应 seed 的原始初始化开始。已有 G0／G1 直接提供静态与原门控对照。训练后继续完成 native／exact 两种权重的六项 TRAIN 均值门控评估。

| 项目 | H1 设置 |
|---|---|
| 框架 | 冻结 OpenAI CLIP ViT-L/14 |
| ρ／β／η／τγ | 0.15／0.075／0.25／0.1 |
| rank／adapter rank／κ | 32／8／20 |
| epochs／batch size | 10／64 |
| 适配器 LR／CP LR | 1e-3→1e-5／3e-4→3e-6 |
| λorth／weight decay／gradient clip | 0.001／1e-4／1 |
| 读出 | 全局 g |
| 门控输入 | 归一化的物体查询特征 TRAIN 均值残差 |

下一步代码工作包括：从服务器 TRAIN 缓存流式统计并保存均值；将该均值作为模型固定 buffer；在门控路径加入残差与归一化；新增 H1 配置、三卡调度和同 seed 汇总。均值、样本数、物体查询顺序及缓存身份随实验保存。

结果按以下顺序比较：

1. **H1−G1**：改变门控输入后，组合排序、未见属性及 CW 集合外竞争的变化。
2. **H1−G0**：新视觉动态相对同配置静态 CP 的训练收益。
3. **H1 dynamic−mean**：同一权重下逐图像变化带来的推理增量。
4. 同时记录 TRAIN 门控图像方差占比、条件属性 NLL、未见候选 top-1 和逐组合正确数，将表示变化与预测结果对应起来。

主结果继续按 VAL CW native AUC 选权重，并保留 exact 选中结果；各条件使用自己的 VAL CW／OW bias。下一轮计算投入集中于这一项输入表示改动。

## 8. 数据核查与复算文件

G 组退出码为 0，六项训练与六项均值评估均完成。核查 **26 个训练运行、12 个 exact 选中条件、24 个均值替换条件**，复算 **344 份预测导出**，核对 **180 处训练源指纹**。96 份动态重评估预测与对应原导出逐元素一致；exact/native 同权重时的 40 份导出逐元素一致，G0/G1 seed3353 的两条选择分别保存不同权重。

服务器保存的 228 行排序诊断均已复算一致；G 组源代码包的 75 份清单文件 SHA256 一致。固定 VAL bias 的 HM 与条件属性准确率复算一致；exact AUC 最大误差约 0.000049 点，native AUC 最大误差约 0.0280 点。正文主指标采用服务器保存值，配对 SD 从同 seed 差值计算。

- [完整指标](analysis_roundG_results_20261020/tables/metrics.csv)、[均值与标准差](analysis_roundG_results_20261020/tables/means.csv)、[同 seed 差值](analysis_roundG_results_20261020/tables/comparisons.csv)、[配对汇总](analysis_roundG_results_20261020/tables/paired_means.csv)。
- [排序诊断](analysis_roundG_results_20261020/tables/ranking_diagnostics.csv)、[排序诊断配对汇总](analysis_roundG_results_20261020/tables/ranking_diagnostic_paired_mean_sd.csv)、[逐组合增量](analysis_roundG_results_20261020/tables/pair_deltas.csv)。
- [TRAIN 门控统计](analysis_roundG_results_20261020/tables/moments.csv)、[训练历史](analysis_roundG_results_20261020/tables/history.csv)、[结构化分析](analysis_roundG_results_20261020/analysis.json)、[预测复算记录](analysis_roundG_results_20261020/audit.json)、[源代码与静态等价验证](analysis_roundG_results_20261020/supplementary_audit.json)。
- [复算脚本](analysis_roundG_results_20261020/analyze_g.py)、[静态等价与诊断核查脚本](analysis_roundG_results_20261020/supplementary_analysis.py)。
'''
    replacements=dict(OVERVIEW=overview,PAIRED=paired_table,MATCHED=matched,MATCHED_SELECTION=matched_selection,
        SELECTION=selection_table,DIAGNOSTICS=diagnostics,GATE=gate)
    for key,value in replacements.items():text=text.replace('{{'+key+'}}',value)
    assert '{{' not in text
    # Use measured per-seed changes in the opening beta comparison.
    values=[delta('G1-F1',s,'test_ow')['AUC_native'] for s in SEEDS]
    text=text.replace('+0.678／+0.703／+0.735','／'.join(f'{v:+.3f}' for v in values))
    (ROOT/(NAME+'.md')).write_text(text,encoding='utf-8',newline='\n')
    for name in ('ranking_diagnostics.csv','ranking_diagnostics_mean_sd.csv','ranking_diagnostic_deltas.csv',
                 'ranking_diagnostic_paired_mean_sd.csv'):
        shutil.copy2(ARCHIVE/'roundG_20261020'/name,OUT/'tables'/name)
    print(ROOT/(NAME+'.md'))


if __name__=='__main__':
    main()
