"""Report I-group evidence and the next separate-readout experiment."""
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
NAME='VG-CPOD_v1_I组结果分析与下一轮方向_20261021'
SEEDS=(3351,3352,3353)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def mean(case,setting):return next(r for r in D['means'] if r['case']==case and r['setting']==setting)
def paired(comp,setting):return next(r for r in D['paired_means'] if r['comparison']==comp and r['setting']==setting)
def metric(case,seed,setting):return next(r for r in D['metrics'] if r['case']==case and r['seed']==seed and r['setting']==setting)
def diag(case):return next(r for r in D['diagnostic_means'] if r['case']==case and r['phase']=='test' and r['group']=='unseen')
def fmt(v):return '%.3f'%v
def delta(comp,setting,key='AUC_native'):
    row=paired(comp,setting);return '%+.3f ± %.3f'%(row[key+'_mean'],row[key+'_sd'])


def main():
    primary=table(['模型','VAL CW AUC','VAL OW AUC','TEST CW AUC','TEST OW AUC','未见条件属性'],[
        [c,*[fmt(mean(c,s)['AUC_native_mean']) for s in ('val_cw','val_ow','test_cw','test_ow')],fmt(mean(c,'test_ow')['gt_object_attr_acc_unseen_mean'])+'%']
        for c in ('D1','G0','G1','H1','I1','I2')])
    comparisons=table(['比较','TEST CW AUC：均值 ± 配对 SD','TEST OW AUC：均值 ± 配对 SD','未见属性增量'],[
        [comp,delta(comp,'test_cw'),delta(comp,'test_ow'),'%+.3f 个百分点'%paired(comp,'test_ow')['gt_object_attr_acc_unseen_mean']]
        for comp in ('I1-H1','I2-H1','I1-I2','I1-G0','I2-G0','I1-G1','I2-G1')])
    seeds=table(['seed','I1 native/exact epoch','I2 native/exact epoch','I2−G1 CW AUC','I2−G1 OW AUC'],[
        [s,*['%d/%d'%(next(r['native'] for r in D['selections'] if r['case']==c and r['seed']==s),next(r['exact'] for r in D['selections'] if r['case']==c and r['seed']==s)) for c in ('I1','I2')],
         '%+.3f'%(metric('I2',s,'test_cw')['AUC_native']-metric('G1',s,'test_cw')['AUC_native']),
         '%+.3f'%(metric('I2',s,'test_ow')['AUC_native']-metric('G1',s,'test_ow')['AUC_native'])] for s in SEEDS])
    gate_rows=[]
    for c in ('G1','H1','I1','I2'):
        moments=[r for r in D['moments'] if r['case']==c]
        gate_rows.append([c,'%.5f'%np.mean([r['image_std'] for r in moments]),
            '%.3f%%'%(100*np.mean([r['image_fraction'] for r in moments])),
            delta(c+'-'+c+'_mean','test_cw'),delta(c+'-'+c+'_mean','test_ow'),
            '%+.3f'%paired(c+'-'+c+'_mean','test_ow')['gt_object_attr_acc_unseen_mean']])
    gates=table(['模型','TRAIN 图像门控标准差','图像方差占比','CW dynamic−mean','OW dynamic−mean','未见属性增量（百分点）'],gate_rows)
    decomposition=[]
    for c in ('H1','I1','I2'):
        vals=[]
        for setting in ('test_cw','test_ow'):
            vals.extend([mean(c+'_mean',setting)['AUC_native_mean']-mean('G0',setting)['AUC_native_mean'],paired(c+'-'+c+'_mean',setting)['AUC_native_mean'],mean(c,setting)['AUC_native_mean']-mean('G0',setting)['AUC_native_mean']])
        decomposition.append([c,*['%+.3f'%v for v in vals]])
    decomposition=table(['模型','CW：mean−G0','CW：dynamic−mean','CW：完整模型−G0','OW：mean−G0','OW：dynamic−mean','OW：完整模型−G0'],decomposition)
    diag_table=table(['TEST 未见图像指标','G0','G1','H1','I1','I2'],[
        [label,*['%.4f'%diag(c)[key+'_mean'] for c in ('G0','G1','H1','I1','I2')]]
        for key,label in [('object_accuracy','物体准确率（%）'),('conditional_accuracy','GT-object 条件属性准确率（%）'),
            ('conditional_nll','GT-object 条件属性 NLL'),('seen_support_pct','TRAIN 组合概率质量（%）'),
            ('extra_support_pct','CW 集合外组合概率质量（%）'),('unseen_rank_ceiling','全部未见候选内真实组合 top-1（%）'),
            ('extra_dominates_valid','CW 集合外最高分超过 CW 内未见最高分（%）')]])
    a=D['audit'];checks=json.loads((OUT/'next_equations_audit.json').read_text())
    identity_error=max(r['tied_readout_max_error'] for r in checks['records'])
    text=r'''# VG-CPOD v1：I 组结果分析与下一轮方向

日期：2026-10-21。框架：冻结 OpenAI CLIP ViT-L/14；数据：UT-Zappos。分析本次下载的 I1／I2 各三个 seed 和十二项均值门控评估。

**下一轮优先做 J1：沿用 I2 的残差门控，分别学习静态 CP 读出与动态 CP 读出。** 属性／物体因子继续共享，仅增加一个 768×32 动态读出矩阵，三 seed 各训练 10 epochs。现有 I2 是共享读出的直接对照，G0／G1 继续提供静态和原门控参照。

I1／I2 相对 H1 的 TEST CW／OW AUC 分别回升 **0.528／0.918 点、0.576／0.983 点**，三个 seed 均回升。I1 与 I2 的差异很小：I2 的 CW／OW 均值高 0.048／0.064 点，TEST 未见属性预测一致率为 99.63%–99.84%。本轮支持继续使用较温和的残差投影尺度；保留原始幅值与归一化后调整温度，两种处理给出的表现接近。

I2 相对静态 G0 的 TEST CW／OW 仍为 **−0.307／−0.330 点**；其动态门控相对自身 TRAIN 均值条件提高 **0.428／0.098 点**。下一步计算集中于静态和动态修正共用读出矩阵这一结构约束。

## 1. 本轮完整结果

主表使用各模型原定的 VAL CW native AUC 选权重规则，数值为三 seed 均值。AUC 为面积乘 100；条件属性使用真实物体标签。

PRIMARY

D1 的 TEST CW／OW 为 34.791／25.453，继续作为当前性能参照。G0／G1 与 I 组使用相同的 β=0.075，适合分析门控与结构变化；D1 的 β=0.1，反映已有方案达到的结果。

COMPARISONS

![完整结果及两种选权重路径](analysis_roundI_results_20261021/figures/01_results_and_selection.png)

I2 的 VAL CW 比 I1 高 0.005 点，VAL OW 低 0.051 点；TEST CW／OW 高 0.048／0.064 点。两组输入尺度相近后，保留幅值与使用单位方向得到非常接近的结果。

在 1,891 张 TEST 未见图像中，I1 相对 I2 的条件属性正确数净变化为 0／−3／−2。正确数和预测一致性共同表明，接下来的主变量可以转向 CP 的静态／动态读出。后续固定 I2 门控，便于与已有 I2 一对一比较。

## 2. 选权重差异解释了部分均值变化

SEEDS

I1 的 seed3351 出现 native 第 8 轮、exact 第 10 轮的差异。I2 两种选择都为 8／8／6；G1 的 native 为 8／8／9、exact 为 8／8／6。

| 比较 | 选权重依据 | TEST CW native AUC 差值 | TEST OW native AUC 差值 | CW exact AUC 差值 | OW exact AUC 差值 |
|---|---|---|---|---|---|
| I1−I2 | native | −0.048 | −0.064 | −0.025 | −0.062 |
| I1−I2 | exact | +0.075 | −0.007 | +0.088 | −0.003 |
| I2−G1 | native | −0.326 | −0.204 | −0.278 | −0.192 |
| I2−G1 | exact | −0.133 | +0.012 | −0.092 | +0.018 |

选权重改变后，I1/I2 的小幅 CW 排名翻转。I2 与 G1 在 exact 路径的 OW 基本持平，CW 仍低 0.133 点。两组已经收回 H1 的主要退化，当前结果更适合支持结构实验，而非继续围绕 I1/I2 的小幅排名细分超参。

## 3. 投影尺度影响明确，动态强弱与整体收益需要分别看

以下门控矩使用各模型选中权重，在全部 22,998 张 TRAIN 图像上统计。dynamic−mean 指同一权重下，逐图像门控相对 TRAIN 物体／模式均值门控的配对差值。

GATES

H1 的图像门控标准差约 0.066，I1/I2 降至约 0.027–0.028，图像方差占比仍接近 100%。本轮调低了实际图像调制尺度，同时保留了中心化门控的动态图像成分。

![门控变化与推理干预收益](analysis_roundI_results_20261021/figures/02_gate_effects.png)

I2 的 CW dynamic−mean 增量为 +0.325／+0.400／+0.559 点，三个 seed 为正；OW 为 −0.053／+0.300／+0.048 点。未见属性正确数净变化为 +10／−3／+5，三 seed 平均提高 0.212 个百分点。

将完整结果写成“均值条件相对 G0 的差距”与“图像动态的推理贡献”之和：

DECOMPOSITION

以 I2 为例，均值条件比独立训练 G0 的 CW／OW 低 0.735／0.429 点，逐图像门控回收 0.428／0.098 点，完整 I2 最后仍低 0.307／0.330 点。

均值条件是已训练模型的推理干预；G0 是独立训练的静态模型。这两项比较分别显示同一模型中的动态贡献和整个训练方案的表现。它们提出下一轮的具体假设：**给静态和动态修正各自的读出矩阵，能否改善整体排序，同时保留动态图像贡献。**

![均值条件、动态贡献和完整训练效果](analysis_roundI_results_20261021/figures/03_static_dynamic_effects.png)

## 4. 属性迁移与组合排序的剩余差距

DIAGNOSTICS

I2 相对 H1 的未见条件属性提高 0.740 个百分点，条件属性 NLL 从 2.3831 降到 2.3433；未见候选内真实组合 top-1 从 47.594% 回升至 49.286%。这些变化与 CW／OW 回升一致。

相对 G0，I2 物体准确率基本持平，未见条件属性低 0.670 个百分点，未见候选内真实组合 top-1 低 0.353 个百分点。下一项结构实验仍聚焦条件属性和组合排序。

## 5. 当前公式核查与 J1 的改动依据

### 5.1 当前模型中两类修正共用 Wc

令 $u_i=A_u e_i^a$、$v_j=A_v e_j^o$，$a_{ij}=u_i\odot v_j$，$h_{bj}=\gamma_{bj}-\mathbf{1}$。其中 $g_b\in\mathbb{R}^{768}$ 为图像全局 CLIP 特征，$\kappa=20$；$i,j,b$ 分别索引属性、物体和图像。当前 I2 的修正可以精确展开为：

$$
\begin{aligned}
\Delta U^{\mathrm{I2}}_{bji}
&=\rho\kappa\,g_b^\top W_c(a_{ij}\odot\gamma_{bj})\\
&=\rho\kappa\,g_b^\top W_c a_{ij}
+\rho\kappa\,g_b^\top W_c(a_{ij}\odot h_{bj}).
\end{aligned}
$$

第一项是静态 CP 修正，第二项是图像动态修正，两者通过同一个 Wc 接收训练更新。这个共享约束来自当前实现。本轮的均值干预与独立静态训练结果，支持将其作为下一项结构变量。

现有概率链式分解、TRAIN 残差、门控中心化、张量维度和损失恒等式复核一致。J1 调整参数共享方式。

### 5.2 J1：分别学习静态和动态读出

设 $W_s,W_d\in\mathbb{R}^{768\times32}$，继续共享 $A_u,A_v$，使用：

$$
\begin{aligned}
U^{\mathrm{J1}}_{bji}
&=U^{\mathrm{base}}_{bji}+\Delta U^{\mathrm{J1}}_{bji},\\
\Delta U^{\mathrm{J1}}_{bji}
&=\rho\kappa\,g_b^\top W_s a_{ij}
+\rho\kappa\,g_b^\top W_d(a_{ij}\odot h_{bj}).
\end{aligned}
$$

静态和动态两路联合训练。$W_s=W_d=W_c$ 时，公式精确恢复 I2；$W_d=0$ 时，CP 修正恢复静态形式。对应类别算子为：

$$
M_{bj}=\left[W_s\operatorname{diag}(v_j)
+W_d\operatorname{diag}(v_j\odot h_{bj})\right]A_u,
\qquad \operatorname{rank}(M_{bj})\le32.
$$

共享属性因子使算子的秩上界继续为 32。单个修正分数对两路矩阵的梯度为：

$$
\begin{aligned}
\nabla_{W_s}\Delta U^{\mathrm{J1}}_{bji}
&=\rho\kappa\,g_b a_{ij}^\top,\\
\nabla_{W_d}\Delta U^{\mathrm{J1}}_{bji}
&=\rho\kappa\,g_b(a_{ij}\odot h_{bj})^\top.
\end{aligned}
$$

当前共享 Wc 接收的分数梯度是上述两项之和；J1 将两项梯度分配给各自矩阵，再由优化器更新。属性／物体因子仍共同训练。该实验检验解除读出共享约束后的整体效果。

新增参数为 **24,576**，可训练参数从 I2 的 **122,880** 变为 **147,456**，增加 20%。I2 作为共享读出对照，J1 的 dynamic−mean 继续测量同一模型中的图像变化贡献。

### 5.3 门控与损失固定

J1 沿用 I2：

$$
\begin{aligned}
r_{bj}&=f^o_{bj}-\mu_j,\qquad
\widehat r_{bj}=\frac{r_{bj}}{\max(\lVert r_{bj}\rVert_2,10^{-12})},\\
q_{bjk}&=2\sigma\!\left(\frac{b_k^\top\widehat r_{bj}}{0.56}\right)-1,\\
h_{bjk}&=\frac{0.25}{2}\left(q_{bjk}-\frac{1}{32}\sum_{\ell=1}^{32}q_{bj\ell}\right),\\
\mathcal{L}&=\frac{3}{2}\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}}
+10^{-3}\mathcal{L}_{\mathrm{orth}}.
\end{aligned}
$$

$b_k$ 是视觉门控矩阵第 $k$ 行的转置。η 保持 0.25，门控 γ 仍在 [0.75,1.25]；ρ=0.15、β=0.075。两路矩阵独立后，修正分数由各自权重共同决定。

## 6. 下一轮具体安排

| 项目 | J1 设置 |
|---|---|
| 主要变量 | 静态 W_s、动态 W_d 分别学习 |
| 属性／物体因子 | 共享原 A_u、A_v |
| CLIP／输入缓存 | 冻结 OpenAI ViT-L/14，复用现有缓存和 TRAIN 均值 |
| 门控输入／τγ／η | I2 归一化残差／0.56／0.25 |
| ρ／β／rank／adapter rank | 0.15／0.075／32／8 |
| epoch／batch | 10／64 |
| 初始化 | 各 seed 原始初始化；W_s=0、W_d=0 |
| CP 学习率 | 两路均 3e-4，原余弦日程降到 3e-6 |
| 适配器学习率 | 1e-3，余弦降到 1e-5 |
| λorth／weight decay／gradient clip | 0.001／1e-4／1 |
| GPU1／GPU2／GPU3 | seed3351／3352／3353 |
| 新增训练量 | 3 次训练、30 个模型 epoch |
| 训练后评估 | native／exact 各三项 TRAIN 均值门控评估，共 6 项 |
| 已有对照 | I2、G0、G1；D1 作性能参照，另保留 I1/H1 结果 |

先比较 J1−I2：解除读出共享约束的完整训练效果；再比较 J1−G0/G1，以及 J1 dynamic−mean。并列记录条件属性、候选排序、两路修正的 RMS 和选权重路径。

主要选择依据继续为 VAL CW native AUC，exact 选择并列保存。输入尺度、β、ρ、rank 等保持上表数值，让下一轮围绕一个结构变量得出结论。

### 6.1 后续代码改动范围

1. `vgcpod/model.py`：将当前 Wc 作为静态 W_s，新增零初始化 W_d，前向分别计算静态项与 h 动态项；共享 A_u/A_v。
2. `mean_gate_eval.py`：均值条件使用同一 W_s，加上 W_d 对 TRAIN 均值 h 的动态项。新前向中两路矩阵需要分别参与评估。
3. 训练日志：增加静态／动态修正的独立 RMS；两路矩阵沿用现有 CP 参数组与学习率规则。
4. 配置／调度／汇总：新增 J1，三卡三个 seed，自动与现有 I2/G0/G1 做配对比较，并保存两条选权重路径。

均值条件的正确表达式为：

$$
\bar h_{jk}=\frac{1}{N}\sum_{b\in\mathrm{TRAIN}}h_{bjk},
\qquad
\Delta U^{\mathrm{J1,mean}}_{bji}
=\rho\kappa g_b^\top W_s a_{ij}
+\rho\kappa g_b^\top W_d(a_{ij}\odot\bar h_j).
$$

本次已用三组真实 I2 权重和合成输入检验共享矩阵恢复、静态特例及梯度公式；float64 最大恢复误差为 IDENTITY_ERROR。这是下一轮公式的数值核查，实验代码保持本轮版本。

## 7. 数据与实现复核

- 六次训练与十二项评估全部完成，整轮退出码为 0。
- H/I 的 TRAIN 均值文件 SHA256 相同；I1/I2 六个模型的 native／exact／末轮 checkpoint 均值 buffer 与保存文件逐元素一致。
- 核对 RELEASE_FILES 个交付文件哈希、SOURCE_FILES 个运行源文件指纹，读取 PREDICTION_FILES 个预测文件并重算 RANKING_ROWS 行排序诊断。
- 保存曲线积分与 AUC 的最大差异为 1.42e-14；概率复算 exact AUC 的最大差异为 4.88e-5 点；固定 VAL bias 的 HM、未见条件属性准确率复算一致。
- float32 归一化概率重算 native 扫描的最大差异为 0.028018 点，主表使用原始分数保存的曲线积分。native／exact 两类结果均在报告中保留。
- 下载包包含 TRAIN 均值与身份信息；原始大特征数组仍保存在服务器。本次复算依据 checkpoint、预测、曲线和日志。

明细位于 `analysis_roundI_results_20261021/`：`analysis.json`、`audit.json`、`next_equations_audit.json`、`tables/` 和 `figures/`。
'''
    replacements=dict(PRIMARY=primary,COMPARISONS=comparisons,SEEDS=seeds,GATES=gates,
        DECOMPOSITION=decomposition,DIAGNOSTICS=diag_table,IDENTITY_ERROR='%.2e'%identity_error,
        RELEASE_FILES=str(a['release_files_verified']),SOURCE_FILES=str(a['source_files_verified']),
        PREDICTION_FILES=str(a['prediction_files']),RANKING_ROWS=str(a['ranking_rows_verified']))
    for key,value in replacements.items():text=text.replace(key,value)
    target=ROOT/(NAME+'.md');target.write_text(text,encoding='utf-8',newline='\n')
    subprocess.run(['C:/Users/llenovo/anaconda3/Scripts/pandoc.exe',str(target),'--standalone','--self-contained','--mathml',
        '--from=markdown+tex_math_dollars+pipe_tables','--to=html5','--css='+str(ROOT/'analysis_20261018/report.css'),
        '--metadata=pagetitle:VG-CPOD v1 I组结果分析与下一轮方向','--output='+str(ROOT/(NAME+'.html'))],check=True)
    print(target)


if __name__=='__main__':main()
