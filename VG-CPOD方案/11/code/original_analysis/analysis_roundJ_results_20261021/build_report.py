"""Write the J1 evidence, optimizer derivation and next two learning-rate experiments."""
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
S=json.loads((OUT/'supplement.json').read_text(encoding='utf-8'))
NAME='VG-CPOD_v1_J组结果分析与下一轮方向_20261021'
SEEDS=(3351,3352,3353)

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])
def mean(case,setting):return next(r for r in D['means'] if r['case']==case and r['setting']==setting)
def paired(comp,setting):return next(r for r in D['paired_means'] if r['comparison']==comp and r['setting']==setting)
def metric(case,seed,setting):return next(r for r in D['metrics'] if r['case']==case and r['seed']==seed and r['setting']==setting)
def diag(case):return next(r for r in D['diagnostic_means'] if r['case']==case and r['phase']=='test' and r['group']=='unseen')

def main():
    main_table=table(['模型','VAL CW AUC','VAL OW AUC','TEST CW AUC','TEST OW AUC','TEST 未见条件属性'],[
        [c,*['%.3f'%mean(c,s)['AUC_native_mean'] for s in ('val_cw','val_ow','test_cw','test_ow')],
         '%.3f%%'%mean(c,'test_ow')['gt_object_attr_acc_unseen_mean']] for c in ('D1','G0','G1','H1','I2','J1')])
    paired_table=table(['比较','CW AUC：均值 ± 配对 SD','OW AUC：均值 ± 配对 SD','未见条件属性增量'],[
        [c,*['%+.3f ± %.3f'%(paired(c,s)['AUC_native_mean'],paired(c,s)['AUC_native_sd']) for s in ('test_cw','test_ow')],
        '%+.3f 个百分点'%paired(c,'test_ow')['gt_object_attr_acc_unseen_mean']] for c in ('J1-I2','J1-G0','J1-G1','J1-D1')])
    seed_table=table(['seed','J1 native/exact epoch','J1−I2 CW','J1−I2 OW','未见条件属性净正确数'],[
        [s,' / '.join(str(next(r for r in D['selections'] if r['case']=='J1' and r['seed']==s)[k]) for k in ('native','exact')),
         *['%+.3f'%(metric('J1',s,t)['AUC_native']-metric('I2',s,t)['AUC_native']) for t in ('test_cw','test_ow')],
         '%+d'%next(r['net_correct'] for r in S['prediction_differences'] if r['comparison']=='J1-I2' and r['seed']==s and r['phase']=='test' and r['group']=='unseen')]
        for s in SEEDS])
    diagnostic_table=table(['TEST 未见图像指标','G0','I2','J1'],[
        [label,*['%.4f'%diag(c)[key+'_mean'] for c in ('G0','I2','J1')]]
        for key,label in [('object_accuracy','物体准确率（%）'),('conditional_accuracy','GT-object 条件属性准确率（%）'),
            ('conditional_nll','GT-object 条件属性 NLL'),('extra_support_pct','CW 集合外组合概率质量（%）'),
            ('unseen_rank_ceiling','全部未见候选内真实组合 top-1（%）'),('extra_dominates_valid','集合外最高分超过 CW 内未见最高分（%）')]])
    mean_table=table(['模型','TEST CW dynamic−mean','TEST OW dynamic−mean','TEST CW 均值条件','TEST OW 均值条件'],[
        [c,*['%+.3f'%(mean(c,t)['AUC_native_mean']-mean(c+'_mean',t)['AUC_native_mean']) for t in ('test_cw','test_ow')],
         *['%.3f'%mean(c+'_mean',t)['AUC_native_mean'] for t in ('test_cw','test_ow')]] for c in ('I2','J1')])
    component_rows=[];adam_rows=[]
    for seed in SEEDS:
        r=next(r for r in D['readout_history'] if r['seed']==seed and r['selected'])
        p=next(r for r in D['parameters'] if r['case']=='J1' and r['seed']==seed and r['stage']=='selected')
        component_rows.append([seed,*['%.3f'%v for v in (p['Wc_norm'],p['Wd_norm'],r['cp_static_centered_rms'],r['cp_dynamic_centered_rms'],r['centered_component_cosine'])]])
        ws=next(r for r in S['adam_steps'] if r['seed']==seed and r['parameter']=='Wc')
        wd=next(r for r in S['adam_steps'] if r['seed']==seed and r['parameter']=='Wd')
        adam_rows.append([seed,'%.4f'%(wd['moment_norm']/ws['moment_norm']),'%.4f'%(wd['adam_adaptive_step_norm']/ws['adam_adaptive_step_norm'])])
    components=table(['seed','静态矩阵范数','动态矩阵范数','静态中心化 RMS','动态中心化 RMS','两路中心化余弦'],component_rows)
    adam=table(['seed','动态/静态：一阶矩范数比','动态/静态：Adam 自适应步长范数比'],adam_rows)
    text=r'''# VG-CPOD v1：J 组结果分析与下一轮方向

日期：2026-10-21。框架：冻结 OpenAI CLIP ViT-L/14；数据：UT-Zappos。

**下一轮优先为动态读出 Wd 单独设置学习率，做 K1=9e-5、K2=3e-5 两档、各三个 seed。** J1 的独立读出已经产生新的图像动态作用；当前收益在不同 seed 与 CW/OW 间分化。checkpoint 显示，小门控对应的小梯度经 Adam 自适应归一化后，仍得到接近静态读出的更新步长。下一步直接检验两路学习速度的配比。

现有 J1 公式与代码核查一致。本报告给出优化器分组和配置建议，实验源代码保持 J 组版本。

## 1. 完整结果：CW 改善，OW 由 seed3351 拉低

三次训练和六项均值评估全部完成，总退出码为 0。主表使用各模型预定的 VAL CW native AUC 选权重，数值为三个 seed 的均值；AUC 为面积乘 100。条件属性使用真实物体标签。

MAIN_TABLE

PAIRED_TABLE

J1 相对 I2 的 TEST CW AUC 提高 0.448 点、未见条件属性提高 0.370 个百分点，OW AUC 下降 0.328 点。相对相同 β/ρ 的 G0，CW 高 0.141 点、OW 低 0.658 点。D1 继续作为已有性能参照；D1 的 β=0.1，其余 G/I/J 对照使用 β=0.075。

![三 seed 结果与历史参照](analysis_roundJ_results_20261021/figures/01_results.png)

SEED_TABLE

seed3352/3353 相对 I2 的 CW/OW 都提高；seed3351 的 OW 下降 2.445 点，是均值下降的主要来源。J1 两种选权重都为 5/6/6，参数及预测完全相同，因此当前 J1 的 native/exact 选权重路径给出同一组模型。对 I2 的配对结果也相同。

J1 的 TEST CW/OW HM@VAL-bias 为 49.544/39.506，相对 I2 下降 0.122/0.527 点。CW AUC 的增量主要体现在扫描整条 seen/unseen 权衡曲线后的面积，固定 VAL bias 的 HM 尚未同步改善。

## 2. 剩余差距：条件属性改善，候选竞争仍有分化

DIAGNOSTIC_TABLE

J1 相对 I2 的条件属性 NLL 从 2.3433 降至 2.3087，准确率有所提高；物体准确率下降 0.529 个百分点，全部未见候选内真实组合 top-1 下降 0.529 个百分点，CW 集合外最高分超过有效未见最高分的比例提高 1.692 个百分点。这些量共同解释了 CW 与 OW 的不同表现。

seed3351 中，物体准确率从 64.093% 降至 62.665%，未见候选 top-1 从 50.608% 降至 47.488%，集合外候选占优比例从 24.167% 升至 28.715%。同一 seed 的条件属性 NLL 略有改善，但条件属性准确率净减少 23 张。其问题同时涉及物体和组合排序。

seed3353 的未见条件属性净增加 41 张，seed3352 净增加 3 张。这一轮表明独立动态读出能够学习有用修正，下一轮需要比较两路更新速度如何影响完整排序结果。上述集合外类别仅用于描述已有预测。

## 3. 小门控已经对应较大的动态分数修正

### 3.1 门控幅值与分数幅值是两个量

J1 的 TRAIN 图像门控标准差三 seed 平均为 0.02686，I2 为 0.02807，门控输入、温度和 η 沿用 I2。两组的动态分数作用明显不同。

对每张 TEST 未见图像，在真实物体条件下计算 dynamic 与 TRAIN 均值门控两种条件的 logit 差，并沿属性维去均值后计算 RMS。I2 三 seed 为 0.0642/0.0631/0.0688；J1 为 0.5870/0.6810/0.6760。三 seed 均值从 0.0653 增至 0.6480，约增大 9.9 倍。

MEAN_TABLE

J1 的 dynamic−mean 达到 CW +8.645、OW +3.592 点。该比较测量已训练 J1 对图像动态项的依赖；J1 相对独立训练 I2 的完整训练增量为 CW +0.448、OW −0.328 点。J1 均值条件的 TEST CW 低于独立静态 G0 约 8.504 点，说明两路在当前联合训练中形成了较强配合。

J1 在 VAL 上的 dynamic−mean 为 CW +0.172、OW −1.287 点，与 TEST 的幅度和方向不同。下一轮继续按 VAL 选权重，并同时报告两种世界的结果和三个 seed 的差值。

![门控、动态 logit 修正与优化器步长](analysis_roundJ_results_20261021/figures/02_optimizer_and_scale.png)

### 3.2 两路修正共同增长

COMPONENTS

表中修正 RMS 为选中 epoch 各训练 batch 的图像数加权统计；中心化沿每个物体的属性维进行。两路中心化余弦约为 0.52–0.57，表示训练输出上两项总体同向。矩阵范数描述参数尺度，中心化分数 RMS 则直接描述条件 softmax 相关的修正尺度。

![静态与动态修正的训练过程](analysis_roundJ_results_20261021/figures/03_readout_history.png)

## 4. 下一轮的依据：Adam 使梯度尺度与步长尺度分离

### 4.1 当前前向公式核查

令 $a_{ij}=(A_ue_i^a)\odot(A_ve_j^o)$，$h_{bj}=\gamma_{bj}-\mathbf{1}$，当前 J1 使用：

$$
U_{bji}=U^{\mathrm{base}}_{bji}
+\rho\kappa g_b^\top W_s a_{ij}
+\rho\kappa g_b^\top W_d(a_{ij}\odot h_{bj}).
$$

$W_s$ 在代码中为 `Wc`。两路共享属性／物体因子；$W_s=W_d=W_c$ 时精确恢复 I2。均值评估分别使用静态项与 $W_d$ 的均值动态项，预测复算一致。当前需要优化两路学习率的配置。

### 4.2 checkpoint 直接显示的优化器尺度

ADAM_TABLE

一阶矩和自适应步长取自选中 checkpoint 最后一个训练 step 的 AdamW 状态。步长范数使用带偏差校正的一阶／二阶矩和该 step 学习率重建，统计自适应项，weight decay 单独执行。Wd 的一阶矩范数约为静态读出的 2%–3%，其自适应步长范数约为静态读出的 95%–100%。

Adam 的核心递推为：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)G_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)G_t^{\odot2},\\
\widehat m_t&=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\\
\Delta\theta_t^{\mathrm{Adam}}&=-\alpha_t
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\end{aligned}
$$

以上除法与开方均逐元素执行。若整段梯度历史统一乘以正数 $c$，从零矩状态开始，则：

$$
\widehat m'_t=c\widehat m_t,\qquad
\widehat v'_t=c^2\widehat v_t,\qquad
\Delta\theta'_t=-\alpha_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon/c}.
$$

当 ε 项较小时，缩小梯度主要被二阶矩归一化抵消。真实两路梯度并非统一常数缩放；上表是实际 checkpoint 中两路优化器状态的测量。它说明 η=0.25 下的小门控，仍可通过独立 Wd 学出较大的分数修正。直接改变 Wd 的学习率可以明确改变其更新步长。

## 5. 下一轮具体安排：K1/K2 动态读出学习率

保留 J1 的前向、损失与独立读出，增加动态读出学习率倍率 $m_d$：

$$
\alpha_{s,t}=\alpha_{\mathrm{CP},t},\qquad
\alpha_{d,t}=m_d\alpha_{\mathrm{CP},t},\qquad
m_d\in\{0.3,0.1\}.
$$

| 实验 | Wd 初始学习率 | Wd 末步学习率 | 作用 |
|---|---|---|---|
| J1：已完成 | 3e-4 | 3e-6 | 原独立读出对照 |
| K1：优先 | 9e-5 | 9e-7 | 动态读出以静态 CP 的 0.3 倍速度更新 |
| K2 | 3e-5 | 3e-7 | 覆盖 0.1 倍更新速度，与 K1 比较尺度趋势 |

两组各三个 seed，直接完整训练：GPU1/2/3 分别运行 seed3351/3352/3353，各卡顺序执行 K1→K2。总计 6 次训练、60 个模型 epoch；native/exact 各自做 TRAIN 均值评估，共 12 项零优化步评估。

| 固定项 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 OpenAI ViT-L/14，复用现有缓存和 TRAIN 均值 |
| 读出／因子 | J1 独立 W_s、W_d，继续共享 A_u/A_v |
| 门控输入／τγ／η | 归一化 TRAIN 残差／0.56／0.25 |
| ρ／β／rank／adapter rank | 0.15／0.075／32／8 |
| 其他 CP 学习率 | 3e-4 → 3e-6 |
| 适配器学习率 | 1e-3 → 1e-5 |
| epochs／batch | 10／64 |
| 初始化 | 各 seed 原始初始化，两路读出均为零 |
| λorth／weight decay／gradient clip | 0.001／1e-4／1 |
| 主要选权重依据 | VAL CW native AUC，独立保存 exact 路径 |
| 对照 | J1、I2、G0/G1；D1 作性能参照 |

### 5.1 下一轮代码改动范围

1. `vgcpod/engine.py::make_optimizer`：将 Wd 放入单独参数组。adapter 组沿用原学习率；原 CP 组包含 A_u/A_v/Bvis/Wc；动态组包含 Wd，`lr_scale = cp_lr_scale * dynamic_readout_lr_multiplier`。每个参数只属于一个组。
2. K1/K2 配置：新增 `dynamic_readout_lr_multiplier=0.3/0.1`，J1 对应默认倍率 1。原余弦日程按各组倍率计算学习率。
3. 日志与汇总：保存各组学习率、两路中心化 RMS、选中 checkpoint 的矩范数／自适应步长、native/exact 配对指标和 dynamic−mean。
4. 三卡调度：六次从对应 seed 初始化的训练，复用 J 组 TRAIN 均值，自动完成十二项评估与历史比较。

### 5.2 结果分析重点

首先比较 K1/K2−J1 的同 seed CW/OW AUC 与固定 VAL bias 的 HM，确认降低 Wd 更新速度后的完整效果；再看相对 I2/G0/G1 的结果。重点跟踪 seed3351 的物体准确率、未见候选 top-1 和集合外候选占优比例，并同时记录其他两个 seed 的变化。

动态 logit RMS、两路矩阵范数和 dynamic−mean 用于解释学习过程；完整训练的 CW/OW、条件属性和候选排序共同决定下一步方向。本轮已有结果支持优先投入两路优化速度的实验。

## 6. 复核记录

- J1 的 3 次训练、6 项评估全部完成，主程序退出码为 0。
- 核对 90 个交付文件哈希、30 项训练源文件指纹。I 组下载包哈希与上一轮独立核查记录一致，历史数据沿用已核验分析。
- 本次复算 48 个预测文件；native 与 exact 同轮选中，12 个对应预测完全相同；24 个动态复评预测与原预测逐元素一致。
- 重算 306 行排序诊断；保存曲线积分最大误差 7.11e-15，概率复算 exact AUC 最大误差 5.17e-5 点；native AUC、固定 VAL bias HM、条件属性准确率复算一致。
- TRAIN 均值与 I 组逐字相同，所有 J1 选中／末轮 checkpoint buffer 一致；损失恒等式和两路 RMS 一致。
- 原始大特征缓存仍在服务器，本次分析使用下载的 checkpoint、预测、曲线、日志与固定 TRAIN 均值。

明细位于 `analysis_roundJ_results_20261021/`：`analysis.json`、`audit.json`、`supplement.json`、`tables/`、`figures/`。本轮分析未修改实验源代码。
'''
    for key,value in dict(MAIN_TABLE=main_table,PAIRED_TABLE=paired_table,SEED_TABLE=seed_table,DIAGNOSTIC_TABLE=diagnostic_table,
        MEAN_TABLE=mean_table,COMPONENTS=components,ADAM_TABLE=adam).items():text=text.replace(key,value)
    target=ROOT/(NAME+'.md');target.write_text(text,encoding='utf-8',newline='\n')
    subprocess.run(['C:/Users/llenovo/anaconda3/Scripts/pandoc.exe',str(target),'--standalone','--self-contained','--mathml',
        '--from=markdown+tex_math_dollars+pipe_tables','--to=html5','--css='+str(ROOT/'analysis_20261018/report.css'),
        '--metadata=pagetitle:VG-CPOD v1 J组结果分析与下一轮方向','--output='+str(ROOT/(NAME+'.html'))],check=True)
    print(target)

if __name__=='__main__':main()
