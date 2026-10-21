"""K results and a change of research direction based on accumulated evidence."""
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
S=json.loads((OUT/'supplement.json').read_text(encoding='utf-8'))
NAME='VG-CPOD_v1_K组结果分析与后续主线决策_20261021'

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])

def metric(case,setting):return next(r for r in D['means'] if r['case']==case and r['setting']==setting)
def paired(comparison,setting):return next(r for r in D['paired_means'] if r['comparison']==comparison and r['setting']==setting)
def candidate(case):return next(r for r in S['candidate_means'] if r['case']==case and r['phase']=='test' and r['world']=='ow')

def main():
    cases=('D1','G0','G1','I2','J1','K1','K2')
    results=table(['模型','VAL CW','VAL OW','TEST CW','TEST OW','TEST OW HM@VAL-bias'],[
        [c,*['%.3f'%metric(c,t)['AUC_native_mean'] for t in ('val_cw','val_ow','test_cw','test_ow')],
        '%.3f'%metric(c,'test_ow')['HM_at_val_bias_mean']] for c in cases])
    paired_table=table(['同 seed 比较','TEST CW AUC 增量 ± SD','TEST OW AUC 增量 ± SD'],[
        [c,*['%+.3f ± %.3f'%(paired(c,t)['AUC_native_mean'],paired(c,t)['AUC_native_sd']) for t in ('test_cw','test_ow')]]
        for c in ('K1-J1','K2-J1','K1-G0','K2-G0','K1-D1','K2-D1','K1-K2')])
    scale_rows=[]
    for case in ('J1','K1','K2'):
        adam=[r for r in S['optimizer'] if r['case']==case and r['stage']=='native_selected' and r['parameter']=='Wd']
        scale_rows.append([case,'%.3f'%np.mean([r['parameter_norm'] for r in adam]),
            '%.3f'%np.mean([r['lr_scaled_adaptive_step_norm_over_static'] for r in adam]),
            '%.4f'%np.mean([r['image_std'] for r in D['moments'] if r['case']==case]),
            '%.3f'%np.mean([r['dynamic_minus_mean_centered_rms_unseen'] for r in S['dynamic_scales'] if r['case']==case and r['phase']=='test']),
            *['%+.3f'%(metric(case,t)['AUC_native_mean']-metric(case+'_mean',t)['AUC_native_mean']) for t in ('test_cw','test_ow')]])
    scales=table(['模型','Wd 范数','Wd/Ws 实际自适应步长','门控图像标准差','动态条件 logit RMS','CW dynamic−mean','OW dynamic−mean'],scale_rows)
    errors=table(['模型','全部未见候选 top-1','真实物体内未见属性 top-1','同物体内排错','跨物体竞争排错'],[
        [c,'%.2f%%'%candidate(c)['unseen_pool_accuracy_mean'],
        '%.2f%%'%candidate(c)['gt_object_unseen_pool_accuracy_mean'],
        '%.2f%%'%(100*candidate(c)['within_object_ranking_errors_mean']/1891),
        '%.2f%%'%(100*candidate(c)['cross_object_competition_errors_mean']/1891)] for c in ('D1','G0','I2','J1','K1','K2')])
    common=table(['seed','G0 排错且后续四组也全部排错','占 G0 原有错误比例'],[
        [r['seed'],r['all_wrong'],'%.2f%%'%r['pct_g0_errors']] for r in S['common_errors'] if r['phase']=='test' and r['metric']=='unseen'])
    exact=table(['选权重规则','G0 CW/OW','J1 CW/OW','K1 CW/OW','K2 CW/OW'],[
        [label,*['%.3f / %.3f'%(metric(c+suffix,'test_cw')['AUC_native_mean'],metric(c+suffix,'test_ow')['AUC_native_mean']) for c in ('G0','J1','K1','K2')]]
        for label,suffix in [('VAL native 选中',''),('VAL exact 选中','_exact')]])
    text=r'''# VG-CPOD v1：K 组结果分析与后续主线决策

日期：2026-10-21。数据：已下载 K1/K2 全部结果及历轮归档。当前框架：冻结 OpenAI CLIP ViT-L/14，UT-Zappos。

**建议结束现有冻结缓存头上的门控与学习率搜索。下一阶段集中研究“物体条件的局部判别证据”：先在同一 patch 比较两个属性假设，再汇聚证据形成组合分数。** CLIP 视觉 Adapter 和静态 CP 作为共同基座；创新集中在判别算子及其可证明性质。G0 保留为静态对照，D1 保留为历史性能参照。正式计划为两数据集、三种读取、三 seed，共 18 次完整训练。

上一轮提出降低 Wd 学习率，是基于 J1 的 Adam 状态。K 组证实这个改动有效控制了更新量，却没有改善整体结果。因此，继续细分 Wd 学习率已失去优先级。更大的问题在于：多轮模型反复保留了同一批组合排序错误，而特征提取和局部查询始终无法接受任务梯度。

本报告完成现有预测、checkpoint 与代码的复算分析；训练代码沿用已交付 K 版。

## 1. K 组结果：更新速度变了，泛化水平没有上移

6 次训练、12 项均值门控评估全部完成，主程序退出码为 0。下表是三 seed 均值，CW/OW 列为 native AUC，单位为点；同一行使用 VAL CW native AUC 选出的同一组权重。

__RESULTS__

K1 相对 J1 的 CW 下降 0.333 点，OW 增加 0.031 点；K2 相对 J1 的 CW/OW 都下降。K1/K2 均低于静态 G0，也低于 D1。K1 的 VAL CW 最高，但 TEST 未形成对应提升。D1 的 β=0.1；G0/G1/I2/J1/K1/K2 的 β=0.075，因而 G0 是本轮更直接的静态对照。

__PAIRED__

SD 为三个同 seed 差值的样本标准差。K1−K2 的 CW/OW 仅约 +0.071/+0.061 点，这两档构成近似相同的性能区间。继续在 0.1 与 0.3 之间插值，投入价值较低。

![历轮模型与三 seed 结果](analysis_roundK_results_20261021/figures/01_history_plateau.png)

### 1.1 双选权重规则也没有改变方向判断

K1/K2 的 seed3352 在 native 规则下选中 epoch8，在 exact 规则下选中 epoch6；其余两个 seed 分别为 epoch5/6，两条规则一致。下表在两种选权重规则下统一展示 native AUC，比较的是选中权重的影响。

__EXACT__

exact 选权重后，K1/K2 的 OW 有小幅回升，仍低于对应的静态 G0。当前结论可以归于训练后的模型表现，而非单个选模分支异常。

## 2. K 组回答了什么：缩小动态更新有效，但没有改善组合排序

__SCALES__

实际自适应步长指选中 checkpoint 最后一次 AdamW 更新中，自适应项的范数比；weight decay 单独执行。动态条件 logit RMS 在 TEST 未见图像的真实物体条件下，计算 dynamic 与 TRAIN 均值门控的 logits 差，再沿属性维去均值。

Wd 范数从 20.894 降到 7.708、2.850；自适应步长比从 0.971 降到 0.307、0.111。门控图像标准差仍在约 0.027 附近，动态分数 RMS 已从 0.648 降到 0.300、0.111。配置、优化器和前向响应都按设计发生了变化。

与此同时，TEST OW AUC 仍集中在 24.50 左右；动态分数变小没有把模型带到 G0 的 25.184。J1、K1、K2 的 dynamic−mean 随动态分支变弱而下降，这反映各模型对自身动态分支的依赖程度。它与相对独立训练 G0 的收益是两种比较。

![动态更新与输出尺度](analysis_roundK_results_20261021/figures/02_update_scale.png)

由此应收束三条支线：Wd 学习率细分、门控温度／幅度细分、通过新增缩放系数重新平衡静态与动态分支。这些量已能控制模型作用幅度，当前缺少的是改善组合排序的新信息处理方式。

## 3. 大路径的依据：共同错误和候选内排序

### 3.1 后续模型反复保留 G0 的同一批错误

在每个 seed 的 1,891 张 TEST 未见组合图像上，只在全部 OW 未见组合候选内取 top-1。以 G0 为参照，统计 I2、J1、K1、K2 也同时预测错误的图像：

__COMMON__

也就是说，G0 原有组合排序错误的 88.3%–92.0%，在后续四组中始终没有被任何一组纠正。这比单看均值相差 0.1–0.5 点更能说明现有路线的状态：改变门控输入、读出绑定和优化速度，主要移动了一小部分决策边界。

这里统计的是各 seed 内相同图像的预测交集，三个 seed 的图像不作为独立样本重复计入。

### 3.2 属性与物体两侧都有实际排序空间

将每张未见图像分成三个互斥情况：

1. 真实组合在全部未见候选中为 top-1。
2. 即使给定真实物体，真实属性仍输给该物体的其他未见属性。
3. 真实组合在其物体内为 top-1，但输给另一个物体的组合。

__ERRORS__

三列“全部未见候选 top-1／同物体内排错／跨物体竞争排错”相加为 100%。G0 的错误由约 30.72% 的同物体属性排序错误和 19.64% 的跨物体竞争错误构成；K1/K2 没有改善这两个部分的整体水平。

![未见候选内的错误分解](analysis_roundK_results_20261021/figures/03_error_decomposition.png)

**下一阶段把主要投入放在可训练的局部属性判别，并继续衡量跨物体竞争。** 只改变组合先验、统一 seen/unseen 偏置或动态分支幅度，无法直接解决同物体内的未见属性排序。

对固定物体 $o$ 和两个未见组合属性 $a_1,a_2$，当前联合对数概率满足：

$$
\begin{aligned}
s(x,a,o)&=\log p(o\mid x)+U(x,a,o)-\log\sum_{a'}\exp U(x,a',o),\\
s(x,a_1,o)-s(x,a_2,o)&=U(x,a_1,o)-U(x,a_2,o).
\end{aligned}
$$

物体概率和条件归一化项在差值中消去；给所有未见组合添加同一偏置也会消去。因此，同物体内约 31% 的排错需要改变属性证据及其打分顺序。这个结论直接来自当前评分公式和已有预测。

### 3.3 重新解释“未见条件属性准确率约 24%”

原报告的条件属性准确率在真实物体下比较全部 16 个属性，采用零 bias。G0 中约 66.86% 的未见图像被预测为该物体的 TRAIN 属性组合；采用已有 VAL bias 后，其条件属性准确率为 57.29%，只在该物体的未见属性中比较则为 69.28%。

因此，24% 同时包含训练组合偏好，不能作为 CLIP 视觉表征的能力上限。候选内排序及共同错误提供了更有针对性的方向依据。本次的判断是：**当前冻结特征与末端读出的组合已出现明显收益停滞，值得把学习能力移入视觉证据提取过程。**

## 4. 当前结构在哪里限制了下一步收益

核查 `vgcpod/features.py` 与 `model.py` 后，可以把当前实际计算路径概括为：

> 固定 CLIP 最末层 patch → 固定文本查询与 softmax 池化 → 固定物体查询向量 → 32 维动态模式系数；组合主读出使用固定全局 CLS 向量。

在 J/K 配置中：

- CLIP 全部冻结，图像和文本向量在训练前生成。
- 局部 patch 经过固定查询池化后保存为 `fo`，空间位置与逐 patch 信息没有进入头部训练。
- `readout_feature=global`，静态和动态 CP 都由全局向量读取。
- `fo` 经残差处理进入 `Bvis`，作用是改变模式权重；任务梯度不能回到 patch，也不能调整查询池化。
- 属性查询缓存 `fa` 在当前 V1 主模型中不参与输出。

这一结构能学习组合分类器及其输入相关系数，但无法让材质／状态监督改变视觉编码器的关注区域。早期固定局部向量替换全局读出没有收益，也不等于“训练局部视觉证据”这条路径已经被实验覆盖。

__RESEARCH_DIRECTION__

## 7. 本次复核与交付

- 核对 98 个交付文件哈希和 60 项训练源码指纹。
- 独立复算 96 份预测，native AUC 和固定 VAL bias HM 与归档一致；exact AUC 最大差异为浮点舍入量级。
- 48 份 dynamic 复评预测与原预测逐元素一致，复算 414 行排序诊断。
- 检查两组全部 epoch 的学习率倍率、checkpoint 中 TRAIN 均值、损失恒等式、RMS 和 native/exact 选权重。
- 逐图计算共同错误、真实物体内未见属性排序和跨物体竞争分解；从真实 checkpoint 重建 J1/K1/K2 的 Adam 更新量。

数据明细：`analysis_roundK_results_20261021/analysis.json`、`audit.json`、`supplement.json`、`tables/`、`figures/`。本报告给出结果判断、近邻文献差异、核心算子推导与正式比较矩阵。训练代码未改动；下一次实现以第 6 节为规格。
'''
    text=text.replace('__RESEARCH_DIRECTION__',(OUT/'research_direction.md').read_text(encoding='utf-8'))
    for key,value in [('RESULTS',results),('PAIRED',paired_table),('SCALES',scales),('ERRORS',errors),('COMMON',common),('EXACT',exact)]:
        text=text.replace('__'+key+'__',value)
    path=ROOT/(NAME+'.md');path.write_text(text,encoding='utf-8',newline='\n')
    subprocess.run(['C:/Users/llenovo/anaconda3/Scripts/pandoc.exe',str(path),'--standalone','--self-contained','--mathml',
        '--from=markdown+tex_math_dollars+pipe_tables','--to=html5','--css='+str(ROOT/'analysis_20261018/report.css'),
        '--metadata=pagetitle:VG-CPOD K组结果与后续主线决策','--output='+str(ROOT/(NAME+'.html'))],check=True)
    print(path)

if __name__=='__main__':main()
