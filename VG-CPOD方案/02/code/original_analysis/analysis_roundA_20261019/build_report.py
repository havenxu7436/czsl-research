"""Write the round-A interpretation and concrete next experiments from audited results."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
CASES=['D0','V1','A1','A2','A3','A4','A5','A6']


def m(case,setting,key='AUC_native'):
    return D['metrics'][case]['metrics'][setting][key]


def dist(case,phase='test'):
    return next(r for r in D['distribution'] if r['case']==case and r['phase']==phase)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    main_table=table(['模型/组','CP 初始 LR','β','最佳 epoch','VAL CW AUC','VAL OW AUC','TEST CW AUC','TEST OW AUC'],[
        [c,'—' if D['chosen'][c]['cp_lr'] is None else f"{D['chosen'][c]['cp_lr']:.0e}",D['chosen'][c]['beta'],
         D['chosen'][c]['epoch']]+[f'{m(c,s):.3f}' for s in ['val_cw','val_ow','test_cw','test_ow']] for c in CASES])
    diff_table=table(['对照','VAL CW ΔAUC','VAL OW ΔAUC','TEST CW ΔAUC','TEST OW ΔAUC'],[
        [hi+'−'+lo]+[f'{m(hi,s)-m(lo,s):+.3f}' for s in ['val_cw','val_ow','test_cw','test_ow']]
        for hi,lo in [('A1','V1'),('A4','V1'),('A5','V1'),('A5','D0'),('A5','A1'),('A5','A6'),('A6','D0')]])
    transfer_table=table(['模型/组','TEST 已见条件属性准确率','TEST 未见条件属性准确率','TEST 未见图像的已见概率质量','TEST 物体准确率','TEST CW NLL'],[
        [c,f"{dist(c)['attr_seen']:.2f}",f"{dist(c)['attr_unseen']:.2f}",f"{100*dist(c)['support_unseen']:.2f}%",
         f"{dist(c)['object_acc']:.2f}",f"{m(c,'test_cw','pair_nll_unbiased'):.3f}"] for c in ['D0','V1','A1','A4','A5','A6']])
    pair=list(csv.DictReader((OUT/'tables/per_pair.csv').open(encoding='utf-8-sig')))
    suede_table=table(['未见组合','图像数','原 D0','原 V1','A1','A5','A6'],[
        [attr+'–'+obj,next(r['images'] for r in pair if r['attribute']==attr and r['object']==obj)] +
        [f"{float(next(r['cond_acc'] for r in pair if r['case']==c and r['setting']=='test_ow' and r['attribute']==attr and r['object']==obj)):.2f}"
         for c in ['D0','V1','A1','A5','A6']]
        for attr,obj in [('Suede','Slippers'),('Suede','Boots.Mid-Calf')]])
    curves=table(['组','最佳 epoch','VAL CW 峰值','第 10 轮 VAL CW','峰值时修正 RMS','峰值时裁剪步骤比例'],[
        [c,D['chosen'][c]['epoch'],f"{D['chosen'][c]['val_peak']:.3f}",f"{D['chosen'][c]['val_last']:.3f}",
         f"{D['chosen'][c]['delta_rms']:.3f}",f"{100*D['chosen'][c]['clip_fraction']:.2f}%"] for c in ['A1','A2','A3','A4','A5','A6']])
    hm_table=table(['模型/组','TEST CW HM（VAL 偏置）','TEST OW HM（VAL 偏置）','TEST CW AUC_exact','TEST OW AUC_exact'],[
        [c,f"{m(c,'test_cw','HM_at_val_bias'):.3f}",f"{m(c,'test_ow','HM_at_val_bias'):.3f}",
         f"{m(c,'test_cw','AUC_exact'):.3f}",f"{m(c,'test_ow','AUC_exact'):.3f}"] for c in ['D0','V1','A1','A5','A6']])
    report=r'''# VG-CPOD v1 A组结果分析与下一步计划

分析日期：2026-10-19。新增实验 A1–A6，全部为 seed3351；原 D0/V1 对照也取 seed3351。以下差值均为绝对点数。

**A5 是下一步的主配置：CP 学习率 3e-4、β=0.1。它在 VAL CW 和 VAL OW 上同时最好，TEST OW 也最好。A1 保留为 CW 性能对照，A6 保留为相同损失下的 D0 对照。**

- A5 相对原 V1：TEST CW AUC **+1.238**，TEST OW AUC **+2.666**。
- A5 相对原 D0：TEST CW AUC **+3.280**，TEST OW AUC **+1.414**。
- A5 相对相同 β 的 A6：TEST CW AUC **+3.916**，TEST OW AUC **+1.343**。
- A1 的 TEST CW AUC 为 **34.495**，高于 A5 的 **34.022**；A5 的验证结果、OW 与条件属性识别更好，因此以 A5 为 B 组基座，同时保留 A1。

**立即可执行：给 A1/A5/A6 补 seed3352、3353，共 6 次训练。结构开发集中在“保留全局读出，由局部证据做幅度可控的模式调制”。**

## 1. 结果与实现核查

下载归档已解压至 `server_results_roundA_20261019/roundA_20261019`。六组均完成 10 epochs、3600 steps，总队列退出码为 0。六组协议的源码哈希均与上次交付代码包一致，特征缓存也与上一轮一致。

逐项核对了配置、最佳/最终 checkpoint、训练历史、学习率比例和最终学习率。最佳权重全部为 VAL CW AUC_native 首次最大值；TEST 没有参与选择。

24 份新增预测文件已独立复算。准确率、VAL 偏置 HM、native AUC 均与保存值一致；exact AUC 最大差异约 `7.11e-15` 点。概率归一化最大误差约 `1.43e-6`，已见概率质量误差约 `8.90e-8`。训练日志中的任务损失与推导恒等式最大误差约 `1.09e-7`。

**本轮观察到的性能变化来自实际实验设置，学习率分组和 β 损失均按预期执行。**

## 2. 完整结果：A5 综合最好，A1 保留 CW 优势

{{MAIN}}

![同 seed 的原基线与 A 组 AUC 比较](analysis_roundA_20261019/figures/01_auc_comparison.png)

A5 的 VAL CW/OW AUC 为 **43.228 / 30.291**，两者均为本轮最高。选择 A5 来自验证集结果。A1 的 TEST CW AUC 和 CW 固定偏置 HM 更高，因此适合作为后续模型的第二个性能参照。

{{HM}}

native 与 exact AUC 得到相同的主要排序。固定偏置 HM 保留为工作点指标，AUC 用来衡量 seen–unseen 权衡曲线。

本轮为一个 seed 的配置筛选；后续用已确定的候选补齐另外两个 seed，报告均值和配对差值。

## 3. 两个超参的作用：降低 CP 学习率有效，β 与学习率需要配合

{{DIFF}}

### 3.1 A1：降低 CP 更新速度，单独就带来了收益

A1 保持 β=0.5，把 CP 初始学习率从 1e-3 降为 3e-4，适配器仍为 1e-3。相对原 V1，VAL CW/OW 提高 **0.330 / 2.221**，TEST CW/OW 提高 **1.711 / 2.485**。

最佳 epoch 从第 5 轮移到第 8 轮，之后验证 AUC 只小幅回落。当前乘法修正分支需要比适配器更慢的更新，这项优化已经体现为 CW/OW 同向收益。

A2 把 CP 学习率进一步降到 1e-4，VAL CW/OW 为 39.675/24.061，低于 A1/A5。下一轮先固定 3e-4，不把延长 A2 或继续降学习率作为主要投入。

### 3.2 A4：单独降低 β 会改善部分属性指标，但组合 AUC 下降

A4 在 CP 学习率仍为 1e-3 时把 β 降至 0.1，相对原 V1，VAL CW 下降 **7.524**，TEST OW 下降 **2.197**。

A4 的 TEST 未见属性准确率从原 V1 的 13.96% 提高到 19.30%，已见概率质量也降低了。这说明损失调整确实改变了模型偏好，但更低的已见概率质量并不自动意味着更好的组合排序。

### 3.3 A5：降低 β 与降低 CP 学习率相互补充

A5 相对 A4 只把 CP 学习率降到 3e-4，VAL CW/OW 提高 **8.837 / 5.825**，TEST CW/OW 提高 **2.170 / 4.863**。

相对 A1，A5 只降低 β，VAL CW/OW 又提高 **0.982 / 0.995**，TEST OW 提高 **0.181**，TEST CW 下降 **0.473**。

对原 V1、A1、A4、A5 构成的 2×2 设置，交互量为：

$$
I=m(\mathrm{A5})-m(\mathrm{A1})-m(\mathrm{A4})+m(\mathrm{V1}).
$$

各自按 VAL 选择 checkpoint 后，VAL CW/OW 的交互量为 **+8.507 / +3.604**，TEST CW/OW 为 **+0.459 / +2.379**。这量化了本轮的非加和关系：两项调整合用的效果优于由各自单独变化简单相加的预测。

## 4. 泛化修复到了哪一步

{{TRANSFER}}

![已见概率质量与未见条件属性识别](analysis_roundA_20261019/figures/03_support_and_transfer.png)

相对原 V1，A5 在 TEST 未见图像上分给已见组合的概率质量从 **84.37% 降到 72.86%**；未见条件属性准确率从 **13.96% 升到 24.38%**；物体准确率从 **68.36% 升到 70.32%**。组合 NLL 从 **3.053 降到 2.477**。

上一轮“修正更偏向已见支持集合”的问题已有明显改善。现在可以把 A5 作为有效的静态 CP 基座，继续研究视觉条件化。

### 4.1 匹配对照 A6 给出了更准确的贡献划分

A5/A6 都使用 β=0.1。它们在 VAL 未见条件属性准确率上分别为 **20.92% / 15.49%**；在 TEST 上分别为 **24.38% / 24.43%**。

TEST 的 1,891 张未见组合图像中，A5 条件属性识别正确 **461** 张，A6 为 **462** 张，几乎相同。A5 的已见组合条件属性则比 A6 多识别正确 **50** 张，物体准确率也高 **0.27** 点。

与此同时，A5 相对 A6 的组合 CW/OW AUC 提高 **3.916 / 1.343**，OW 未见组内排序准确率从 **49.34% 到 50.29%**。CP 对组合预测仍有增量，但 TEST 未见条件属性识别的整体收益主要由新的训练设置恢复，CP 自身还没有在该指标上拉开差距。

**因此，B 组应把 A5 和 A6 都作为参照，集中检验视觉证据能否进一步改善未见属性判别。**

### 4.2 Suede 有明显恢复，也仍有针对性的改进空间

下表为给定真实物体后的条件属性准确率：

{{SUEDE}}

A5 相对原 V1 已恢复了这两个组合的大部分损失；相对匹配的 A6，`Suede–Slippers` 仍少识别 **37** 张，`Suede–Boots.Mid-Calf` 少 **25** 张。另一方面，A5 在未见的 `Canvas–Shoes.Loafers` 上比 A6 高 **47.83** 点，在 `Satin–Sandals` 上高 **12.39** 点。

这些相互抵消的类别变化解释了为什么 A5/A6 的未见总体准确率几乎相同。视觉模块的下一项任务是让材质证据在不同物体上更一致地生效。训练与配置选择继续使用完整 VAL 集，这些 TEST 组合用来解释已有误差。

## 5. 学习曲线支持保留当前训练尺度

{{CURVES}}

![学习曲线与条件 logits 的修正幅度](analysis_roundA_20261019/figures/02_learning_curves.png)

A5 的修正 RMS 为 **1.218**，小于 A4 的 **1.986**。A5 第 8 轮达到 VAL CW 峰值 43.228，第 10 轮仍为 43.093，已进入较平稳阶段。A1 同样在第 8 轮取得峰值。

下一步固定 **CP LR=3e-4、适配器 LR=1e-3、β=0.1、rank32、10 epochs、batch64**，把实验变量集中到视觉证据怎样进入修正。A1 保留 β=0.5，作为另一条损失设置参照。

## 6. 现在首先做：补齐 A1/A5/A6 的三 seed

需要新增的只有六组：

| 物理 GPU | 第一组 | 第二组 |
|---|---|---|
| GPU1 | A1 / seed3352 | A1 / seed3353 |
| GPU2 | A5 / seed3352 | A5 / seed3353 |
| GPU3 | A6 / seed3352 | A6 / seed3353 |

三组均直接使用已有配置和代码，不用上传新包或重新编码 CLIP。A1 对照原三 seed V1/D0，A5 对照同 β 的三 seed A6。

在服务器执行：

```bash
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/vgcpod_v1_roundA

export VG_CACHE=/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14
export VG_A_OUT=/data/xhw/vgcpod_v1_runs/roundA_20261019
export VG_A_SESSION=vgcpod_A_confirm

bash scripts/start_round_a_tmux.sh \
  --gpus 1 2 3 --cases A1 A5 A6 --seeds 3352 3353

tmux attach -t vgcpod_A_confirm
```

这里使用新的 tmux 会话名，避免原 A 组已完成的会话阻止启动。输出沿用同一结果根目录，新 seed 写入各自目录。

按 **Ctrl+b，松开，再按 d** 脱离会话。另一个终端查看：

```bash
tail -f /data/xhw/vgcpod_v1_runs/roundA_20261019/launcher.log
```

```bash
python round_a.py summarize \
  --out /data/xhw/vgcpod_v1_runs/roundA_20261019
```

当前汇总会保留所有已完成 seed 的逐组指标。补齐后分析三 seed 均值和相同 seed 的差值；`v1_val_cw_ranking` 是逐运行排序，多 seed 的结论使用按组聚合结果。

## 7. 紧接着投入的方向：B 组保留全局读出，局部证据只调制模式

B 组采用 A5 的固定训练设置。三组均先跑 seed3351：

| 组 | 读出 | 门控输入 | 模式系数 | 目的 |
|---|---|---|---|---|
| B1 | 归一化的 0.5 全局 + 0.5 物体查询特征 | 关闭 | 全 1 | 判断局部信息混入读出能否改善材质识别 |
| B2 | 全局特征 g | 物体查询特征 f_o | 原幅度，η=1 | 让局部证据只负责模式调制 |
| B3 | 全局特征 g | 物体查询特征 f_o | 小幅度，η=0.25 | 检验较弱的视觉调制是否更合适 |

候选前向为：

$$
\begin{aligned}
\gamma_{bj}^{(\eta)}
&=\mathbf{1}+\eta\left[2\sigma(B_{\mathrm{vis}}f^o_{bj}/\tau_\gamma)-\mathbf{1}\right],\\
\Delta U_{bji}
&=\rho\kappa\,g_b^\top W_c
\left(u_i\odot v_j\odot\gamma_{bj}^{(\eta)}\right).
\end{aligned}
$$

η=0.25 时每个模式系数位于 `(0.75,1.25)`。全局 CLIP 信息负责读出，局部特征负责调整组合模式的使用程度。

实现 B 组时需要把 `model.py` 中当前共用的 `local` 拆成 `readout_feature` 与 `gate_input`，再加入混合比例和门控幅度两个配置。现有 V3 同时使用全局读出、全局门控，不能直接替代 B2/B3。A 组的损失与学习率分组代码可直接沿用。

B 组重点比较相对 A5 的 CW/OW AUC 增量、未见条件属性准确率与类别变化。B2/B3 的对照只改变调制幅度；一个代表 checkpoint 的真实门控与训练均值门控比较即可补充动态作用证据。诊断随训练输出，保持实验规模集中。

补 seed 可以马上运行，B 组实现与准备可同时进行。A5 已提供明确训练基座，下一轮结构实验按这套设置推进。

## 8. 本轮之后的超参决定

| 项目 | 决定 |
|---|---|
| CP 学习率 | 主配置固定 3e-4 |
| 适配器学习率 | 保持 1e-3，余弦终点 1e-5 |
| β | 主配置 0.1；A1 保留 0.5 作参照 |
| rank / adapter rank | 保持 32 / 8 |
| epochs / batch | 保持 10 / 64 |
| ρ、κ、weight decay | 保持 0.1、20、1e-4 |
| 视觉伙伴 | 排在 B 组之后；本轮改善来自静态分支的训练设置 |
| 下一项主要变量 | 全局/混合读出，以及局部模式调制幅度 |

数学层面，这轮支持继续使用已经实施的 β 归一化损失与分组学习率。新的公式变化集中于读出和门控通路分离；当前数据没有要求改动 CLIP 编码、概率链式分解或 CP 收缩本身。

## 9. 文件与复核入口

- [全部指标](analysis_roundA_20261019/tables/all_metrics.csv)。
- [逐组差值](analysis_roundA_20261019/tables/comparisons.csv)。
- [已见/未见与概率质量](analysis_roundA_20261019/tables/distribution.csv)。
- [逐组合指标](analysis_roundA_20261019/tables/per_pair.csv)。
- [十轮历史](analysis_roundA_20261019/tables/history.csv)。
- [完整审计](analysis_roundA_20261019/audit.json)、[结构化分析](analysis_roundA_20261019/analysis.json)。
- [分析脚本](analysis_roundA_20261019/analyze_round_a.py)、[报告生成脚本](analysis_roundA_20261019/build_report.py)。

本地归档指纹已记录于审计 JSON。此次没有修改训练代码，也没有在本机重新训练或重新提取图像特征。
'''
    replacements={'MAIN':main_table,'DIFF':diff_table,'TRANSFER':transfer_table,'SUEDE':suede_table,'CURVES':curves,'HM':hm_table}
    for k,v in replacements.items():report=report.replace('{{'+k+'}}',v)
    assert '{{' not in report
    p=ROOT/'VG-CPOD_v1_A组结果分析与下一步计划_20261019.md'
    p.write_text(report,encoding='utf-8')
    print('REPORT',p,'characters',len(report))


if __name__=='__main__':main()
