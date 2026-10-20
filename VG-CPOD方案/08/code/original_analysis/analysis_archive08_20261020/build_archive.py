"""Write experiment 08's five parallel READMEs and update both route indexes."""
import json
from pathlib import Path
import re

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parent/'CZSL'
STAGE=REPO/'VG-CPOD方案/08'
TITLE='支持集权重折中与门控训练对照'
D=json.loads((WORK/'analysis_roundG_results_20261020/analysis.json').read_text(encoding='utf-8'))
INVENTORY=json.loads((STAGE/'configs/experiment_inventory.json').read_text(encoding='utf-8'))


def write(relative,text):
    (STAGE/relative).write_text(text,encoding='utf-8',newline='\n')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])


def mean(case,setting,key='AUC_native_mean'):
    return next(r[key] for r in D['means'] if r['case']==case and r['setting']==setting)


def main():
    seed_table=table(['实验','seed','native／exact最佳epoch','VAL CW','VAL OW','TEST CW','TEST OW'],
        [[r['case'],r['seed'],str(r['selected_epoch'])+'／'+str(r['exact_selected_epoch'])]+
         [f"{r['metrics'][k]['AUC_native']:.4f}" for k in ('val_cw','val_ow','test_cw','test_ow')] for r in INVENTORY])
    exact_table=table(['实验','seed','exact最佳epoch','VAL CW native','VAL CW exact','TEST CW native','TEST OW native'],
        [[r['case'],r['seed'],r['exact_selected_epoch']]+
         [f"{r['exact_selection_metrics'][setting][key]:.4f}" for setting,key in
          [('val_cw','AUC_native'),('val_cw','AUC_exact'),('test_cw','AUC_native'),('test_ow','AUC_native')]] for r in INVENTORY])
    means=table(['三 seed 方案','ρ','β','VAL CW AUC','TEST CW AUC','TEST OW AUC','TEST 未见条件属性'],
        [[case,rho,beta]+[f"{mean(case,k):.3f}" for k in ('val_cw','test_cw','test_ow')]+
         [f"{mean(case,'test_ow','gt_object_attr_acc_unseen_mean'):.3f}%"]
         for case,rho,beta in [('D1',.125,.1),('E1',.15,.1),('F1',.15,.05),('G0',.15,.075),('G1',.15,.075)]])
    gate_table=table(['seed','G1动态−均值 TEST CW AUC','TEST OW AUC','未见属性正确数变化'],
        [[seed]+[f"{next(r['AUC_native'] for r in D['comparisons'] if r['comparison']=='G1-G1_mean' and r['seed']==seed and r['setting']==setting):+.4f}" for setting in ('test_cw','test_ow')]+
         [f"{next(r['net_correct'] for r in D['similarity'] if r['comparison']=='G1-G1_mean' and r['seed']==seed and r['phase']=='test' and r['group']=='unseen'):+d}"] for seed in (3351,3352,3353)])
    main_text=r'''# 08 · 支持集权重折中与门控训练对照

**G 组实验，VG-CPOD v1，归档日期 2026-10-20。** 使用冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 G0/G1 各三 seed、共六次训练和 60 个模型 epoch；保存 native／exact 两种选择的 48 份评价与预测，以及 G1 六项动态／TRAIN 均值替换评估的 48 份评价与预测，共 **96 份预测导出、24 份 checkpoint 文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [07 F组](../07/README.md) · [06 E组](../06/README.md)

## 1. 实验目的与修订范围

07 的 F1 使用 β=0.05、ρ=0.15。相对 E1 的 β=0.1，未见条件属性提高、组合 AUC 回落。本轮取 β=0.075，检验支持集权重的中间强度，并在同一 β、ρ 下训练静态 G0 与动态 G1，量化视觉门控模块的独立训练收益。

G1 相对 F1 只改变 `task_aux_beta` 与 `experiment_id`；G0 在 G1 配置上将 `gate_input` 从 `local` 改为 `none`，并使用自己的实验编号。全局 g 读出、ρ=0.15、rank32、适配器、CP 学习率及十轮日程沿用 F 组。

训练核心、模型与损失、CLIP 特征提取、均值评估器和 `run.py` 共十份文件与 F 包逐字节一致。新增代码集中于 G0/G1 配置、三卡双任务队列、排序诊断及同 seed 汇总。

## 2. 公式修订：支持集系数改为 9/86

张量轴为 `[batch, object, attribute]`，属性索引为 i、物体索引为 j；TRAIN 组合集合按 $(i,j)$ 记为 $\mathcal{S}$。当前链式联合分布与损失为：

```math
\begin{aligned}
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
Z_b&=\sum_{(i,j)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],\\[4pt]
\mathcal{L}_{\mathrm{support}}&=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_a+\mathcal{L}_o&=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_a+\mathcal{L}_o)]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

条件 softmax 遍历全部 16 个属性，物体 softmax 遍历全部 12 个物体；完整联合分布在 192 个组合上归一化，Z 是其中 83 个 TRAIN 组合的总概率质量。代入本轮 β：

```math
\begin{aligned}
\mathcal{L}^{\mathrm{E1}}_{\mathrm{task}}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{3}{22}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{\mathrm{G0/G1}}_{\mathrm{task}}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{\mathrm{F1}}_{\mathrm{task}}&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac1{14}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

本轮支持集系数为 **0.104651163**，位于 E1 的 0.136363636 与 F1 的 0.071428571 之间；pair 项保持 1.5。反向传播继续使用原 objective 表达式；原始、加权损失及其重构误差日志沿用 F 实现。

## 3. 静态与动态修正、权重选择及诊断公式

令 $u_i=A_ue_i^a$、$v_j=A_ve_j^o$，r=32、d=768。G1 保留局部门控和模式中心化；G0 关闭视觉门控输入：

```math
\begin{aligned}
H_r&=I_r-\frac1r\mathbf{1}_r\mathbf{1}_r^\top,\\[4pt]
q_{bj}&=2\sigma(B_{\mathrm{vis}}f^o_{bj}/\tau_\gamma)-\mathbf{1}_r,\\[4pt]
\gamma^{\mathrm{G1}}_{bj}&=\mathbf{1}_r+\frac\eta2H_rq_{bj},
\qquad\eta=0.25,\quad\tau_\gamma=0.1,\\[4pt]
\gamma^{\mathrm{G0}}_{bj}&=\mathbf{1}_r,\\[4pt]
U^{\mathrm{G1}}_{bji}&=U^{\mathrm{base}}_{bji}+0.15\kappa g_b^\top W_c(u_i\odot v_j\odot\gamma^{\mathrm{G1}}_{bj}),\\[4pt]
U^{\mathrm{G0}}_{bji}&=U^{\mathrm{base}}_{bji}+0.15\kappa g_b^\top W_c(u_i\odot v_j).
\end{aligned}
```

G0 仍训练 CP 分解和适配器，Bvis 冻结；G1 训练视觉门控并沿用其正交项：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{G0}}&=\mathcal{L}_{\mathrm{task}}(0.075),\\[4pt]
\mathcal{L}_{\mathrm{G1}}&=\mathcal{L}_{\mathrm{task}}(0.075)
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_r\right\rVert_F^2.
\end{aligned}
```

两组 prim、cent 分支关闭；AdamW 的 weight decay 独立执行。G0 与 G1 的可训练参数量分别为 98,304 与 122,880。该对照比较视觉门控模块及其配套正交项的训练效果。

两种选权重依据分别保存首次严格最大 VAL CW AUC；G1 的选中权重再统计全部 TRAIN 图像的门控均值：

```math
\begin{aligned}
t_m^*&=\min\{t\in\{1,\ldots,T\}:A_t^m=\max_{1\le s\le T}A_s^m\},
\qquad m\in\{\mathrm{native},\mathrm{exact}\},\\[4pt]
\bar\gamma_{jk}&=\frac{1}{N_{\mathrm{train}}}\sum_b\gamma_{bjk},
\qquad\gamma^{\mathrm{mean}}_{bjk}=\bar\gamma_{jk}.
\end{aligned}
```

本轮 seed3353 的 native／exact 分别选中第 9／6 轮，因此两套结果独立保存、各自按 VAL CW／OW native bias 评价 TEST。

新增诊断从保存的完整 OW 分布读取。令 $\mathcal{P}=\mathcal{A}\times\mathcal{O}$、$\mathcal{C}_h=\mathcal{S}\cup\mathcal{Y}_h$，$\mathcal{E}_h=\mathcal{P}\setminus\mathcal{C}_h$：

```math
\begin{aligned}
M_b^{\mathrm{seen}}&=\sum_{(i,j)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
M_b^{\mathrm{extra}}&=\sum_{(i,j)\in\mathcal{E}_h}\exp(s_{bji}),\\[4pt]
D_b^{\mathrm{extra}}&=\mathbf{1}\!\left[
\max_{(i,j)\in\mathcal{E}_h}s_{bji}
>\max_{(i,j)\in\mathcal{C}_h\setminus\mathcal{S}}s_{bji}\right].
\end{aligned}
```

程序输出概率质量、CW 集合外最高分超越比例、全部未见候选内真实组合 top-1，以及物体／GT-object 条件属性的准确率与 NLL；均按 all/seen/unseen 图像分组。这些诊断描述原预测，不改变模型分数。

## 4. 具体代码修订与验证

| 文件 | 本轮具体改动 | 作用 |
|---|---|---|
| [G1.json](configs/round_g/G1.json) | β=0.075、experiment_id=G1 | 支持集权重中间点 |
| [G0.json](configs/round_g/G0.json) | G1 配置的 gate_input 改为 none | 相同目标下的静态训练对照 |
| [round_g.py](code/vgcpod_v1_roundG/round_g.py) | 每卡 G0→G1 队列、续训、六项均值评估、两组与历史配对汇总 | 六项训练的完整编排 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundG/ranking_diagnostics.py) | `diagnose()`、逐组合统计、同 seed 诊断差值及 SD | 连接属性、概率质量与组合排序 |
| [tmux启动](code/vgcpod_v1_roundG/scripts/start_round_g_tmux.sh) | G 组会话、输出目录、日志和退出码 | 三 GPU 各负责一个 seed |
| [G组测试](code/vgcpod_v1_roundG/tests/test_round_g.py) | 配置范围、静态极限、β=0.075 梯度、排序诊断 | 累计39项发布测试 |
| [pre_g_complete.json](configs/references/pre_g_complete.json) | 20份已有训练运行的指标快照 | E1/F1 与历史同 seed 对照 |

[相对07的源码补丁](code/source_changes.patch)记录实际文件差异。模型、objective、优化器与 CLIP 缓存读取沿用 F 包；[服务器身份核对](results/server_source_identity.json)与[原包76成员映射](package_members.json)保留源字节身份。

[原发布验证](results/package_validation/G/validation/result.json)跑通六项训练、六项评估及重复启动复用。与原 F 训练器使用相同 G0/G1 配置比较，native／exact／last 的模型参数、AdamW 状态及 Torch RNG 逐元素一致。新增诊断在 E1/F1 真实预测上复算 36 行，与独立 F 分析的最大差异小于 1e-8。

## 5. 超参数选取策略与实际矩阵

1. **β 中间点。** 在固定 ρ=0.15 下，已有 β=0.1 与 0.05 的直接对照；本轮取 β=0.075，系数按 1.5β/(1+β) 精确计算。
2. **补同配置模块对照。** G0/G1 使用相同 β、ρ、读出、rank 和优化日程，区别为静态／视觉门控模块及动态分支的正交项。
3. **保持十轮和三 seed。** 两组均从各自原始初始化训练，seed3351/3352/3353 分别在 GPU1/2/3 顺序运行 G0→G1，各3600次更新。
4. **同 seed 配对。** G1−E1/F1 测量目标权重变化，G1−G0 测量门控模块，G1 dynamic−mean 测量已训练权重对图像变化的依赖。
5. **保留原选模规则。** 主结果取首次严格最大 VAL CW native AUC；exact 选中结果独立保存，各条件使用自己的 VAL native bias。

{{SEEDS}}

主表为 native 选择的 native AUC，面积×100。G0/G1 的 native 选中轮次均为8/8/9，exact 均为8/8/6；两种权重完整指标见[六项训练清单](configs/experiment_inventory.json)。

| 固定项 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 ViT-L/14，224px、768维，同一缓存 |
| ρ／β／η／τγ | 0.15 / **0.075** / 0.25 / 0.1 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |
| eval batch／workers | 64 / 4 |

内部 `variant=V1`；通过 `gate_input=none/local` 区分静态和动态。G0 的 CP 组为 Au/Av/Wc，G1 增加 Bvis；`cp_lr_scale=0.3`，Wc 零初始化。

## 6. 仓库结构与复现入口

```text
08/
  README.md          # 修订、公式、超参策略与末节数据分析
  code/              # 原G代码、相对F补丁、原分析源码
  configs/           # G0/G1、兼容配置、资产、参考与运行清单
  results/           # 六次训练、双选择、均值评估、日志及原归档
  analysis/          # 数据分析、图表、静态等价验证与仓库复算
```

恢复原运行包并复用服务器现有 CLIP 缓存：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 08 \
  --dest /data/xhw/vgcpod_project/replay_08
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_08/vgcpod_v1_roundG

export VG_G_OUT=/data/xhw/vgcpod_v1_runs/replay_08_G
export VG_G_SESSION=vgcpod_08_G
bash scripts/start_round_g_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020 \
  --f-out /data/xhw/vgcpod_v1_runs/roundF_20261020
tmux attach -t vgcpod_08_G
```

每卡顺序完成 G0/G1 及本 seed 的 G1 两项均值评估。按 Ctrl+b、d 脱离 tmux。`launcher.exitcode=0` 且日志出现 `ROUND_G_COMPLETE ... training_jobs 6 mean_gate_jobs 6` 表示整轮完成。外部输入见 [external_inputs.json](external_inputs.json)，命令见 [runbook.json](runbook.json)。

从仓库保存的预测、checkpoint 与缓存元数据复算结果及均值门控的静态等价关系：

```bash
python VG-CPOD方案/08/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`，原分析保存在 `analysis/confirmed/`。materialize 按映射恢复原包布局与字节。

## 7. 数据分析与下一轮方向

{{MEANS}}

- **β=0.075 回收 F1 的排序损失并保留部分属性收益。** G1−F1 的 TEST CW/OW AUC 为 +0.501±0.278／+0.705±0.029 点；G1−E1 为 +0.103／−0.124 点，未见条件属性 +1.322 个百分点。
- **门控模块的 TEST 排序收益较小。** G1−G0 的 native 选择结果为 CW +0.019±0.037、OW −0.127±0.098 点；相同权重的 exact AUC 为 −0.019／−0.138 点。两组 native 轮次逐 seed 对齐。
- **seed3353 首次出现两种选择分歧。** G0/G1 的 native 选择 epoch9，exact 选择 epoch6；G1 exact 选中结果相对 native 的 TEST CW/OW native AUC 平均低0.193／0.215点。
- **门控方差约98%为固定物体／模式成分。** G1 native 的图像变化占比为2.02%／1.95%／1.83%；dynamic−mean 的 TEST CW/OW AUC 为 +0.135±0.179／−0.037±0.187点。
- **固定均值门控可吸收进静态 CP 物体因子。** 12×768 物体文本矩阵满行秩；六套选中权重的数值验证中，折叠前后的修正 logits 误差小于1e-14。
- **下一轮建议 H1：用相对 TRAIN 均值的局部视觉残差驱动门控。** 固定 β=0.075、ρ=0.15、η=0.25，检验图像相关成分；H1 为后续表示设计，本阶段实际归档为已完成的 G0/G1。

![同配置静态与动态训练对照](analysis/confirmed/figures/02_matched_static_dynamic.png)

完整分组、双选择、逐组合变化、门控方差与静态等价推导，以及 H1 的公式和实践安排，见[数据分析 README](analysis/README.md)。
'''
    for old,new in [(r'\frac1r',r'\frac{1}{r}'),(r'\frac\eta2',r'\frac{\eta}{2}')]:main_text=main_text.replace(old,new)
    write('README.md',main_text.replace('{{SEEDS}}',seed_table).replace('{{MEANS}}',means))
    code=r'''# 08 · 代码与修订

原 G 包源码位于 [vgcpod_v1_roundG](vgcpod_v1_roundG/)。配置、资产、原说明和验证记录按平行结构分区保存，materialize 恢复原始布局。本轮相对 [07 F组](../../07/code/README.md)的实际改动见[源码补丁](source_changes.patch)。

| 入口 | 改动与用途 |
|---|---|
| [G0/G1配置](../configs/round_g/) | β=0.075；gate_input=none/local |
| [round_g.py](vgcpod_v1_roundG/round_g.py) | `worker()` 每卡顺序 G0→G1；`launch()` 三个队列并行；`summarize()` 输出主指标、双选择、β与静态配对及诊断 |
| [ranking_diagnostics.py](vgcpod_v1_roundG/ranking_diagnostics.py) | `diagnose()` 从完整OW预测计算排序量；`summarize_entries()` 输出分组、逐组合及同seed比较 |
| [tmux启动](vgcpod_v1_roundG/scripts/start_round_g_tmux.sh) | 记录激活环境 Python、G会话和退出码 |
| [G组测试](vgcpod_v1_roundG/tests/test_round_g.py) | 配置范围、静态极限、β=0.075目标梯度、已知概率样例；累计39项 |
| [模型](vgcpod_v1_roundG/vgcpod/model.py)／[训练器](vgcpod_v1_roundG/vgcpod/engine.py) | 与 F 包逐字节一致，通过已有配置分支实现本轮目标 |
| [均值评估器](vgcpod_v1_roundG/mean_gate_eval.py) | 沿用 F 包；本轮只对 G1 的两种权重执行均值替换 |

## 1. 静态和动态训练分支

G0/G1 都训练适配器和全局读出 CP。`gate_input=none` 时 Bvis 冻结、gamma 全为1、orth为0；local 时使用中心化视觉门控并训练 Bvis。两组 task 相同：

```math
\mathcal{L}_{\mathrm{task}}(0.075)=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

G1 保留 λorth=0.001。原模型和训练运算顺序保持源包字节身份，实际配置与训练参数数目见各运行的 protocol。

## 2. 三 GPU 双任务队列

- seed3351/3352/3353 使用物理GPU1/2/3。每个队列顺序训练G0→G1，然后做G1 native/exact均值评估；共六次训练、六项评估。
- 已完成任务直接复用，未完成任务从完整epoch继续。`round_g_status.json` 按三条seed队列记录两次训练及两项评估。
- 每个模型独立保存 `best_native.pt`、兼容别名 `best.pt`、`best_exact.pt` 和 `last.pt`，两条选择分别导出。
- `round_g_paired_*` 首先比较G1−G0、G1−E1/F1；`beta_native_*` 比较同ρ下β=0.05/0.075/0.1。

## 3. 保存预测上的排序诊断

`diagnose()` 使用全部OW组合的 log 概率恢复物体边际，使用导出的GT-object条件logits计算属性准确率和NLL。按split区分TRAIN、CW与CW集合外候选，统计概率质量、最高分超越关系和未见候选内top-1；另按组合记录正确数与NLL。

`summarize_entries()` 核对图像与标签顺序，按selection、condition、phase、group汇总。G1−G0、G1−E1/F1/D1/A5和dynamic−mean保存同seed差值、均值及样本SD。诊断不改变训练或预测。

## 4. 原发布验证与仓库复算

- [39项测试](../results/package_validation/G/validation/unittest.txt)、[六训练六评估流程](../results/package_validation/G/validation/result.json)、[流程日志](../results/package_validation/G/validation/cli_output.txt)。
- 原F训练器设置为相同G0/G1配置时，native/exact/last参数、AdamW状态和Torch RNG逐元素一致；重复启动跳过全部已完成任务。
- [真实E1/F1诊断复核](../results/package_validation/G/validation/real_prediction_diagnostics.json)、[代码范围](../results/package_validation/G/validation/source_scope.json)、[服务器身份](../results/server_source_identity.json)。
- [原包76成员映射](../package_members.json)、[原分析源码](original_analysis/)、[仓库复算入口](../analysis/recompute.py)、[归档验证](../analysis/repository_validation.json)。

服务器使用Python3.8.18、PyTorch1.13.1+cu117、RTX3090；发布数学及调度测试使用CPU合成特征，正式训练结果来自服务器。CLIP许可见[CLIP_LICENSE](vgcpod_v1_roundG/vendor/CLIP_LICENSE)。
'''
    write('code/README.md',code)
    config=r'''# 08 · 配置与超参数策略

| 方案 | gate_input | ρ | β | η | native／exact最佳epoch |
|---|---|---|---|---|---|
| [G0静态](round_g/G0.json) | none | 0.15 | 0.075 | 配置0.25，静态分支不使用 | 8/8/9；8/8/6 |
| [G1动态](round_g/G1.json) | local | 0.15 | 0.075 | 0.25 | 8/8/9；8/8/6 |

每列轮次按seed3351/3352/3353顺序。六次训练各十轮、3600次更新；两组逐seed轮次对齐。完整配置与两套指标见[experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. **β取中间值。** 在已有E1 β=0.1与F1 β=0.05、相同ρ=0.15下取β=0.075，测量属性泛化与组合排序折中。
2. **精确解释系数。** pair系数1.5不变，support系数为9/86≈0.104651163。
3. **加入同配置静态训练。** G0/G1同β、ρ、rank、读出、初始化规则与优化日程；G0冻结Bvis且orth为0，G1使用视觉门控与对应正交项。
4. **固定剩余超参数。** η=0.25、τγ=0.1、CP分组LR、batch64和十轮日程沿用F；三GPU各负责一个seed的两组队列。
5. **沿用选权重规则。** native主结果取首次严格最大VAL CW native AUC；exact路径独立保存。各条件使用各自VAL native bias并固定到TEST。
6. **同seed配对。** 先读G1−E1/F1，再读G1−G0，随后读G1 dynamic−mean及分组排序诊断。

```math
\mathcal{L}_{\mathrm{task}}(0.075)=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结OpenAI ViT-L/14，224px、768维 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.1 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 梯度裁剪／动态正交项／workers | 1 / 0.001 / 4 |

内部variant=V1；由gate_input决定是否启用视觉门控，中心化模式保留。CP组G0为Au/Av/Wc，G1增加Bvis；cp_lr_scale=0.3，Wc零初始化，prim/cent分支关闭。

## 配置、资产与对照索引

- [G0/G1实际配置](round_g/)、[F1](round_f/F1.json)、[E1](round_e/E1.json)、[D1](round_d/D1.json)、[默认配置](default.json)。
- [六项训练清单](experiment_inventory.json)、[六项均值评估清单](intervention_inventory.json)。
- [G前20份参考指标](references/pre_g_complete.json)、[标签资产](assets/)、[已有阶段与外部输入](../external_inputs.json)。
- 缓存manifest SHA256为 `bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

后续H1门控输入表示设计见[数据分析](../analysis/README.md)，本阶段实际配置为G0/G1。
'''
    write('configs/README.md',config)
    results=r'''# 08 · G组训练结果

本阶段新增G0/G1各三seed，共六次训练、60个模型epoch。native与exact两种选择分别保存24份评价／预测；六项G1均值替换评估包含12个条件、48份评价／预测。合计 **96份预测导出、24份checkpoint文件**。

## native选择主结果

{{SEEDS}}

AUC为服务器保存的native AUC、面积×100。每次训练均完成10轮、3600次更新。每个模型的best.pt与best_native.pt为相同字节，last.pt保存完整训练状态。

## exact选择结果

{{EXACT}}

seed3351/3352两种选择均为epoch8；G0/G1的seed3353分别选中native epoch9与exact epoch6。对应权重和预测独立保留。G0/G1在前两个seed上的16份exact/native预测逐元素一致，第三个seed使用不同权重。

## 原运行、日志与汇总

- [G0原运行](raw_export/roundG_20261020/G0/ut-zap50k/V1/)与[G1原运行](raw_export/roundG_20261020/G1/ut-zap50k/V1/)包含protocol、history、metrics、曲线、预测和四份checkpoint，exact结果位于各seed的exact_selection目录。
- [三条队列状态](raw_export/roundG_20261020/round_g_status.json)、[六项任务表](raw_export/roundG_20261020/round_g_jobs.json)、[主日志](raw_export/roundG_20261020/launcher.log)、[退出码0](raw_export/roundG_20261020/launcher.exitcode)。
- [主结果](raw_export/roundG_20261020/round_g_summary.csv)、[均值与SD](raw_export/roundG_20261020/round_g_mean_sd.csv)、[配对增量](raw_export/roundG_20261020/round_g_paired_deltas.csv)、[配对汇总](raw_export/roundG_20261020/round_g_paired_mean_sd.csv)。
- [三档β主表](raw_export/roundG_20261020/beta_native_summary.csv)、[支持损失表](raw_export/roundG_20261020/support_training_summary.csv)、[exact指标](raw_export/roundG_20261020/exact_selection_summary.csv)、[选择差异](raw_export/roundG_20261020/exact_selection_effects.csv)。
- [排序诊断](raw_export/roundG_20261020/ranking_diagnostics.json)、[诊断配对汇总](raw_export/roundG_20261020/ranking_diagnostic_paired_mean_sd.csv)、[逐组合概率](raw_export/roundG_20261020/pair_probability_diagnostics.csv)。

## G1动态与TRAIN均值替换

{{GATE}}

[native均值评估](raw_export/roundG_20261020/mean_gate/)和[exact均值评估](raw_export/roundG_20261020/mean_gate_exact/)各保存TRAIN均值、图像方差及dynamic/train_mean两种条件的四种评价、预测与曲线。统计使用22,998张TRAIN图像，均值形状[12,32]，优化步数为零；条件各用自己的VAL bias。

## 原交付、来源与分析

- [原服务器TAR](archives/vgcpod_roundG_results_20261020.tar.gz)、[原G训练ZIP](archives/vgcpod_v1_roundG_20261020.zip)，旁附SHA256；二进制沿用Git LFS。
- [原发布验证](package_validation/G/)、[原包说明字节](package_support/G/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包76成员映射](../package_members.json)、[新增结果成员与哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[已有结果引用](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原TAR中C/D/E/F结果与源码引用[04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)、[07](../../07/results/README.md)已有归档；A5/A6引用[02](../../02/results/README.md)。
- [仓库复算验证](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。静态CP折叠为保存权重上的算子等价验证，未增加训练或原预测导出。

G1−G0的TEST CW/OW native AUC为+0.019/−0.127点；β中间值与门控训练收益分别分析。H1为后续建议，本阶段归档为已完成G0/G1。
'''
    write('results/README.md',results.replace('{{SEEDS}}',seed_table).replace('{{EXACT}}',exact_table).replace('{{GATE}}',gate_table))
    analysis=(WORK/'VG-CPOD_v1_G组结果分析与下一轮方向_20261020.md').read_text(encoding='utf-8')
    analysis=analysis.replace('# VG-CPOD v1：G 组结果分析与下一轮方向','# 08 · 数据分析',1)
    intro='日期：2026-10-20。框架：冻结CLIP ViT-L/14，UT-Zappos。原运行见[结果索引](../results/README.md)，修订见[主README](../README.md)。本页先分析已完成G0/G1，再列后续H1表示设计。'
    analysis=re.sub(r'日期：[^\n]+',lambda _:intro,analysis,count=1)
    analysis=re.sub(r'^\$\$\n(.*?)\n\$\$',lambda m:'```math\n'+m[1]+'\n```',analysis,flags=re.M|re.S)
    analysis=analysis.replace('analysis_roundG_results_20261020/','confirmed/')
    for name in ('analyze_g.py','supplementary_analysis.py'):
        analysis=analysis.replace('(confirmed/'+name+')','(../code/original_analysis/analysis_roundG_results_20261020/'+name+')')
    analysis+='''\n## 9. 仓库复算与归档核查

- [复算入口](recompute.py)顺序运行[预测与指标](recompute_metrics.py)、[排序诊断及静态CP等价](recompute_supplementary.py)、[图表](recompute_figures.py)。读取02/04/05/06/07已有结果、08新增结果与01缓存元数据。
- [仓库验证](repository_validation.json)核对原包成员、TAR结果、README数值、公式及链接和运行包恢复；[原始复算记录](confirmed/audit.json)、[诊断及等价验证](confirmed/supplementary_audit.json)。

```bash
python VG-CPOD方案/08/analysis/recompute.py
python tools/reproduce.py verify
```

输出至analysis/recomputed，原分析在analysis/confirmed。本阶段实际完成G0/G1各三seed，H1为后续输入表示实验建议。
'''
    write('analysis/README.md',analysis)
    path=REPO/'VG-CPOD方案/README.md';text=path.read_text(encoding='utf-8')
    row='| 08 | '+TITLE+'（G 组）；G0/G1各三seed、β=0.075、双准则选择与六项均值评估 | [修订说明](08/README.md) · [代码](08/code/README.md) · [配置](08/configs/README.md) · [结果](08/results/README.md) · [数据分析](08/analysis/README.md) |'
    if '| 08 |' not in text:text=text.rstrip()+'\n'+row+'\n'
    path.write_text(text,encoding='utf-8',newline='\n')
    path=REPO/'README.md';text=path.read_text(encoding='utf-8')
    latest='最新归档（2026-10-20）：[VG-CPOD 08 · '+TITLE+'](VG-CPOD方案/08/README.md)，包含G0/G1（ρ=0.15、β=0.075）各三seed、静态与视觉门控训练对照、native／exact双准则及六项均值评估，共96份预测、24份checkpoint文件；附公式与代码修订、超参数策略及[数据分析](VG-CPOD方案/08/analysis/README.md)。前序[07 F组](VG-CPOD方案/07/README.md)、[06 E组](VG-CPOD方案/06/README.md)、[05 D组](VG-CPOD方案/05/README.md)、[04 C组](VG-CPOD方案/04/README.md)、[03 B组](VG-CPOD方案/03/README.md)、[02 A组](VG-CPOD方案/02/README.md)与[01原正式结果](VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/README.md)继续保留。'
    text=re.sub(r'^最新归档[^\n]+',lambda _:latest,text,flags=re.M)
    text=text.replace('| VG-CPOD | 01–07 |','| VG-CPOD | 01–08 |').replace('02–07 使用对应编号','02–08 使用对应编号').replace('02–07 可直接复用','02–08 可直接复用')
    text=text.replace('07 恢复原 F 包，包含支持损失日志与 β 配对汇总。','07 恢复原 F 包，包含支持损失日志与 β 配对汇总；08 恢复原 G 包，包含静态与动态配对队列及排序诊断。')
    path.write_text(text,encoding='utf-8')
    print('Five experiment READMEs and two indexes updated.')


if __name__=='__main__':
    main()
