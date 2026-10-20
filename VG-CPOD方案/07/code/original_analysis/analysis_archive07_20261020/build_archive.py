"""Write experiment 07's five parallel READMEs and update repository indexes."""
from pathlib import Path
import json
import re

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parent/'CZSL'
STAGE=REPO/'VG-CPOD方案/07'
TITLE='支持集权重下调与跨组合排序诊断'
D=json.loads((WORK/'analysis_roundF_results_20261020/analysis.json').read_text(encoding='utf-8'))
INVENTORY=json.loads((STAGE/'configs/experiment_inventory.json').read_text(encoding='utf-8'))


def write(relative,text):
    (STAGE/relative).write_text(text,encoding='utf-8',newline='\n')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])


def mean(case,setting,key='AUC_native_mean'):
    return next(r[key] for r in D['means'] if r['case']==case and r['setting']==setting)


def main():
    seed_table=table(['seed','native／exact最佳epoch','VAL CW','VAL OW','TEST CW','TEST OW'],
        [[r['seed'],str(r['selected_epoch'])+'／'+str(r['exact_selected_epoch'])]+
         [f"{r['metrics'][k]['AUC_native']:.4f}" for k in ('val_cw','val_ow','test_cw','test_ow')] for r in INVENTORY])
    means=table(['三 seed 方案','ρ','β','VAL CW AUC','TEST CW AUC','TEST OW AUC','TEST 未见条件属性'],
        [[case,rho,beta]+[f"{mean(case,k):.3f}" for k in ('val_cw','test_cw','test_ow')]+
         [f"{mean(case,'test_ow','gt_object_attr_acc_unseen_mean'):.3f}%"]
         for case,rho,beta in [('A5',.1,.1),('D1',.125,.1),('E1',.15,.1),('F1',.15,.05),('F1_mean',.15,.05)]])
    gate_rows=[]
    for s in (3351,3352,3353):
        values=[next(r['AUC_native'] for r in D['comparisons'] if r['comparison']=='F1-F1_mean' and r['seed']==s and r['setting']==k) for k in ('test_cw','test_ow')]
        correct=next(r['net_correct'] for r in D['similarity'] if r['comparison']=='F1-F1_mean' and r['seed']==s and r['phase']=='test' and r['group']=='unseen')
        gate_rows.append([s]+[f'{v:+.4f}' for v in values]+[f'{correct:+d}'])
    gate_table=table(['seed','动态−均值 TEST CW AUC','TEST OW AUC','未见条件属性正确数变化'],gate_rows)
    main_text=r'''# 07 · 支持集权重下调与跨组合排序诊断

**F 组实验，VG-CPOD v1，归档日期 2026-10-20。** 使用冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 F1 三 seed、30 个模型 epoch，保存 native／exact 两种选择的 24 份评价与预测，以及六项动态／TRAIN 均值替换评估的 48 份评价与预测，共 72 份预测导出、12 份 checkpoint 文件。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [06 E组](../06/README.md) · [05 D组](../05/README.md)

## 1. 实验目的与修订范围

06 的 E1 使用 ρ=0.15、β=0.1。其 VAL CW AUC 三 seed 均值最高，TEST 未见条件属性低于 D1，同时未见图像上的 TRAIN 组合概率质量增加。本轮以 E1 为基座，只把 β 下调为 0.05，测量支持集损失权重对属性泛化和组合排序的作用。

F1 与 E1 的配置仅 `task_aux_beta`、`experiment_id` 两个字段不同。ρ=0.15、中心化门控 η=0.25、τγ=0.1、全局 g 读出、rank、学习率和 10 epochs 均沿用 E1。

模型与损失文件 `vgcpod/model.py`、CLIP 特征实现和 `mean_gate_eval.py` 与 E 组逐字节一致。代码新增四个加权损失日志字段、F1 三卡编排、β 对照表及支持集训练诊断；两个选权重轨道继续沿用。

## 2. 公式修订：降低支持集项系数，pair 项保持 1.5

完整联合分布按物体与条件属性分解。记张量轴为 `[batch, object, attribute]`，真实标签为 $(i_b,j_b)$，TRAIN 组合集合为 $\mathcal{S}$：

```math
\begin{aligned}
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
Z_b&=\sum_{(j,i)\in\mathcal{S}}\exp(s_{bji}),\\[4pt]
\mathcal{L}_{\mathrm{pair}}
&=\frac{1}{B}\sum_b[-s_{bj_bi_b}+\log Z_b],\\[4pt]
\mathcal{L}_{\mathrm{support}}&=-\frac{1}{B}\sum_b\log Z_b,\\[4pt]
\mathcal{L}_a+\mathcal{L}_o
&=-\frac{1}{B}\sum_b s_{bj_bi_b}
=\mathcal{L}_{\mathrm{pair}}+\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

条件 softmax 遍历全部 16 个属性，物体 softmax 遍历全部 12 个物体，联合概率在 192 个组合上归一化；Z 是其中 83 个 TRAIN 组合的总概率质量。

现有归一化训练目标可精确改写为：

```math
\begin{aligned}
\mathcal{L}_{\mathrm{task}}(\beta)
&=\frac{1.5}{1+\beta}
\left[\mathcal{L}_{\mathrm{pair}}+\beta(\mathcal{L}_a+\mathcal{L}_o)\right]\\[4pt]
&=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}^{\mathrm{E1}}
&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{3}{22}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}_{\mathrm{task}}^{\mathrm{F1}}
&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac{1}{14}\mathcal{L}_{\mathrm{support}}.
\end{aligned}
```

因此本轮支持集项系数从 0.136363636 降为 0.071428571，比例为 11/21，即原来的 **52.38%**；pair 项系数保持 1.5。这是对已有 objective 的系数调整与恒等分解，反向传播沿用原表达式及运算顺序。

F1 的动态正交项和 AdamW weight decay 沿用原实现：

```math
\mathcal{L}_{\mathrm{F1}}
=\frac32\mathcal{L}_{\mathrm{pair}}+\frac1{14}\mathcal{L}_{\mathrm{support}}
+\frac{0.001}{r^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^{\top}-I_r\right\rVert_F^2.
```

原子属性与伙伴辅助分支关闭，prim 和 cent 为零。权重衰减由 AdamW 执行。

## 3. 沿用的视觉修正与新增日志公式

令 r=32，d=768，属性／物体文本模式为 $u_i=A_u e_i^a$、$v_j=A_v e_j^o$。沿用模式中心化和全局读出：

```math
\begin{aligned}
q_{bj}&=2\sigma(B_{\mathrm{vis}}f^o_{bj}/\tau_\gamma)-\mathbf1_r,\\[4pt]
H_r&=I_r-\frac1r\mathbf1_r\mathbf1_r^\top,\\[4pt]
\gamma_{bj}&=\mathbf1_r+\frac\eta2H_rq_{bj},
\qquad\eta=0.25,\quad\tau_\gamma=0.1,\\[4pt]
U_{bji}&=U^{\mathrm{base}}_{bji}
+0.15\kappa\,g_b^\top W_c(u_i\odot v_j\odot\gamma_{bj}).
\end{aligned}
```

每个图像／物体查询的模式均值为 1。CLIP 图文编码器冻结，全局与物体查询特征从同一缓存读取。

`engine.py` 在 `optimizer.step()` 后，从已脱离计算图的标量记录：

```math
\begin{aligned}
w_{\mathrm{support}}&=\frac{1.5\beta}{1+\beta},\\[4pt]
\ell_{\mathrm{pair,weighted}}&=1.5\mathcal{L}_{\mathrm{pair}},\\[4pt]
\ell_{\mathrm{support,weighted}}&=w_{\mathrm{support}}\mathcal{L}_{\mathrm{support}},\\[4pt]
\epsilon_{\mathrm{task}}&=\left|\mathcal{L}_{\mathrm{task}}
-\ell_{\mathrm{pair,weighted}}-\ell_{\mathrm{support,weighted}}\right|.
\end{aligned}
```

四个字段为 `support_weight`、`pair_weighted`、`support_weighted`、`task_reconstruction_error`。epoch 按每 batch 的实际图像数加权；最后一个字段是平均 batch 绝对误差。汇总表另列 epoch 均值的重构误差，二者分别记录。

## 4. 具体代码修订与验证

| 文件 | 本轮具体改动 | 作用 |
|---|---|---|
| [F1.json](configs/round_f/F1.json) | `task_aux_beta=0.05`、`experiment_id=F1` | 单独测量支持集项系数 |
| [engine.py](code/vgcpod_v1_roundF/vgcpod/engine.py) | 新增 `task_decomposition()` 和四个标量日志字段 | 直接记录原始与加权损失的关系 |
| [round_f.py](code/vgcpod_v1_roundF/round_f.py) | 三卡训练、续训、六项均值评估、β 主表及 F1−E1 等配对汇总 | 输出完整 F 组结果 |
| [pre_f_complete.json](configs/references/pre_f_complete.json) | 17 份已有 A5/A6/C1/C2/D1/E1 指标和选中 epoch TRAIN 快照 | 提供已有对照 |
| [tmux启动](code/vgcpod_v1_roundF/scripts/start_round_f_tmux.sh) | F 组独立会话、输出与日志路径 | GPU1/2/3 分别运行一个 seed |
| [F组测试](code/vgcpod_v1_roundF/tests/test_round_f.py) | 配置差异、损失及梯度恒等式、有限差分、加权日志 | 累计 36 项发布测试 |

[相对06的源码补丁](code/source_changes.patch)保留实际差异。新增代码不改变模型初始化、objective、优化器或学习率的实现。

[原 CPU 完整流程](results/package_validation/F/validation/result.json)完成三项训练、六项均值评估与重复启动复用；将原 E 引擎设为相同 β=0.05 后，native best／last 的模型参数、AdamW 状态与 Torch 随机状态逐元素一致。[服务器源码核对](results/server_source_identity.json)及[原包69成员映射](package_members.json)保留来源身份。

## 5. 超参数选取策略与实际矩阵

1. **从 E1 单独改变 β。** E1 按既定 VAL CW 三 seed 均值规则作为基座，ρ=0.15 固定，β 从 0.1 下调到 0.05。
2. **用精确系数解释调整。** pair 系数固定 1.5，support 系数约减半；由此测量 TRAIN 组合质量与未见属性的变化。
3. **保持相同初始化和训练日程。** 三个 seed 从各自原始初始化开始，各 10 epochs、3600 次更新。
4. **同 seed 配对。** 首要比较 F1−E1，随后列 D1、A5、A6、C2 和已有两个 seed 的 C1；配对使用相同特征缓存。
5. **保留两种选权重和均值评估。** native 主结果、exact 选择单列，各条件使用自己的 VAL native bias 并固定到 TEST。

{{SEEDS}}

| 固定项 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 ViT-L/14，224px，768 维，同一缓存 |
| ρ／η／τγ／β | 0.15 / 0.25 / 0.1 / **0.05** |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| 适配器 LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |
| eval batch／workers | 64 / 4 |

CP 参数组为 Bvis、Au、Av、Wc，`cp_lr_scale=0.3`，Wc 从零初始化。内部 `variant=V1`，动态行为由 `gate_input=local` 和 `gate_center_modes=true` 显式启用。实际全字段配置和两套指标见[实验清单](configs/experiment_inventory.json)。

## 6. 仓库结构与复现入口

```text
07/
  README.md          # 修订、公式、超参策略与末节数据分析
  code/              # 原F代码、相对E补丁、原始分析源码
  configs/           # F1、兼容配置、资产、参考与运行清单
  results/           # 三seed、双选择、均值评估、状态及原归档
  analysis/          # 数据分析、机制分解、图表及仓库复算入口
```

恢复本轮训练包并复用服务器已有 CLIP 特征：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 07 \
  --dest /data/xhw/vgcpod_project/replay_07
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_07/vgcpod_v1_roundF

export VG_F_OUT=/data/xhw/vgcpod_v1_runs/replay_07_F
export VG_F_SESSION=vgcpod_07_F
bash scripts/start_round_f_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020
tmux attach -t vgcpod_07_F
```

每卡先训练对应 seed，再运行 native／exact 两项均值评估。按 Ctrl+b、d 脱离 tmux。`launcher.exitcode=0` 且三项状态均为 complete 表示整轮完成。已有对照优先读取对应运行，包内 17 份快照提供同 seed 标量对比。外部输入见 [external_inputs.json](external_inputs.json)，命令见 [runbook.json](runbook.json)。

从仓库预测、权重和缓存元数据复算本次分析与概率因子交换：

```bash
python VG-CPOD方案/07/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`，原分析保存于 `analysis/confirmed/`。运行包的原 README、说明及验证字节分区保留，materialize 恢复原始布局。

## 7. 数据分析与下一轮方向

{{MEANS}}

AUC 为面积×100，条件属性使用 GT-object。**F1 相对 E1 的 TEST CW／OW AUC 为 −0.397 ± 0.232／−0.829 ± 0.756 点，未见条件属性为 +1.816 ± 0.424 个百分点。** 三个 seed 均呈现属性提高、组合 AUC 下降；E1 和 F1 同 seed 均选中 8／8／6 轮。

- TEST 未见图像上的 TRAIN 组合概率质量从 73.698% 降至 70.770%，条件属性 NLL 从 2.3872 降至 2.2884；调整支持集权重确实改善了属性泛化。
- 未见图像的物体准确率由 63.353% 升至 63.741%；全部未见候选内真实组合 top-1 由 49.921% 降至 48.017%。CW 集合外候选最高分压过 CW 内未见最高分的比例由 23.621% 升至 25.771%。
- 概率因子交换中，E1 物体×F1 条件属性的 OW exact AUC 为 24.548，F1 物体×E1 条件属性为 25.131；原 E1／F1 为 25.261／24.451。排序变化主要随条件分支出现。
- F1 dynamic−TRAIN mean 的 TEST CW／OW AUC 为 +0.016／−0.003 点，配对 SD 为 0.096／0.076；门控 TRAIN 图像变化占比为 1.51%—1.88%。
- F1 的 OW HM 均值为 40.349，相对 E1 的 39.792 提高 0.556 点；三个 seed 的增量为 −0.588／−1.389／+3.645。已见准确率提高、未见准确率下降，工作点向已见侧移动。
- **下一轮建议 G1：ρ=0.15、β=0.075、η=0.25；并补 G0：相同 β 和 ρ，gate_input=none 的静态训练对照。** 各三个 seed、10 epochs，共六次训练。该项为后续计划；本阶段归档的实际训练为 F1、β=0.05。

![F1 相对 E1 的属性与组合排序变化](analysis/confirmed/figures/06_f1_e1_paired.png)

完整配对结果、损失日志核查、逐组合概率质量、概率因子交换、动态／均值分析及下一轮设计见[数据分析 README](analysis/README.md)。
'''
    main_text=main_text.replace(r'\mathbf1_r',r'\mathbf{1}_r').replace(r'\frac1r',r'\frac{1}{r}').replace(r'\frac\eta2',r'\frac{\eta}{2}')
    write('README.md',main_text.replace('{{SEEDS}}',seed_table).replace('{{MEANS}}',means))
    code=r'''# 07 · 代码与修订

原 F 训练包源码位于 [vgcpod_v1_roundF](vgcpod_v1_roundF/)。配置、资产、原说明和验证记录按仓库分区保存，materialize 恢复原布局。本轮模型与损失沿用 [06 E组](../../06/code/README.md)，实际差异见[源码补丁](source_changes.patch)。

| 入口 | 具体改动与作用 |
|---|---|
| [F1配置](../configs/round_f/F1.json) | β=0.05、experiment_id=F1；其余与 E1 一致 |
| [engine.py](vgcpod_v1_roundF/vgcpod/engine.py) | 增加 `task_decomposition()` 和四个标量日志字段 |
| [round_f.py](vgcpod_v1_roundF/round_f.py) | 三卡调度、自动续训、六项均值评估、β 对比和支持损失表 |
| [tmux启动](vgcpod_v1_roundF/scripts/start_round_f_tmux.sh) | F 组会话、输出路径、日志和退出码 |
| [F组测试](vgcpod_v1_roundF/tests/test_round_f.py) | 配置差异、损失与梯度恒等式、有限差分和加权日志；累计36项 |
| [模型与损失](vgcpod_v1_roundF/vgcpod/model.py) | 与 E 组逐字节一致，已有 `task_aux_beta` 直接读取新值 |
| [均值评估](vgcpod_v1_roundF/mean_gate_eval.py) | 沿用 E 组 native／exact 选择与 TRAIN 均值实现 |

## 1. 加权损失日志

训练反向传播继续调用已有 `objective()`。`optimizer.step()` 后，loss 张量脱离计算图并转为 Python 标量，再计算：

```python
def task_decomposition(logs, beta):
    support_weight = 1.5 * beta / (1.0 + beta)
    pair_weighted = 1.5 * logs['pair']
    support_weighted = support_weight * logs['support']
    return dict(support_weight=support_weight, pair_weighted=pair_weighted,
                support_weighted=support_weighted,
                task_reconstruction_error=abs(logs['task'] - pair_weighted - support_weighted))
```

对应恒等式：

```math
\mathcal{L}_{\mathrm{task}}=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
```

epoch 按 batch 的实际图像数加权。`task_reconstruction_error` 为平均 batch 绝对误差，`support_training_summary.csv` 的 `task_epoch_reconstruction_error` 由 epoch 均值另行计算。原日志的 task、pair、cond、object、support、orth 等字段继续保存。

## 2. 三 GPU 编排与汇总

- seed3351／3352／3353 分别使用物理 GPU1／2／3；每个子进程内部为 `cuda:0`。
- 每卡先完成本 seed 的 F1 训练，再运行 native／exact 两项均值替换，合计三次训练、六项评估。
- 两个最优 checkpoint 的首次严格最大值、`best.pt` 兼容副本及双轨续训沿用 E 实现。
- `beta_native_summary.csv` 对比 E1／F1；`round_f_paired_*` 对比 F1−E1/D1/A5/A6/C2/C1；`support_training_summary.csv` 保存选中及最后 epoch 的原始与加权损失。
- 原始预测支持逐组合对比；同 seed 和缓存指纹用于配对。17份参考指标见 [pre_f_complete.json](../configs/references/pre_f_complete.json)。

## 3. 原发布验证与仓库复算

- [36项测试输出](../results/package_validation/F/validation/unittest.txt)、[CPU完整流程](../results/package_validation/F/validation/result.json)、[流程日志](../results/package_validation/F/validation/cli_output.txt)。
- 相同 β=0.05、初始化及缓存下，与原 E 训练器的 native best／last 比较，模型参数、AdamW 状态和 Torch RNG 逐元素一致。三次训练、六项评估的重复启动全部复用已有结果。
- [代码范围核查](../results/package_validation/F/validation/source_scope.json)、[原文档公式排版](../results/package_validation/F/validation/document_layout.json)、[服务器源码核对](../results/server_source_identity.json)。
- [69个原包成员映射](../package_members.json)、[原分析源码](original_analysis/)、[仓库复算入口](../analysis/recompute.py)、[归档核查](../analysis/repository_validation.json)。

服务器使用 Python 3.8.18、PyTorch 1.13.1+cu117、RTX3090；原数学与编排测试使用 CPU 合成特征，真实训练结果来自服务器原始导出。CLIP 许可保留于 [CLIP_LICENSE](vgcpod_v1_roundF/vendor/CLIP_LICENSE)。
'''
    write('code/README.md',code)
    config=r'''# 07 · 配置与超参数策略

| 方案 | 读出 | 门控 | ρ | β | η | 新增seed／native最佳epoch |
|---|---|---|---|---|---|---|
| [F1](round_f/F1.json) | 全局 g | 物体查询输入、模式中心化 | 0.15 | **0.05** | 0.25 | 3351/8、3352/8、3353/6 |

F1 与 [E1](round_e/E1.json) 的完整配置仅 `task_aux_beta` 与 `experiment_id` 不同。三次训练各 10 epochs、3600 次更新；native／exact 均选中 8／8／6 轮。实际全字段与两套指标见[experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. E1 按既定 VAL CW 三 seed 均值规则作为基座，固定 ρ=0.15；以 β=0.05 测量支持集权重下调的效果。
2. 精确保持 pair 项系数 1.5；support 系数从 3/22 降到 1/14，为原来的 52.38%。
3. 固定模型结构、η=0.25、τγ=0.1、rank32、CP分组学习率与10轮日程，直接与 E1 同 seed 配对。
4. seed3351／3352／3353 分配到物理 GPU1／2／3，每个 seed 从对应原始初始化训练。
5. 主结果按首次严格最大 VAL CW native AUC 选权重；exact 权重单独保存。每种选择完成一次 TRAIN 全集均值替换，条件分别用各自 VAL bias。
6. 对 F1−E1 的 AUC、HM、条件属性、支持集质量联合分析，后续建议 β=0.075 的 G1 与同配置静态 G0；本阶段实际配置为 F1、β=0.05。

```math
\mathcal{L}_{\mathrm{task}}^{\mathrm{F1}}
=\frac32\mathcal{L}_{\mathrm{pair}}+\frac1{14}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结 OpenAI ViT-L/14，224px，768维 |
| ρ／η／τγ／β | 0.15 / 0.25 / 0.1 / 0.05 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 梯度裁剪／正交项／workers | 1 / 0.001 / 4 |

CP 参数组为 Bvis、Au、Av、Wc，`cp_lr_scale=0.3`，Wc 从零初始化。内部 `variant=V1`；通过 `gate_input=local`、`gate_center_modes=true` 启用视觉模式。prim 与 cent 分支关闭，兼容字段保留于原配置。

## 配置、资产与对照索引

- [F1实际配置](round_f/F1.json)、[E1直接对照](round_e/E1.json)、[D1配置](round_d/D1.json)、[C配置](round_c/)、[A配置](round_a/)、[B配置](round_b/)、[默认配置](default.json)。
- [三项训练清单](experiment_inventory.json)、[六项均值评估清单](intervention_inventory.json)。
- [F前17份参考指标](references/pre_f_complete.json)、[E前参考](references/pre_e_complete.json)、[A组三seed参考](references/round_a_confirm.json)、[标签资产](assets/)。
- [外部输入与已有阶段](../external_inputs.json)。缓存 manifest SHA256 为 `bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

后续 β=0.075 的系数与对照策略见[数据分析](../analysis/README.md)。
'''
    write('configs/README.md',config)
    results=r'''# 07 · F组训练结果

本阶段新增 F1 三次训练、30 个模型 epoch。native／exact 两种选择共保存 24 份 VAL／TEST CW／OW 评价与预测；六项均值替换评估共 12 个条件、48 份评价与预测。合计 72 份预测导出、12 份 checkpoint 文件。

{{SEEDS}}

AUC 使用服务器保存的 native AUC，面积×100。每次训练完成 10 epochs、3600 次更新；主权重按首次严格最大 VAL CW native AUC 选择。本轮 exact 与 native 的选中参数相同，12份对应预测数组逐元素一致。

## 运行、日志与β汇总

- [三个seed原运行](raw_export/roundF_20261020/F1/ut-zap50k/V1/)保存协议、完整历史、指标、曲线、预测与四份 checkpoint；exact 结果在各 seed 的 `exact_selection/`。
- [三卡状态](raw_export/roundF_20261020/round_f_status.json)、[任务表](raw_export/roundF_20261020/round_f_jobs.json)、[主日志](raw_export/roundF_20261020/launcher.log)、[退出码0](raw_export/roundF_20261020/launcher.exitcode)。三次训练及六项评估均完成。
- [原运行汇总](raw_export/roundF_20261020/round_f_summary.csv)、[均值与SD](raw_export/roundF_20261020/round_f_mean_sd.csv)、[同seed增量](raw_export/roundF_20261020/round_f_paired_deltas.csv)、[配对均值](raw_export/roundF_20261020/round_f_paired_mean_sd.csv)。
- [E1/F1 β主表](raw_export/roundF_20261020/beta_native_summary.csv)、[β均值](raw_export/roundF_20261020/beta_native_mean_sd.csv)、[支持损失训练表](raw_export/roundF_20261020/support_training_summary.csv)。
- [exact选择](raw_export/roundF_20261020/exact_selection_summary.csv)、[选权重差异](raw_export/roundF_20261020/exact_selection_effects.csv)、[逐组合增量](raw_export/roundF_20261020/per_pair_gains.csv)。

每个 seed 的 `best.pt` 与 `best_native.pt` 为逐字节相同的兼容入口；`best_exact.pt` 保留第二种选择的身份，`last.pt` 保存完整训练状态。12份 checkpoint 文件对应三次训练。

## 动态与TRAIN均值替换

{{GATE}}

[native评估](raw_export/roundF_20261020/mean_gate/)和[exact评估](raw_export/roundF_20261020/mean_gate_exact/)各保存 TRAIN 均值及图像方差、dynamic／train_mean 两种条件的指标、预测与曲线。统计使用22,998张TRAIN图像，均值形状 `[12,32]`，优化步数为零。条件各自使用自己的 VAL bias。

## 原交付、来源与分析

- [原服务器TAR](archives/vgcpod_roundF_results_20261020.tar.gz)、[原F训练ZIP](archives/vgcpod_v1_roundF_20261020.zip)，旁附 SHA256；二进制沿用 Git LFS。
- [原发布验证](package_validation/F/)、[原包说明字节](package_support/F/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包69成员映射](../package_members.json)、[新增结果成员与哈希](raw_export_members.json)、[服务器源码身份](server_source_identity.json)、[已有结果引用](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原 TAR 中 C／D／E 的结果与源码已与 [04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)核对并建立来源映射；A5／A6 三seed引用 [02](../../02/results/README.md)。
- [仓库复算核查](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。本轮分析额外从已保存概率计算两个分支因子交换，共24条标量评价，不增加训练或原始预测导出计数。

F1 相对 E1 的 TEST CW／OW AUC 为 −0.397／−0.829 点，未见条件属性 +1.816 个百分点。后续 G0／G1 属于分析建议；本阶段实际训练配置为 F1、β=0.05。
'''
    write('results/README.md',results.replace('{{SEEDS}}',seed_table).replace('{{GATE}}',gate_table))
    analysis=(WORK/'VG-CPOD_v1_F组结果分析与下一轮方向_20261020.md').read_text(encoding='utf-8')
    analysis=analysis.replace('# VG-CPOD v1：F 组结果分析与下一轮方向','# 07 · 数据分析',1)
    intro='日期：2026-10-20。框架：冻结 CLIP ViT-L/14，UT-Zappos。原运行见[结果索引](../results/README.md)，公式与修订见[主README](../README.md)。本页先分析已完成 F1，再记录 G0/G1 下一轮建议。'
    analysis=re.sub(r'日期：[^\n]+',lambda _:intro,analysis,count=1)
    analysis=re.sub(r'^\$\$\n(.*?)\n\$\$',lambda m:'```math\n'+m[1]+'\n```',analysis,flags=re.M|re.S)
    analysis=analysis.replace('analysis_roundF_results_20261020/','confirmed/')
    for name in ('analyze_f.py','analyze_mechanism.py'):
        analysis=analysis.replace('(confirmed/'+name+')','(../code/original_analysis/analysis_roundF_results_20261020/'+name+')')
    analysis+='''\n## 8. 仓库复算与归档核查

- [复算入口](recompute.py)顺序运行[预测复算](recompute_metrics.py)、[排序分解与因子交换](recompute_mechanism.py)、[图表](recompute_figures.py)。读取02/04/05/06已有结果、07新增结果和01缓存元数据，数值计算沿用原分析。
- [归档验证](repository_validation.json)记录原包与服务器成员、README数值、公式、链接和运行包恢复；[原始复算](confirmed/audit.json)、[新增日志核查](confirmed/supplementary_audit.json)、[机制分解](confirmed/mechanism.json)。

```bash
python VG-CPOD方案/07/analysis/recompute.py
python tools/reproduce.py verify
```

输出至 `analysis/recomputed/`；原分析在 `analysis/confirmed/`。本阶段已完成 F1 三seed、β=0.05，G0/G1 为后续实验建议。
'''
    write('analysis/README.md',analysis)
    path=REPO/'VG-CPOD方案/README.md'
    text=path.read_text(encoding='utf-8')
    row='| 07 | '+TITLE+'（F 组）；F1 三 seed、加权损失日志、双准则选权重及六项均值评估 | [修订说明](07/README.md) · [代码](07/code/README.md) · [配置](07/configs/README.md) · [结果](07/results/README.md) · [数据分析](07/analysis/README.md) |'
    if '| 07 |' not in text:
        text=text.rstrip()+'\n'+row+'\n'
    path.write_bytes(text.encode('utf-8'))
    path=REPO/'README.md';text=path.read_text(encoding='utf-8')
    latest='最新归档（2026-10-20）：[VG-CPOD 07 · '+TITLE+'](VG-CPOD方案/07/README.md)，包含 F1（ρ=0.15、β=0.05）三 seed、加权支持集损失日志、native／exact 选权重和六项均值评估，共72份预测、12份checkpoint文件；附公式与代码修订、超参数策略及[数据分析](VG-CPOD方案/07/analysis/README.md)。前序 [06 E组](VG-CPOD方案/06/README.md)、[05 D组](VG-CPOD方案/05/README.md)、[04 C组](VG-CPOD方案/04/README.md)、[03 B组](VG-CPOD方案/03/README.md)、[02 A组](VG-CPOD方案/02/README.md)与[01 原正式结果](VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/README.md)继续保留。'
    text=re.sub(r'^最新归档[^\n]+',lambda _:latest,text,flags=re.M)
    text=text.replace('| VG-CPOD | 01–06 |','| VG-CPOD | 01–07 |').replace('02–06 使用对应编号','02–07 使用对应编号').replace('02–06 可直接复用','02–07 可直接复用')
    text=text.replace('06 恢复原 E 包，包含双准则选权重和均值评估入口。','06 恢复原 E 包，包含双准则选权重和均值评估入口；07 恢复原 F 包，包含支持损失日志与 β 配对汇总。')
    path.write_text(text,encoding='utf-8')
    print('Five experiment READMEs and two indexes updated.')


if __name__=='__main__':
    main()
