"""Write experiment 09's five parallel READMEs and route indexes."""
import json
from pathlib import Path
import re

WORK=Path(__file__).resolve().parents[1]
REPO=WORK.parent/'CZSL'
STAGE=REPO/'VG-CPOD方案/09'
TITLE='训练均值中心化与归一化残差门控'
D=json.loads((WORK/'analysis_roundH_results_20261021/analysis.json').read_text(encoding='utf-8'))
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
         ['%.4f'%r['metrics'][k]['AUC_native'] for k in ('val_cw','val_ow','test_cw','test_ow')] for r in INVENTORY])
    exact_table=table(['实验','seed','exact最佳epoch','VAL CW native','VAL CW exact','TEST CW native','TEST OW native'],
        [[r['case'],r['seed'],r['exact_selected_epoch']]+
         ['%.4f'%r['exact_selection_metrics'][setting][key] for setting,key in
          [('val_cw','AUC_native'),('val_cw','AUC_exact'),('test_cw','AUC_native'),('test_ow','AUC_native')]] for r in INVENTORY])
    means=table(['三 seed 方案','VAL CW AUC','VAL OW AUC','TEST CW AUC','TEST OW AUC','TEST 未见条件属性'],
        [[case]+['%.3f'%mean(case,k) for k in ('val_cw','val_ow','test_cw','test_ow')]+
         ['%.3f%%'%mean(case,'test_ow','gt_object_attr_acc_unseen_mean')] for case in ('G0','G1','H1')])
    members=len(json.loads((STAGE/'package_members.json').read_text(encoding='utf-8'))['members'])
    main=r'''# 09 · 训练均值中心化与归一化残差门控

**H 组实验，VG-CPOD v1，归档日期 2026-10-21。** 使用冻结 OpenAI CLIP ViT-L/14，数据为 UT-Zappos natural split。本阶段新增 H1 三个 seed、三次训练和 30 个模型 epoch；保存 native／exact 两种选择的 24 份预测，以及六项动态／TRAIN 均值替换评估的 48 份预测，共 **72 份预测文件、12 份 checkpoint 文件**。

[代码与具体改动](code/README.md) · [超参数配置](configs/README.md) · [全部训练结果](results/README.md) · [数据分析](analysis/README.md) · [08 G组](../08/README.md) · [07 F组](../07/README.md)

## 1. 实验目的与修订范围

08 的 G1 使用物体查询特征直接驱动门控；选中权重的 TRAIN 门控总方差中，图像变化约占 2%，其余主要为固定物体／模式差异。本轮先统计每个查询在全部 TRAIN 图像上的均值，再把归一化视觉残差输入门控，检验图像变化对属性与组合识别的贡献。

H1 相对 G1 将 `gate_input=local` 改为 `train_residual`，新增 `gate_residual_eps=1e-12` 和固定均值 buffer，实验编号为 H1。β=0.075、ρ=0.15、η=0.25、全局 g 读出、rank32、适配器和 CP 学习率均沿用 G1。三 seed 共用同一 TRAIN 统计资产，模型可训练参数仍为 **122,880**；新增 buffer 为 12×768 个 float32 数。

## 2. 公式修订：TRAIN 查询均值与残差门控

令 $f^o_{bj}\in\mathbb{R}^{768}$ 为冻结 CLIP 的物体查询视觉特征，j 遍历 12 个查询，b 遍历 22,998 张 TRAIN 图像。均值对全部图像、全部查询计算，不读取图像标签：

```math
\begin{aligned}
\mu_j&=\frac{1}{N_{\mathrm{train}}}\sum_{b\in\mathrm{TRAIN}}f^o_{bj},\\[4pt]
r_{bj}&=f^o_{bj}-\mu_j,\\[4pt]
\widehat r_{bj}&=\frac{r_{bj}}{\max(\lVert r_{bj}\rVert_2,\epsilon)},
\qquad\epsilon=10^{-12}.
\end{aligned}
```

统计代码以 float64 分块累计，保存 float64 均值及 float32 副本；模型 forward 使用固定 float32 buffer `gate_fo_mean`。VAL／TEST 和两个权重选择轨道使用同一 TRAIN 均值。数学上，TRAIN 原始残差均值为零；逐向量单位化后的方向均值由数据决定。

令模式数 R=32，$B_{\mathrm{vis}}\in\mathbb{R}^{R\times768}$，其第 k 行转置为 $b_k$。H1 保留原模式中心化：

```math
\begin{aligned}
z_{bjk}&=b_k^\top\widehat r_{bj}/\tau_\gamma,
\qquad \tau_\gamma=0.1,\\[4pt]
q_{bjk}&=2\sigma(z_{bjk})-1,\\[4pt]
\gamma_{bjk}&=1+\frac{\eta}{2}\left(q_{bjk}-\frac1R\sum_{\ell=1}^Rq_{bj\ell}\right),
\qquad\eta=0.25,\\[4pt]
\frac1R\sum_k\gamma_{bjk}&=1,
\qquad 0.75\le\gamma_{bjk}\le1.25.
\end{aligned}
```

残差为零时，单位化结果为零、所有门控模式均为 1。中心化门控的视觉基梯度为：

```math
\nabla_{b_\ell}\gamma_{bjk}
=\frac{\eta}{\tau_\gamma}\left(\delta_{k\ell}-\frac1R\right)
\sigma(z_{bj\ell})[1-\sigma(z_{bj\ell})]\widehat r_{bj}.
```

## 3. CP 读出、训练目标与评价公式

张量轴为 `[batch, object, attribute]`。冻结文本特征定义 $u_i=A_ue_i^a$、$v_j=A_ve_j^o$，$W_c\in\mathbb{R}^{768\times R}$。CP 仍由全局图像特征 g 读出：

```math
\begin{aligned}
U^{\mathrm{H1}}_{bji}&=U^{\mathrm{base}}_{bji}
+0.15\kappa g_b^\top W_c(u_i\odot v_j\odot\gamma_{bj}),\\[4pt]
s_{bji}&=\log p(o_j\mid x_b)+\log p(a_i\mid o_j,x_b),\\[4pt]
\mathcal{L}_{\mathrm{task}}(0.075)&=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}},\\[4pt]
\mathcal{L}^{\mathrm{H1}}&=\mathcal{L}_{\mathrm{task}}(0.075)
+\frac{0.001}{R^2}\left\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_R\right\rVert_F^2.
\end{aligned}
```

κ=20；pair/support 依照 [08 的损失定义](../08/README.md)，support 是 TRAIN 组合概率质量的负对数。内部 variant=V1，prim／cent 分支为零；AdamW weight decay 独立执行。Wc 零初始化，初始 CP 修正为零。固定均值不进入优化器。

native／exact 两种权重分别取首次最大 VAL CW AUC，并对每种权重统计 TRAIN 门控均值：

```math
\begin{aligned}
t_m^*&=\min\{t\in\{1,\ldots,T\}:A_t^m=\max_{1\le s\le T}A_s^m\},
\qquad m\in\{\mathrm{native},\mathrm{exact}\},\\[4pt]
\bar\gamma_{jk}&=\frac1{N_{\mathrm{train}}}\sum_b\gamma_{bjk},
\qquad \gamma^{\mathrm{mean}}_{bjk}=\bar\gamma_{jk}.
\end{aligned}
```

本轮两个选择轨道均选中 5／8／6 epoch，各自保存权重和导出。动态与均值条件各自选取 VAL CW／OW 的 native bias，再固定用于 TEST；均值干预优化步数为零。

## 4. 具体代码修订与验证

| 文件 | 具体改动 | 作用 |
|---|---|---|
| [gate_features.py](code/vgcpod_v1_roundH/vgcpod/gate_features.py) | `prepare_gate_statistics()` 和 `load_gate_mean()` | 分块统计、复用、身份检查与加载 |
| [model.py](code/vgcpod_v1_roundH/vgcpod/model.py) | 固定 `gate_fo_mean` buffer；`train_residual` 分支 | 残差与 L2 单位化后进入原门控 |
| [engine.py](code/vgcpod_v1_roundH/vgcpod/engine.py) | 构建模型加载均值；`evaluate_run()` 从 checkpoint 恢复 | 训练与离线评价共享同一均值 |
| [mean_gate_eval.py](code/vgcpod_v1_roundH/mean_gate_eval.py) | 从 checkpoint 模型状态传入固定均值 | 对 H1 的两套权重做均值替换 |
| [H1.json](configs/round_h/H1.json)／[运行配置](configs/runtime/H1.json) | 模板与服务器生成配置分开保存 | 运行配置补充均值路径及 SHA256 |
| [round_h.py](code/vgcpod_v1_roundH/round_h.py) | 先统计一次，再三卡训练与六项评估 | H1−G0/G1、双选择及逐组合汇总 |
| [ranking_diagnostics.py](code/vgcpod_v1_roundH/ranking_diagnostics.py) | H1 及 exact 历史对照、dynamic−mean | 保存分组排序量与同 seed 差值 |
| [tmux启动](code/vgcpod_v1_roundH/scripts/start_round_h_tmux.sh)／[H组测试](code/vgcpod_v1_roundH/tests/test_round_h.py) | 会话、日志与退出码；五项新增测试 | 累计 44 项发布测试 |

[相对08的源码补丁](code/source_changes.patch)保留完整差异。[源码身份](results/server_source_identity.json)逐文件核对服务器源字节；均值资产、12 份 checkpoint 和全部预测进入本轮结果目录。

原发布验证完成 44 项测试、三项 CPU 合成训练、六项均值评估及重复启动复用。相同 G1 配置在新旧核心上的模型参数、AdamW 状态和 Torch RNG 逐元素一致；H1 续训与固定 buffer 恢复通过核对。[原验证明细](results/package_validation/H/validation/result.json)、[测试日志](results/package_validation/H/validation/unit_tests.txt)、[仓库复算](analysis/recompute.py)。

## 5. 超参数选取策略与实际矩阵

1. **围绕门控输入做单项表示比较。** H1 延续 G1 的 β、ρ、η、CP 读出与优化设置，变化集中于 TRAIN 中心化和残差单位化。
2. **均值来自固定 TRAIN 缓存。** 三 seed 共用 22,998 张图像、12 个查询的均值；float64 累计、float32 buffer，统计身份随配置与 checkpoint 保存。
3. **保持原门控强度。** η=0.25、τγ=0.1、模式中心化保留，门控范围沿用 [0.75,1.25]；ε=1e-12 定义零范数分支。
4. **保留十轮与三 seed 配对。** seed3351/3352/3353 分别在物理 GPU1/2/3，从各自原始初始化训练，每次 3600 次更新。
5. **分开读取三类比较。** H1−G1 测输入变换，H1−G0 测相对静态模型的训练效果，H1 dynamic−mean 测同一权重对图像变化的依赖。
6. **沿用原选权重规则。** native 为主表，exact 独立归档；各条件使用自己的 VAL native bias。

SEED_TABLE

| 固定项目 | 设置 |
|---|---|
| CLIP／缓存 | 冻结 OpenAI ViT-L/14，224px、768维，复用原缓存 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.1 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，原余弦日程 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |

CP 优化组为 Au/Av/Wc/Bvis，`cp_lr_scale=0.3`。`gate_fo_mean` 是固定 buffer，未增加可训练参数。

## 6. 仓库结构与复现入口

```text
09/
  README.md          # 实验修订、公式、超参策略，末节数据分析
  code/              # H原代码、相对G补丁、原分析源码
  configs/           # H1模板、服务器运行配置、资产、参考与清单
  results/           # 三seed、双选择、六项干预、TRAIN均值、日志及原归档
  analysis/          # 数据分析、图表、保存预测复算与归档核对
```

恢复原运行包，复用服务器现有 CLIP 缓存：

```bash
git lfs pull
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 09 \
  --dest /data/xhw/vgcpod_project/replay_09
conda activate vgcpod_v1
cd /data/xhw/vgcpod_project/replay_09/vgcpod_v1_roundH

export VG_H_OUT=/data/xhw/vgcpod_v1_runs/replay_09_H
export VG_H_SESSION=vgcpod_09_H
bash scripts/start_round_h_tmux.sh \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 --gpus 1 2 3 \
  --a-out /data/xhw/vgcpod_v1_runs/roundA_20261019 \
  --c-out /data/xhw/vgcpod_v1_runs/roundC_20261020 \
  --d-out /data/xhw/vgcpod_v1_runs/roundD_20261020 \
  --e-out /data/xhw/vgcpod_v1_runs/roundE_20261020 \
  --f-out /data/xhw/vgcpod_v1_runs/roundF_20261020 \
  --g-out /data/xhw/vgcpod_v1_runs/roundG_20261020

tmux attach -t vgcpod_09_H
```

脚本先计算并保存共享 TRAIN 均值，再生成输出目录下的 `configs/H1.json`，随后三个 seed 并行训练及评价。按 Ctrl+b、d 脱离 tmux。`launcher.exitcode=0` 且日志出现 `ROUND_H_COMPLETE ... training_jobs 3 mean_gate_jobs 6` 表示整轮完成。外部输入见 [external_inputs.json](external_inputs.json)，命令见 [runbook.json](runbook.json)。

从仓库保存的预测、checkpoint、均值和缓存元数据复算指标与图表：

```bash
python VG-CPOD方案/09/analysis/recompute.py
python tools/reproduce.py verify
```

复算依赖 NumPy、PyTorch、Matplotlib，输出至 `analysis/recomputed/`，原分析保存在 `analysis/confirmed/`。materialize 按映射恢复原包 80 个成员的布局与字节。

## 7. 数据分析与下一轮方向

MEANS_TABLE

- **完整 H1 的 TEST 排序下降。** H1−G1 的 TEST CW／OW AUC 为 −0.902±0.617／−1.186±0.641 点，三个 seed 两项均下降；exact 选权重结果为 −0.708／−0.971 点。
- **图像变化的推理贡献增强。** H1 dynamic−mean 的 TEST CW／OW AUC 为 +2.146±0.486／+1.516±0.630 点，未见条件属性 +1.340 个百分点；该量与独立训练 G0/G1 的总效果分别报告。
- **TRAIN 门控变化几乎全部转向图像维度。** 图像方差占比约 99.9%，图像标准差约为 G1 的 5.8 倍；代表残差 RMS 约 0.17–0.19，单位化对应代表放大倍数约 5.4–5.9。
- **主要下降位于条件属性和未见组合排序。** H1−G1 的 TEST 未见条件属性下降 1.375 个百分点、NLL 增加 0.0545，全部未见候选内真实组合 top-1 下降 1.851 个百分点；物体准确率下降 0.053 个百分点。
- **后续建议 I1/I2 拆分幅值因素。** I1 保留 TRAIN 去均值并去掉残差 L2；I2 保留归一化残差、τγ=0.56，作为 TRAIN 代表尺度对照。两组固定 η=0.25、ρ=0.15、β=0.075，各三个 seed。它们为下一轮方案，本阶段实际结果为已完成 H1。

![同配置静态、原门控与残差门控](analysis/confirmed/figures/01_matched_results.png)

完整逐 seed 差值、动态／均值干预、双选择、残差尺度推导与下一轮计划，见[数据分析 README](analysis/README.md)。
'''
    write('README.md',main.replace('SEED_TABLE',seed_table).replace('MEANS_TABLE',means))
    code=r'''# 09 · 代码与修订

原 H 包源码位于 [vgcpod_v1_roundH](vgcpod_v1_roundH/)。配置、资产、原说明和验证记录按平行结构分区保存，materialize 恢复原始布局。相对 [08 G组](../../08/code/README.md)的完整差异见[源码补丁](source_changes.patch)。

## 1. TRAIN 均值统计与固定 buffer

[gate_features.py](vgcpod_v1_roundH/vgcpod/gate_features.py) 的 `prepare_gate_statistics()` 只读取 TRAIN 的 `fo.npy`，以 float64 分块求和，对每张图像的全部物体查询求均值；`train_object_query_mean.npz` 保存 `mean_fo_float64`、float32 `mean_fo` 与统计身份，JSON 保存样本数、均值范数、查询词表和 SHA256。

`load_gate_mean()` 读取运行配置中的均值路径与哈希。`VGCPOD(..., gate_mean=...)` 将均值注册为 `gate_fo_mean` buffer；该张量不进入优化器，并随 checkpoint 保存。三个 seed 使用同一统计文件，forward 使用 float32。

## 2. 门控输入与 checkpoint 评价

[model.py](vgcpod_v1_roundH/vgcpod/model.py) 新增的核心路径为：

```python
gate_input = F.normalize(
    fo - self.gate_fo_mean[None], dim=-1,
    eps=c.get('gate_residual_eps', 1e-12))
```

随后继续执行原 sigmoid、η=0.25 的模式中心化和全局 g 的 CP 读出。联合概率、objective、训练步、AdamW、学习率日程及裁剪沿用 G1。

[engine.py](vgcpod_v1_roundH/vgcpod/engine.py) 的 `make_model()` 在训练构建时加载均值；`evaluate_run()` 将 checkpoint 中的 `gate_fo_mean` 传回构造器。[mean_gate_eval.py](vgcpod_v1_roundH/mean_gate_eval.py) 同样从权重恢复均值，再统计训练后门控矩并替换成均值条件。原包测试中移除外部均值文件后，native／exact 评价与原导出一致。

## 3. 三 GPU 调度与配对汇总

- [round_h.py](vgcpod_v1_roundH/round_h.py)：`prepare_runtime_config()` 先生成共享均值和运行配置，`launch()` 再开启三个 seed 队列；每队列训练 H1，依次执行 native、exact 均值评估。
- [tmux启动](vgcpod_v1_roundH/scripts/start_round_h_tmux.sh)：固定激活环境 Python，记录主日志、退出码与会话；完成任务复用，未完成训练从最后完整 epoch 继续。
- `summarize()` 分别比较 native H1−G0/G1 和 exact H1−G0/G1；同时保存同配置主表、支持损失日志、逐组合差值与 dynamic−mean。
- [ranking_diagnostics.py](vgcpod_v1_roundH/ranking_diagnostics.py)：扩展 H1 native／exact 的完整 OW 概率诊断，按 all/seen/unseen 分组，记录物体／属性、概率质量与未见候选排序。

| 原运行入口 | 仓库入口 |
|---|---|
| H1 模板 | [configs/round_h/H1.json](../configs/round_h/H1.json) |
| 原服务器运行配置 | [configs/runtime/H1.json](../configs/runtime/H1.json) |
| H前26个历史 native 指标及G0/G1 exact结果 | [pre_h_complete.json](../configs/references/pre_h_complete.json) |
| 源码差异／服务器核对 | [补丁](source_changes.patch) · [源码身份](../results/server_source_identity.json) |
| 发布包原布局恢复 | [package_members.json](../package_members.json) |

## 4. 发布验证与仓库复算

- [44项测试](../results/package_validation/H/validation/unit_tests.txt)、[结构化测试摘要](../results/package_validation/H/validation/unit_test_summary.json)、[三训练六评估流程](../results/package_validation/H/validation/result.json)、[流程日志](../results/package_validation/H/validation/cli_output.txt)。
- 新增五项 H 测试覆盖配置差异、全部 TRAIN 查询均值、显式 CP／门控公式与有限差分、前两步梯度、续训及 checkpoint buffer 恢复；原测试同时保留。
- 原 G1 路径在新旧核心中的参数、AdamW 与 Torch RNG 逐元素一致；重复启动复用三项训练、六项均值评估。
- [仓库复算入口](../analysis/recompute.py)、[原分析源码](original_analysis/)、[归档验证](../analysis/repository_validation.json)。

正式训练环境为 Python3.8.18、PyTorch1.13.1+cu117、NumPy1.24.4、RTX3090。原发布数学与调度验证使用 CPU 合成缓存，正式结果来自服务器。CLIP 许可见 [CLIP_LICENSE](vgcpod_v1_roundH/vendor/CLIP_LICENSE)。
'''
    write('code/README.md',code)
    configs=r'''# 09 · 配置与超参数策略

| 方案 | gate_input | ε | ρ | β | η／τγ | native／exact最佳epoch |
|---|---|---|---|---|---|---|
| [H1](round_h/H1.json) | train_residual | 1e-12 | 0.15 | 0.075 | 0.25／0.1 | 5/8/6；5/8/6 |

轮次按 seed3351/3352/3353 顺序。三次训练各十轮、3600 次更新。完整实际配置与两套指标见 [experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. **保持已有目标权重。** β=0.075，pair 系数为 3/2、support 系数为 9/86；ρ=0.15，CP 全局 g 读出保留。
2. **变化集中于门控表示。** 将 G1 的原查询特征改为 TRAIN 去均值后的归一化残差，η=0.25、τγ=0.1、模式中心化不变。
3. **固定统计来源。** 每个物体查询由全部 22,998 张 TRAIN 图像求均值，三个 seed 共用同一 [12,768] 资产；标签不参与统计。
4. **保持初始化与优化日程。** 三 seed 从原始初始化开始，rank32、adapter rank8、batch64、十轮、CP LR scale0.3、λorth0.001。
5. **分开保存选权重路径。** 主表使用首次最大 VAL CW native AUC；exact 路径独立保存，各条件使用自己的 VAL native bias。
6. **按同 seed 比较。** H1−G1、H1−G0 与 H1 dynamic−mean 分别衡量输入变化、整体训练效果及图像动态的推理贡献。

```math
\mathcal{L}_{\mathrm{task}}(0.075)=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结 OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项／workers | 1 / 0.001 / 4 |

内部 variant=V1，通过 `gate_input=train_residual` 开启门控。CP 参数组为 Au/Av/Wc/Bvis；`gate_fo_mean` 为固定 buffer。prim／cent 分支关闭。

## 模板、运行配置与统计资产

- [H1模板](round_h/H1.json) 是原 ZIP 的配置，保留实验超参数；[runtime/H1.json](runtime/H1.json) 为原服务器实际运行配置，新增均值文件路径及 SHA256。
- [TRAIN均值及身份](../results/raw_export/roundH_20261020/feature_statistics/)与 [原输出运行配置](../results/raw_export/roundH_20261020/configs/H1.json)随结果保存。三 seed checkpoint 记录相同均值和哈希。
- 新的复现运行由调度器在指定输出目录生成自己的均值资产及运行配置。原运行配置保留服务器当时的路径。
- [三项训练](experiment_inventory.json)、[六项均值评估](intervention_inventory.json)、[26份历史参考](references/pre_h_complete.json)、[标签资产](assets/)、[外部输入](../external_inputs.json)。
- 直接对照为 [G1](round_g/G1.json)，静态对照为 [G0](round_g/G0.json)。缓存 manifest SHA256：`bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

下一轮 I1/I2 的输入与温度建议见[数据分析](../analysis/README.md)；本阶段实际训练配置为 H1。
'''
    write('configs/README.md',configs)
    effects=[]
    for seed in (3351,3352,3353):
        r=json.loads((STAGE/'results/raw_export/roundH_20261020/mean_gate/H1'/('seed%d'%seed)/'metrics.json').read_text())
        net=next(v['net_correct'] for v in D['similarity'] if v['comparison']=='H1-H1_mean' and v['seed']==seed and v['phase']=='test' and v['group']=='unseen')
        effects.append([seed,'%+.4f'%r['dynamic_minus_train_mean']['test_cw']['AUC_native'],
                        '%+.4f'%r['dynamic_minus_train_mean']['test_ow']['AUC_native'],'%+d'%net])
    effects=table(['seed','H1动态−均值 TEST CW AUC','TEST OW AUC','未见属性正确数变化'],effects)
    results=r'''# 09 · H组训练结果

本阶段新增 H1 三 seed，共三次训练、30 个模型 epoch。native／exact 两种选择各保存 12 份评价与预测；六项均值替换评估包含 12 个条件、48 份评价与预测。合计 **72 份预测文件、12 份 checkpoint 文件**；TRAIN 查询均值与六套门控矩另行保存。

## native选择主结果

SEED_TABLE

AUC 为服务器原始分数保存的 native AUC，面积×100。每次训练完成十轮、3600 次更新，物体／GT-object 条件属性、CW／OW、seen／unseen 以及 NLL 的完整指标见各运行 JSON。

## exact选择结果

EXACT_TABLE

H1 的两条轨道均选择 5／8／6 epoch，对应模型参数与预测一致；best.pt 与 best_native.pt 为相同字节，best_exact.pt 作为独立 checkpoint 文件保留。last.pt 保存完整模型、优化器与随机状态。

## 原运行、TRAIN均值与汇总

- [H1三seed运行](raw_export/roundH_20261020/H1/ut-zap50k/V1/)包含 protocol、history、metrics、曲线、预测、四份 checkpoint 及 exact_selection。
- [共享TRAIN均值](raw_export/roundH_20261020/feature_statistics/)、[实际运行配置](raw_export/roundH_20261020/configs/H1.json)：均值形状 [12,768]，统计 22,998 张 TRAIN 图像，float64 累计、float32 模型 buffer。
- [三卡状态](raw_export/roundH_20261020/round_h_status.json)、[任务表](raw_export/roundH_20261020/round_h_jobs.json)、[主日志](raw_export/roundH_20261020/launcher.log)、[退出码0](raw_export/roundH_20261020/launcher.exitcode)。
- [主结果](raw_export/roundH_20261020/round_h_summary.json)、[均值与SD](raw_export/roundH_20261020/round_h_mean_sd.csv)、[配对差值](raw_export/roundH_20261020/round_h_paired_deltas.csv)、[配对汇总](raw_export/roundH_20261020/round_h_paired_mean_sd.csv)。
- [同ρ／β主表](raw_export/roundH_20261020/matched_native_summary.csv)、[支持损失](raw_export/roundH_20261020/support_training_summary.csv)、[exact指标](raw_export/roundH_20261020/exact_selection_summary.csv)、[exact历史配对](raw_export/roundH_20261020/exact_reference_paired_mean_sd.csv)。
- [排序诊断](raw_export/roundH_20261020/ranking_diagnostics.json)、[诊断差值](raw_export/roundH_20261020/ranking_diagnostic_paired_mean_sd.csv)、[逐组合概率](raw_export/roundH_20261020/pair_probability_diagnostics.csv)、[逐组合增量](raw_export/roundH_20261020/per_pair_gains.csv)。

## H1动态与TRAIN均值替换

EFFECTS_TABLE

[native均值评估](raw_export/roundH_20261020/mean_gate/)和[exact均值评估](raw_export/roundH_20261020/mean_gate_exact/)分别保存 TRAIN 门控均值 [12,32]、图像方差及 dynamic/train_mean 两个条件的四种评价、预测与曲线。每项统计全部 22,998 张 TRAIN 图像，优化步数为零；条件各用自己的 VAL bias。

## 原交付、来源与分析

- [原服务器TAR](archives/vgcpod_roundH_results_20261020.tar.gz)、[原H训练ZIP](archives/vgcpod_v1_roundH_20261020.zip)，旁附 SHA256；二进制沿用 Git LFS。
- [原发布验证](package_validation/H/)、[原包说明字节](package_support/H/)、[原实施与分析报告字节](package_support/original_reports/)。
- [原包 MEMBER_COUNT 个成员映射](../package_members.json)、[H结果成员与哈希](raw_export_members.json)、[服务器源码核对](server_source_identity.json)、[历史引用](reused_reference_members.json)、[阶段SHA256清单](../file_manifest.json)。
- 原 TAR 的 C/D/E/F/G 历史结果与源码映射到 [04](../../04/results/README.md)、[05](../../05/results/README.md)、[06](../../06/results/README.md)、[07](../../07/results/README.md)、[08](../../08/results/README.md)；A5/A6 引用 [02](../../02/results/README.md)。
- [仓库复算与归档验证](../analysis/repository_validation.json)、[完整数据分析](../analysis/README.md)。

H1−G1 的 TEST CW／OW AUC 为 −0.902／−1.186 点，H1 dynamic−mean 为 +2.146／+1.516 点；完整训练效果与同一权重的推理干预分别分析。I1/I2 为后续建议，本阶段归档为已完成 H1。
'''
    write('results/README.md',results.replace('SEED_TABLE',seed_table).replace('EXACT_TABLE',exact_table).replace('EFFECTS_TABLE',effects).replace('MEMBER_COUNT',str(members)))
    analysis=(WORK/'VG-CPOD_v1_H组结果分析与下一轮方向_20261021.md').read_text(encoding='utf-8')
    analysis=analysis.replace('# VG-CPOD v1：H 组结果分析与下一轮方向','# 09 · 数据分析与下一轮方向',1)
    analysis=analysis.replace('analysis_roundH_results_20261021/figures/','confirmed/figures/')
    analysis=re.sub(r'^\$\$\n(.*?)\n\$\$',lambda m:'```math\n'+m[1]+'\n```',analysis,flags=re.M|re.S)
    analysis=analysis.replace('`analysis_roundH_results_20261021/`','[confirmed/](confirmed/)')
    analysis+='''
## 8. 仓库复算与归档核对

[主说明](../README.md)记录公式、代码修订与超参策略；[原分析数据](confirmed/analysis.json)、[原审计](confirmed/audit.json)、[仓库验证](repository_validation.json)、[源码补丁](../code/source_changes.patch)按平行结构保存。

```bash
python VG-CPOD方案/09/analysis/recompute.py
python tools/reproduce.py verify
```

复算读取本轮 H1、前序 D/E/F/G、缓存元数据及固定均值，验证 156 份预测、198 行排序诊断和 checkpoint 均值副本，输出至 `recomputed/`。原始字节映射与配置路径适配由复算入口完成。
'''
    write('analysis/README.md',analysis)
    path=REPO/'VG-CPOD方案/README.md';text=path.read_text(encoding='utf-8')
    row='| 09 | '+TITLE+'（H 组）；H1三seed、共享TRAIN均值、双准则选择与六项均值评估 | [修订说明](09/README.md) · [代码](09/code/README.md) · [配置](09/configs/README.md) · [结果](09/results/README.md) · [数据分析](09/analysis/README.md) |'
    if '| 09 |' not in text:text=text.rstrip()+'\n'+row+'\n'
    path.write_text(text,encoding='utf-8',newline='\n')
    path=REPO/'README.md';text=path.read_text(encoding='utf-8')
    latest='最新归档（2026-10-21）：[VG-CPOD 09 · '+TITLE+'](VG-CPOD方案/09/README.md)，包含H1三个seed、共享TRAIN查询均值、归一化残差视觉门控、native／exact双准则与六项均值评估，共72份预测、12份checkpoint文件；附公式与代码修订、超参数策略及末节[数据分析](VG-CPOD方案/09/analysis/README.md)。前序[08 G组](VG-CPOD方案/08/README.md)、[07 F组](VG-CPOD方案/07/README.md)、[06 E组](VG-CPOD方案/06/README.md)、[05 D组](VG-CPOD方案/05/README.md)、[04 C组](VG-CPOD方案/04/README.md)、[03 B组](VG-CPOD方案/03/README.md)、[02 A组](VG-CPOD方案/02/README.md)与[01原正式结果](VG-CPOD方案/v1/results/ut-zap50k-3seeds-20261018/README.md)继续保留。'
    text=re.sub(r'^最新归档（.*$',latest,text,flags=re.M)
    text=text.replace('| VG-CPOD | 01–08 |','| VG-CPOD | 01–09 |').replace('02–08','02–09')
    text=text.replace('08 恢复原 G 包，包含静态与动态配对队列及排序诊断。','08 恢复原 G 包，包含静态与动态配对队列及排序诊断；09 恢复原 H 包，包含共享 TRAIN 均值统计、归一化残差门控及三 seed 调度。')
    path.write_text(text,encoding='utf-8',newline='\r\n')
    print('Wrote 09 five READMEs and both indexes')


if __name__=='__main__':main()
