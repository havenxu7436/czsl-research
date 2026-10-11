# 下一步实践路线：冻结名称锚点的 EKR 小规模探索

日期：2026-10-07。适用范围：UT-Zappos、原 CPF/GloVe、现有 UT v2 资产与权重。

这份路线及配套脚本用于检验一个新假设：**冻结名称模型后，图像和候选对象条件下选择属性描述，能否产生优于同参数量名称适配及错配知识的增量。**

原静态文本归因 gate=false 保持不变。这是另立协议的有限探索，不是原正式 EKR 门槛已经通过。当前 VAL 已用于诊断和提出假设，后续 VAL 结果属于开发证据。配套代码不提供 test 入口。

## 1. 今天先做什么

按顺序完成“上传 → 准备 → 18 次冻结分数导出 → 检查诊断”。这一阶段不训练。

随后只运行 seed 0 的三项分支试验，每项 3 轮。它们通过新探索筛选后，才补 seed 3351、2026。主试验总预算是 **9 次分支训练 × 3 轮**；已有 12 项原生训练不重跑，基础模型不解冻。

这条线继续利用现有数据划分、GloVe、v2 描述字典、训练权重和评估代码。主要改变是冻结锚点改为 A_unit，并将所有路由的对象条件统一为对象名称。

## 2. 为什么选择这个调整

已完成的 v2 结果显示：C_text 相对 A_unit 的平均 CW AUC +3.067 pp，但相对固定 C_shuffle 只有 +0.379 pp，原 gate 未通过。OW、bias=0 时，Boat Shoes 有明确局部退化，Flats 有改善。这些结果支持检查“知识是否应条件使用”，没有证明条件选择一定有效。

原 EKR 是 `C_text + residual`，关闭分支仍是 C_text。新探索改成 `A_unit + residual`，使关闭分支真正回到名称参考。A_unit 并非各指标上的最强模型；选择它是为了明确隔离新增描述知识。

本轮仅研究 **AE 属性描述选择**。OE 描述继续保留在原资产中，但新路由的对象条件只使用对象名称向量；不会一边用对象描述、一边把收益全部归因于属性描述。描述错配是负对照，与 USG 的随机图、边采样、传播路线分开。

## 3. 实验组与复用关系

| 标识 | 本轮作用 | 是否新训练 |
| --- | --- | --- |
| A_unit / C_text / C_shuffle | 既有三 seed、10 轮权重，冻结后导完整分数 | 否 |
| N0 | 同 seed 的冻结 A_unit，分支强制关闭 | 否，只评价 |
| N_name | 相同路由结构及参数量，只读属性名称，四个重复名称槽 | 是，3 轮 |
| N_clean | 图像 × 候选对象名称条件化，选择原 v2 正确属性描述残差，带 null | 是，3 轮 |
| N_shuffle | 与 N_clean 相同，仅将属性描述换成冻结 donor 的错配描述；名称不置换 | 是，3 轮 |
| N_noobj | N_clean 去掉对象条件 | 仅主试验确认后，3 轮 |
| N_nonull | N_clean 去掉 null 通道 | 仅主试验确认后，3 轮 |

N_name/N_clean/N_shuffle 的注册与可训练参数都是 **28,834**。它们共享同 seed 的锚点、相同新模块初始化种子、训练顺序种子、批次、优化器和评价规则。消融保留对应层的参数声明，但被移除的连接不参与前向；“登记参数相同”不意味着每项消融的有效计算相同。

N_name 不使用 `name - description_mean`，因为那仍包含描述信息；其四个槽都是原属性名称，值为名称向量的一半。它控制增加可训练分支的效果。四个重复槽是预算匹配手段，不声称代表四种名称语义。

## 4. 模型、公式与训练约定

令 $e_{a,0}$ 为属性名称，$e_{a,1},e_{a,2},e_{a,3}$ 为冻结的三个描述单位向量。保留原 EKR 的中心化残差：

$$
\bar e_a=\frac{1}{4}\sum_{j=0}^{3}e_{a,j}
$$

$$
r_{a,j}=e_{a,j}-\bar e_a
$$

令 $u_x$ 为冻结 A_unit 的归一化属性视觉特征，$e_o^{name}$ 为候选对象的单位名称向量：

$$
q_{x,o}=\operatorname{norm}\left(W_xu_x\odot\sigma(W_oe_o^{name})\right)
$$

$$
k_{a,j}=\operatorname{norm}(W_kr_{a,j})
$$

注意力在四个知识槽和一个可学习 null 槽上共同 softmax，温度 $T=0.2$。null 不贡献知识值：

$$
\Delta(x,a,o)=\sum_{j=0}^{3}\pi_{x,o,a,j}\left\langle u_x,\frac{r_{a,j}}{2}\right\rangle
$$

$$
\eta=0.1\tanh(\gamma)
$$

$$
s(x,a,o)=s_{A\_unit}(x,a,o)+\eta\Delta(x,a,o)
$$

初始 $\gamma=0$，所以初始输出与 A_unit 相同。知识值有界，新增每个组合分数的绝对幅度不超过 0.1。强制关闭时新增分数为零。**幅度有界不是准确率不退化的保证；null 也需要训练，不能预先宣称已经学会拒绝坏知识。**候选对象不是图像的真实对象，不使用 VAL 真标签决定门控。

纯文本版：`score = A_unit_score + 0.1*tanh(gain)*selected_attribute_residual`。

仅对 TRAIN 的 83 个候选组合优化交叉熵：

$$
\mathcal L_{train}=\operatorname{CE}(s/0.05,y)
$$

基础模型冻结且处于 eval 模式。训练参数固定：batch=64，AdamW，lr=0.001，weight_decay=0.0001，rank=32，epochs=3，workers 沿用原配置。暂不加入受扰 KL 正则、解冻、静态 K、CLIP 或新损失。

每个分支在自身 3 轮中选择 **VAL CW AUC 的第一个严格最大值**；OW 不用于选择轮次。不同分支可以选择不同的最佳轮次，这属于相同的选模规则。三轮里无收益时不临时追加轮数。

## 5. 部署：Windows 上传，Linux 执行

本地交付包含 `name_anchor_ekr_20261007.zip`。ZIP 只有脚本、冻结参考记录和说明，不包含图像、GloVe 或模型权重。服务器必须保留原 UT v2 的 `/var/tmp/czsl_xhw_ut_v2_20261006`，特别是三个 seed 的 A_unit/C_text/C_shuffle 权重。

在 **Windows PowerShell** 执行：

```powershell
scp "C:\Users\llenovo\Desktop\UT_description_review\name_anchor_ekr_20261007.zip" xhw@211.87.232.122:/tmp/name_anchor_ekr_20261007.zip
```

上传地址沿用此前的服务器；若你实际 SSH 用了其他地址或端口，scp 应使用相同地址和端口。

在 **Linux 服务器**执行以下代码块。用 CPF 的绝对 Python 路径，不依赖当前提示符是 `(base)` 还是 `(cpf)`，不需要管理员权限：

```bash
CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
"$CPF_PY" - <<'PY'
from pathlib import Path
import zipfile
archive=Path('/tmp/name_anchor_ekr_20261007.zip')
parent=Path('/tmp').resolve()
destination=parent/'name_anchor_ekr_20261007'
if destination.exists():
    raise FileExistsError('Bundle already exists; verify it or use a new version')
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:
        raise ValueError('ZIP CRC failed')
    for member in z.infolist():
        target=(parent/member.filename).resolve()
        if destination not in target.parents and target != destination:
            raise ValueError('Unexpected archive path')
    z.extractall(parent)
print('UNPACKED',destination)
PY
"$CPF_PY" /tmp/name_anchor_ekr_20261007/verify_release.py
```

首次解压才执行 Python 解压段；已解压后直接校验即可。文件采用 LF 换行，SHA256SUMS 使用 ASCII，避免此前校验文件尾部 `\r` 的问题。不要用 Windows 编辑器另存脚本后覆盖 ZIP 中的版本。

## 6. 阶段 A：准备与冻结分数诊断

```bash
bash /tmp/name_anchor_ekr_20261007/prepare.sh
bash /tmp/name_anchor_ekr_20261007/replay_scores.sh
```

默认新工作目录是 `/var/tmp/czsl_xhw_name_ekr_20261007`，继续使用此前有空间的系统盘。脚本要求至少 5 GiB 空余。不会在已满的 `/data` 写入新训练结果；原数据和权重只读。

准备阶段校验原始源码、12 组配置中的相关配置、原 gate、已下载审阅包的 24 次评价记录、bank/cache manifest，以及要复用的九个权重。源码复制到独立目录；原 gate 检查和旧训练代码保留。任何来源不一致会停止，不能直接关闭检查继续。

诊断共 18 次推理：3 个输入组 × 3 个 seed × CW/OW。每次导出 float32 `scores.dat`，总原始分数约 34 MiB。你原报告中的 18 次评价合计约 233 秒，单次约 10–18 秒；新运行另有模型加载及校验开销，实际时间以日志为准。可以直接运行；网络不稳时放 tmux。

输出：

```text
/var/tmp/czsl_xhw_name_ekr_20261007/diagnostics/full_scores/
  replay_audit.json
  replay_metric_deltas.csv
  fixed_bias_grid.csv
  object_groups_common_bias.csv
  semantic_ut-zappos_.../cw/ 或 ow/
    scores.dat  metrics.json  labels.npz  candidates.json  image_ids.json
```

复现检查：权重、有效配置、bank 的身份一致；VAL 图像顺序、候选顺序、gold 与旧输出一致；分数有限、字节数正确、raw argmax 一致；六项原生指标与旧记录的最大差不超过 **0.05 pp**。这个容差用于识别推理数值差异，不是降低科研收益门槛。超出时停止并复核，不训练。

诊断使用固定 bias 网格 [-1,1]、步长 0.025，以及 bias=0 和同 seed 的 A_unit OW 最佳 HM 偏置。各模型在相同偏置点比较 Boat Shoes 等对象组。网格仅帮助理解，**不替代原生 AUC**；A_unit 的偏置点来自既有 VAL，仅作开发诊断。

查看：

```bash
cat /var/tmp/czsl_xhw_name_ekr_20261007/diagnostics/full_scores/replay_audit.json
head -n 8 /var/tmp/czsl_xhw_name_ekr_20261007/diagnostics/full_scores/replay_metric_deltas.csv
```

读完这阶段重点回答：零点损失是否随共同偏置变化？对象错误能否在改善 seen/unseen 取舍后仍观察到？这些结果用于解释新假设，不用于按 VAL 改写句子、挑对象排除或重选旧 checkpoint。即使局部损失主要随偏置改变，也不等于正确语义已得到归因证明。

## 7. 阶段 B：只跑 seed 0 的三项 pilot

```bash
tmux new -s name_ekr_pilot
bash /tmp/name_anchor_ekr_20261007/run_pilot.sh
```

用 `Ctrl+B` 后按 `D` 离开，重新查看用 `tmux attach -t name_ekr_pilot`。服务器没有 tmux 时可直接运行；不要因缺少 tmux 安装管理员组件。

顺序是 N_name → N_clean → N_shuffle。每项训练前自动用两张 TRAIN 图做 GPU smoke：初始分数等于真实 A_unit；三次更新后分支梯度有限、基础权重不变；关闭分支仍等于 A_unit。smoke 权重不会带入正式训练，正式训练重新初始化。

**预先固定的 pilot 条件**：N_clean 的 CW AUC 相对 N0、N_name、N_shuffle 均至少 +0.2 pp；相对 N0、N_name 的 CW HM 不低于 -0.5 pp、OW AUC 不低于 -0.5 pp、OW best_unseen 不低于 -2 pp。这是计算预算筛选，不是统计显著性。

查看决定：

```bash
cat /var/tmp/czsl_xhw_name_ekr_20261007/reports/pilot/decision.json
```

`proceed=false`：结束本轮预算，保留三项原始候选结果及负结果，不继续补 seed，不临时调学习率、幅度或轮数。它表示该小预算配置没有通过筛选，不能据此断言所有 EKR 都不可能成功。

`proceed=true`：补齐下一阶段。单 seed 成功不能写成方法已成立。

新分支训练时间尚未在服务器测得。每个 run 的 `train_complete.json` 有包含逐轮 CW 评价的 `train_seconds`，`metrics.json` 有最终评价时长；用首项实测估计其余预算，不把既有原生 CPF 耗时直接套过来。

## 8. 阶段 C：相同三轮预算，补另两个 seed

```bash
bash /tmp/name_anchor_ekr_20261007/run_confirm.sh
```

脚本只新增 seed 3351、2026 的六项训练；保留 seed 0，三 seed 全部按相同预算汇总。

**预先固定的确认条件**：N_clean 相对 N0、N_name、N_shuffle 的三 seed 平均 CW AUC 均至少 +0.5 pp，且各比较至少 2/3 seed 为正。CW HM/OW AUC/OW best_unseen 的保护条件与 pilot 相同，并对均值和每个 seed 同时检查，防止均值隐藏大幅退化。

```bash
cat /var/tmp/czsl_xhw_name_ekr_20261007/reports/confirm/decision.json
head -n 15 /var/tmp/czsl_xhw_name_ekr_20261007/reports/confirm/paired_deltas.csv
```

这里的 +0.2/+0.5 pp 是本次新探索的预先筛选阈值，不是根据已看到的新 EKR 结果拟定，也没有修改旧静态文本 gate。报告给出每 seed 配对差；三个 seed 不能自动宣称统计显著。

若只超过 N0、不超过 N_name，收益可能由一般适配解释；若不超过 N_shuffle，正确描述归属仍缺乏支持。若门槛未通过，停止当前配置，不展开三数据集训练。

## 9. 阶段 D：确认通过后再做机制与受扰诊断

两项消融各三个 seed，共 6 次训练、18 个分支 epoch：

```bash
bash /tmp/name_anchor_ekr_20261007/run_ablations.sh
```

报告在 `reports/mechanisms/paired_deltas.csv`。看 N_clean 相对 N_noobj/N_nonull 的配对差是否支持对象条件与 null 的作用。若去掉模块也一样好，应采用更简单版本并缩小机制主张。

随后可在固定已训练权重上做错配诊断，不新增训练：

```bash
bash /tmp/name_anchor_ekr_20261007/run_noise.sh
```

N_name/N_clean、三个训练 seed、名义 10%/30% 属性错配、噪声 seeds 11/22/33、CW/OW，共 72 次额外 VAL 推理。16 个属性中实际错配分别为 2/16=12.5%、5/16=31.25%，报告必须用实际比例。改动只进入新增分支，基础名称模型和 bank 文件保持原样。

N_name 应不受描述错配影响；这是隔离检查。N_clean 若干净有效、受扰损失更大，说明知识可靠性问题仍存在。null 未改善受扰行为时不能声称“已经具备可靠拒绝”。本阶段不自动加入 R6 的受扰 KL 训练，避免同时改变方法和损失。

`reports/noise/raw_evaluations.csv` 保存原值，`per_model_noise_means.csv` 先在同一训练模型内平均三个噪声 seed。不能把 3 个训练 seed × 3 个噪声 seed 当成 9 次独立训练。

## 10. 后续扩大范围的条件

只有主确认和机制检查支持该方法，才考虑第二数据集。优先 MIT-States 的小规模匹配验证，之后再考虑昂贵的 C-GQA。其他数据集必须有自己的同 seed A_unit 锚点及匹配知识负对照，不能把 UT 的归因证明移过去。

当前工具包只实现 UT，不提供尚未核实的 MIT/C-GQA“一键命令”。不要直接修改 `--datasets` 或用其他数据集旧 C 权重代替名称锚点。

正式论文前还需核对最近邻方法、固定描述置换的敏感性，并在方案与超参数冻结后使用独立的最终评价证据。现有 VAL 已被研究过程多次使用，不能包装成盲测。已有一次固定 shuffle 加三个训练 seed，仍不等于三个独立 shuffle 重复。

## 11. 日志、结果保存与下载

新 SSH 窗口可执行：

```bash
source /tmp/name_anchor_ekr_20261007/env.sh
nvidia-smi
tail -n 30 "$NAME_WORK/experiments/name_ekr/runs/name_ut-zappos_N_clean_s0_e3/train.log"
cat "$NAME_WORK/experiments/name_ekr/runs/name_ut-zappos_N_clean_s0_e3/status.json"
```

训练日志会输出 `epoch/batch/loss/eta/grad_params`；`EVALUATE val ...` 是评价。`train_complete.json` 表示训练完成，`status=train_and_val_complete` 才是最终 CW/OW 及锚点评价均完成。

阶段 A 后即可导出诊断；完成 pilot 或确认后也可导出，输出使用新时间戳，不覆盖：

```bash
source /tmp/name_anchor_ekr_20261007/env.sh
cd "$NAME_WORK/src"
OUT="$NAME_WORK/name_ekr_review_$(date +%Y%m%d_%H%M%S).zip"
"$CPF_PY" -m transition.name_plan pack --work "$NAME_WORK" --out "$OUT"
printf 'DOWNLOAD THIS FILE: %s\n' "$OUT"
```

在 Windows PowerShell 用日志显示的确切 ZIP 名替换下面的文件名：

```powershell
scp xhw@211.87.232.122:/var/tmp/czsl_xhw_name_ekr_20261007/name_ekr_review_YYYYMMDD_HHMMSS.zip "C:\Users\llenovo\Desktop\UT_description_review\"
```

导出包含完整诊断分数、配置、日志、指标和 hash，不含 `.pth/.pt` 权重。服务器原权重和新分支权重都保留。

报告同时保留候选模型的原始指标和关闭残差的参考。不能只报告退回名称模型后的非负结果，再声称新方法没有退化。

## 12. 中断与停止规则

- 推理完成的项会核对复用；若单项留下半个输出目录，会拒绝覆盖。确认没有同项进程在写后，将报错的**那一个目录**改名留存再重试，不删除原实验。
- 分支训练完成、最终评价中断：重新运行同一阶段脚本会使用 finish-only 补评价，不重训；若某项评价残留半个目录，同样只留存该评价目录。
- 分支训练本身中断：当前实现没有 optimizer/RNG 恢复，不能从某个 batch 接着跑。保留部分 run，先排查空间/进程，再建立新的完整匹配探索版本；不要把半程权重当完成结果。新版本仍复用原 A_unit。
- 代码、字典、权重、原 gate 或冻结配置改变：工具停止。应说明变更并另立版本，不修改冻结记录来“通过校验”。

最小建议投入到阶段 B。只有 pilot 通过才补另外两个 seed；确认通过才做消融和受扰诊断。此路线没有保证收益，也不要求因为已经投入时间就继续增加预算。

## 13. 当前交付验证范围

配套脚本已在本地做 CPU 检查：名称对照独立于 AE/OE 描述、对象查询不泄露 OE 描述、正确知识分支能响应描述改变、参数量、零增量、强制关闭、幅度界、候选重排/对象分块、初始增益更新后的梯度。安装器在真实归档源码上验证，并保留原源码与旧 gate 检查；Python 源码按 3.8 语法解析。

本地环境是 Python 3.12 / torch 2.5.1 CPU；**未在服务器执行 GPU smoke、18 次新推理或任何新 EKR 训练**。这些由上述服务器命令完成。准备阶段检查 Python 3.8 / torch 1.13.1 / torchvision 0.14.1 / CUDA，训练前执行真实名称锚点的 GPU smoke。不要把本地工具验证写成 EKR 实验成功。
