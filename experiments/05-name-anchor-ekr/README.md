# 05 · 名称锚点 EKR

状态：已训练：seed0、三组各3轮。

## 修改与结论

将零增量回退锚点改为 A_unit，避免关闭动态分支仍保留静态描述混淆。

正确描述未胜名称和错配；先诊断机制，再决定是否扩跑。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 05 --dest /srv/czsl-reproduce-05
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-05
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

执行包：[name_anchor_ekr_20261007](<code/name_anchor_ekr_20261007>)。

运行时入口：`/srv/czsl-reproduce-05/name_anchor_ekr_20261007/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../04-semantic-v2/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

原始结果：[final_metrics.csv](<results/final_metrics.csv>)。

<details>
<summary>模型定义、对照与协议边界</summary>

### 冻结名称锚点的 EKR 协议

日期：2026-10-07。适用范围：UT-Zappos、原 CPF/GloVe、现有 UT v2 资产与权重。

这份路线及配套脚本用于检验一个新假设：**冻结名称模型后，图像和候选对象条件下选择属性描述，能否产生优于同参数量名称适配及错配知识的增量。**

原静态文本归因 gate=false 保持不变。这是另立协议的有限探索，不是原正式 EKR 门槛已经通过。当前 VAL 已用于诊断和提出假设，后续 VAL 结果属于开发证据。配套代码不提供 test 入口。

#### 执行范围

先准备并进行 18 次冻结分数回放，再检查诊断；回放本身不训练。

随后只运行 seed 0 的三项分支试验，每项 3 轮。它们通过新探索筛选后，才补 seed 3351、2026。主试验总预算是 **9 次分支训练 × 3 轮**；已有 12 项原生训练不重跑，基础模型不解冻。

这条线继续利用现有数据划分、GloVe、v2 描述字典、训练权重和评估代码。主要改变是冻结锚点改为 A_unit，并将所有路由的对象条件统一为对象名称。

#### 为什么选择这个调整

已完成的 v2 结果显示：C_text 相对 A_unit 的平均 CW AUC +3.067 pp，但相对固定 C_shuffle 只有 +0.379 pp，原 gate 未通过。OW、bias=0 时，Boat Shoes 有明确局部退化，Flats 有改善。这些结果支持检查“知识是否应条件使用”，没有证明条件选择一定有效。

原 EKR 是 `C_text + residual`，关闭分支仍是 C_text。新探索改成 `A_unit + residual`，使关闭分支真正回到名称参考。A_unit 并非各指标上的最强模型；选择它是为了明确隔离新增描述知识。

本轮仅研究 **AE 属性描述选择**。OE 描述继续保留在原资产中，但新路由的对象条件只使用对象名称向量；不会一边用对象描述、一边把收益全部归因于属性描述。描述错配是负对照，与 USG 的随机图、边采样、传播路线分开。

#### 实验组与复用关系

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

#### 模型、公式与训练约定

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

#### 阶段 A：准备与冻结分数诊断

```bash
bash ${REPLAY}/name_anchor_ekr_20261007/prepare.sh
bash ${REPLAY}/name_anchor_ekr_20261007/replay_scores.sh
```

需要前序 v2 的 A_unit/C_text/C_shuffle 权重，按协议核对九份权重身份；输入只读。

准备阶段校验原始源码、12 组配置中的相关配置、原 gate、归档的 24 次评价记录、bank/cache manifest，以及要复用的九个权重。源码复制到独立目录；原 gate 检查和旧训练代码保留。任何来源不一致会停止，不能直接关闭检查继续。

诊断共 18 次推理：3 个输入组 × 3 个 seed × CW/OW。每次导出 float32 `scores.dat`，总原始分数约 34 MiB。实际运行耗时以日志为准。

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

#### 阶段 B：只跑 seed 0 的三项 pilot

```bash
bash ${REPLAY}/name_anchor_ekr_20261007/run_pilot.sh
```

顺序是 N_name → N_clean → N_shuffle。每项训练前自动用两张 TRAIN 图做 GPU smoke：初始分数等于真实 A_unit；三次更新后分支梯度有限、基础权重不变；关闭分支仍等于 A_unit。smoke 权重不会带入正式训练，正式训练重新初始化。

**预先固定的 pilot 条件**：N_clean 的 CW AUC 相对 N0、N_name、N_shuffle 均至少 +0.2 pp；相对 N0、N_name 的 CW HM 不低于 -0.5 pp、OW AUC 不低于 -0.5 pp、OW best_unseen 不低于 -2 pp。这是计算预算筛选，不是统计显著性。

查看决定：

```bash
cat /var/tmp/czsl_xhw_name_ekr_20261007/reports/pilot/decision.json
```

`proceed=false`：结束本轮预算，保留三项原始候选结果及负结果，不继续补 seed，不临时调学习率、幅度或轮数。它表示该小预算配置没有通过筛选，不能据此断言所有 EKR 都不可能成功。

`proceed=true`：补齐下一阶段。单 seed 成功不能写成方法已成立。

#### 阶段 C：相同三轮预算，补另两个 seed

```bash
bash ${REPLAY}/name_anchor_ekr_20261007/run_confirm.sh
```

脚本只新增 seed 3351、2026 的六项训练；保留 seed 0，三 seed 全部按相同预算汇总。

**预先固定的确认条件**：N_clean 相对 N0、N_name、N_shuffle 的三 seed 平均 CW AUC 均至少 +0.5 pp，且各比较至少 2/3 seed 为正。CW HM/OW AUC/OW best_unseen 的保护条件与 pilot 相同，并对均值和每个 seed 同时检查，防止均值隐藏大幅退化。

```bash
cat /var/tmp/czsl_xhw_name_ekr_20261007/reports/confirm/decision.json
head -n 15 /var/tmp/czsl_xhw_name_ekr_20261007/reports/confirm/paired_deltas.csv
```

这里的 +0.2/+0.5 pp 是本次新探索的预先筛选阈值，不是根据已看到的新 EKR 结果拟定，也没有修改旧静态文本 gate。报告给出每 seed 配对差；三个 seed 不能自动宣称统计显著。

若只超过 N0、不超过 N_name，收益可能由一般适配解释；若不超过 N_shuffle，正确描述归属仍缺乏支持。若门槛未通过，停止当前配置，不展开三数据集训练。

#### 阶段 D：确认通过后再做机制与受扰诊断

两项消融各三个 seed，共 6 次训练、18 个分支 epoch：

```bash
bash ${REPLAY}/name_anchor_ekr_20261007/run_ablations.sh
```

报告在 `reports/mechanisms/paired_deltas.csv`。看 N_clean 相对 N_noobj/N_nonull 的配对差是否支持对象条件与 null 的作用。若去掉模块也一样好，应采用更简单版本并缩小机制主张。

随后可在固定已训练权重上做错配诊断，不新增训练：

```bash
bash ${REPLAY}/name_anchor_ekr_20261007/run_noise.sh
```

N_name/N_clean、三个训练 seed、名义 10%/30% 属性错配、噪声 seeds 11/22/33、CW/OW，共 72 次额外 VAL 推理。16 个属性中实际错配分别为 2/16=12.5%、5/16=31.25%，报告必须用实际比例。改动只进入新增分支，基础名称模型和 bank 文件保持原样。

N_name 应不受描述错配影响；这是隔离检查。N_clean 若干净有效、受扰损失更大，说明知识可靠性问题仍存在。null 未改善受扰行为时不能声称“已经具备可靠拒绝”。本阶段不自动加入 R6 的受扰 KL 训练，避免同时改变方法和损失。

`reports/noise/raw_evaluations.csv` 保存原值，`per_model_noise_means.csv` 先在同一训练模型内平均三个噪声 seed。不能把 3 个训练 seed × 3 个噪声 seed 当成 9 次独立训练。

#### 后续扩大范围的条件

只有主确认和机制检查支持该方法，才考虑第二数据集。优先 MIT-States 的小规模匹配验证，之后再考虑昂贵的 C-GQA。其他数据集必须有自己的同 seed A_unit 锚点及匹配知识负对照，不能把 UT 的归因证明移过去。

当前工具包只实现 UT，不提供尚未核实的 MIT/C-GQA“一键命令”。不要直接修改 `--datasets` 或用其他数据集旧 C 权重代替名称锚点。

正式论文前还需核对最近邻方法、固定描述置换的敏感性，并在方案与超参数冻结后使用独立的最终评价证据。现有 VAL 已被研究过程多次使用，不能包装成盲测。已有一次固定 shuffle 加三个训练 seed，仍不等于三个独立 shuffle 重复。

#### 中断与停止规则

- 推理完成的项会核对复用；若单项留下半个输出目录，会拒绝覆盖。确认没有同项进程在写后，将报错的**那一个目录**改名留存再重试，不删除原实验。
- 分支训练完成、最终评价中断：重新运行同一阶段脚本会使用 finish-only 补评价，不重训；若某项评价残留半个目录，同样只留存该评价目录。
- 分支训练本身中断：当前实现没有 optimizer/RNG 恢复，不能从某个 batch 接着跑。保留部分 run，先排查空间/进程，再建立新的完整匹配探索版本；不要把半程权重当完成结果。新版本仍复用原 A_unit。
- 代码、字典、权重、原 gate 或冻结配置改变：工具停止。应说明变更并另立版本，不修改冻结记录来“通过校验”。

最小建议投入到阶段 B。只有 pilot 通过才补另外两个 seed；确认通过才做消融和受扰诊断。此路线没有保证收益，也不要求因为已经投入时间就继续增加预算。

</details>
