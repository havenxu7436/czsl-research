# EKR-CPF v2：冻结名称底座与共享值对齐的完整实践方案

日期：2026-10-07。主实验：UT-Zappos，seed 0，每个新增分支 3 轮。本文对应同目录内实际实现的 `core.py`、`pilot.py` 和 `run.sh`。

## 1. 最终选择与理由

**下一步优先做：冻结已有 A_unit + N_name，把描述作为一个新的语义评分增量，保留原始槽向量，并学习所有属性共享的低秩 value 对齐。** 先验证增量是否超过名称与错配对照，再决定是否有必要继续做可靠性训练和更强对象条件。

这一选择是依据当前实验作出的工程与研究判断，不保证收益，也不等于已经具有论文级创新。

| 已有证据 | 对新实现的要求 |
|---|---|
| N_name CW AUC 43.123195%，强于 N_clean 的 42.866431% | 保留已有效的名称模型；所有新对照共享同一个底座 |
| 正确映射被替换后 CW/OW AUC 平均下降 0.170232/0.113058 pp | 描述值得做一次独立增量验证；不能据此宣布语义贡献成立 |
| clean 在 TRAIN、VAL 都改善 margin，但两边都弱于 name | 优先检查 value 表达能力与名称信息保护；不直接增加轮数 |
| 原 clean 值为各槽减去四槽均值，name 值为名称/2 | 新增分支不再随槽均值改写名称向量；主对照完全同构 |
| 原分支空槽几乎没有实际作用，对象上下文影响很小 | 第一轮保留共有接口，暂不新增拒识损失或更复杂对象模块 |

我对前次“描述减名称”建议作了进一步修正：如果将该差向量作为唯一 value，并把名称对照也直接代入同一公式，名称减名称会变成零向量，对照无法学习。这种实现不能充分排除新增容量收益。当前采用原始语义值，名称对照具有非零输出和可训练梯度。

选择冻结已训练 N_name，是为了检验能否在当前较强底座上增加描述贡献。它额外使用了先前 3 轮名称训练，所有新分支都共享这份训练成本；不能把与旧 A_unit 的差值当作相同总训练预算下的收益。

## 2. 与原三条路径、随机语义图的关系

- 复用路径 1 已审核的名称/AE 字典与 GloVe bank，不再改描述文本或重建词向量。
- 路径 2 的固定 K 校准继续关闭，`lambda=0`，不为新结果再次扫描 K 强度。
- 当前属于 EKR 的条件特征驱动评分增量验证：使用冻结 CPF 属性特征查询文本槽，输出候选组合分数修正。没有解冻图像主干，也不能写成已经完成了全部联合特征学习路线。
- 固定错配字典只作为归因对照；没有随机图、图边、GNN 或图采样。USG 随机语义图仍是独立路线。
- 原阶段 B、原语义 gate 均保持 `false`。这是独立的 v2 协议，不执行旧 `run_confirm.sh`，不覆盖旧代码或原实验结果。

## 3. 精确模型定义

### 3.1 共同名称底座

每个新分支均使用相同 seed 0 的 A_unit 与已选出的 N_name 第 3 轮权重：

$$
B(x,a,o)=s_{A\_unit}(x,a,o)+\Delta_{N\_name}(x,a,o).
$$

两者所有参数均冻结，并保持 `eval()`，包括 BatchNorm 等运行状态。新分支读取原 `attr_feature`，不向基础模型或名称分支传递梯度。

这里的“名称保护”指底座参数和分数生成方式不被新字典改写；新增残差仍可能改变最终预测，所以仍需要性能保护条件。

### 3.2 固定文本槽

每个属性使用原 bank 中四个 300 维单位向量：名称 + 三条 AE 描述。记为 $e_{a,r}$，其中 $r=0,1,2,3$。对象上下文仍是对象名称单位向量 $n_o$。

向量不再减去四槽均值，不因更换 AE 而重新计算名称槽的中心化向量。名称对照把自身名称重复四次；错配对照保留自身名称，AE 使用已冻结的原 shuffle 映射。

### 3.3 共享低秩 value 对齐

令 $D\in\mathbb{R}^{32\times300}$、$U\in\mathbb{R}^{300\times32}$：

$$
\widetilde e_{a,r}=e_{a,r}+UD e_{a,r},
$$

$$
v_{a,r}=\frac{1}{2}\operatorname{norm}(\widetilde e_{a,r}).
$$

`value_down=D` 用 PyTorch Linear 默认随机初始化，`value_up=U` 初始化为零。因此初始 value 精确采用原始语义向量的方向，随后允许共享映射适配 CPF 属性特征空间。**不能把 U、D 同时初始化为零，否则这一乘积的两边都缺乏有效起始梯度。**

投影共享于全部属性和槽；不引入属性独立的投影、自由组合参数表或对象专属 value 表。value 的范数固定为 0.5，防止靠无界放大向量获得表面收益。

### 3.4 条件查询与共有空槽

冻结属性特征归一化后为 $u(x)$。沿用 rank 32 的条件查询结构：

$$
q(x,o)=\operatorname{norm}\left(W_xu(x)\odot\sigma(W_on_o)\right),
$$

$$
k_{a,r}=\operatorname{norm}(W_ke_{a,r}).
$$

四个文本槽的 logit 为 $q^\top k_{a,r}/T$，空槽 logit 为 $q^\top\operatorname{norm}(k_{\varnothing})/T+b_{\varnothing}$；$T=0.2$。五项 softmax 得到权重 $p_{\varnothing},p_0,p_1,p_2,p_3$。

空槽 value 为零。第一轮它只是共有门控接口，**没有另加坏知识拒识目标，也不将其权重解释为可靠性概率**。保留它也使重复名称对照的图像/对象查询仍能经由门控学习，避免所有查询参数成为无效占位。

### 3.5 分数增量与零初始化

$$
\Delta_{v2}(x,a,o)=\eta\sum_{r=0}^{3}p_r(x,a,o)\,u(x)^\top v_{a,r},
$$

$$
\eta=0.1\tanh(g),\qquad g_{init}=0,
$$

$$
s_{v2}(x,a,o)=B(x,a,o)+\Delta_{v2}(x,a,o).
$$

新增分支初始输出精确为零；关闭分支也精确返回共同名称底座。因为权重和不超过 1、value 范数为 0.5，新增单个候选残差绝对值不超过 0.05。

增量系数允许负值，这是分数修正的设计选择，负号本身不表示训练失败。零初始化的首步通常只有 `raw_gain` 有非零梯度；随后查询/value_up 开始学习，再随后 value_down 获得梯度。已用 CPU 检查验证这个梯度顺序。

## 4. 四项实际训练与一个共同基线

| 项目 | 新增分支输入 | 共享 value 对齐 | 新增可训练参数 | 角色 |
|---|---|---|---:|---|
| B_name | 新增分支关闭 | 无新增训练 | 0 | 共同冻结基线，CW/OW 只回放一次 |
| V_name_align | 四槽重复自身名称 | 可训练 | 48,034 | 排除额外适配容量收益 |
| V_clean_align | 名称 + 正确 AE | 可训练 | 48,034 | 预先固定的主候选 |
| V_shuffle_align | 名称 + 固定错配 AE | 可训练 | 48,034 | 正确描述归因对照 |
| V_clean_fixed | 名称 + 正确 AE | U、D 冻结于初始化 | 28,834 | 检验 value 对齐是否必要 |

三个主对照注册容量、可训练参数数目、初始参数指纹相同，只替换固定 bank 输入。`V_clean_fixed` 是刻意减少可训练能力的模块消融，不冒充等可训练参数归因对照；其注册参数仍为 48,034。

参数计数：三项 300→32 的查询/key Linear 共 28,800；空槽 key 32、空槽 bias 1、gain 1；value 低秩两项 Linear 共 19,200。

主候选始终是 `V_clean_align`。不能看到四项结果后，把最好的一项改名为预设主方法。`V_clean_fixed` 若表现更好，要如实记录；若考虑换成更简单模型，另立后续协议。

## 5. 训练、选择和检查

### 5.1 固定预算

| 参数 | 固定值 |
|---|---|
| 数据与 split | 原 UT-Zappos TRAIN；只评价 VAL CW/OW |
| seed | 0；四项都从头训练各自新增分支 |
| 轮数 | 每项完整 3 轮，共 12 个提交完成的训练 epoch |
| batch size | 64 |
| optimizer | AdamW |
| learning rate / weight decay | 0.001 / 0.0001 |
| 分类温度 | 0.05 |
| rank / value rank | 32 / 32 |
| 查询温度 / 最大 gain | 0.2 / 0.1 |
| 数据增强 / workers | 复用原 TRAIN dataset 与原配置 workers；各项同一 seed 和 loader generator |
| 优化范围 | 仅新增 router；A_unit 与原 N_name 冻结 |

损失仅使用原组合监督：

$$
\mathcal L=\operatorname{CE}\left(s_{v2}(x)/0.05,y_{pair}\right).
$$

不新增 KL、伪标签、人工样本筛选或拒识损失。本轮检验 shared value 对齐与描述的增量作用，避免多个新损失同时改变归因。

### 5.2 选择策略

先回放共同名称底座，必须复现旧 N_name：六个核心原生指标最大差异 ≤0.05 pp，真实标签及原始 argmax 完全一致。

每轮训练后评价 CW AUC。最佳 checkpoint 是 **epoch 0、1、2、3 中第一个严格最大值**；epoch 0 就是新增 gain=0 的共同底座。若三轮都没有超过底座，选择 epoch 0，结果仍如实保存。最后只对这个选中 checkpoint 评价 CW 和 OW，不按 OW 重新扫描训练轮次。

保留每次评价的 float32 全量 `scores.dat`、候选顺序、图像 ID、真实标签和指标。标准流程包括 2 次共同基线回放、12 次逐轮 CW、8 次最终 CW/OW，共 22 次原生评价。

### 5.3 新 pilot 门槛

预先固定 `V_clean_align` 的 CW AUC 同时满足：

```text
clean_align - B_name          >= +0.2 pp
clean_align - V_name_align    >= +0.2 pp
clean_align - V_shuffle_align >= +0.2 pp
```

相对 B_name 和 V_name_align，都需满足：CW best-HM 不低于 −0.5 pp、OW AUC 不低于 −0.5 pp、OW best-unseen 不低于 −2 pp。

全部成立才得到新 pilot 的 `proceed=true`。这是一项单 seed 的运营筛选，不是统计证明，不改变任何旧 gate。对齐消融另记录 `clean_align - clean_fixed >= +0.1 pp`；如果该项不成立，即使性能筛选通过，也不能声称 value 对齐的增量贡献已得到支持。

### 5.4 本地已验证与服务器待验证

已通过 12 项 CPU 检查：同容量/同初始化、零增量、非零名称对照与查询梯度、候选排序和对象分块不变性、原始名称槽不随 AE 输入变化、bank 隔离、分阶段梯度、残差范数界、冻结底座、恢复 RNG/optimizer 后同一步结果一致、归因门槛和保护条件。

另用实际 `pilot.py` 控制器做了合成 CPU 流程检查：在第 2 轮评价前注入中断，从第 1 个已提交 epoch 恢复；保留失败评价目录，完成四项训练、配对汇总与无权重 ZIP 指纹校验。

Python 3.8 语法、Bash 脚本和文档命令会随发布检查。**本地没有跑服务器实际权重的 GPU 训练，不把合成测试当作性能结果。** 服务器 `prepare` 会重新核对旧源码、字典、权重、诊断证据及实际共同基线回放，失败即停止。

## 6. 文件与位置

| 内容 | 路径 |
|---|---|
| 本地发布 ZIP | `C:/Users/llenovo/Desktop/UT_description_review/ekr_v2_anchor_alignment_20261007.zip` |
| 服务器 ZIP | `/tmp/ekr_v2_anchor_alignment_20261007.zip` |
| 解压脚本目录 | `/tmp/ekr_v2_anchor_alignment_20261007/` |
| 只读原名称实验 | `/var/tmp/czsl_xhw_name_ekr_20261007/` |
| 已完成机制诊断 | `/var/tmp/czsl_xhw_name_ekr_diag_20261007_v1/` |
| 新 v2 实验输出 | `/var/tmp/czsl_xhw_ekr_v2_align_20261007/` |
| 新审阅 ZIP | `/var/tmp/czsl_xhw_ekr_v2_align_20261007_review.zip` |

固定使用 `/data/xhw/miniconda3/envs/cpf/bin/python`；终端显示 `(base)` 也可以运行。实际要求仍是已工作的 CPF PyTorch 1.13.1+cu117 与 GPU。新盘需至少 5 GiB 空余。没有在已满 `/data` 新建训练输出。

预计成本比上一轮三项、每项三轮略高：这里四项，仍以图像主干计算为主要开销，按上一轮速度可能约 1–2 小时；以实际日志为准，不把估算作为运行时承诺。

## 7. 从上传到训练的具体命令

### 7.1 本地 Windows PowerShell 上传

```powershell
scp "C:/Users/llenovo/Desktop/UT_description_review/ekr_v2_anchor_alignment_20261007.zip" xhw@211.87.232.122:/tmp/ekr_v2_anchor_alignment_20261007.zip
```

### 7.2 服务器 SSH 解压

首次解压执行下面整个代码块。已经解压则跳过这一段，直接校验；不覆盖既有 bundle。

```bash
/data/xhw/miniconda3/envs/cpf/bin/python - <<'PY'
from pathlib import Path, PurePosixPath
import stat
import zipfile

archive = Path('/tmp/ekr_v2_anchor_alignment_20261007.zip')
target = Path('/tmp/ekr_v2_anchor_alignment_20261007')
if target.exists():
    raise SystemExit('BUNDLE EXISTS; VERIFY THE EXISTING BUNDLE')
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:
        raise SystemExit('ZIP CRC FAILED')
    for item in z.infolist():
        p = PurePosixPath(item.filename)
        if p.is_absolute() or '..' in p.parts or '\\' in item.filename:
            raise SystemExit('UNSAFE ZIP PATH')
        if not p.parts or p.parts[0] != target.name:
            raise SystemExit('WRONG ZIP ROOT')
        if stat.S_ISLNK(item.external_attr >> 16):
            raise SystemExit('ZIP LINKS ARE NOT ALLOWED')
    z.extractall(target.parent)
print('UNPACKED', target)
PY
```

### 7.3 校验发布包

两行是一条命令，第一行末尾的 `\` 必须保留，后面不要加空格。

```bash
/data/xhw/miniconda3/envs/cpf/bin/python \
  /tmp/ekr_v2_anchor_alignment_20261007/verify_release.py
```

正常输出为 `V2 RELEASE CHECKSUM PASSED` 和文件数。

### 7.4 新建 tmux 会话

```bash
tmux new -s ut_ekr_v2_align
```

当前若已在 tmux 中，不必嵌套新建，直接执行下一步。

### 7.5 准备并回放共同名称底座

```bash
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh prepare
```

这一阶段运行 CPU 检查、校验旧输入、加载实际权重、回放 CW/OW；不进行正式训练。看到下面这一行后继续：

```text
V2 PREPARED; OLD GATE REMAINS FALSE
```

### 7.6 跑四项 pilot 并汇总

```bash
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh train
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh report
```

四项顺序是 name_align、clean_align、shuffle_align、clean_fixed。正式训练结束后输出 `V2 PILOT DECISION` 和 `DOWNLOAD V2 REVIEW ZIP`。**先看新 decision，不运行旧阶段 C。**

全新输出目录也支持一次执行全部阶段；与上面的分步方法二选一，不要已经 prepare 后再次使用 `all`：

```bash
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh all
```

detach 用 `Ctrl+B` 后按 `D`；重新连接：

```bash
tmux attach -t ut_ekr_v2_align
```

### 7.7 阅读结果

```bash
cat /var/tmp/czsl_xhw_ekr_v2_align_20261007/reports/decision.json
cat /var/tmp/czsl_xhw_ekr_v2_align_20261007/reports/results.csv
```

结果路径包括：

```text
protocol.json
prepare_checks.json
baseline/val_cw/  baseline/val_ow/
runs/V_name_align/
runs/V_clean_align/
runs/V_shuffle_align/
runs/V_clean_fixed/
reports/decision.json  reports/results.csv  reports/integrity.json
review_manifest.json
```

### 7.8 下载到本地 Windows PowerShell

```powershell
scp "xhw@211.87.232.122:/var/tmp/czsl_xhw_ekr_v2_align_20261007_review.zip" "C:/Users/llenovo/Desktop/UT_description_review/"
```

审阅 ZIP 不含权重；保留服务器所有 `checkpoints/*.pth`。如实际 SSH 使用另外的端口，scp 沿用同一端口的 `-P`。

## 8. 中断、磁盘与重复运行

训练中断时保留所有原文件，使用：

```bash
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh resume
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh report
```

恢复位置是最后一个完整、提交成功的 epoch 边界，不是中断 batch。保存新增分支、optimizer、Python/NumPy/Torch/CUDA RNG、loader generator 和最佳 checkpoint 指针。未完成的 epoch 会重新跑；这部分额外计算和失败日志保留，不假装从未发生。

已完成的项目只有在指纹验证通过后才跳过。失败的评价目录保留，重新评价使用新的 attempt 目录；不会把旧不完整目录当作成功结果。若缺少已提交状态、状态/权重指纹不符或准备阶段未完成，停止，不强行续训。

如果需要全新输出版本，先在 tmux 中设置一次变量，再按 prepare/train/report 三步运行：

```bash
export V2_OUT=/var/tmp/czsl_xhw_ekr_v2_align_20261007_v2
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh prepare
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh train
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh report
```

这会创建新的实验版本，不删除旧版本。对应审阅 ZIP 会是 `${V2_OUT}_review.zip`，下载时使用实际输出路径。

如果只在最后 ZIP 打包时失败，而 report 已完成，可在空间恢复后仅重新打包：

```bash
bash /tmp/ekr_v2_anchor_alignment_20261007/run.sh pack
```

已有完整 ZIP 时使用已有 ZIP，不重复覆盖。原实验、机制诊断、失败尝试和负结果均保留；汇总依据当前 v2 的 `complete.json` 指针，不扫描旧 v1 的指标。

## 9. pilot 后如何作决定

| 观察 | 下一步 |
|---|---|
| clean_align 不能超过名称/错配对照 | 停止这次预定预算；不增加轮次，不放宽门槛 |
| clean_align 超过两个对照，所有保护条件通过 | 冻结新的多 seed 确认协议；准备 3351/2026 各自同规则名称底座后配对确认 |
| 上述通过，但 clean_align 不优于 clean_fixed | 描述增量可能成立；不能声称对齐模块贡献成立，后续须明确是否采用简化方法 |
| clean_align 的 gain 最终接近零，或选中 epoch 0 | 如实报告新增分支未获得可采纳增量，不把关闭分支当作方法成功 |
| gate 通过但对象错配效应仍小 | 可以报告描述增量，不报告已经学会强对象条件选择 |

确认阶段仍需报告每个 seed 的配对差、均值与离散程度；三 seed 和运营门槛不构成充分的统计证明。seed 0 的名称权重不能复制冒充其余两个 seed 的名称底座。

当前只使用一份冻结的 shuffle 映射。多训练 seed 不能替代多份输入映射；如果 pilot 通过，确认协议还应预先固定至少两份额外无固定点的 AE 映射，并完整报告结果。额外映射应在新分支的输入副本上构建，保留自身名称，不修改原 bank 文件；这部分尚未纳入本轮四项 pilot。

只有描述增量在确认阶段稳定后，才另立可靠性版本：用 TRAIN 内的输入干预验证坏知识下能否回退到共同底座，分别检查正确输入损失与错配输入损失。不能仅因文本 receiver/donor 不同就假定每条描述都语义错误，也不能把一个输入错配检测器直接称为视觉可靠性判断。

更强对象交互放在后面，先用同图同属性跨对象残差、去对象重训和上下文错配区分真实条件贡献。跨 MIT-States/C-GQA 的推广只在 UT 的方法与超参数冻结后进行；不使用 test 调方法。

## 10. 创新性与相关方法的边界

共享低秩对齐、门控、残差连接本身不足以作为完整创新。Troika 已研究分支视觉文本对齐与 Cross-Modal Traction；CAMS 已采用门控 cross-attention；Retrieval-Augmented Primitive Representations 则从训练图像数据库检索属性/对象表示。[Troika](https://arxiv.org/abs/2303.15230)、[CAMS](https://arxiv.org/abs/2511.16378)、[Retrieval-Augmented Primitive Representations](https://ojs.aaai.org/index.php/AAAI/article/view/28043)。

本方案当前的价值是：针对 CPF/GloVe 上已观察到的描述分支弱势，建立一个保持名称底座、具有同构归因对照、能够零增量回退的实证验证。论文主张要依赖最终数据：描述是否超越名称与错配对照；value 对齐是否必要；对象条件和拒识是否各有可复现增量。本文只作相关方法边界核查，不声称完成全面新颖性检索。

建议给导师的表述：

> 旧 EKR pilot 未通过。机制诊断发现正确描述映射有相对作用，但描述分支弱于名称对照，空槽与对象条件尚未充分发挥功能。下一步冻结已有名称模型，以原始语义槽和共享低秩 value 对齐检验描述增量，使用等参数名称/错配对照及固定值消融，先限制在 UT、seed 0、每项 3 轮。通过后再确认多 seed，保留所有负结果。
