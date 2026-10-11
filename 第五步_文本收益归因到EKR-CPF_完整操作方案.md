# 第五步：从文本收益归因过渡到 EKR-CPF，再验证创新机制

版本：2026-10-05，`ekr0601_transition_v1`。依据《Week5许瀚文.md》及导师“先走第一条路线”的回复制定。导师确认的是先复核文本贡献、再尝试条件知识增量；没有要求同时开展随机语义图。

**执行主线：补名称单位向量和描述打乱对照 → 三个配对 seed 确认文本贡献 → 冻结 C，训练 EKR 小分支 → 强对照与机制消融 → 跨数据集验证 → 有依据时增加知识受扰退回训练 → 冻结协议后测试。**

本方案附带 [ekr_cpf_transition_20261005_v1.zip](ekr_cpf_transition_20261005_v1.zip)，其中的命令对应实际提供的 Python 工具，不需要你自行猜测或编写 `train_ekr.py`。这是待服务器验证的研究原型，尚无 EKR 性能结果，也没有预先可靠涨点保证。

**路径修订（2026-10-05）：**原训练器的权重保存在两层checkpoint目录中。此前文档漏写了中间层；请额外上传 [checkpoint路径修复脚本](ekr_checkpoint_path_fix_v1/normalize_legacy_checkpoints.py)，按第2.4节先核对hash并建立统一路径，再沿原步骤继续。原工具包和旧权重不需要替换。

## 0. 从哪里接着做

### 0.1 已完成的部分不重做

周报确认三个数据集基线、UT 的 A/C/D/E、MIT/C-GQA 的 A/C 均已完成。昨天方案中任何“另外两个数据集尚未跑 A/C”的表述，以本周完整归档为准。

| 数据集 | 已有 seed | A：val CW AUC% | C：val CW AUC% | C−A CW/OW AUC（pp） | 当前解释 |
| --- | --- | --- | --- | --- | --- |
| UT-Zappos | 3351 | 39.470 | 44.262 | +4.791 / +6.587 | 明显的单 seed 改善，需分离范数和内容 |
| MIT-States | 0 | 13.707 | 13.392 | −0.315 / −0.010 | 混合/近持平，不能声称收益 |
| C-GQA | 3351 | 8.689 | 8.742 | +0.053 / +0.251 | 幅度小，需要配对重复 |

UT 的 Dpost/Epost 最终均选 `lambda=0`。本路线固定 `kcc_calib=False, kcc_lambda=0`，不继续扩大固定 K 校准网格。历史 test 与这里的 validation 分开记录。三个已有 A 权重与历史基线相同，复用只算一个 seed，不能增加独立重复次数。

另一个必须保留的事实：UT 的 OW 原始 `bias=0` top-1 仅净纠正4张图像，val unseen 净减少3张。AUC/HM 改善涉及扫描偏置后的整条曲线，不能写成“所有未见图像的原始预测均改善”。

### 0.2 新旧目录及名称

| 名称 | 服务器路径/含义 |
| --- | --- |
| ROOT | `/data/xhw/czsl_repro` |
| SRC | `$ROOT/src/KCC0601_v1`，已有源代码 |
| LEGACY | `$ROOT/experiments/kcc0601_glove_mvp_v1`，本周已有结果 |
| KNOWLEDGE | `$ROOT/knowledge/kcc0601_v1/trusted_v1`，实际使用的知识源 |
| OLD_CACHE | `$ROOT/knowledge/kcc0601_v1/cache_trusted_v1` |
| PROJ | `$ROOT/src/EKR0601_transition_v1`，新代码副本 |
| EXP | `$ROOT/experiments/ekr0601_transition_v1`，新实验 |
| BANK_ROOT | `$ROOT/knowledge/ekr0601_transition_v1/description_bank_v1` |
| CPF_PY | `/data/xhw/miniconda3/envs/cpf/bin/python` |

训练 ID `ut-zappos` 对应数据/知识文件夹 `ut-zap50k`；`mit-states`、`cgqa` 两者一致。始终使用上面的绝对路径，避免当前目录或 SSH 重连后变量为空造成错误。

已有 run ID：`mvp_<数据集>_<A或C>_legacy_s<seed>_l0`。原始最佳权重在 `checkpoints/<config_name>_<train_name>/best_auc.pth`；UT C实际是 `checkpoints/ut-zappos-czsl_mvp_ut-zappos_C_legacy_s3351_l0/best_auc.pth`。第2.4节修复脚本读取原日志并核对CW/OW权重hash后，才建立后文使用的 `checkpoints/best_auc.pth` 相对符号链接，不移动或覆盖权重。UT C 的完整 SHA256 应为：

```text
7a87ed1e0bcd8440950efeee2ced2d3a7f3c8c603ffcdf0f073a5cb0029e42ae
```

## 1. 先把假设和公式定下来

### 1.1 第一阶段：原语尺度与文本语义分开

令原基线冻结的名称均值向量为 $v_p\in\mathbb R^{300}$。别名、分词、GloVe 词表和 OOV 规则使用旧代码，不能重新换词表。

$$
w_p^{A_{raw}}=v_p,\qquad
w_p^{A_{unit}}=\frac{v_p}{\max(\|v_p\|_2,\epsilon)}.
$$

每个原语有自己的名称单位向量及3条描述的单位向量，记为 $t_{p1},\ldots,t_{p4}$，第一槽为名称。描述先按旧 tokenizer 取 GloVe token 均值，再做 L2；全 OOV/近零描述按旧规则退回自己的名称。

$$
\mu_p=\frac14\sum_{r=1}^{4}t_{pr},\qquad
w_p^C=\operatorname{norm}(\mu_p).
$$

`C_text` 直接读取旧 `W_a_K.pt/W_o_K.pt`，不因重构产生新数值版本。新 bank 先重算上式并检查最大误差不超过 $10^{-6}$，再保留旧张量的原值。

`C_shuffle` 保留接收者的名称，把整组3条描述用固定无不动点置换分配到别的同类原语。属性和对象分别置换，种子固定1729和1730，所有训练 seed 使用同一置换。保持全局条数、文本集合、维度与 OOV 流程；不要在每个 epoch 重新随机打乱。

| 对照 | 改动 | 能回答什么 |
| --- | --- | --- |
| A_raw | 原始名称向量 | 本周 A 的延续 |
| A_unit | 仅将相同名称向量单位化 | 输入尺度改变能解释多少 C−A |
| C_text | 正确归属的名称+描述聚合 | 相对 A_unit 的额外内容贡献 |
| C_shuffle | 名称保留，描述归属固定打乱 | 内容归属是否优于随机代码/附加向量 |

主要配对差为 $C-A_{unit}$ 和 $C-C_{shuffle}$。$C-A_{raw}$ 是总输入效果，不能单独证明知识语义作用。描述打乱是负对照，不是随机语义图。

### 1.2 第二阶段：只选择 C 平均表示之外的差异

对属性描述库中心化，不把差异向量重新 L2 单位化：

$$
d_{ar}=t_{ar}-\mu_a,\qquad \sum_{r=1}^{R_a}d_{ar}=0,
\qquad d_{a0}=0.
$$

真实库目前 $R_a=4$，包含名称；$r=0$ 是新增空选项。区分名称槽与空槽。一般变长库用有效槽 mask，均值只除以有效槽数量，padding 不参加 softmax。

保持旧 C 的三个 raw cosine 分数相加：

$$
s_C=s_{comp}+s_{attr}+s_{obj}.
$$

取旧 `val_forward` 中 `attr_emb = self.attrc_embedder(fused_vis)` 的300维输出，定义 $u(x)=\operatorname{norm}(attr\_emb)$。原模型其他计算顺序不变。

候选对象条件使用该候选的旧 C 对象向量 $w_o^C$，不是图像的真值对象：

$$
q(x,o)=\operatorname{norm}\left[(Q_xu(x))\odot\sigma(Q_ow_o^C)\right],
\qquad Q_x,Q_o,P_k\in\mathbb R^{32\times300}.
$$

$$
k_{ar}=\operatorname{norm}(P_kd_{ar}),\quad
\ell_r=q^\top k_{ar}/T_r,\quad
\ell_0=q^\top\operatorname{norm}(k_0)/T_r+b_0,
\quad \alpha=\operatorname{softmax}(\ell_0,\ell_1,\ldots).
$$

固定 $T_r=0.2$，学习空 key $k_0$ 和偏置 $b_0$；$b_0$ 初值为0。这里的注意力权重不等于知识正确概率，空权重也不是校准过的不确定性。

$$
e_{ar}(x)=\frac{u(x)^\top d_{ar}}2,\quad e_{a0}=0,
\qquad
s_{EKR}=s_C+\eta\sum_{r=1}^{R_a}\alpha_r(x,o,a)e_{ar}(x),
\quad\eta=0.1\tanh(g),\ g_{init}=0.
$$

描述单位向量范数不超过1，故 $\|d\|\le2$、$|e|\le1$、$|s_{EKR}-s_C|\le0.1$。除以2固定用于与普通 raw-cosine 对照匹配最坏扰动范围。$\eta$ 可以为负，表示有界分数修正；它不是固定 K 的 lambda。要记录它的符号，不能只按“更关注某描述”解释负幅度。

三个验收恒等：初始化 $g=0$、显式关闭分支均严格恢复 C；均匀选择真实中心化描述时增量为零；仅剩名称一个有效槽时中心化差异全为零。空选项软权重本身不保证每个受扰样本精确恢复 C。

### 1.3 新分支损失与原 CPF 损失分开

冻结 C 参数且使 C 保持 `eval()`，训练图像仍使用原训练增强。候选只用 train seen pairs $\mathcal P_{tr}$：

$$
L_{task}=CE\left(\{s_{EKR}(x,a,o)/\tau\}_{(a,o)\in\mathcal P_{tr}}, y_{pair}\right),
\qquad\tau=0.05.
$$

这是新增联合 CE，不是原 CPF 的组合/对象/属性三支损失。第一阶段 A_raw/A_unit/C_text/C_shuffle 继续用原 CPF 三支损失；第二阶段 R0/R3/R5 等全部用同一新联合 CE，保证对照公平。推理比较 raw 分数；新增残差只加一次，不额外再除温度。

固定配置：分支 rank32、AdamW `lr=1e-3, weight_decay=1e-4`，批次64、workers4，对象分块16；初筛3轮，正式10轮，无分支学习率网格。C 的训练状态、参数及 BN buffer 全冻结，工具核对训练前后完整 state hash。

### 1.4 代码实际改动位置

下面都是安装后新PROJ中的文件；旧KCC源码保留。需要修改机制时在新协议版本中改对应文件，不把函数直接粘进旧模型的训练/评价混合路径。

| 文件/入口 | 实现内容 |
| --- | --- |
| `train_semantic.py` | 从旧train.py生成，只替换模型import，原损失/训练流程保留 |
| `models/semantic_control.py` | 原语向量查询处切换raw/unit/correct/shuffle，不改其他层 |
| `models/ekr_base.py::raw_forward` | 从原KCC val_forward前缀生成，暴露原始pred和attr_emb |
| `transition/bank.py::construct/build` | 单条描述、中心化、固定置换、旧C一致性与hash |
| `transition/router.py::Router.forward` | 图像/对象query、keys、空选项、values、分块及有界幅度 |
| `transition/runtime.py::BranchModel` | C冻结状态、原语输入、分支加载与临时知识错配 |
| `transition/run.py::branch` | train seen pair联合CE、R6的KL、CW选轮与OW保护 |
| `transition/evaluate.py::evaluate` | 原evaluator_ge评价、CW/OW候选、指标与逐图保存 |
| `transition/summary.py`、`noise_summary.py` | 配对统计、语义门槛、受扰统计分表 |

模型每次前向都计算各候选对象的条件，不使用真值对象作路由；噪声修改只在context中临时替换新增分支buffer，原C和磁盘bank不变。安装器拒绝源码结构不同的自动接入，完整模型接入还需第4.2/6.2节GPU验证。

## 2. 部署工具包、恢复环境和运行变量

### 2.1 本地 PowerShell 上传

只在你自己的 Windows PowerShell 执行此段。服务器账号/IP 来自此前 SSH 截图；若学校入口已改变，修改 `$Server`。不用管理员权限。

```powershell
$Package = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\ekr_cpf_transition_20261005_v1.zip'
$Server = 'xhw@211.87.232.122'
Get-FileHash -LiteralPath $Package -Algorithm SHA256
ssh $Server 'mkdir -p /data/xhw/czsl_repro/tools'
scp $Package "${Server}:/data/xhw/czsl_repro/tools/"
$Repair = 'C:\Users\llenovo\Documents\ChatGPT\CZSL\ekr_checkpoint_path_fix_v1\normalize_legacy_checkpoints.py'
scp $Repair "${Server}:/data/xhw/czsl_repro/tools/normalize_legacy_checkpoints.py"
```

### 2.2 服务器解包、安装到新副本

以下代码块都是服务器 Bash 命令，每段完整粘贴，等待回到提示符后再粘下一段。括号中的 `set -euo pipefail` 只影响子 shell，失败后不会主动关闭你的交互 SSH。

```bash
(
  set -euo pipefail
  ROOT=/data/xhw/czsl_repro
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  cd "$ROOT/tools"
  sha256sum ekr_cpf_transition_20261005_v1.zip
  unzip -n ekr_cpf_transition_20261005_v1.zip
  cd ekr_cpf_transition_20261005_v1
  sha256sum -c SHA256SUMS.txt
  "$CPF_PY" install.py \
    --source "$ROOT/src/KCC0601_v1" \
    --destination "$ROOT/src/EKR0601_transition_v1"
)
```

压缩包 hash 应与本地输出相同。预期显示各文件 `OK` 及 `INSTALLED .../EKR0601_transition_v1`。安装器先检查9个关键源码的 SHA256；源代码与 Week5 快照不同会停止，不能通过删除 hash 检查来继续。它复制旧项目，在新副本添加模块，旧目录不修改。

若提示 `Destination already exists`，先检查它是否已经成功安装；成功则跳过安装。若源代码不一致，先保存差异并对照本周 `meta/source_snapshot.tar.gz`，在新版本目录处理，不覆盖旧实验源码。

### 2.3 一次性保存变量

```bash
(
  set -euo pipefail
  EXP=/data/xhw/czsl_repro/experiments/ekr0601_transition_v1
  mkdir -p "$EXP/protocol" "$EXP/audit" "$EXP/reports" "$EXP/launch_logs"
  if [ ! -e "$EXP/protocol/session_env.sh" ]; then
    cat > "$EXP/protocol/session_env.sh" <<'SH'
export ROOT=/data/xhw/czsl_repro
export SRC="$ROOT/src/KCC0601_v1"
export LEGACY="$ROOT/experiments/kcc0601_glove_mvp_v1"
export PROJ="$ROOT/src/EKR0601_transition_v1"
export EXP="$ROOT/experiments/ekr0601_transition_v1"
export KNOWLEDGE="$ROOT/knowledge/kcc0601_v1/trusted_v1"
export OLD_CACHE="$ROOT/knowledge/kcc0601_v1/cache_trusted_v1"
export BANK_ROOT="$ROOT/knowledge/ekr0601_transition_v1/description_bank_v1"
export GLOVE="$ROOT/data/glove/glove.6B.300d.txt"
export CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
export GPU=2
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJ"
SH
  fi
  cat "$EXP/protocol/session_env.sh"
)
```

`GPU=2` 是原来的物理卡编号，必须根据 `nvidia-smi` 核对。若2号卡已被占用，修改 `session_env.sh` 中这一行再开始本轮，不同时在同一卡上启动多个训练。模型内部 `cuda:0` 指当前可见卡，外部 `CUDA_VISIBLE_DEVICES=2` 映射到物理2号卡。

每次重连/新 tmux 窗口先执行：

```bash
source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
```

### 2.4 使用 tmux 和原 Python 检查环境

```bash
tmux new -As ekr-attr
```

进入后：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  nvidia-smi
  "$CPF_PY" -u -m transition.cli preflight --root "$ROOT"
  "$CPF_PY" -u tests/test_router_cpu.py
  "$CPF_PY" -u "$ROOT/tools/normalize_legacy_checkpoints.py" \
    --legacy-exp "$LEGACY" --apply \
    2>&1 | tee "$EXP/audit/legacy_checkpoint_paths_v1.log"
  sha256sum "$LEGACY/runs/mvp_ut-zappos_C_legacy_s3351_l0/checkpoints/best_auc.pth"
)
```

预期 `PREFLIGHT PASSED`、`ALL CPU ROUTER CHECKS PASSED`。使用原 Python3.8、torch1.13.1+cu117、torchvision0.14.1+cu117，直接调用解释器即可；不依赖当前提示符显示 `(base)` 还是 `(cpf)`。官方版本配对见 [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。

路径修复成功时有 `VERIFIED`、`CHECKPOINT_LINKED`及最后 `LEGACY CHECKPOINT PATHS READY`。它先核对三个数据集六个A/C权重，再建立六个别名；已经正确的路径显示 `EXISTING_OK`，不同hash或悬空链接会停止，绝不覆盖。脚本只用Python标准库，不占GPU、无需sudo，实际hash读取耗时随磁盘速度变化。

如果再次出现文件可读却 `torch.nn.modules.pooling` 导入失败，停止此处并按已保留的系统故障证据找管理员；该问题不能靠反复重装研究依赖解决。不要在这个方案中升级 torch/Python 来绕过旧协议。

训练期间按 `Ctrl-b` 后按 `d` 脱离；重连用 `tmux attach -t ekr-attr`。不要用 `Ctrl-c` 代替脱离。

## 3. 构建可追溯的单条描述 bank

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -u -m transition.cli build-bank \
    --project "$PROJ" --cache-root "$OLD_CACHE" \
    --knowledge-root "$KNOWLEDGE" --out-root "$BANK_ROOT" \
    --glove "$GLOVE" --datasets ut-zap50k mit-states cgqa \
    --permutation-seed 1729 \
    2>&1 | tee "$EXP/launch_logs/build_bank_v1.log"
)
```

它一次扫描原 GloVe，同时为三个数据集保留所需 token，不读取图像。运行一般远短于训练，但需扫描大文本文件，放在 tmux 中更方便。完成时应有三个 `BANK VERIFIED`。

```text
description_bank_v1/<folder>/
  bank.pt             # a/o 的单条 units、mask、均值、差异及固定打乱
  labels.json         # 与旧缓存一致的属性/对象行序
  descriptions.json   # 原文本、tokens、OOV、fallback
  report.json         # 重构误差、差异范数、近零比例、OOV计数
  manifest.json       # 知识来源、旧缓存/GloVe/hash、置换seed、输出hash
```

每个属性/对象当前4个有效槽；UT属性/对象为16/12，MIT115/245，C-GQA413/674。bank 源知识的7个必需文件和 GloVe hash 必须匹配旧 C 缓存；不会自动把 `bulk_trusted_unverified` 改成正式审核。

检查 UT 报告：

```bash
source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
"$CPF_PY" -m json.tool "$BANK_ROOT/ut-zap50k/report.json"
```

验收：`old_C_max_abs_error <= 1e-6`，`residual_sum_max_abs` 接近浮点误差；若大部分差异范数接近0，先停止 EKR 投入，真实描述已经近乎重复，注意力很难找出被平均消去的差异。

知识质量抽查建议：UT仅28个原语，直接检查全部文本；MIT/C-GQA先固定抽取32个属性+32个对象，优先包含多义、非视觉和 `label_template_only` 条目，记录“标签、文本、问题、依据、处理决定”。本轮先观察冻结版本；若需修订，另建 `knowledge_v2/bank_v2` 并重训全部匹配对照，不能改 bank 后继续用旧 C 当完全同协议结果。

## 4. UT 归因验证：先只新增两次训练

### 4.1 生成四种配置，复用已完成的 A/C

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage semantic --datasets ut-zappos --seeds 3351 --epochs 10 \
    --variants A_raw A_unit C_text C_shuffle
)
```

配置从旧 C `train.log` 开头的有效配置恢复，避免复用未合并覆盖项的旧 YAML。保留10轮、64批次、workers4、原学习率、ViT-B/GloVe及原训练损失；增强起始轮37仍不触发。新文件名是 `semantic_ut-zappos_<variant>_s3351_e10.yml`。

### 4.2 GPU 恒等检查必须先通过

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -u -m transition.cli smoke \
    --cfg "$EXP/configs/semantic_ut-zappos_A_raw_s3351_e10.yml" \
    --checkpoint "$LEGACY/runs/mvp_ut-zappos_A_legacy_s3351_l0/checkpoints/best_auc.pth" \
    --out "$EXP/audit/semantic_A_fidelity_ut_v1.json"
  "$CPF_PY" -u -m transition.cli smoke \
    --cfg "$EXP/configs/semantic_ut-zappos_C_text_s3351_e10.yml" \
    --checkpoint "$LEGACY/runs/mvp_ut-zappos_C_legacy_s3351_l0/checkpoints/best_auc.pth" \
    --out "$EXP/audit/semantic_C_fidelity_ut_v1.json"
)
```

两项均须显示 `GPU SMOKE PASSED`，与原模型分数最大误差不超过 $10^{-6}$。这里只检查真实模型的接入，不训练并保存新的 A/C 权重。

### 4.3 仅跑 A_unit 和 C_shuffle

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -u -m transition.cli run-plan --configs \
    "$EXP/configs/semantic_ut-zappos_A_unit_s3351_e10.yml" \
    "$EXP/configs/semantic_ut-zappos_C_shuffle_s3351_e10.yml" \
    2>&1 | tee "$EXP/launch_logs/semantic_ut_native_v1.log"
)
```

顺序执行，原 A/C 不重复训练。根据周报 UT 每个完整原训练流程约74–82分钟，两次可预留约2.5–3小时；新实现和服务器负载会改变时长。新工具保留原逐轮评价，末尾另外保存统一格式的 CW/OW 验证结果，实际可能多一次评价开销。

完成后：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/semantic_ut_native_v1"
  cat "$EXP/reports/semantic_ut_native_v1/paired_summary.csv"
)
```

重点读 `C_text-A_unit`、`C_text-C_shuffle`，同时看 `A_unit-A_raw`。这里只是单 seed 方向检查，不用它直接宣布 EKR 成功或投入三个数据集大矩阵。

## 5. 三个配对 seed 与进入 EKR 的门槛

### 5.1 完成 UT 三个 seed 的四组对照

预先固定 seeds `{0,3351,2026}`。3351的 A/C复用旧权重，A_unit/C_shuffle使用上一步；0和2026的四种变体全部重训。总共12个实验身份，其中2个复用，因此 UT归因阶段总新增10个 run；上一步2个已经包含在其中。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage semantic --datasets ut-zappos --seeds 0 2026 --epochs 10 \
    --variants A_raw A_unit C_text C_shuffle
  configs=()
  for seed in 0 2026; do
    for variant in A_raw A_unit C_text C_shuffle; do
      configs+=("$EXP/configs/semantic_ut-zappos_${variant}_s${seed}_e10.yml")
    done
  done
  "$CPF_PY" -u -m transition.cli run-plan --configs "${configs[@]}" \
    2>&1 | tee "$EXP/launch_logs/semantic_ut_remaining_v1.log"
)
```

约需再预留10–12小时；适合夜间 tmux 队列。不同变体仍使用旧原生训练流程，其 cuDNN benchmark 行为与新增冻结分支不同，分别报告协议，不混作同一训练方式。

### 5.2 汇总和冻结工程门槛

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/semantic_ut_3seed_v1"
  "$CPF_PY" -m transition.cli gate \
    --report "$EXP/reports/semantic_ut_3seed_v1" \
    --out "$EXP/protocol/semantic_gate_ut_v1.json" \
    --datasets ut-zappos --seeds 0 3351 2026
)
```

预设进入 EKR 的工程条件全部满足才 `proceed=true`：

1. 三个完整配对 seed；每种变体有 CW/OW，而不是只挑两个涨点 seed。
2. 平均 CW AUC：$C-A_{unit}\ge0.5$ pp 且 $C-C_{shuffle}\ge0.5$ pp。
3. 上述两类差值各至少2/3个 seed 为正；C平均 CW AUC不低于 A_raw。
4. 相对 A_unit，平均 CW HM不低于−0.5 pp、OW AUC不低于−0.5 pp、OW最佳 Unseen不低于−2 pp。

这是投入门槛，0.5 pp 是预先约定的工程幅度，不是统计显著性或论文录用标准。三个 seed 的均值、样本标准差和每个差值必须一起展示。

| 门槛结果 | 下一步 |
| --- | --- |
| 正确描述稳定超过 A_unit 与打乱 | 执行第6节冻结分支 |
| A_unit几乎解释全部 C收益 | 定位为尺度/表示输入改善，暂缓描述语义创新主张 |
| C与打乱近似相同 | 调查模板/随机代码效应，先改善知识质量 |
| seed方向不稳定 | 保留结果，不选择性丢弃负 seed；先复查协议和文本质量 |
| 报告缺项 | 补完相应训练/评价，再输出新版本门槛文件 |

门槛不通过不是数学上证明 EKR绝不可能有效；它表示当前证据不足以按导师确认的这条路径继续大规模投入。工具会阻止正式分支配置，接口 smoke仍可单独检查。不要手工把 JSON 的 false改成true。

## 6. EKR 最小原型：同一个 C 起点，R0/R3/R5 三组

### 6.1 对照定义

| ID | 机制 | 用途 |
| --- | --- | --- |
| C0 | 冻结 C，分支关闭 | 同 checkpoint 的原始起点，不能当一次独立新训练 |
| R0 | 聚合 C 属性向量作单 key/value，图像/对象条件+空选项 | 普通条件适配与参数/损失控制 |
| R3 | 名称/描述原始 units作 key/value，同 query+空选项 | 普通描述注意力强对照 |
| R5 | 中心化差异作 key/value，图像/对象条件+空选项 | 主 EKR候选 |

R0/R3/R5都有三个300→32投影、32维空 key、空偏置和幅度参数，新增注册参数为28834；工具还记录实际 trainable数量。三者计算次数不同，不能宣称 FLOPs相同。R0输出为 $\eta(1-\alpha_0)\cos(u,w_a^C)$；它已有 C聚合知识，缺少单条描述差异。

### 6.2 先做 GPU 分支检查

此接口检查可以在归因实验运行前或等待结果时做，`--smoke-only` 配置不允许正式训练。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 3351 --epochs 3 \
    --variants R0 R3 R5 --use-legacy-anchor --smoke-only
  for variant in R0 R3 R5; do
    "$CPF_PY" -u -m transition.cli smoke \
      --cfg "$EXP/configs/smoke_ut-zappos_${variant}_s3351_e3.yml" \
      --out "$EXP/audit/${variant}_gpu_smoke_ut_v1.json"
  done
)
```

每项都须通过：旧 C原始分数一致、初始分支零增量、三个小更新后路由器至少两个参数组有非零有限梯度、C完整 state不变、关闭分支仍恢复 C。这几个更新只做审计，不保存为正式权重。

### 6.3 门槛通过后，训练三组3轮

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 3351 --epochs 3 \
    --variants R0 R3 R5 --use-legacy-anchor \
    --gate "$EXP/protocol/semantic_gate_ut_v1.json"
  "$CPF_PY" -u -m transition.cli run-plan --configs \
    "$EXP/configs/branch_ut-zappos_R0_s3351_e3.yml" \
    "$EXP/configs/branch_ut-zappos_R3_s3351_e3.yml" \
    "$EXP/configs/branch_ut-zappos_R5_s3351_e3.yml" \
    2>&1 | tee "$EXP/launch_logs/branch_ut_pilot_e3_v1.log"
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/branch_ut_pilot_e3_v1"
)
```

它冻结旧 C，全程使用该 seed的同一 C checkpoint。训练顺序、数据 worker种子和新优化器设置配对；不把 C已有涨点计入 EKR新增收益。

每轮只跑 val CW，取第一个严格最高 CW AUC的轮；训练后才对该权重跑一次 CW/OW，并计算关闭分支的 C0。这样减少 OW开销。新训练第一批只有幅度参数梯度非零是正常初始化现象；随后应有路由参数梯度。`eta`长期0、所有残差近零需要排查，不能把完全未工作的小分支当好结果。

### 6.4 看哪些文件、如何决定是否继续

```bash
source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
"$CPF_PY" -m json.tool "$EXP/runs/branch_ut-zappos_R5_s3351_e3/selected_state.json"
cat "$EXP/reports/branch_ut_pilot_e3_v1/paired_summary.csv"
```

关键比较：`R5-C0`、`R5-R0`、`R5-R3`。每个候选先选 CW最佳轮，再相对 C0检查 CW HM≥−0.5 pp、OW AUC≥−0.5 pp、OW最佳 Unseen≥−2 pp；CW AUC必须严格改善。若未满足，`selected_state.json`记录 `residual_off`，采用结果退回 C，候选的失败指标仍保存。

这里的规则是“先 CW选轮、后保护检查”，不是看所有轮 OW再选一个最有利组合；不通过后不重新扫描其他 epoch来救结果。

继续投入的初筛建议：R5相对 C0有正向幅度，且高于 R0/R3，主要保护条件通过。若只有0.0x pp微差，则不急于全面扩展。如果3轮末 loss/eta仍明显变化，可将同一组三个版本共同做一次5轮验证：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 3351 --epochs 5 \
    --variants R0 R3 R5 --use-legacy-anchor \
    --gate "$EXP/protocol/semantic_gate_ut_v1.json"
  "$CPF_PY" -u -m transition.cli run-plan --configs \
    "$EXP/configs/branch_ut-zappos_R0_s3351_e5.yml" \
    "$EXP/configs/branch_ut-zappos_R3_s3351_e5.yml" \
    "$EXP/configs/branch_ut-zappos_R5_s3351_e5.yml"
)
```

**这是从相同 C重新开始的5轮 run，不是3+2断点续训。**当前包未保存 optimizer续训状态，不伪装成连续训练。最多做这一次共同延长；已经收敛且仍不能超过强对照时，停止扩展。

## 7. 创新验证：配对重复和能排除替代解释的消融

### 7.1 正式三 seed，匹配各自 C起点

只有第6节有值得复查的趋势，执行10轮正式 R0/R3/R5：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 0 3351 2026 --epochs 10 \
    --variants R0 R3 R5 --use-legacy-anchor \
    --gate "$EXP/protocol/semantic_gate_ut_v1.json"
  configs=()
  for seed in 0 3351 2026; do
    for variant in R0 R3 R5; do
      configs+=("$EXP/configs/branch_ut-zappos_${variant}_s${seed}_e10.yml")
    done
  done
  "$CPF_PY" -u -m transition.cli run-plan --configs "${configs[@]}" \
    2>&1 | tee "$EXP/launch_logs/branch_ut_3seed_e10_v1.log"
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/branch_ut_3seed_e10_v1"
)
```

seed3351使用原 C，0和2026使用第5节重训 C。不能从一个 C checkpoint起3次分支训练，然后宣称已经有3次独立 C重训证据；本工具把 `anchor_seed/anchor_sha256` 写入每个结果。

### 7.2 消融表及执行命令

| ID | 改动 | 若不低于 R5，意味着什么 |
| --- | --- | --- |
| R1 | 图像/对象 query换成共享学习常量 | 静态重加权可能足够 |
| R2 | 移除候选对象条件 | 对象条件尚未证明必要 |
| R4 | 移除空选项 | 空选项的独立作用尚未证明 |
| R7 | 仅新分支的属性描述归属打乱，C输入不改 | 正确文本可能只是随机代码/容量效应 |
| Rname | 只读取名称槽的中心化差异 | 重新偏向名称可能已经解释提升 |
| Rvalue | 保持中心化 keys，仅 values用原始 units | 与 R3相比能单独检验 value中心化 |

Rname非常关键：EKR可能只是抵消不合适的平均描述，重新偏向类别名称，而非利用某条描述的新证据。R3同时改变keys和values，不能单独说明哪一处中心化有用；Rvalue补这个缺口。R1/R2/R4属于功能消融，其实际使用参数可能减少，不能替代 R0/R3的容量控制。

先在3351检查这些版本，再在有结论的关键消融上重复全部三个 seed。完整命令如下，若尚处初筛可将 `--seeds` 改为仅3351，正式报告时补齐：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 0 3351 2026 --epochs 10 \
    --variants R1 R2 R4 R7 Rname Rvalue --use-legacy-anchor \
    --gate "$EXP/protocol/semantic_gate_ut_v1.json"
  configs=()
  for seed in 0 3351 2026; do
    for variant in R1 R2 R4 R7 Rname Rvalue; do
      configs+=("$EXP/configs/branch_ut-zappos_${variant}_s${seed}_e10.yml")
    done
  done
  "$CPF_PY" -u -m transition.cli run-plan --configs "${configs[@]}"
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/branch_ut_mechanisms_v1"
)
```

这18个 run是后续完整消融上限，不是今日必须全部启动的队列。每一项都与同 seed、10轮、相同 anchor的 R5比较；不得用3轮 R0去衬托10轮 R5。

### 7.3 逐图配对，检查收益和代价

工具保存 `image_ids.json`、`candidates.json`、`labels.npz`。下例检查 R5相对同 checkpoint关闭分支的变化：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  RID=branch_ut-zappos_R5_s3351_e10
  "$CPF_PY" -m transition.cli bias0 \
    --reference "$EXP/runs/$RID/anchor_val_ow" \
    --treatment "$EXP/runs/$RID/eval_val_ow" \
    --train-pairs "$ROOT/data/ut-zap50k/compositional-split-natural/train_pairs.txt" \
    --out "$EXP/reports/bias0_R5_minus_C0_ut_s3351_v1"
)
```

输出逐图CSV、seen/val-unseen分组CSV、逐属性CSV。程序先确认图像、候选和真值顺序一致；seen身份只看train pairs。

$$
\Delta Acc_{bias0}=100\frac{N_{wrong\to correct}-N_{correct\to wrong}}{N}.
$$

它不同于 best-HM点、最佳 Unseen和 AUC。先取纠正/退化数量最多的前5个属性，成对检查成功与失败图像，不只展示成功案例。注意力权重只可称路由权重，不能当作因果解释。

## 8. MIT/C-GQA：先补归因，再验证跨数据集迁移

### 8.1 先补原生 seed的两项对照

这两个数据集已有 A/C，不重做。MIT seed0、C-GQA seed3351分别补 A_unit/C_shuffle：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  for dataset in mit-states cgqa; do
    seed=3351
    if [ "$dataset" = mit-states ]; then seed=0; fi
    "$CPF_PY" -m transition.cli make-configs \
      --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
      --stage semantic --datasets "$dataset" --seeds "$seed" --epochs 10 \
      --variants A_raw A_unit C_text C_shuffle
    "$CPF_PY" -u -m transition.cli smoke \
      --cfg "$EXP/configs/semantic_${dataset}_C_text_s${seed}_e10.yml" \
      --checkpoint "$LEGACY/runs/mvp_${dataset}_C_legacy_s${seed}_l0/checkpoints/best_auc.pth" \
      --out "$EXP/audit/semantic_C_fidelity_${dataset}_v1.json"
    "$CPF_PY" -u -m transition.cli run-plan --configs \
      "$EXP/configs/semantic_${dataset}_A_unit_s${seed}_e10.yml" \
      "$EXP/configs/semantic_${dataset}_C_shuffle_s${seed}_e10.yml"
  done
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/semantic_all_native_v1"
)
```

C-GQA每个原流程约155–159分钟，其中最终 OW约31分钟。新 evaluator使用磁盘 memmap，但原 evaluator还会创建多个 CPU矩阵；OW的原始 score单份约7.55 GiB，临时磁盘建议至少20 GiB，空闲 RAM建议至少64 GiB，并用系统实际占用确认。磁盘 memmap不等于整个评价都没有内存复制。

```bash
df -h /data/xhw/czsl_repro
free -h
```

不要把几分钟无新输出直接判断为卡死。优先在另一个窗口看 `nvidia-smi`、进程与最新日志；工具每25个 batch打印 `EVAL_IMAGES`，偏置曲线扫描阶段可能暂时没有这条进度。

### 8.2 有投入价值时补三 seed归因

MIT还需3351/2026四组，C-GQA还需0/2026四组；包含各自原生两项对照后，每个数据集总新增10个归因 run。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  for dataset in mit-states cgqa; do
    seeds=(0 2026)
    if [ "$dataset" = mit-states ]; then seeds=(3351 2026); fi
    "$CPF_PY" -m transition.cli make-configs \
      --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
      --stage semantic --datasets "$dataset" --seeds "${seeds[@]}" --epochs 10 \
      --variants A_raw A_unit C_text C_shuffle
    configs=()
    for seed in "${seeds[@]}"; do
      for variant in A_raw A_unit C_text C_shuffle; do
        configs+=("$EXP/configs/semantic_${dataset}_${variant}_s${seed}_e10.yml")
      done
    done
    "$CPF_PY" -u -m transition.cli run-plan --configs "${configs[@]}"
  done
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/semantic_all_3seed_v1"
  "$CPF_PY" -m transition.cli gate \
    --report "$EXP/reports/semantic_all_3seed_v1" \
    --out "$EXP/protocol/semantic_gate_all_v1.json" \
    --datasets ut-zappos mit-states cgqa --seeds 0 3351 2026
)
```

三个数据集分别统计，禁止跨数据集平均涨点掩盖 MIT的负结果。若延长 MIT/C-GQA原训练预算，应对四组归因版本共同延长并另立协议，不能只延长 C。

### 8.3 跨数据集 EKR，区分两类证据

若某数据集自身文本门槛通过，可用它自己的 gate正式训练。若该数据集没有通过，但 UT明确通过，可以按预先声明的“UT设计迁移验证”开展 R0/R3/R5，必须写明这是检验方法在弱/噪声知识上的边界，不声称该数据集语义收益已成立。

下例是原生 seed的10轮迁移检查，仍使用目标数据集自身 C checkpoint，`--transfer-from ut-zappos`只允许采用 UT通过的投入依据，不会用 UT权重替代目标 C：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  for dataset in mit-states cgqa; do
    seed=3351
    if [ "$dataset" = mit-states ]; then seed=0; fi
    "$CPF_PY" -m transition.cli make-configs \
      --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
      --stage branch --datasets "$dataset" --seeds "$seed" --epochs 10 \
      --variants R0 R3 R5 --use-legacy-anchor \
      --gate "$EXP/protocol/semantic_gate_ut_v1.json" --transfer-from ut-zappos
    configs=()
    for variant in R0 R3 R5; do
      configs+=("$EXP/configs/branch_${dataset}_${variant}_s${seed}_e10.yml")
    done
    "$CPF_PY" -u -m transition.cli run-plan --configs "${configs[@]}"
  done
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/branch_transfer_native_v1"
)
```

继续三 seed时，将每个数据集的 `--seeds` 改为 `0 3351 2026`，并补目标数据集对应 C训练，第8.2节会产生这些 anchor；队列按每个 seed展开。自身 gate通过时去掉 `--transfer-from`、使用 `semantic_gate_all_v1.json`。已生成的同名配置绑定旧 gate hash，不能覆盖：若要改训练身份，使用新的 EXP/BANK版本并保存差异，而不是直接修改既有配置。

跨数据集成功要同时与 A_raw/A_unit和 C比较，不能只超过一个弱 C。跨协议/预算的比较另列，强对照 R0/R3才是同冻结训练协议下的机制比较。MIT持平、C-GQA微差都如实保留；只有 UT有效就限定为UT证据，不宣称普遍方法。

## 9. 再创新：在机制成立后验证受扰知识的退回训练

### 9.1 为什么后做

先证实 R5的干净知识收益，再研究“知识错配时能否减小新增分支损伤”。这一步利用已有批量信任知识的真实局限，提出可测目标，而不是同时堆多个 loss求涨点。

训练错配只改变新增分支属性描述，保留自己的名称、旧 C输入和磁盘 bank；按同类属性跨标签替换3条描述，固定每批噪声 seed。每5个 batch取一次，错配比例10%，UT16个属性实际取 `round(16×0.1)=2`个，即12.5%；30%评价实际取5个，即31.25%。评价JSON记录名义/实际比例、接收者和供体属性ID，不把整数取整误差隐藏。

$$
p_C=\operatorname{softmax}(s_C/\tau),\quad
p_{noise}=\operatorname{softmax}((s_C+\eta\Delta_{noise})/\tau),
$$

$$
L_{fallback}=KL(\operatorname{stopgrad}(p_C)\Vert p_{noise}),
\qquad L=L_{task,clean}+0.05L_{fallback}.
$$

其目的不是要求干净知识始终复制 C，而是压制受扰分支对已有 C的无依据改变。若模型只通过关闭全部分支满足正则，clean没有超过 C，则不算成功。

| ID | 版本 |
| --- | --- |
| R6 | R5 +上述退回训练 |
| R0noise | R0 +完全相同错配与 KL |
| R3noise | R3 +完全相同错配与 KL |

R6必须与 R5、R0noise、R3noise比较，避免把普通正则化收益写成 EKR特有创新。

### 9.2 先 UT3351，再配对重复

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --datasets ut-zappos --seeds 3351 --epochs 10 \
    --variants R6 R0noise R3noise --use-legacy-anchor \
    --gate "$EXP/protocol/semantic_gate_ut_v1.json"
  "$CPF_PY" -u -m transition.cli run-plan --configs \
    "$EXP/configs/branch_ut-zappos_R6_s3351_e10.yml" \
    "$EXP/configs/branch_ut-zappos_R0noise_s3351_e10.yml" \
    "$EXP/configs/branch_ut-zappos_R3noise_s3351_e10.yml"
)
```

通过后相同命令扩展 `--seeds 0 3351 2026`，然后按这三个 seed、三种变体循环队列；已完成3351项由 `run-plan`校验后复用，不重训。

### 9.3 固定模型后的受扰评价

先 UT：干净比例0一次，10%/30%各用固定噪声 seeds `{11,22,33}`，R0/R3/R5/R6及两个噪声强对照都做。0%不因噪声 seed改变而重复计作三个观测。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  for variant in R0 R3 R5 R6 R0noise R3noise; do
    RID="branch_ut-zappos_${variant}_s3351_e10"
    for ratio in 0.1 0.3; do
      for noise_seed in 11 22 33; do
        for setting in cw ow; do
          "$CPF_PY" -u -m transition.cli eval \
            --cfg "$EXP/configs/$RID.yml" \
            --load "$EXP/runs/$RID/checkpoints/best_branch.pth" \
            --phase val --setting "$setting" \
            --noise-ratio "$ratio" --noise-seed "$noise_seed" \
            --out "$EXP/runs/$RID/noise_v1/p${ratio}_n${noise_seed}_${setting}"
        done
      done
    done
  done
)
```

干净结果直接取原 `eval_val_cw/ow`。上述72次 UT评价适合作为一个 tmux队列；**不直接乘到 C-GQA**，其每次 OW约半小时，需另行预估预算。所有 noise目录独立，输出存在就停止，修复/再试使用 `noise_v2`。

比较：同模型相对自己干净结果的下降、受扰时相对 C0的损伤，以及 R6对 R0noise/R3noise的差值。比如

$$
D_v(q)=AUC_v(0)-\frac13\sum_{n\in\{11,22,33\}}AUC_v(q,n).
$$

噪声 seed是同一个训练模型的扰动重复，不是新的训练 seed。只有 R6保持 clean收益且减少损伤，并优于同正则强对照，才支持“条件知识选择+受扰退回”的贡献。用独立命令汇总，不混入主干净结果表：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli noise-summary --exp "$EXP" \
    --out "$EXP/reports/noise_ut_s3351_v1" --noise-version noise_v1 \
    --runs branch_ut-zappos_R0_s3351_e10 branch_ut-zappos_R3_s3351_e10 \
      branch_ut-zappos_R5_s3351_e10 branch_ut-zappos_R6_s3351_e10 \
      branch_ut-zappos_R0noise_s3351_e10 branch_ut-zappos_R3noise_s3351_e10
)
```

输出每次扰动原值、每个训练seed内部的噪声均值/样本标准差，以及R6对强对照的相同噪声配对差。矩阵缺项会报错。多训练seed正式报告先对每个模型的三个噪声seed求均值，再在训练seed之间计算均值/SD，不把3×3当成9次独立训练。

## 10. 可选终版：解冻继续训练必须补 C_budget

第一版冻结 C是过渡实验，有助于定位知识差异的作用。若它成立再做解冻，否则不增加新训练变量。

解冻版本仍从同一 C起点，使用新联合 CE，统一再训练10轮；编码器 lr取旧 `1e-5`，其他原模型参数 `1e-4`，新分支 `1e-3`，统一 AdamW wd `1e-4`。新增条件残差保持上界0.1。C_budget同样优化这个联合 CE，只关闭分支，控制额外训练与损失切换。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  "$CPF_PY" -m transition.cli make-configs \
    --root "$ROOT" --exp "$EXP" --legacy-exp "$LEGACY" --bank-root "$BANK_ROOT" \
    --stage branch --regime joint --datasets ut-zappos --seeds 0 3351 2026 --epochs 10 \
    --variants C_budget R0 R3 R5 --use-legacy-anchor \
    --gate "$EXP/protocol/semantic_gate_ut_v1.json"
  configs=()
  for seed in 0 3351 2026; do
    for variant in C_budget R0 R3 R5; do
      configs+=("$EXP/configs/joint_ut-zappos_${variant}_s${seed}_e10.yml")
    done
  done
  "$CPF_PY" -u -m transition.cli run-plan --configs "${configs[@]}"
  "$CPF_PY" -m transition.cli summarize --exp "$EXP" --legacy-exp "$LEGACY" \
    --out "$EXP/reports/joint_ut_3seed_e10_v1"
)
```

必须比较 `joint R5−joint C_budget/R0/R3`。此处分支关闭只恢复“已经共同更新的基础模型”，不能保证恢复最初冻结 C权重；最终选择原 C需要另载原 anchor。工具对 joint记录 `candidate_requires_joint_controls`，不会自动宣称已采用。上述12次训练是可选终版预算，与冻结分支分开汇报。

## 11. 记录、统计和最终 test

### 11.1 每个 run的实际文件

```text
runs/<run_id>/
  effective_config.yml
  meta/code_manifest.json  meta/environment.json
  meta/cache_manifest.json meta/bank_manifest.json
  meta/branch_identity.json                 # 分支参数量/anchor/训练方式
  train.log  status.json  train_complete.json
  epoch_metrics.jsonl                       # 分支逐轮记录
  checkpoints/best_auc.pth                  # 原训练归因组
  checkpoints/best_branch.pth               # 分支/joint；joint含基础参数
  eval_val_cw/  eval_val_ow/                 # 候选指标，不隐藏负结果
  anchor_val_cw/ anchor_val_ow/              # 冻结分支对应 C0
  selected_state.json                       # 分支是否采用
```

每个 eval目录有 metrics、图像顺序、候选顺序、真值和原始bias0 top-1。默认不长期保存大score矩阵；需要完整矩阵时对独立评价加 `--save-scores`，输出为float32 `scores.dat`，shape见JSON，先算磁盘预算。

`meta/environment.json`与有效配置证明环境和参数；`branch_identity.json`证明真正增加/训练的参数；训练前后基础 state hash证明冻结。`train_seconds`表示训练阶段加逐轮CW评价的总时长，不是纯GPU前向时间；最终评价另记 `eval_seconds`。峰值为当前进程PyTorch allocated MiB，不是整卡总占用。

报告器产生：`runs.csv`、`paired_deltas.csv`、`paired_summary.csv`与 `collection.json`。其输入有hash；新数据产生后使用新的报告目录，不覆盖已冻结报告。相同输入可重复校验。

### 11.2 单位和配对关系

JSON的 AUC/HM为0–1，展示乘100；差值为百分点，不能写成相对百分比：

$$
\delta_s=100(M_{treatment,s}-M_{reference,s}),\quad
\bar\delta=\frac1n\sum_s\delta_s,\quad
SD=\sqrt{\frac{\sum_s(\delta_s-\bar\delta)^2}{n-1}}.
$$

每数据集、CW/OW、训练方式、预算、seed/anchor分别配对。单 seed不填样本标准差，三个seed不保证统计显著性。主比较表至少包含A_raw/A_unit/C/C_shuffle、C0/R0/R3/R5，再按完成情况追加消融和R6。

Seen/Unseen的各自最大值不一定处于同一偏置点。只有 JSON中的 `hm_seen/hm_unseen`可以重算对应 best-HM：$HM=2SU/(S+U)$。不要用 `best_seen/best_unseen`重算 HM。

候选未采用时有两套数：候选权重的原始指标，及采用的 residual_off结果C0。报告器保留候选并记录 `adopted`；不能只发布退回 C后的非负结果然后说 EKR“不降点”。

### 11.3 冻结后一次性 test：以冻结分支终版为例

先用所有 validation结果决定主版本、轮数、强对照与报告范围，不能见 test再改描述、调系数或选seed。下例冻结 UT的三个seed、四项归因对照和R0/R3/R5，原生A/C允许从旧run复用；它们不会被当作新训练。

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  # 3351四项配置已在第4节生成；另两个seed在第5节生成。
  runs=()
  for seed in 0 3351 2026; do
    for variant in A_raw A_unit C_text C_shuffle; do
      runs+=("semantic_ut-zappos_${variant}_s${seed}_e10")
    done
    for variant in R0 R3 R5; do
      runs+=("branch_ut-zappos_${variant}_s${seed}_e10")
    done
  done
  "$CPF_PY" -m transition.cli freeze-final --exp "$EXP" --legacy-exp "$LEGACY" \
    --runs "${runs[@]}" --out "$EXP/protocol/final_test_ut_v1.json"
)
```

若确定最终主方法为R6，应在首次冻结前把R6及R0noise/R3noise加入同一run数组。若决定采用joint终版，则纳入C_budget与对应joint强对照，不能只纳入joint R5。这里的固定分支 off规则按 `selected_state`绑定；joint需要你先审查其强对照结果再把对应run纳入最终协议。

冻结包括配置、权重、代码清单、验证指标与选择文件hash；冻结协议不能覆盖。随后从协议中取精确配置/权重运行：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  PROTOCOL="$EXP/protocol/final_test_ut_v1.json"
  "$CPF_PY" - "$PROTOCOL" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
protocol = Path(sys.argv[1])
exp = Path(os.environ['EXP'])
for run in json.loads(protocol.read_text())['runs']:
    for setting in ('cw', 'ow'):
        out = exp / 'final_test_v1' / run['run_id'] / setting
        subprocess.run([sys.executable, '-u', '-m', 'transition.cli', 'eval',
            '--cfg', run['config'], '--load', run['checkpoint'],
            '--phase', 'test', '--setting', setting, '--protocol', str(protocol),
            '--out', str(out)], check=True)
PY
)
```

工具校验协议内配置/权重/代码/证据hash，对没有冻结身份的test拒绝执行；分支若选off，test按off运行。旧历史test保留，当前最终test另列。现有dataset reader会读取标准划分元数据中的全部pair表以定义词表/候选，这是原协议行为；训练梯度、选择依据只使用train/val，不加载test图像用于调参。

## 12. 中断、检查及安全重试

### 12.1 正常运行怎么查看

新SSH窗口执行，替换RID为正在运行的项：

```bash
source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
RID=branch_ut-zappos_R5_s3351_e3
nvidia-smi
ps -u "$USER" -o pid,etime,pcpu,pmem,args
tail -n 30 "$EXP/runs/$RID/train.log"
cat "$EXP/runs/$RID/status.json"
```

分支训练应周期性输出 `epoch/batch/loss/eta/grad_params`；进入评价会打印 `EVALUATE val ...`。`train_complete.json`存在表示训练已成功结束，尚可能在最终OW或C0评价。`status=train_and_val_complete`才是训练+评价完成。

### 12.2 训练完成、评价中断：只补评价

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  RID=branch_ut-zappos_R5_s3351_e3
  "$CPF_PY" -u -m transition.cli train-branch \
    --cfg "$EXP/configs/$RID.yml" --finish-only
)
```

归因组把 `train-branch`换成 `run-semantic`。它检查原训练的有效配置、代码、权重和完整评价身份；不会重新训练。

若提示输出非空而没有metrics，说明该单项评价留有中断文件。先确认没有相同进程仍在写、没有 `metrics.json`，只移动这一项到保留目录，再finish：

```bash
(
  set -euo pipefail
  source /data/xhw/czsl_repro/experiments/ekr0601_transition_v1/protocol/session_env.sh
  RID=branch_ut-zappos_R5_s3351_e3
  ITEM="$EXP/runs/$RID/eval_val_ow"
  test -d "$ITEM"
  test ! -e "$ITEM/metrics.json"
  mv -T -- "$ITEM" "${ITEM}.interrupted_$(date +%Y%m%d_%H%M%S)"
  "$CPF_PY" -u -m transition.cli train-branch --cfg "$EXP/configs/$RID.yml" --finish-only
)
```

若实际中断的是 `anchor_val_ow`或epoch评价，按错误中的准确目录处理；epoch中断属于训练未完成，不适用finish-only。不要删除整个EXP或旧结果。

### 12.3 训练本身中断

当前包没有optimizer恢复，`train_complete.json`不存在时拒绝假续训。保留原run，以新实验版本重新开始。最稳妥是在修正原因后用 `EXP=.../ekr0601_transition_v2`重新生成同预算配置，保留旧失败记录；所需anchor可通过 `--use-legacy-anchor`或明确复用已成功的同seed C处理，不能把未完成的分支权重当已完成run。

OOM先确认别的GPU进程；对象分块可在**新配置版本**把 `MODEL.branch_object_chunk`改为8，分块数学恒等已有CPU检查。不要只给R5降低训练批次而留下R0/R3原批次；批次、梯度累积或损失变化需要共同更新协议。C-GQA OW的CPU内存不足与GPU OOM分开处理。

`Partial output exists`/`Output is not empty`是保护已有实验的停止，不是服务器卡死；检查阶段和完成标记再按上述分支处理。源码/银行内容改变后不能finish旧run，需另列评价版本。

## 13. 能形成什么创新，以及何时止损

EKR的候选贡献是：**围绕已有聚合知识模型，利用均值消去的描述差异，在图像和候选对象条件下作有界修正，并验证无额外证据及受扰知识时的退回行为。**这是一项可检验的研究定位，不能先宣称新的完整条件概率分解、首次条件属性建模或首次语义选择。

条件属性已有 [CANet（Learning Conditional Attributes, CVPR 2023）](https://arxiv.org/abs/2305.17940)；语义变体选择和专家结构已有 [EVA](https://arxiv.org/abs/2506.20986)。因此“加对象条件”“加注意力”“选一个描述”各自不足以单独构成新颖性。本文的差异保留、C恒等、有界值、语义归属控制、受扰退回需要组合成明确问题，并与这些最近邻方法进一步比较；当前检索不构成全面新颖性证明。

| 结果 | 可支持的结论 | 下一步 |
| --- | --- | --- |
| C超过A_unit和打乱、3seed方向一致 | 当前正确文本内容有可复查价值 | 开始EKR pilot |
| R5超过C0，但未超过R0/R3 | 普通适配/注意力可解释新增收益 | 优先简单版本，EKR机制尚未成立 |
| R5超过R0/R3，却不超过Rname/R7 | 名称偏好/随机代码仍可解释结果 | 修订假设，不宣称描述增量机制 |
| R5超过强对照且消融/多seed支持 | 差异选择机制值得继续研究 | 跨数据集与终版验证 |
| R6保持clean收益并优于同正则强对照 | 受扰知识退回具有额外价值 | 聚焦可用知识与可靠性边界 |
| 最佳采用状态是off | 本轮没有采用新增模块 | 保存负结果，停止堆复杂度 |

每次只引入一个研究变量。第一版不增加 CLIP编码器、不换视觉backbone、不加入OI图传播、不引入固定K；这些变化都需要新协议和相应强对照。USG涉及边保留概率、图采样、消息传播与跨图平均，另设目录和实验；本方案的描述置换与知识错配均是负对照/鲁棒性诊断，不构成USG运行结果。

## 14. 今日执行范围和交付检查

今日先完成第2–4节：部署、环境检查、bank、A/C恒等smoke、UT的A_unit/C_shuffle两次10轮训练；可提前检查第6.2节EKR接口。不要一次启动后续所有条件阶段。

第一阶段UT三seed归因总新增10次训练，参考周报约12–14小时，先获得是否值得继续的证据。其他数据集归因、正式9次分支、消融18次、鲁棒性9次、joint12次都是通过前一阶段后逐步投入的预算。新分支运行时长尚未在GPU测得，首个pilot用实际 `train_seconds/eval_seconds`再估计后续，不把CPU测试时长当训练耗时。

- [ ] 所有旧run、知识、缓存与完整导出 `czsl_all_completed_20261005_000428_2772161`保留。
- [ ] bank与旧C聚合一致，标签顺序/hash无误；知识仍标记批量信任状态。
- [ ] 原模型GPU恒等检查通过；新分支初始/关闭恒等、梯度与冻结通过。
- [ ] 四项归因对照按预设seed完成，数据集分别统计，不替换负seed。
- [ ] EKR超过C0及R0/R3，关键消融支持机制；参数、预算与候选协议配对。
- [ ] 受扰诊断与主干净指标分开；噪声seed与训练seed分开。
- [ ] test前冻结版本与规则，保留候选失败指标和off选择。

配套工具的本地检查已通过：9种router的零增量/幅度界/重排/分块/梯度、扰动隔离与RNG保持；真实Week5源码的安装和原特征前缀一致；原GloVe构建函数的聚合/置换/禁止覆盖；配置、配对汇总、缺数据门槛及最终协议冻结。另用真实UT归档复核bias0配对，得到周报同样的43纠正/39退化、净+4与未见净−3。

这些检查在本地Python3.12/CPU torch2.5.1执行，源码也按Python3.8语法解析。**服务器Python3.8/torch1.13.1/CUDA的完整模型smoke和实际训练还未执行**；必须由本方案中的服务器检查确认后才开始新实验。不要把工具检查通过写成EKR实验成功。

路径修订验证：用户截图已显示服务器 `PREFLIGHT PASSED` 与 torch1.13.1下的全部CPU router检查通过，失败发生在旧权重的简化路径检查。修复脚本在本地完整归档中只读核对六个真实A/C权重，全部与原CW/OW声明hash一致；另检查相对链接生成和禁止覆盖逻辑。服务器实际链接创建及完整模型GPU smoke仍需执行。
