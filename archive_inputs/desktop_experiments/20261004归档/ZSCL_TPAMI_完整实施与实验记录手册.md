# ZSCL / TPAMI 扩写：完整实施与实验记录手册

版本：2026-09-30　｜　输入设计：[ZSCL实验设计0601.md](<C:/Users/llenovo/Desktop/ZSCL实验设计0601.md>)　｜　状态：**实施规范，代码尚未按本手册改造**

## 0. 使用范围与必须先讲清的结论

本手册假设 CPF、Troika、Logic-Troika 基线已经各自完成训练、CW/OW 测试和归档。新方法以**实际在服务器跑通的 CPF 修补版** `/data/xhw/czsl_repro/src/CPF_fixed` 为起点。当前本地能核对到的是原压缩包 `work/sources/CRMask_copy` 和既有修补手册，**无法从本机确认服务器现存 `CPF_fixed` 的最终逐行内容**。下列文件和接口是实施目标，开发前必须以服务器实际源码核对函数签名，不能把文中的伪接口当成已存在的命令。

研究问题：结构化知识 `K` 是否通过图像相关、对象条件化的属性预测提升组合零样本泛化，而不仅仅改变文本长度、加一个静态 pair 偏置或增加参数。

正式声明分成三级：

| 层级 | 可以表述的结论 | 必要证据 |
|---|---|---|
| 工程完成 | A–H 及评估流程正确运行 | 形状、梯度、无泄露、日志、checkpoint 回归检查 |
| 实验支持 | Full Model 相对匹配对照有稳定增益 | 三数据集、CW/OW、配对多 seed、置信区间、负对照 |
| 论文主张 | 知识确实参与条件属性预测 | `H>A/B/F`、`G>E`、图像与知识干预、错误分解；不能只凭 `H>A` |

**不能预设 H 一定超过论文基线或设置虚构的目标 AUC。** 设计中的 Gate 是继续投入实验的内部阈值；论文数值必须如实报告。外部论文值用于检查基线量级，不是新方法必达成绩。

---

## 1. 固定实验契约

### 1.1 固定的数据、训练和评估口径

| 项目 | 固定规则 |
|---|---|
| 数据 | MIT-States、UT-Zappos50K、C-GQA 的同一 natural compositional split；保存 metadata 和 split SHA256 |
| 数据图像 | 每个 metadata 引用均可读取；实际样本数和缺图数写入审计文件 |
| 候选 pair | 训练 CE 仅用 train pair；CW/OW 按**现有通过基线验证的 evaluator**定义，所有变体共用同一候选表和顺序 |
| 选模 | 每 epoch 验证，按预先固定的验证集 AUC 保存最佳 checkpoint；测试集不能参与选模/调参 |
| CW/OW | 同一个训练 checkpoint 分别评估；若特定原论文协议要求不同训练设置，要作为另一条独立实验并注明 |
| 主指标 | Top-1 AUC、HM、Seen、Unseen；原始 `[0,1]` 和乘 100 后百分点都记录 |
| 训练条件 | 同一数据集的 A–H 固定 ViT-B/16 权重、数据增强、optimizer、epoch、batch、验证频率、精度与硬件；只变变体开关 |
| 负对照 | 与所比较的方法保持额外参数量、文本编码器、描述数量、训练步数一致；无法完全匹配时列明差异 |
| 多 seed | MIT 建议 `0,1,2`；UT/C-GQA 建议 `3351,3352,3353`。这只是新实验预注册方案，不是声称作者使用这些 seed |
| 调参 | 先在 UT 验证集选择有限网格，冻结后迁移到 MIT/C-GQA；各数据集若单独调参须另列选择过程 |
| 测试 | 一组冻结配置和 checkpoint 对测试集运行一次；后续探索需标为 exploratory |

基线数字可从 [CPF/LOGICZSL复现实验方案](<C:/Users/llenovo/Documents/Codex/2026-09-28/new-chat/outputs/CPF_LOGICZSL_paper_reproduction_experiment_plan.md>) 导入。已有 CPF UT 示例是 CW `36.51/52.59/61.49/68.48`、OW `27.07/43.87/61.49/51.30`（AUC/HM/Seen/Unseen，百分点）；只可作为对应源码与数据 SHA 相同的 A 基线。

### 1.1.1 三数据集执行矩阵

以下是本地原始 CPF 配置的起始值，不等同于服务器修补版的最终有效配置。正式运行前导出各数据集的 `config_resolved.yml`，以基线实际跑通的设置为准；A–H 在同一数据集内必须一致。

| 数据集 | 原始配置文件 | 原始训练 batch | 原始 epoch | 数据目录核对重点 |
|---|---|---:|---:|---|
| UT-Zappos50K | `config/ut-zappos.yml` | 64 | 10 | `ut-zap50k` 图像、natural split、GloVe 路径 |
| MIT-States | `config/mit-states.yml` | 32 | 20 | `mit-states/images` 是否指向真实图像而非 `__MACOSX`；metadata 引用零缺图 |
| C-GQA | `config/cgqa-vit.yml` | 64 | 20 | 图像数及 metadata 引用；大 pair 空间的显存与评估分块 |

每个数据集依次做：A 回归 → 知识审计 → B–F → Gate 1 → G/H → Gate 2 → 负对照 → 多 seed → CW/OW 冻结测试。若 batch 64 在 C-GQA 或其他设置 OOM，先降低知识计算峰值和评估 chunk；必须改训练 batch 时，该数据集的 A–H 一起改并重跑匹配基线，不得混比。

### 1.2 实验编号和目录

推荐：

```text
/data/xhw/czsl_repro/
├── src/CPF_fixed/                         # 已通过的基线，保持原样
├── src/CPF_KCC/                           # 新方法独立开发目录
├── data/<dataset>/knowledge_kcc/v1/       # 按数据集冻结的知识
└── runs/kcc_v1/<dataset>/<variant>/seed_<seed>/
    ├── config_resolved.yml
    ├── command.txt
    ├── train.log
    ├── val_by_epoch.csv
    ├── eval_cw.log
    ├── eval_ow.log
    ├── metrics_cw.json
    ├── metrics_ow.json
    ├── manifest.json
    ├── checkpoint_sha256.txt
    └── checkpoints/best_val_auc.pth
```

建议 Run ID：`kcc_v1__utz__G__seed3351__knowledge_v1`。同名 Run ID 不覆盖已有输出；参数、知识或代码变化后生成新 Run ID。

### 1.3 源码冻结和差异审计

在服务器执行，先确认工作路径与现有修改；不使用 `git reset` 或覆盖原版：

```bash
export CZSL_ROOT=/data/xhw/czsl_repro
export CPF_BASE="$CZSL_ROOT/src/CPF_fixed"
export KCC_CODE="$CZSL_ROOT/src/CPF_KCC"

test -f "$CPF_BASE/train.py" && test -f "$CPF_BASE/models/ddm.py"
git -C "$CPF_BASE" status --short
git -C "$CPF_BASE" rev-parse HEAD 2>/dev/null || true
sha256sum "$CPF_BASE/train.py" "$CPF_BASE/test.py" "$CPF_BASE/models/ddm.py"
```

若 `CPF_KCC` 尚不存在，使用 `cp -a "$CPF_BASE" "$KCC_CODE"` 创建副本；若已存在，先检查并保留现有工作，不执行覆盖复制。将原基线 checkpoint、配置、split 和关键源码 SHA 写入 `baseline_manifest.json`。对 A 开关关闭的新代码进行数值回归：相同权重、同一固定 batch、`model.eval()` 下 pair scores 应与原修补版在浮点容差内一致；若不一致，先定位原因，再做新实验。

---

## 2. 设计稿中需要落地修正的四点

1. **原 CPF 并无可直接复用的 `[B,N,M]` 对象条件属性 logits。** 本地原压缩包 `models/ddm.py` 训练时分别算 object/attribute/combination logits，验证时组合分数为三者相加。新头必须显式构造 `z_attr_given_obj[b,o,a]`，不能只给旧变量改名。
2. **设计稿的 `pair_bias = h(e_ao)` 只依赖文本知识。** 它对每张图相同，属于静态 pair prior，无法独立证明图像条件概率模型。G/H 的主新项必须包含图像特征、对象特征与知识的相互作用。
3. **原 CPF 用 GloVe 300维，不能直接塞入 CLIP 文本向量。** 主消融统一用现有 GloVe/同维聚合生成知识向量；若另用 CLIP 文本编码器，应建立同编码器的无知识匹配对照，并列为独立实验。
4. **LogiCZSL `LogicLoss` 接收 logits，函数内部再 `softmax`，且原实现内部已有 `0.01` 系数。** 不可先把适配器结果 softmax 一次再送进去，也不可在外部又无意重复乘 `0.01`。必须把缩放集中到一个明示的 `beta`，并验证梯度到达 CPF 可训练层。

另一个科学风险是设计稿中 `K_text = cos(pair_embedding, mean(object_embedding,attribute_embedding))` 可能天然接近常数。先绘制直方图、计算标准差和独特值数，再决定它能否作为有信息量的先验；接近常数时不能通过 Gate 1。

---

## 3. Phase 0：把修补版整理成可扩展的 CPF 基底

### 3.1 逐文件修改清单

下表以服务器 `CPF_fixed` 为目标；文件名如与本地原压缩包不同，以实际文件为准。LogiCZSL 的代码仅作为知识 schema 和逻辑损失参照，不在其已跑通的 Troika 基线上直接加本方法。

| 文件 | 需要修改或新增 | 验收证据 |
|---|---|---|
| `models/ddm.py` | 保留 A 的原始路径；将 object、attribute、composition 原始 logits 和图像特征以内部结构返回；接入可开关的 Text K、conditional head、prior、logic head | 所有开关关时与 CPF_fixed 输出数值对齐 |
| `models/knowledge_encoder.py`（新增） | 按固定 `attrs/objs` 顺序载入 W/K，做维度、finite、checksum 检查；注册为 buffer 或受控参数 | 错序、缺项、NaN 明确报错 |
| `models/conditional_head.py`（新增） | 用图像、对象知识和 pair 文本知识产生真正的 `z_attr_given_obj`；支持 pair index / object chunk | 输入改变时输出改变；无 `[B,N,M,D]` 大张量 |
| `models/prior.py`（新增） | centered / clamped-log 归一化、形状和数值检查、候选 pair gather | uniform K 不改变排名及指标 |
| `models/logic_adapter.py`（新增） | 产生 object `[B,N]`、parent `[B,H]`、OE `[B,N,3]`、attribute `[B,M]`、AE `[B,M,3]` 的原始 logits | shape、非零梯度、权重系数一致 |
| `models/logicloss.py`（新增或经授权复制） | 移植并审计 LogiCZSL HC/OE/AE 公式；统一 `beta`，输出分项损失 | B/F/H 有非零 loss 和可回传梯度 |
| `dataset.py` | 保留既有样本返回与 split；增加可选知识 ID 对齐；不让 val/test 标签进入训练先验 | A 数据顺序、长度与 CPF_fixed 一致 |
| `train.py` | 统一 A–H 配置、train pair CE、分项日志、val AUC 选模、恢复训练状态 | 仅验证集决定 best epoch；训练期间不窥 test |
| `test.py` / `evaluator_ge.py` | 仍用现有经过基线检查的 AUC/HM 计算；只更换 score 来源；CW/OW 共用同一 best checkpoint | A 的 CW/OW 与旧脚本一致 |
| `config/kcc_*.yml`（新增） | 保存 Text K、Prior、Feature K、Logic、lambda、beta、chunk 等开关；输出解析后完整配置 | 运行目录中有 `config_resolved.yml` |
| `tools/build_knowledge.py`（新增） | 从固定知识 JSON/CSV 生成有序向量和先验 | 产物形状、覆盖率、checksum |
| `tools/audit_knowledge.py`（新增） | 漏项、重复项、OOV、泄露、分布、各数据集覆盖审计 | 产出机器可读报告且零硬错误 |
| `tools/run_variant.py`（新增，可选） | 以变体编号和数据集/seed 生成 config、run ID、命令 | 单个 Run ID 对应唯一配置 |
| `tools/evaluate_variant.py`（新增） | 从指定 run 恢复验证最佳权重、配置和知识版本，再调用现有 CW/OW evaluator | 两种 setting 均来自同一 checkpoint，指标落盘为 JSON |
| `tools/collect_results.py`（新增） | 只从冻结日志/JSON 提取结果、对照论文和组内差值 | 汇总可追溯到 checkpoint SHA |

注意本地原版 `train.py` 曾每 epoch 访问 test、`test.py` 导入失效 evaluator；这些在 `CPF_fixed` 的修补工作中已处理。新目录应**继承现存修补结果并做回归**，不要重新引入原压缩包代码。原版 `models/ddm_v3.py` 和 `models/ddm.py` 同时存在，确认 `train.py` 实际 import 的类后，只改被调用的实现。

### 3.2 模型统一输出契约

新模型在训练/验证内部统一提供（不强求全部公开给旧 evaluator）：

```text
obj_logits:          [B,N]
attr_logits_base:    [B,M]
pair_logits_base:    [B,P_candidate]   # 已按 dset.pairs 顺序 gather
attr_given_obj:      [B,P_candidate]   # G/H 的图像条件知识项
pair_logits_final:   [B,P_candidate]
logic_raw_logits:    {obj[B,N], parent[B,H], oe[B,N,3], attr[B,M], ae[B,M,3]}
loss_parts:          loss_comp, loss_obj, loss_attr, loss_cond_attr, loss_oi, loss_oe, loss_ae
```

`P_candidate` 在训练为 train pair 数，评估为 evaluator 的 CW/OW candidate 数。以 `pair_obj_idx[P]`、`pair_attr_idx[P]` 显式 gather；**不要假设 pair tensor 是 `[N,M]` 顺序或先 obj 后 attr**。原数据定义的 pair 元组顺序是 `(attribute, object)`，而 `K_oa` 存储顺序为 `[object, attribute]`。

### 3.3 原 CPF 分数保留和新分数接入

以已经复现成功的 CPF 分数 `s_base(x,a,o)` 作为主基底。A 应严格使用它。D 增加静态 prior；G/H 增加图像相关的知识条件项：

```text
s_A(x,a,o) = s_base(x,a,o)
s_D(x,a,o) = s_base(x,a,o) + lambda_k * prior_K[o,a]
s_G(x,a,o) = s_base_Ktext(x,a,o)
             + gamma * delta_cond(x,o,a,K)
             + lambda_k * prior_K[o,a]
```

其中 `s_base_Ktext` 是 C 的同一路径文本替换输出；`delta_cond` 必须随图像变化。`gamma` 建议初始化为 0 或很小值以保持训练初始接近 E，并记录训练后数值；若始终约等于 0，即使 G/H 指标提高，也不能把收益归因于条件头。基底的 object/attribute/combination 三项尺度、CosineClassifier 温度和候选顺序先保持不变。不要直接把未校准的 log probability 与原 cosine logits 相加；若改为 `log_softmax` 体系，必须另设一整套公平重训的 A 对照。

---

## 4. Phase 1：知识文件、向量和先验

### 4.1 输入文件与命名映射

设计稿中的 `hc/oda/ada` 分别对应先前 LogiCZSL 修补材料中的 `oi/oe/ae`。导入程序通过显式映射接受两种命名，但以一种规范 schema 写入中间文件：

| 规范项 | 可接受原始名 | 目标值 | 硬性检查 |
|---|---|---|---|
| OI 父类 | `hc_dict.json` 或 `oi_dict.json` | object → parent/group | 每个对象均有映射；多父类需明确建模策略 |
| OE 描述 | `oda_dict.json` 或 `oe_dict.json` | object → list[str] | 全对象覆盖；原 LogicLoss 若仍要求3条则恰好3条 |
| AE 描述 | `ada_dict.json` 或 `ae_dict.json` | attribute → list[str] | 全属性覆盖；原 LogicLoss 若仍要求3条则恰好3条 |
| LLM 兼容性 | `compatibility.csv` 等 | `[N,M]` 或含名称表 | 行列顺序、值域、来源、缺失规则都明确 |

使用数据集 `attrs/objs` 的**精确字符串**作为唯一键；保留名称规范化前后映射表。知识来源必须写明：作者原件、导师提供、人工重建或 LLM 生成；这几种来源不能混写成“官方知识”。保留 prompt、模型/版本、生成日期、原始输出、清洗脚本、人工改动和 SHA256。若只有某个数据集的知识，先做该数据集，其他数据集标为 `not available`。

### 4.2 生成规范化产物

```text
knowledge_kcc/v1/
├── raw/oi.json, oe.json, ae.json, compatibility.csv
├── normalized/oi.json, oe.json, ae.json
├── objects.txt, attributes.txt, pair_index.csv
├── W_o_K.pt                 # [N,300]，主实验与 CPF GloVe 维度一致
├── W_a_K.pt                 # [M,300]
├── W_ao_K/                  # 每个 object 一个 [M,300] chunk，或 CPU memmap
├── K_oa_text.pt             # [N,M]
├── K_oa_llm.pt              # 只有完整可审计覆盖时才生成主实验版
├── K_oa_mix.pt              # 明确混合权重和缺失策略
├── prior_centered.pt
├── prior_log.pt
├── coverage_report.json
└── generation_manifest.json
```

主消融建议用原 CPF 的冻结 GloVe 300维词空间处理词组，并保持 A–H 相同的分词/OOV 规则。描述文本：每条先编码并 L2 归一，再与名称向量聚合，最后归一。`W_ao_K` 对**全部属性×对象组合**建立同数量模板，保证 seen/unseen 覆盖对称。单纯词均值对语序不敏感，报告中要注明；CLIP 文本编码器版本作为独立扩展，配一个同 CLIP 编码器但只用名称/长度匹配模板的对照，不能把 encoder 收益当作知识收益。

### 4.3 Pair 模板与 C-GQA 内存

统一模板，例如 `the {attr} {obj}`、`a {obj} that is {attr}`、`{attr} appearance of {obj}`；所有 pair 使用同样三条。若添加 LLM pair 描述，主实验要求对**全部候选 pair**用同一协议生成；若只覆盖 seen pair，这一版本只能作为偏差诊断实验，不能作为 Full Model 主结果。

C-GQA `674×413=278,362` pairs：一个 FP32 `[N,M,300]` 张量约319 MiB，一个 batch 64 的 FP32 `[B,N,M]` score 约68 MiB，而 `[B,N,M,300]` 会达到约20 GiB（还不含梯度与模型）。因此预计算 pair 知识保存在 CPU/chunk 或半精度只读缓存；训练只 gather train pairs，评估按 object 或 pair chunk 计算。任何阶段不得构造 `[B,N,M,D]` 中间张量。

### 4.4 `K_oa` 构建与泄露审计

至少保存 text-only、完整 LLM-only（若有）、mix 三版。`K_text` 可以先按设计稿的文本相似度实现，但必须检查 `std`、分位数、unique 数、seen/unseen 分布和与文本名称长度的相关性。若标准差接近零，则该公式不合格，重新定义独立且有区分性的相似度/可行性模型，重新冻结版本；不能在测试集上挑一个“效果好”的 K。

`K_llm` 只能基于属性/对象词和公开语义描述，不传入 split 身份、图像标签或测试频数。C-GQA 若没有覆盖全部组合，主实验只用 `K_text`；缺失值填0.5只能做单列的缺失覆盖实验，需报告 mask 与 seen/unseen 比例。先验采用 soft 分数：

```python
centered = (K - K.mean()) / (K.std(unbiased=False) + 1e-8)
log_prior = torch.log(K.clamp(min=0.05, max=1.0))
```

静态 K 对训练 train pairs 可参与 loss，但不读取 val/test 是否存在的标记。评估 CW/OW 统一先按同一 pair 索引 gather `K[o,a]`，再交现有 evaluator 应用候选 mask。若先验加入时与 unseen bias 校准顺序改变，所有 A–H 和对照必须统一，并留下 evaluator 版本哈希。

### 4.5 Phase 1 验收

| 检查 | 通过条件 |
|---|---|
| 覆盖 | `N_obj`/`N_attr` 全部名称一一匹配，主实验 pair 模板覆盖100%；缺失知识项计数0 |
| 对齐 | 随机抽10个 `(a,o)`，人工打印 `attr_idx,obj_idx,K[o,a],pair_text` 与原名称一致 |
| 数值 | embedding/K/prior 均 finite，维度和值域正确，归一化向量范数接近1 |
| 先验 | 不是常数；uniform prior 回归应使 pair 排名及指标不变 |
| 泄露 | 产物可由词表、原始知识和预先固定脚本重建；不读取 val/test 图像/标签/组合存在性 |
| 可追溯 | 原始知识、生成脚本、标准化结果及所有产物有 SHA256 |

---

## 5. Phase 2：A–F MVP 实现

### 5.1 A–F 的精确定义

| 变体 | Text K | Prior K | 图像条件 Feature K | External logic | 实际改动 |
|---|:---:|:---:|:---:|:---:|---|
| A CPF-only | 0 | 0 | 0 | 0 | 旧修补版行为；新代码全关时必须回归它 |
| B CPF+external K | 0 | 0 | 0 | 1 | A + OI/OE/AE concept heads 与逻辑 loss，最终预测仍用 CPF pair score |
| C richer text | 1 | 0 | 0 | 0 | 在 CPF 词向量接点按顺序替换 W_o/W_a；维度与温度不变 |
| D calibration | 0 | 1 | 0 | 0 | `s_base + λ·prior[o,a]`，训练/验证/测试一致 |
| E text+calib | 1 | 1 | 0 | 0 | C 的分数 + D 的 prior |
| F naive full K | 1 | 1 | 0 | 1 | E + 与 B 同一个逻辑头/损失；即设计稿的 naive combination |

`λ` 先试 `{0,0.025,0.05,0.1,0.2}`，仅按 UT 验证集预设汇总规则确定；0 用来校验 A/C。`beta` 先试 `{0,0.001,0.01}`，**确认只乘一次**。具体网格只是实施建议，应在首次查看测试集前登记到 config。若某值触发 NaN，记录失败，不从报告中抹掉。

### 5.2 输入端改动

在 `DDM._setup_word_composer` 所建立的 `attr_embedder/obj_embedder` 处，C/E/F 和 G/H 使用同顺序的 `W_a_K/W_o_K` 初始化或只读 buffer；A/B/D 仍用原 GloVe 向量。确认下游 `compose_word_embeddings`、text-guided object attention 和属性分类器实际读取到替换后的向量。冻结/可训练策略必须固定并写入配置；若原 CPF embeddings 为冻结，就先沿用冻结。

### 5.3 输出端改动

现有 `DDM.val_forward` 输出按 candidate pair 的 `scores` 字典。D/E/F 将 prior 按 `pair_obj_idx,pair_attr_idx` gather 成 `[P]`，与已有 `pred[:,p]` 相加；训练 train pairs 同样 gather，不能先 `argmax object`。全局常数 K 不改变任意两个 pair 的排序，作为必要回归检查。λ=0 时 D=A、E=C 的 logits/指标在确定性条件下应相同。

### 5.4 Logic adapter 的可执行契约

原 `logicloss.py` 期望的是**原始 logits**：object `[B,N]`、parent `[B,H]`、OE `[B,N,3]`、attribute `[B,M]`、AE `[B,M,3]`。B/F/H 使用相同 concept head，输入可复用 CPF 图像特征，但没有扩展描述的视觉分支必须明确定义新的图文匹配头。对 OE/AE 的三条描述分别编码，并保证张量最后一维为3；若数据不满足3条，要先修 schema 或改 loss 实现，不能靠重复一条占位。

逻辑总损失定义一次：

```text
L_logic_raw = L_oi + L_oe + L_ae
L_total = L_CPF + alpha_cond·L_cond_attr + beta(t)·L_logic_raw
```

原 `LogicLoss` 内部固定 `×0.01` 时，移植后要移除内部系数或令外部 beta=1，二选一并记录。主实验 `alpha_cond=0` 用于 A–F；G/H 的 `alpha_cond` 必须在验证集定好。`LogicLoss` 接收 logits，自行做 softmax；adapter 不做第二次 softmax。必须对 `L_logic_raw` 单独 `backward` 检查可训练视觉/概念参数梯度范数大于0，并在关闭 logic 时保持 A/C/E 的原损失数值。

若 B/F/H 只有 OI loss 跑通、OE/AE 未实现，这些实验标作 `OI-only`，不能写为设计稿定义的 B/F/H；三类均通过后再填完整 A–H 表。

### 5.5 Gate 1

| 条件 | 最低标准 |
|---|---|
| 工程 | A–F 在 UT 1 seed 跑满论文对齐 epoch；各模块启闭和日志相符；无 NaN/Inf/OOM |
| 回归 | A 新代码与归档 CPF 基线的同权重推理分数接近；D(λ=0)=A、E(λ=0)=C |
| 梯度 | B/F 的三类 logic loss 可以分别产生非零梯度；λ≠0 的 prior 使 pair 排序在非均匀 K 下变化 |
| 值得继续 | E 或 F 至少一个在 UT 验证集 AUC **和** HM 优于 A；OW unseen 不出现事前定义的严重退化（建议超过5百分点视为严重，属于内部门槛） |
| 异常处理 | F 明显低于 E 时记录并在验证集排查 beta/梯度；不要把失败配置从正式消融表中删除 |

Gate 1 只是筛选继续研发，不代表 TPAMI 结论。可记录单 seed 筛选结果，但论文主结论仍需多 seed 和三数据集。

---

## 6. Phase 3：G/H 的条件知识模块

### 6.1 必须有真正图像相关的知识交互

设计稿示例中的 `h(e_ao)` 是 `[N,M]` 静态偏置。主模型改为下面的结构之一，并写明选定版本：

```text
q_attr[b]      = normalize(P_attr(v_attr[b]))              [B,d]
q_obj[b,o]     = normalize(P_obj(v_obj[b]) + U(W_o_K[o]))   [B,N,d]
t_pair[o,a]    = normalize(T(W_ao_K[o,a]))                 [N,M,d]
gate[b,o]      = sigmoid(g(q_obj[b,o], W_o_K[o]))          [B,N,d]
q_cond[b,o]    = normalize(q_attr[b] + gate[b,o] * q_obj[b,o])
delta_cond[b,o,a] = temperature * dot(q_cond[b,o],t_pair[o,a])
```

其中 `v_attr/v_obj` 使用 CPF 的真实视觉特征；`W_o_K/W_ao_K` 来自 Phase 1，`gate` 可训练。以 object chunk 或候选 pair gather 计算 dot，不展开四维大张量。输出 `delta_cond` 随图像和 `o,a` 变化；打乱知识与固定图像应改变输出，打乱图像与固定知识也应改变输出。训练时可只针对 train pairs 计算 CE，对 GT object 计算额外条件属性 loss：

```python
z_attr_gt_obj = z_attr_given_obj[batch_index, obj_label, :]  # [B,M]
L_cond_attr = cross_entropy(z_attr_gt_obj, attr_label)
```

不能把完整 `[B,N,M]` 直接送入属性 CE。`GT-object` 只用于训练辅助 loss；推理对所有候选 `(a,o)` 算分，不读取 GT object。把 `alpha_cond` 与该辅助损失在 E 的参数匹配对照中考虑，否则 `G>E` 可能只是多了辅助监督。

### 6.2 G/H 定义与控制

| 变体 | 与谁共用代码 | 新增项 |
|---|---|---|
| G | E 的 Text K、K prior、训练协议、候选/评估代码 | `gamma·delta_cond`，预注册的条件属性辅助 loss |
| H | G 的全部结构 | 与 B/F 完全同定义的 OI/OE/AE logic loss |

为证明收益不是参数量或另一个注意力模块带来的，增加 `G-randomK`：相同头、相同参数量和训练步数，但在预先固定的 seed 下打乱知识映射。若 G 与 `G-randomK` 无差异，不能把提升归因于语义知识。另保留 `G-template-only`（全部 pair 都只有同长度模板）和 `G-no-pairK`（只含对象/属性知识）。

### 6.3 Gate 2

| 条件 | 最低标准 |
|---|---|
| 数值/显存 | UT G/H 单 seed 跑满；C-GQA batch 64 或有详细记录的等效方案能在 RTX 3090 完成前向、反向和评估 |
| 条件性 | 固定图像交换 `o`、固定 `(o,a)` 换图像，`delta_cond` 都变化；新头梯度非零；`gamma` 非零 |
| 性能筛选 | UT 验证集 G 相对 E、H 相对 F 不显著下降；正式实验中还需比较 H vs G/B/A |
| 负对照 | `G-randomK`、uniform prior、shuffled descriptions 不应系统性复制真实 K 的增益 |

“不显著下降”建议先按预注册的验证集容差（例如 AUC/HM ≤1百分点）筛选，最终采用多 seed 配对区间，不用单次测试结果决定是否修改模型。

---

## 7. 实验全集与执行次序

### 7.1 主消融 A–H

| 编号 | Text | Prior | Conditional | Logic | 核心检验 |
|---|:---:|:---:|:---:|:---:|---|
| A | 0 | 0 | 0 | 0 | 原 CPF 参照 |
| B | 0 | 0 | 0 | 1 | 外部 logic 单独作用 |
| C | 1 | 0 | 0 | 0 | 描述增强文本 |
| D | 0 | 1 | 0 | 0 | compatibility 单独作用 |
| E | 1 | 1 | 0 | 0 | 输入+输出 MVP |
| F | 1 | 1 | 0 | 1 | naive combination |
| G | 1 | 1 | 1 | 0 | 条件知识头的净贡献 |
| H | 1 | 1 | 1 | 1 | Full Model |

每个 `(dataset, variant, seed)` 训练一次，并评估 CW 与 OW。三数据集×八变体×三 seed = **72次训练、144次最终评估**，还未计负对照和超参试验。已有 A 基线只有在与新实验的源码、split、权重、训练策略和 seed 匹配时才可直接复用；否则重跑 A。

### 7.2 必做控制实验

| ID | 实验 | 具体操作 | 成功控制了什么 |
|---|---|---|---|
| S1 | shuffled descriptions | 在同类型、尽量相同描述条数/长度的 object 或 attribute 间置换文本；固定随机种子 | 文本长度、额外语句数量 |
| S2 | random K | 对 K 的行列/值分布做固定 seed 置换，保留均值方差；整套训练/推理协议一致 | 任意静态 pair 偏置 |
| S3 | uniform K | K 全常数；λ 值不变；应得到与无 prior 完全相同的 pair 排名/指标 | 常数偏移代码错误 |
| S4 | template-only pair K | 所有 pair 只用同样三条模板，无 LLM pair 描述 | pair 描述内容贡献 |
| S5 | no-pair-K | 去除 `W_ao_K`，保留 `W_o_K/W_a_K`；使用参数匹配的空输入 | pair 条件知识贡献 |
| S6 | matched text encoder | 若使用 CLIP 文本编码器，保持该 encoder/投影，改用名称或长度匹配无语义描述 | 编码器与参数量变化 |
| S7 | condition head with shuffled K | 保持 G/H 完整结构，打乱知识映射 | 条件头结构 vs 真实语义 |
| S8 | no conditional attr loss | G/H 去掉 `alpha_cond`，其他不变 | 额外辅助监督贡献 |

S1–S5 是附件设计稿的必做项，S6–S8 是为了让因果解释更强。主报告至少完成 UT 的全部 S1–S8；若预算允许，在 MIT/C-GQA 验证关键 S1/S2/S7。所有控制实验使用对应主方法的同一 seed 列表，不按测试集反复筛选。

### 7.3 设计稿中的附加分析逐项实施

| 分析 ID | 输入和运行方式 | 记录的输出 | 完成标准 |
|---|---|---|---|
| P1 K质量 | 固定 E/G，分别 `K_text`、完整 `K_llm`、`K_mix`，统一 λ 调参预算 | K 分布、coverage、CW/OW 四指标、unseen bias | 三版来源可审计；没有完整 LLM 版就标 `N/A` |
| P2 calibration/bias | 保存 seen/unseen 原始 pair score 分布，统一偏置扫描 | AUC 曲线、最佳 bias、seen/unseen 错误率 | evaluator 与主实验一致，不通过 test 选 λ |
| P3 error decomposition | 按最终 pair 预测划分 `pair 正确`、`pair 错且 object 错`、`pair 错且 object 对/attr 错` 三个互斥主类别；独立 object/attr head 的正确性及 prior 改对/改错另做交叉标注 | 各主类别计数、比例、样本 ID；分支与 prior 效应交叉表 | 主类别互斥且总和等于测试样本数；交叉标注不强行计入主类别 |
| P4 noise sensitivity | 对 OE/AE 描述按 10/30/50% 随机替换或删除，固定多 seed mask；可同时测训练期与测试期扰动，分开报 | AUC/HM/Seen/Unseen 随噪声曲线 | 每点至少3个噪声 seed，知识覆盖率记录 |
| P5 描述数 | 每概念取 1/3/5 条，固定选取规则 | 性能、coverage、文本长度、耗时 | 5条仅在真实具有≥5条描述时跑；不能复制/补空行凑5 |
| P6 树深度 | 对真实多级 OI 树使用 depth 1/2/3 | parent 数、覆盖率、性能 | 只有真实存在多级层级才跑；单层知识标 `N/A` |
| P7 失败案例 | 对 H 相对 A/F 的改善、退化、共同错误分层抽样，人工盲审 | 类别、样例 ID、理由、图像使用权限 | 预先定义 taxonomy 与抽样 seed；不只挑成功例子 |
| P8 跨基准迁移 | 固定知识生成 prompt、清洗和模型超参，从 UT 迁移至 MIT/C-GQA | AUC/HM 配对差、coverage | 不用目标测试集反调 prompt；可列为次要分析 |
| P9 效率 | A/E/G/H 同硬件同 batch 同图像 | 每 epoch 时间、推理吞吐、峰值显存、参数量 | 同配置可复现，计时方法一致 |

### 7.4 正式执行次序

1. Phase 0：基线冻结、A 回归、环境和评估锁定。
2. Phase 1：三个数据集各自的知识审计与向量/先验生成；缺一数据集知识就明确缩小可执行范围。
3. UT 单 seed：A–F；只用验证集过 Gate 1。
4. UT 单 seed：G/H、S1–S8；过 Gate 2 和条件性干预。
5. 锁定模型与超参数：输出 `protocol_lock.json` 和 Git commit；以后不得依据测试表现调整。
6. UT 三 seed 全主消融，随后 MIT、C-GQA 三 seed 全主消融；每个 run 结束统一 CW/OW 评估。
7. P1–P9 分析、配对统计、图表和失败案例。
8. 核验所有表格可由 `metrics.json` 与 manifest 自动重新生成，最后写研究结论和限制。

---

## 8. 新程序的命令接口规范

下列命令是**实施后应提供的 CLI 约定**；当前 `CPF_fixed` 只有 `train.py --cfg ... opts` 和 `test.py --cfg ... --load ... opts`，不能直接运行本节的新选项。完成 §3 中脚本与配置后，以下命令才可执行。服务器中的参数名以最终实现保持一致，不能在日志中隐藏覆盖项。

### 8.1 生成/检查知识

```bash
source /data/xhw/miniconda3/etc/profile.d/conda.sh
conda activate cpf
cd /data/xhw/czsl_repro/src/CPF_KCC

python tools/audit_knowledge.py \
  --dataset ut-zappos \
  --data-root /data/xhw/czsl_repro/data/ut-zap50k \
  --knowledge-root /data/xhw/czsl_repro/data/ut-zap50k/knowledge_kcc/v1

python tools/build_knowledge.py \
  --dataset ut-zappos \
  --data-root /data/xhw/czsl_repro/data/ut-zap50k \
  --knowledge-root /data/xhw/czsl_repro/data/ut-zap50k/knowledge_kcc/v1 \
  --encoder glove300 \
  --prior centered \
  --pair-chunk 32
```

MIT 的 dataset/root 设为 `mit-states` / `.../data/mit-states`；C-GQA 设为 `cgqa` / `.../data/cgqa`。初次生成后冻结产物 SHA；变更 prompt、知识条目或编码器必须升版本 `v2`。

### 8.2 单组训练与评估

新程序 `tools/run_variant.py` 应只接受已定义的 A–H 或控制实验 ID，解析配置并保存完整 CLI；有重名 Run ID 时拒绝覆盖。示例：

```bash
export KCC_RUN=/data/xhw/czsl_repro/runs/kcc_v1/ut-zappos/G/seed_3351
mkdir -p "$KCC_RUN"
set -o pipefail

python -u tools/run_variant.py \
  --dataset ut-zappos \
  --variant G \
  --seed 3351 \
  --config config/kcc_ut_zappos.yml \
  --data-root /data/xhw/czsl_repro/data/ut-zap50k \
  --knowledge-root /data/xhw/czsl_repro/data/ut-zap50k/knowledge_kcc/v1 \
  --run-dir "$KCC_RUN" \
  2>&1 | tee "$KCC_RUN/train.log"
```

训练后：

```bash
test -s "$KCC_RUN/checkpoints/best_val_auc.pth"

python -u tools/evaluate_variant.py \
  --run-dir "$KCC_RUN" --setting cw \
  2>&1 | tee "$KCC_RUN/eval_cw.log"

python -u tools/evaluate_variant.py \
  --run-dir "$KCC_RUN" --setting ow \
  2>&1 | tee "$KCC_RUN/eval_ow.log"

sha256sum "$KCC_RUN/checkpoints/best_val_auc.pth" \
  > "$KCC_RUN/checkpoint_sha256.txt"
```

`evaluate_variant.py` 必须从 checkpoint 内的 `config_resolved`、数据/知识哈希和代码版本恢复，拒绝配置不匹配；不能靠当前 shell 的隐式变量重建实验。运行时 `CUDA_VISIBLE_DEVICES=0`，同一张 RTX 3090 顺序执行；tmux 可用 `tmux new-session -s kcc_utz_g_3351` 创建会话，重新进入同一 pane 不需重设变量。

### 8.3 批量运行前的 dry run

实现新 CLI 后先用 `--dry-run` 打印72个主实验的 Run ID、数据、变体、seed 和解析后开关；逐一确认 A–H 表与实际开关完全一致，再启动批量。批量脚本不得自动跳过失败后伪装完成；每个失败保留 exit code、日志和原因。所有命令行和环境信息写入各自 `manifest.json`。

---

## 9. 每个阶段达到什么标准

### 9.1 工程验收

| 级别 | 必须通过的检查 |
|---|---|
| E0 数据 | split 图像可读取、0缺图、类别/组合索引稳定；所有知识文件来源与 SHA 已登记 |
| E1 回归 | A 与 CPF_fixed 同权重输出、CW/OW指标一致；旧逻辑和验证选模不回退 |
| E2 数值 | 所有变体一次前后向无 NaN/Inf；梯度可达新增模块；λ=0/β=0 等价关系成立 |
| E3 候选 | train loss 只用 train pairs，CW/OW mask 与原 evaluator 完全一致；unseen pair 分数 finite |
| E4 资源 | C-GQA 在计划 batch/precision/chunk 下跑完整 epoch；若变更 batch/AMP，另列协议差异 |
| E5 完成 | 每 run 有最佳验证 epoch、checkpoint、CW/OW四指标、配置/代码/知识哈希和完成状态 |

### 9.2 学术判断与统计

主比较：`H−A`、`H−B`、`H−F`、`G−E`、`H−G`，每个数据集 CW/OW 都记录四指标百分点差。三 seed 使用相同 seed 的**配对差**，给平均值±样本标准差和95%配对区间；可通过 seed 级 bootstrap 或配对 t 区间计算，但 `n=3` 的区间很不稳定，建议资源允许时追加到5个 seed。若研究要声称“稳定优于”，至少主要 AUC/HM 比较的均值为正、多个数据集趋势一致、区间尽量排除0，并呈现全部 seed。不要把同一测试集图像的 bootstrap 当作独立训练 seed 的替代。

如果 H 只比 A 好、对 F 或 G 没有稳定增益，只能支持“组合有效”或“新增 head/logic 不确定”，不能宣称知识条件机制胜过 naive combination。若 `G-randomK`≈G 或 shuffle 文本≈真文本，应缩小知识语义贡献表述。若 OW Unseen 明显下降、但 CW 提升，应说明开放空间代价。论文外部 target 值属于不同代码/知识/硬件协议时，只作参照，不替代本实验内部配对比较。

### 9.3 基线参照值（百分点）

| 方法 | 数据集 | CW AUC/HM/Seen/Unseen | OW AUC/HM/Seen/Unseen |
|---|---|---|---|
| CPF 论文 | MIT-States | 11.2 / 26.8 / 41.3 / 34.8 | 4.4 / 15.1 / 40.8 / 14.4 |
| CPF 论文 | UT-Zappos | 41.4 / 55.7 / 66.4 / 71.1 | 31.2 / 47.6 / 64.6 / 56.1 |
| CPF 论文 | C-GQA | 8.2 / 23.9 / 39.6 / 23.5 | 2.1 / 9.5 / 38.4 / 6.8 |

这些是外部参考数值。新方法论文主表应列 A–H 的本地同协议结果与差值，同时说明本地 A 相对 CPF 论文的差距。参考来源：[CPF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf)、[LOGICZSL CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf)；数值曾在既有复现方案中核对。

---

## 10. 实验记录表：可直接复制填写

### 表 R0：知识与协议总账（每 dataset/knowledge version 一行）

| dataset | split_sha256 | image_missing | N_attr | N_obj | train/val/test_n | knowledge_source | raw_sha256 | prompt_version | encoder | pair_coverage | OE/AE_coverage | K_mean/std/min/max | prior_mode | leakage_audit | reviewer |
|---|---|---:|---:|---:|---|---|---|---|---|---:|---:|---|---|---|---|
| UT-Zappos |  |  | 16 | 12 | 22998/3214/2914 |  |  |  | glove300 |  |  |  | centered |  |  |
| MIT-States |  |  | 115 | 245 | 30338/10420/12995 |  |  |  | glove300 |  |  |  | centered |  |  |
| C-GQA |  |  | 413 | 674 | 26920/7280/5098 |  |  |  | glove300 |  |  |  | centered |  |  |

### 表 R1：运行登记（每训练 run 一行）

| run_id | dataset | variant | seed | code_commit | knowledge_sha | config_sha | backbone_sha | epochs | batch | precision | lr | λ | β | γ | α_cond | best_val_epoch | best_val_auc | checkpoint_sha | status | reason |
|---|---|---|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 表 R2：逐 epoch（每 run 每 epoch 一行）

| run_id | epoch | train_total | comp | obj | attr | cond_attr | oi | oe | ae | val_auc | val_hm | val_seen | val_unseen | lr_groups | gamma | gpu_peak_MiB | sec | is_best | anomaly |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### 表 R3：主结果 A–H（每数据集、Setting、seed 各填8行）

| dataset | setting | seed | variant | test_AUC% | HM% | Seen% | Unseen% | ΔAUC_vs_A | ΔHM_vs_A | ΔUnseen_vs_A | best_epoch | checkpoint_sha | knowledge_version |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UT-Zappos | CW | 3351 | A |  |  |  |  | 0 | 0 | 0 |  |  |  |
| UT-Zappos | CW | 3351 | B |  |  |  |  |  |  |  |  |  |  |
| UT-Zappos | CW | 3351 | C |  |  |  |  |  |  |  |  |  |  |
| UT-Zappos | CW | 3351 | D |  |  |  |  |  |  |  |  |  |  |
| UT-Zappos | CW | 3351 | E |  |  |  |  |  |  |  |  |  |  |
| UT-Zappos | CW | 3351 | F |  |  |  |  |  |  |  |  |  |  |
| UT-Zappos | CW | 3351 | G |  |  |  |  |  |  |  |  |  |  |
| UT-Zappos | CW | 3351 | H |  |  |  |  |  |  |  |  |  |  |

OW 与 MIT/C-GQA 按同一格式复制；`Δ` 均为**复现−同 seed 本地 A**，单位百分点。

### 表 R4：核心配对证据（三数据集×CW/OW×seed）

| dataset | setting | seed | H−A AUC/HM | H−B AUC/HM | H−F AUC/HM | G−E AUC/HM | H−G AUC/HM | H−A Unseen | 结论 |
|---|---|---:|---|---|---|---|---|---:|---|
|  |  |  |  |  |  |  |  |  |  |

### 表 R5：多 seed 汇总（平均值±样本 SD + 配对区间）

| dataset | setting | contrast | seeds | ΔAUC mean±SD | ΔAUC 95% CI | ΔHM mean±SD | ΔHM 95% CI | ΔSeen mean±SD | ΔUnseen mean±SD | 支持命题? |
|---|---|---|---|---|---|---|---|---|---|---|
|  |  | H−A |  |  |  |  |  |  |  |  |
|  |  | H−F |  |  |  |  |  |  |  |  |
|  |  | G−E |  |  |  |  |  |  |  |  |

### 表 R6：控制实验

| dataset | setting | seed | control_ID | reference_variant | control_AUC/HM/S/U | reference_AUC/HM/S/U | paired_ΔAUC/HM | parameter_match | description_count_match | knowledge_coverage | 解释 |
|---|---|---:|---|---|---|---|---|---|---|---:|---|
|  |  |  | S1 |  |  |  |  |  |  |  |  |

### 表 R7：附加分析与资源

| analysis_ID | dataset | setting | variant | seed | factor | level | AUC | HM | Seen | Unseen | train_hours | eval_img_per_sec | GPU_peak_MiB | params_M | artifact |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P4 |  |  | H |  | noise_fraction | 0.1 |  |  |  |  |  |  |  |  |  |

### 表 R8：错误案例审计

| dataset | setting | image_id | GT_attr | GT_obj | A_pred | F_pred | H_pred | A_obj_correct | H_obj_correct | prior_help_or_hurt | taxonomy | annotator | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

所有表格的 `AUC/HM/Seen/Unseen` 用百分点。日志中 `0.3651` 填 `36.51`，不能把 validation 最优 AUC 当作 test AUC。`best_seen` 与 `best_unseen` 可能对应 evaluator 偏置扫描的特定口径，HM 直接抄 evaluator 输出，不手算重建。

---

## 11. 测试、排错和复现归档

### 11.1 必要的数值回归（建议自动化）

1. 固定随机输入与 CPF_fixed 权重：A 分数及 loss 在允许的浮点容差内一致。
2. `λ=0`：D=A、E=C；`β=0`：B=A、F=E、H=G（注意新增未用参数不应扰动共同模块初始化；需先固定模块初始化顺序或加载同一共同权重）。
3. uniform prior：`pair_logits_final−pair_logits_base` 在所有 pair 上相同，指标完全一致。
4. 随机打乱 `pair_index`：通过按名称重新 gather 后分数不变；不允许索引静默错位。
5. OI/OE/AE 分支独立开启：loss finite、非零梯度、输入维度及三描述顺序符合 schema。
6. `delta_cond`：换图/换对象/换知识后按预期变化；断开知识输入后 `G` 退回预定义对照。
7. C-GQA 窄 chunk 与宽 chunk：同权重/同输入得到近似相同分数，避免 chunk 边界改变预测。
8. 模型在 train pair 上的 CE 标签均能映射到合法列；验证/测试没有任何训练标签泄露。

### 11.2 常见失败的定位

| 现象 | 首先检查 |
|---|---|
| A 与 CPF_fixed 不同 | 模型 import、初始化、候选顺序、baseline score 是否变过、随机/确定性设置 |
| D 没影响 | K 是否近常数、λ 是否真的传到 final score、候选 gather 是否错序 |
| G≈E 且 gamma≈0 | 条件头梯度、初始化/学习率、pair K 有效性、辅助 loss |
| H≈G 且 logic loss 非零 | loss 是否进入最终 `loss_total`、beta 重复/过小、gradient norm |
| G/H 只提升 seen | pair 描述覆盖偏差、K 对 seen 偏置、验证目标、OW mask |
| C-GQA OOM | 是否构造 `[B,N,M,D]`，是否对全量 pairs 训练 CE，object/pair chunk、缓存 dtype |
| Test 比 Val 大幅异常 | checkpoint 加载、val/test候选定义、知识产物版本、数据图像、eval前 `model.eval()` |

### 11.3 每次运行的完整归档

除 §1.2 的输出外，保存 `git diff` 或源码 snapshot、`conda env export --no-builds`、`pip freeze`、`nvidia-smi -q`、预训练 ViT 权重 SHA、GloVe SHA、知识原件/生成脚本 SHA、CSV 表格和所有训练/验证/测试原始日志。压缩包中不必复制全部图片，但必须有数据 manifest 和 split 哈希。任何修复后重跑的实验用新 Run ID，不能原地覆盖旧失败记录。

---

## 12. 最终论文叙事与限制

完整证据链应按照 `A → B → E/F → G/H → 负对照 → 三数据集多 seed → 错误/效率分析` 报告。结论应分别回答：

1. 描述、compatibility 和外部 logic 各贡献多少？
2. G/H 的图像相关知识条件项是否比静态 prior 与 naive F 更有用？
3. 收益是否在 OW Unseen 上保留，是否伴随 Seen 下降？
4. 打乱知识或长度匹配模板后收益是否消失？
5. 性能、显存和训练时间成本是多少？

若知识来源为人工重建/LLM 生成，应在方法、实验设置和限制中明确；这套新方法可以是独立研究实验，但不能宣称拿到了原 LOGICZSL 作者未公开的原始知识文件。C-GQA 若无法得到完整 LLM compatibility，就以 text-only K 完成主结果，LLM/mix 留空并说明范围。树深度与5条描述若数据源本身不支持，标为 `N/A`，不虚构实验。

参考：用户提供的 [实验设计稿](<C:/Users/llenovo/Desktop/ZSCL实验设计0601.md>)、本地 [CPF/LOGICZSL 修补手册](<C:/Users/llenovo/Documents/Codex/2026-09-28/new-chat/outputs/CPF_LOGICZSL_baseline_repair_manual.md>)、[已有复现方案](<C:/Users/llenovo/Documents/Codex/2026-09-28/new-chat/outputs/CPF_LOGICZSL_paper_reproduction_experiment_plan.md>)、CPF 与 LOGICZSL 正式论文 PDF（链接见 §9.3）。
