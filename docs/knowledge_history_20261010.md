# CZSL 知识恢复、构建与处理完整档案：来源、文件与作用

整理日期：2026-10-10。范围：已保存的原方案、源知识、构建代码、服务器回传 manifest、实验资产与知识审查结果。本文独立于 [全部实验总档案](<../output/截至20261010_CZSL全部实验_公式修正_决策与数据分析总档案.md>)，只完整追踪“知识从哪里来、怎样成为模型输入、哪些部分实际被用过”，不重复所有训练主表。

本文是已有资产的整理与来源核查，没有重新生成知识、改写已有字典、启动训练或改变旧实验门槛。文件位置以本次实际存在的本地副本为准；Linux 路径是历史运行位置，不代表本次远程核实了服务器文件仍然存在。

## 1. 先明确：恢复了什么，没有恢复什么

现在具有的是非官方、可追溯的知识重建，以及 CPF 对这些知识的处理实现；不是找回了 LogiCZSL 原作者遗失的历史数据。

| 项目 | 当前事实 | 不应使用的表述 |
| --- | --- | --- |
| LogiCZSL 原作者历史 GPT 输出、原始知识 JSON | 未恢复；论文无法确定每一条历史输出 | “原作者知识已经精确恢复” |
| OI/OE/AE 的方法与接口 | 按论文思路重建，并适配 CPF | “原论文全部结果已经复现” |
| 早期 UT 源知识 | 2026-09-29 的论文引导语义重建，4 个父类 | “官方 GPT-4 知识” |
| MIT-States / C-GQA 源知识 | WordNet 3.0 首义及规则模板草稿 | “LLM 全量审核知识” |
| UT-v2 | 在固定 TRAIN 拼图上审查、改写描述；不是人类专家全量验收 | “所有材质描述均可由图像确定” |
| B/F 新逻辑知识 | 2026-10-08 另行按论文方法重建，5 个父类 | “UT-v2 的逻辑损失版本” |
| GloVe、CLIP 知识张量 | 不同编码与聚合方式得到的模型输入 | “同一个 bank 换了文件名” |
| 支持掩码、图文分数、教师概率 | 由既有知识与 TRAIN/VAL 特征派生的证据 | “新的外部语义真值” |
| 原 0601 的 G/H | 已有后续执行实现；本档案未核实新的 G/H 完整训练产物 | “G/H 已取得收益” |

方法核对、原代码问题与论文来源见 [LogiCZSL 方法重建手册](<../output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md>)。这份旧手册中“尚未训练”等文字是发布时状态；本档案和全部实验总档案记录的是后续已回传证据。

直接方法来源是 [LogiCZSL 原论文 PDF](<../archive_inputs/referenced_sources/CZSL/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf>) 的 §3.2–3.4、Fig.4 与 Eq.6/9/12/13；代码参照来源是 [本地作者代码副本](<../archive_inputs/referenced_sources/CZSL/CZSL_code/LogiCZSL_copy.tar.gz>)。这些原件此前已做方法核对，本次只整理和确认文件身份，不重复解析论文或运行作者代码。知识 JSON 的实际构建来源则必须分别写作“assistant 论文方法重建”“WordNet 规则草稿”或“TRAIN 审查后的修订”，不能只把来源写成论文标题。

## 2. 从原方案到实际资产：知识不是一个文件

[最初 0601 方案](<../archive_inputs/referenced_sources/0601origin/ZSCL实验设计0601.md>) 预期包含 OI、OE、AE、对象—属性兼容性先验，并有 Text、Calibration、Feature、Logic 四类用途。实际推进时，先完成了能对接原 CPF/GloVe 的确定性预处理，之后才增加视觉支持筛选和逻辑 grounding。

```text
数据集 primitive 标签 / 精确 split 元数据
  ├─ UT 手工语义重建、WordNet 规则草稿 → v1 OI/OE/AE
  │    ├─ 词 / 字符 TF-IDF → lexical K（候选，未用于主训练）
  │    └─ GloVe → 名称 / primitive K / 全组合模板块 → template K → centered prior
  ├─ v1 UT 描述 + 固定 TRAIN 拼图审查 → UT-v2 AE/OE
  │    ├─ 同一 GloVe 构建器 → v2 cache → EKR 四槽 bank
  │    │    ├─ 均值 / 中心化残差 / 固定错配 → EKR、条件属性头
  │    │    └─ TRAIN 视觉特征 → SG-EKR support mask
  │    └─ 固定 prompt + 冻结 CLIP → 图文分数 / 教师概率 → TRAIN 逐句审查
  └─ 2026-10-08 独立论文方法重建 → B/F knowledge.json
       └─ GloVe 父类 / OE / AE 独立节点 → logic bank → CPF 三项逻辑损失
```

学习到的视觉投影、路由参数、head checkpoint 在这些固定资产之外。它们可能改变知识如何被读取，不会自动改写 JSON，也不等于构建了新的外部知识。

## 3. 知识版本与时间顺序

| 时间 / 阶段 | 构建或处理动作 | 得到的知识资产 | 版本边界 |
| --- | --- | --- | --- |
| 2026-09-29 | 根据论文结构与 UT 标签重建 OI/OE/AE | 最早 UT 三个 JSON，4 个父类 | 非官方；未找回历史 GPT 输出 |
| 后续早期准备、至 2026-10-02 | 扩展三数据集；冻结精确标签与来源记录；抽样后 bulk trust | draft / reviewed_v1 / trusted_v1；lexical 候选 | 批量放行不等于逐条人工语义审核 |
| 至 2026-10-05 | GloVe 编码 primitive 与全组合模板；构建 K、P；开展 A/C/D/E | `cache_trusted_v1/<dataset>` | 主训练用 template，不是 lexical 或旧候选掩码 |
| 2026-10-05 起 | 保存名称＋三描述的槽、均值、中心化残差与错配 | EKR description bank | 从已冻结 C cache 扩展，检查聚合可回放 |
| 2026-10-06 | 固定 TRAIN 拼图审查；84 句保留 30、改写 54 | UT-v2 AE/OE、`item_review.json`、审图证据 | 标签与 OI 语义不变；v1 留存 |
| 2026-10-07 至 08 | 在 v2 bank 上做名称锚点、对齐、预算与视觉支持处理 | EKR 参数与 SG support mask | 基础描述未再次改写 |
| 2026-10-08 | 独立重建论文逻辑节点，并适配 CPF B/F | 5 父类 `knowledge.json` 和 logic bank | 不覆盖 v1/v2；F 同时保留旧 v1 Text/Prior |
| 2026-10-09 | 固定 C_text 底座，缓存视觉与分数；训练条件 / generic 小头 | StageA、条件输入控制、固定混合读出 | 缓存不叫源知识；25% 不叫知识权重 |
| 2026-10-10 | 将相同 v2 原文组织为 CLIP prompt，做冻结图文探针 | prompts、text features、image features、q | 没有生成新的 AE/OE；无教师训练 |
| 2026-10-10 | 重编码单槽文本，在 TRAIN 上检查句子位置与适用性 | 单句结果、固定拼图、空白人审表 | 审查未产出已验收的 v3 字典 |
| 2026-10-10 后续实现 | 回到 0601，新增 Full Feature 代码对接旧 pair block | G/H 执行包与受控资产变体 | 实现存在不等于训练已完成 |

## 4. 最早的 UT 恢复：原始语义 JSON 与旧 compatibility 掩码

直接来源记录：[最早 UT / generation_manifest.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/generation_manifest.json>)；构建规则：[最早 UT / build_compatibility.py](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/build_compatibility.py>)。

早期 UT 的 OI/OE/AE 是按论文知识树结构、UT 的 16 个属性和 12 个对象标签进行的语义重建。对象共 36 条 OE，属性共 48 条 AE。最早原文文件为 [最早 UT / oi_dict.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/oi_dict.json>)、[最早 UT / oe_dict.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/oe_dict.json>)、[最早 UT / ae_dict.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/ae_dict.json>)。没有可用于宣称“逐字复现原作者”的完整历史 prompt、raw GPT response、模型版本或采样参数。

旧 [最早 UT / compatibility.csv](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/compatibility.csv>) 的含义必须单列。它由 train、val、test 三份 pair 文件的并集构造；旧脚本的矩阵轴为 **attribute × object，16×12**：

$$
\begin{aligned}
\mathcal{C}_{\mathrm{split}}
  &=\mathcal{C}_{\mathrm{train}}
    \cup\mathcal{C}_{\mathrm{val}}
    \cup\mathcal{C}_{\mathrm{test}},\\
C^{\mathrm{mask}}_{a,o}
  &=\mathbf{1}\!\left[(a,o)\in\mathcal{C}_{\mathrm{split}}\right].
\end{aligned}
$$

这是数据集声明的 closed-world 候选集合掩码，不是现实世界可组合性判断。未出现在并集中的组合不能据此标成“语义不可能”；其轴也与后续 object×attribute 的 K 相反。

后续知识构建明确禁止把这个 split-derived mask 当作语义 prior。早期 A/C/D/E、B/F 的静态校准使用另行构建的 template K；不能根据旧文件存在，就说这些训练使用了 `compatibility.csv`。也不能把最早 README 中留下的旧生成命令当成现在推荐的知识构建路线。

## 5. 三数据集 v1：具体构建来源

### 5.1 原始标签来自哪里

源生成器 [generate.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/generate.py>) 从 `compositional-split-natural` 的 train/val/test pair 清单提取属性与对象并去重排序。UT/MIT 的初始来源是本地 `compositional_split_natural.tar.gz`；C-GQA 初始来源在 manifest 中记录为更新后的 pair-list 镜像。脚本也支持 `--source-data-root` 使用精确服务器 split 文件。

后续 GloVe 构建器再次调用实际项目 `CompositionDataset.parse_split()` 检查标签顺序；不实例化图片读取流程。初始镜像的来源记录与服务器实际训练轴校验是两件事，不能省掉后者。

使用 test pair 元数据确定公开 primitive 词表，不等于读取 test 图片；但如果把 test pair 存在性写入语义兼容性先验，就超出了这个用途。构建器将 split membership 仅用于标签对齐和覆盖审计。

### 5.2 UT：复用既有人工式重建；MIT/C-GQA：WordNet 自动草稿

| 数据集 | 属性 | 对象 | OI 父类 | OE 句数 | AE 句数 | 逐条来源记录 |
| --- | --- | --- | --- | --- | --- | --- |
| ut-zap50k | 16 | 12 | 4 | 36 | 48 | manual_reconstruction_unreviewed：28 |
| mit-states | 115 | 245 | 19 | 735 | 345 | wordnet_auto_unreviewed：360 |
| cgqa | 413 | 674 | 23 | 2022 | 1239 | label_template_only：6<br>wordnet_auto_unreviewed：1081 |

WordNet 数据包的归档 SHA256 为：

```text
cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59
```

这是现有 manifest 记录的源包身份；本文没有重新下载 WordNet 或重新执行生成器。

对象查询先将点和下划线转为空格，再转换为 WordNet 短语；查 noun sense，失败时尝试末尾中心词。属性按 adjective、adjective satellite、verb、noun 顺序查询。找到多个义项时，脚本选首义，不做任务特定词义消歧。

| 产物 | WordNet 有命中时 | 无命中时 | 主要风险 |
| --- | --- | --- | --- |
| OE 第 1 条 | 标签＋synset definition | `A visible object labeled ...` | 首义未必是图像里的概念 |
| OE 第 2 条 | `is a kind of`＋上位类 | `The object category is ...` | 不是直接可观察视觉特征 |
| OE 第 3 条 | 所属 lexname 概念组 | `An instance of ... category` | 可能只是抽象类别 / 占位模板 |
| OI 父类 | noun lexname 去掉 `noun.` 前缀 | `unmapped_object` | 词典粗分类，不是作者的原分组 |
| AE 第 1 条 | 标签＋definition | `An object described as ...` | 非形容词义可能不适合材质/状态 |
| AE 第 2 条 | similar/attribute/also-see 相关词，否则标签模板 | observable-state 模板 | 同义、相关与可见性不是一回事 |
| AE 第 3 条 | `visual or semantic state ...` 模板 | `property or material named ...` | “visual” 字样不保证视觉可辨 |

`entry_provenance.csv` 保存 kind、label、query、synset、sense_count、中心词 fallback、parent、quality、review_reason。MIT 的 360 条均记为 `wordnet_auto_unreviewed`；C-GQA 有 1,081 条该类记录和 6 条 `label_template_only`。这些是逐 primitive 的记录数，不是描述句数。

例：C-GQA 的 `air` 记录了 `air.n.01`、9 个义项、parent=`substance`，并注明首义多义风险。工程上能编码、JSON 覆盖完整，不会消除这个语义风险。

### 5.3 draft → reviewed → trusted 的真实含义

draft 的状态为 `complete_coverage_unreviewed_draft`。后续 trusted manifest 记录 `fixed_seed_sample_audit_then_bulk_trust`，seed=20261002，reviewer=`xhw`，`item_level_manual_review=false`，状态为 `bulk_trusted_unverified`。

`review_status.csv` 中的 `approved` 是允许进入构建的工程标记。它不证明每条描述经过人类专家检查，更不证明知识提高了性能。manifest 的 `compatible_with_formal_results=false` 也不能被改写成“官方结果兼容”：这些非官方资产应独立命名和报告。

## 6. 源知识文件：内容、来源和作用

| 文件 | 内容 / 单位 | 相关构建来源 | 实际作用 |
| --- | --- | --- | --- |
| `attributes.txt` | 固定属性标签顺序 | split 的 primitive 词表＋项目 parse_split | 对齐所有属性行；不是描述文本 |
| `objects.txt` | 固定对象标签顺序 | 同上 | 对齐所有对象行 |
| `oi_dict.json` | parent → child objects | UT 重建 / WordNet lexname | 早期结构校验；B/F 另版本中才形成逻辑层级约束 |
| `oe_dict.json` | 每对象 3 条描述 | UT 重建 / WordNet / v2 TRAIN 审查 | 构建对象 K、对象描述槽或 prompt |
| `ae_dict.json` | 每属性 3 条描述 | 同上 | 构建属性 K、属性描述槽或 prompt |
| `entry_provenance.csv` | primitive 级出处与风险 | 初始生成器 | 区分词典命中、首义、占位、手工重建；早期 cache 未把它作为独立冻结语义输入 |
| `review_status.csv` | 每 primitive 的状态 | 审查 / 批量放行流程 | 构建前工程准入，不等于语义真值 |
| 源 `manifest.json` | 版本、来源、计数、源文件 SHA | 源生成 / 冻结步骤 | 可追溯性与输入完整性，不进入模型前向 |
| `K_oa_text_lexical.csv` | object×attribute 的词面相似度 | 词 / 字符 TF-IDF 构建 | 可选 prior；主训练未选用 |
| `K_oa_text_report.json` | lexical 数值统计与警告 | 同一 lexical 脚本 | 质量审计，不是模板 K 的报告 |
| v2 `item_review.json` | 逐句保留/改写、支持/例外、来源类型 | 固定 TRAIN 拼图审查 | 解释描述为何改变与哪些仍不可靠 |
| `review_evidence/*` | 拼图、样本身份、路径别名、原文本 | TRAIN 固定抽样 | 审查证据，不直接作为模型知识输入 |

UT-v1 三个源 JSON：[v1 ut-zap50k / oi_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/oi_dict.json>)、[v1 ut-zap50k / oe_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/oe_dict.json>)、[v1 ut-zap50k / ae_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/ae_dict.json>)。MIT/C-GQA 的全部源文件在附录文件索引中逐个给出。

每次构建都必须把“目录名字”“manifest 内容”“实际文件 hash”一起保存。目录名含 `trusted`、`reviewed`，不替代内部的真实状态字段。

## 7. GloVe 文本处理：名称、描述与 primitive K

实际冻结构建器：[冻结 GloVe / build_knowledge.py](<../reproduce_0601_20261010/reference_source/tools/build_knowledge.py>)。名称别名来源：[CPF 名称别名 / word_embedding_utils.py](<../reproduce_0601_20261010/reference_source/models/word_embedding_utils.py>)。编码器固定为 GloVe 6B 300d，float32，完整文本身份为：

```text
91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
```

构建时一次扫描 GloVe，仅保留所需 token，但 SHA 对完整文本字节计算。因此“只留部分 token”不等于无法确认完整编码器来源。

本地可用原始词向量压缩包为 本地 GloVe 6B 原始 ZIP（原存储位置：`C:/Users/llenovo/Desktop/CZSL/CZSL_code/downloads/glove.6B.zip`）。ZIP 的 hash 与内部 `glove.6B.300d.txt` 的 hash 是两种不同身份；本文保留包的实际文件索引/指纹，但没有重新解压并扫描全部成员。WordNet 的许可随初始生成包保存为 [WordNet 原始许可](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/WORDNET_LICENSE.txt>)；本档案以源 manifest 的 WordNet 包指纹追溯，不声称当前索引已经备份了整个词典 corpus。

### 7.1 名称 token 与描述 token 的规则不同

| 输入 | 实际处理 | 示例 / 注意 |
| --- | --- | --- |
| primitive 名称 | 用原 CPF `custom_map`；lowercase；有下划线才拆 token 并应用 token alias | `Boots.Ankle → ankle_boots` |
| 其他名称别名 | 保留原初始化语义，不能自行换新同义词 | `Nubuck → grainy_leather`；`Patent.Leather → shiny_leather` |
| GloVe 描述 | `re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())` | 不删除停用词、不做词形还原 |
| 名称 OOV | 构建失败 | 不能悄悄用随机向量替代 |
| 描述 token OOV | 跳过并记录 | 可编码覆盖率不代表整句语义完整 |
| 描述全 OOV / 近零 | v1 builder 退回名称向量并计数 | v2 外层检查禁止任何回退 |

B/F 节点 builder 使用另一条 `[a-z]+` 正则并检查覆盖率；CLIP prompt 还使用显式自然语言名称表。三者不能声称 token 化完全一样。

### 7.2 名称＋三描述聚合的确切公式

令 $g(t)$ 为 300 维词向量，$\mathcal{N}(v)=v/\lVert v\rVert_2$。对 primitive $c$，名称 token 序列为 $T_c^{\mathrm{name}}$，第 $r$ 条描述中可编码 token 序列为 $T_{c,r}^{\mathrm{IV}}$。所有 token 序列保留重复，求和按出现次数累加，基数表示序列长度：

$$
\begin{aligned}
n_c
 &=\frac{1}{\lvert T_c^{\mathrm{name}}\rvert}
   \sum_{t\in T_c^{\mathrm{name}}}g(t),\\
u_{c,0}&=\mathcal{N}(n_c),\\
u_{c,r}
 &=\mathcal{N}\!\left(
   \frac{1}{\lvert T_{c,r}^{\mathrm{IV}}\rvert}
   \sum_{t\in T_{c,r}^{\mathrm{IV}}}g(t)\right),
   \quad r=1,2,3,\\
\mu_c&=\frac14\sum_{r=0}^{3}u_{c,r},\\
w_c^{\mathcal{K}}&=\mathcal{N}(\mu_c).
\end{aligned}
$$

名称占 1 个槽，三条描述各占 1 个槽；不是把全部句子连起来按 token 一次平均。句子先单位化、四槽平均后再单位化；这与原 CPF 原始名称均值 $n_c$ 的尺度不同。

实际消费代码 [CPF Text/Prior 前向 / kcc_cpf.py](<../reproduce_0601_20261010/reference_source/models/kcc_cpf.py>) 使用固定 `W_o_K/W_a_K` 替代原名称表的读取：组合类文本由两个向量拼接后经原 `compose` 模块生成；对象文本参与原文字引导视觉选择；两表还作为对象/属性分类权重。固定输入本身不优化，原视觉、compose 等模块继续按 CPF 协议学习，因此 Text K 并非只在输出分数上加一个常数。

因此 `C_text` 相对 `A_raw` 的变化同时包括语义内容与归一化/聚合。后续 `A_unit` 和 `C_shuffle` 对照就是为拆分这些因素；不能把所有差值归因于“描述更有知识”。

## 8. 全组合模板块：怎样构建，为什么覆盖所有候选

固定三个模板：

```text
a {object} that is {attribute}
the {attribute} {object}
{attribute} appearance of {object}
```

对全部对象—属性笛卡尔积建立模板，不区分 seen/unseen。UT 是 192 个组合，MIT 是 28,175 个，C-GQA 是 278,362 个；不使用“只有训练组合才有描述，未见组合置零”的构建方式。

实际 GloVe 计算使用名称别名 token 和三个模板的 filler token；**这里没有加入 pair-level LLM 描述，也没有把 OE/AE 直接附在模板里**。OE/AE 通过 primitive K 影响后面的相似度参照。

令 $T_{o_i}$、$T_{a_j}$ 为原名称规则产生的 token，$F_\ell$ 为第 $\ell$ 个模板 filler：

$$
\begin{aligned}
v^{(\ell)}_{i,j}
 &=\frac{
     \sum_{t\in T_{o_i}}g(t)
     +\sum_{t\in T_{a_j}}g(t)
     +\sum_{t\in F_\ell}g(t)}
    {\lvert T_{o_i}\rvert+\lvert T_{a_j}\rvert+\lvert F_\ell\rvert},\\
b_{i,j}
 &=\mathcal{N}\!\left(
     \frac13\sum_{\ell=1}^{3}\mathcal{N}(v^{(\ell)}_{i,j})
   \right).
\end{aligned}
$$

模板向量是词袋均值，不具有 CLIP 文本编码器的句序建模能力。存为 `W_ao_K/obj_00000.pt` 等对象分块，每块是 attribute×300。早期 A/C/D/E 中它们用于生成 K，而没有直接进入可训练 Feature head；新 G/H 实现才增加这一消费路径，见第 17 节。

## 9. 静态 template K 与 centered prior：不是概率、不是图

### 9.1 模板相似度 → min-max K

$$
\begin{aligned}
p_{i,j}&=\mathcal{N}\!\left(w_{o_i}^{\mathcal{K}}+w_{a_j}^{\mathcal{K}}\right),\\
R_{i,j}&=b_{i,j}^{\mathsf{T}}p_{i,j},\\
K^{\mathrm{template}}_{i,j}
 &=\frac{R_{i,j}-\min R}{\max R-\min R}.
\end{aligned}
$$

其中 min/max 为整个 object×attribute 矩阵的全局统计。若 raw range 或总体标准差小于 $10^{-8}$，模板 K 全部设置为 0.5。`K_oa_text_raw.pt` 存 R，`K_oa_text_template.pt` 存 min-max 后的模板 K；`K_oa_text.pt` 存本次实际选择的 K。

这不是每行和为 1 的概率，不是经图像真值校准的可靠 feasibility，也不是 OI 邻接图。将矩阵称为“文本候选 prior”比“真实知识概率”更准确。

### 9.2 K → 标准化 P → 组合分数

$$
\begin{aligned}
\bar K&=\frac1{NM}\sum_{i=1}^{N}\sum_{j=1}^{M}K_{i,j},\\
\sigma_K^2
 &=\frac1{NM}\sum_{i=1}^{N}\sum_{j=1}^{M}(K_{i,j}-\bar K)^2,\\
P_{i,j}&=\frac{K_{i,j}-\bar K}{\sigma_K},\\
s_{a_j,o_i}(x)
 &=s_{a_j,o_i}^{\mathrm{CPF}}(x)+\lambda P_{i,j}.
\end{aligned}
$$

若 $\sigma_K<10^{-8}$，P 为零。实际代码用 `std(unbiased=False)`，非退化时没有再加 epsilon；不能拿原方案的示例 `std()` 或“总是加 epsilon”替代运行事实。P 可以为负，整体均值约 0、总体标准差约 1。

原 legacy 实现训练时将 prior 加在温度缩放后的 composition logits，评价时则加在原 composition＋attribute＋object 的未缩放组合分数上；同样的 λ 在两处的相对强度不能简单看作相同。它既进入训练组合 CE，也进入评价，不是所有相关实验都“只做后处理”。本文用上式表达加法接口，不假定 legacy 分数已是统一标定的 log probability。

全局 min-max 是正仿射变换，非退化时继续做总体标准化，等价于直接标准化 R（除浮点误差）。min-max 不会创造额外语义信息。

UT 轴固定：object 0=`Boots.Ankle`，attribute 6=`Leather`；对应 `K[0,6]`。候选身份常写为 `(attribute, object)`，查 K 时需要交换索引位置。UT-v2 的 P 范围为 −2.663768 至 2.294513，均值约 −2.98e−8、总体标准差 1；这份 v2 文件被构建出来，不代表 v2 C_text/后续所有 EKR 都启用了它，实际 λ=0 的路径不加 prior。

### 9.3 可选 lexical K 与未构建的 LLM K

[build_text_prior.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/build_text_prior.py>) 以对象名称＋OE、属性名称＋AE 为文档，删除指定英文停用词，计算词级 TF-IDF 与带词边界标记的字符三元组 TF-IDF：

$$
K^{\mathrm{lexical}}_{i,j}
 =0.75\,\operatorname{cos}(v_{o_i}^{\mathrm{word}},v_{a_j}^{\mathrm{word}})
  +0.25\,\operatorname{cos}(v_{o_i}^{\mathrm{char}},v_{a_j}^{\mathrm{char}}).
$$

TF 使用 $1+\log\mathrm{count}$，IDF 使用 $1+\log((D+1)/(\mathrm{df}+1))$，每个文档向量单位化。CSV 值保留 8 位小数并限制在 [0,1]。其 tokenizer/停用词处理与 GloVe builder 不同。

只有显式选择 `--prior-source lexical`，构建器才把它作为 `K_oa_text.pt` 的来源，再使用相同 centered 转换。已完成主训练的 `prior_source=template`；本文没有找到支持“已完成 lexical K 主训练”的证据。

原方案允许 `K_oa_llm.pt` 和 `K_oa_mix.pt`，但当前没有可核实的可靠 LLM feasibility 资产及相应完整实验。不能以全零矩阵或 split 并集冒充 LLM K，再混合后报成真实语义知识。

### 9.4 已保存编码统计

| 资产 | 模板 K 均值 | 模板 K 总体标准差 | 描述 token OOV | 回退句数 |
| --- | --- | --- | --- | --- |
| ut-zappos-v1 | 0.508405 | 0.244008 | 0.361011% | 0 |
| mit-states-v1 | 0.722081 | 0.120815 | 0.259794% | 0 |
| cgqa-v1 | 0.756356 | 0.092073 | 0.337334% | 0 |
| UT-v2 | 0.537236 | 0.201683 | 0.283286% | 0 |

低 OOV、0 回退只表明编码可进行。K 全局 min-max 后都覆盖 [0,1]，不是“物理兼容性概率已经标定”。三个数据集的 K 分布不同，也不意味着其知识质量可以直接按标准差排序。

## 10. cache 文件全清单与用途

设 N=对象数、M=属性数、d=300。UT 为 N=12、M=16。

| 输出文件 | 形状 / 内容 | 生成来源 | 消费者 / 用途 |
| --- | --- | --- | --- |
| `labels.json` | `attrs` / `objs` 顺序 | 实际 dataset parse_split | 所有表的身份、轴与候选对齐 |
| `W_a_name_raw.pt` | M×300 | 原 CPF 名称 token 均值 | A_raw 与原初始化一致性检查 |
| `W_o_name_raw.pt` | N×300 | 同上 | 同上 |
| `W_a_name_unit.pt` | M×300 | raw 的单位化 | A_unit 控制、名称锚点 |
| `W_o_name_unit.pt` | N×300 | 同上 | 同上 |
| `W_a_K.pt` | M×300 | 名称＋3 条 AE 的单位槽聚合 | Text K 属性表；新 G/H 属性输入 |
| `W_o_K.pt` | N×300 | 名称＋3 条 OE 的单位槽聚合 | Text K 对象表；新 G/H 对象输入 |
| `W_ao_K/obj_XXXXX.pt` | 每对象 M×300 | 3 个全覆盖模板 | 构建 R；新 G/H 的 pair bias 输入 |
| `K_oa_text_raw.pt` | N×M | 模板与 primitive 原型余弦 R | 审计原始相似度；早期不是直接 score prior |
| `K_oa_text_template.pt` | N×M | 全局 min-max R | 模板 K 的保留副本 |
| `K_oa_text.pt` | N×M | 选择 template 或 lexical | 实际 prior 来源冻结与验证 |
| `K_oa_prior_centered.pt` | N×M | K 的总体标准化 | λ 非零时加到候选组合分数 |
| `report.json` | 编码、OOV、K/P、top/bottom、覆盖 | builder 的统计 | 质量和数值诊断，不进入前向 |
| `knowledge_coverage_report.json` | 同一 builder 报告 | 与 report 的同内容副本 | 工程接口 / 覆盖审计 |
| `manifest.json` | 输入、输出、builder SHA 与配方 | 构建冻结 | 输入身份、模型读取校验 |
| `cache_manifest.json` | 当前 builder 的 manifest 同副本 | 同上 | run 元数据与旧接口 |

历史 v1 服务器位置：`/data/xhw/czsl_repro/knowledge/kcc0601_v1/cache_trusted_v1/<dataset>`。v2 历史位置：`/var/tmp/czsl_xhw_ut_v2_20261006/cache/ut-zap50k`。

本地轻量证据目录通常仅回传 labels、manifest、report，不含所有 `.pt`。本次可直接打开的 v2 cache 元数据为 [UT-v2 cache / manifest.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/cache/manifest.json>)、[UT-v2 cache / report.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/cache/report.json>)。不能向用户承诺这些目录已经备份了所有张量或原始 GloVe；manifest 记录输出 SHA 不等于相应大文件已下载到此目录。

构建器先在同父目录临时 stage 生成，再原子改名。输出不得与源知识目录重叠；非空目标拒绝覆盖。重建同一方法应新建版本，而不是删除原 cache 后覆盖旧实验。

## 11. UT-v2：TRAIN 审查到底改变了什么

来源证据：[TRAIN 审图 / manifest.json](<../ut_semantic_v2_20261006/review_evidence/manifest.json>)、[TRAIN 审图 / selected_train_images.json](<../ut_semantic_v2_20261006/review_evidence/selected_train_images.json>)、[UT-v2 / item_review.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/item_review.json>)、[UT-v2 / manifest.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/manifest.json>)。

| 检查项 | 已保存事实 |
| --- | --- |
| 覆盖 | 16 属性＋12 对象；84 条描述；28 张 class contact sheet |
| 固定样本 | 560 个 panel，528 张不同 TRAIN 图片，每类最多 20 个 panel |
| 抽样策略 | other primitive 的 round-robin 覆盖＋固定 class-local RNG，seed=20261006 |
| 描述决策 | 保留 30 句，改写 54 句 |
| 标签与 OI | 标签未改，OI 分组的 JSON 语义未改 |
| 状态 | `assistant_reviewed_train_images`；`human_reviewed=false` |
| 拒绝过度主张 | 图片可见性不是材质真值；样本不是出现率估计 |
| VAL/TEST | 未审查这些图片；但已知先前 VAL 汇总，不能称盲验证 |

路径别名只修复两个品牌目录的实际路径编码，不修改样本标签或 pair 身份。审图脚本区分 metadata identity 字段与实际路径字段；只改变如何找到同一张图片。

改写动机是减少“定义听起来合理，但鞋图未必支持”的断言，把类别/材质来源与可见形态线索分开，并记录反例。例如 `Rubber` 的“鞋底或外壳”、`Faux.Fur` 的“开口处毛边”，可能描述附属区域而非标签定义的主体材料；保留这类限定和例外，不应升级为所有图片都满足的逻辑事实。

### 11.1 v1→v2 的可追溯关系

| 文件 | UT-v1 SHA256 | UT-v2 SHA256 | 语义变化 |
| --- | --- | --- | --- |
| oi_dict.json | `f048e790c4effb8a61797bf903cad4a93371fbe1bfbeac75b445d7dd1e19ba6f` | `af696e4612fa069555ea32d2b787e8ebf1c9d3eefec7b8e38cb5950a982172c8` | 内容未改；序列化字节改变 |
| oe_dict.json | `ca72dddc8645c85b8dc2f7346d10aa9b6a68da35876cfd0e75bdc287335bafaa` | `75c0ee111d29e2d486d4bd4a623c9bdd7e5ace7e47f79040ed49a5bb65de4e8c` | TRAIN 审查后的描述改写 |
| ae_dict.json | `f161f0021c0ccbf7d672d584b6d3ab4257f849f8ad419e094023ad915946235c` | `13a2c0826571a3db4e20e44b83b263eac89e9c26975de7f987e1f0b295848193` | TRAIN 审查后的描述改写 |

OI 的 hash 不同但 JSON 对象语义相同，原因是序列化字节改变（UTF-8/LF 等），不是增加/删除了父类。OE/AE 发生描述内容变化。新源目录保留 previous source 与 knowledge SHA；名称 raw 表按原 CPF 规则应保持不变。

v2 外层准备检查：任何描述 fallback 停止；单句 OOV 占比不得超过 50%；每个轴全部描述 OOV 不得超过 10%。这些检查位于 [UT-v2 / v2_prepare.py](<../ut_semantic_v2_20261006/transition/v2_prepare.py>)，不是说 `bank.py` 本身永远不会产生 fallback。

## 12. EKR description bank：从聚合表恢复描述槽，不是原作者数据恢复

构建代码 [EKR / bank.py](<../ut_semantic_v2_20261006/transition/bank.py>) 从与 C cache 一致的源 AE/OE、原始名称表、同一 GloVe 重新得到四个单位槽：名称 slot 0，描述 slot 1/2/3。这里的“恢复槽”是把我们自己已有的源句子重新编码，不是从一个均值向量反演出遗失的 GPT 文本。

### 12.1 四槽、mask、均值与中心化残差

$$
\begin{aligned}
U_c&=[u_{c,0},u_{c,1},u_{c,2},u_{c,3}],\\
\mu_c&=\frac{\sum_r m_{c,r}u_{c,r}}{\sum_r m_{c,r}},\\
d_{c,r}&=m_{c,r}(u_{c,r}-\mu_c),\\
\sum_r d_{c,r}&\approx 0.
\end{aligned}
$$

当前三个描述完整，mask 全为 1；代码还支持 padding 与 masked mean。**不能再把每个 residual 单独单位化**，否则破坏中心化关系与原来的幅度。

| `bank.pt` 内键（分别在 `a` / `o` 下） | UT 形状 | 构建 / 作用 |
| --- | --- | --- |
| `units` | 16×4×300 / 12×4×300 | 名称与三句各自单位向量 |
| `mask` | 16×4 / 12×4 | 有效槽身份，当前全真 |
| `mean` | 16×300 / 12×300 | 未再次单位化的四槽平均 |
| `residual` | 16×4×300 / 12×4×300 | 槽减各自均值后的中心化表示 |
| `correct` | 16×300 / 12×300 | 重新聚合验证后保留原 C cache 的 literal K 张量 |
| `unit` | 16×300 / 12×300 | 仅名称单位槽 |
| `shuffle_units` / `shuffle_mask` | 同 units / mask | 名称不动，借入另一类别三描述 |
| `shuffle_mean` / `shuffle` | 对应 mean / correct | 错配后重新平均 / 单位化 |
| `permutation` | 16 / 12 | 固定 recipient→donor 的对应关系 |

外层文件为 `bank.pt`、`labels.json`、`descriptions.json`、`report.json`、`manifest.json`。本地真实文件：[UT-v2 EKR bank / bank.pt](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/bank.pt>)；逐句 token/OOV 记录：[UT-v2 EKR bank / descriptions.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/descriptions.json>)；数值审计：[UT-v2 EKR bank / report.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/report.json>)。

v2 报告：属性 OOV 0/181、对象 1/172，回退均为 0；重新聚合与原 C 表最大绝对差均为 0。residual 槽和最大绝对值约为 7.45e−8 / 5.96e−8，residual 范数中位数约 0.547679 / 0.617751。当前本次核查验证了文件 hash 与报告身份，未重新执行 GloVe 全扫描或 GPU 前向。

### 12.2 错配控制有两套算法，不能合并 seed

| 处理链 | 确定方法 | seed | 保留 / 移动什么 |
| --- | --- | --- | --- |
| 早期 bank / C_shuffle / EKR | NumPy RandomState 置换顺序，再环移构造无固定点 donor | 属性 1729，对象 1730 | 每类名称 slot 0 保留；三描述整体换 donor；重新聚合 |
| 条件 head / generic / CLIP | Torch 独立 Generator，反复 randperm 直到无固定点 | perm1 属性 910011，对象 911011；perm2 属性 910029，对象 911029 | 名称保留；额外描述的归属错配 |

两者都不等于“把知识全删掉”，也不是“随机语义图”。它们检验额外描述与类别身份的对应是否重要，并控制文本量 / 输入结构；不检验任意随机 GNN 的效果。

### 12.3 同一知识槽在后续 EKR 中的处理修正

| 路径 | key 的源 | value 的源 | 对象上下文 / 底座 |
| --- | --- | --- | --- |
| 旧主 EKR | 中心化 residual，投影后单位化 | residual / 2 | 对象 `correct`；增量加在 C_text 上 |
| N_clean / N_shuffle | 正确 / 错配的中心化 residual | 对应 residual / 2 | 对象仅 `unit` 名称；A_unit 底座 |
| N_name | 将属性名称重复为 4 个槽 | 属性名称 / 2 | 同样仅对象名称，避免名称控制变成零值 |
| V_clean_align / V_shuffle_align | 原始正确 / 错配 units | 原始 units 经共享低秩对齐、单位化 / 2 | 冻结 A_unit＋已训 N_name 的公共锚点 |
| V_name_align | 4 个重复名称 | 相同对齐结构处理名称 / 2 | 同上；非零名称容量控制 |
| V_clean_fixed | 原始 units | 单位原始 units / 2；value 投影固定为恒等 | 同上；刻意降容量消融 |
| SG | 原始 units，只打开 mask 槽 | 原始单位槽 / 2，固定 | 同一名称锚点、TRAIN mask；不继续 value 对齐 |

旧路径实现 [旧 EKR / router.py](<../ut_semantic_v2_20261006/transition/router.py>)；名称锚点修正 [名称锚点 / name_router.py](<../name_anchor_ekr_20261007/transition/name_router.py>)；共享值处理 [共享值对齐 / core.py](<../ekr_v2_anchor_alignment_20261007/core.py>)。null 节点不是外部知识条目：它是可学习的“不给新增证据”的选项，value=0。query/key/null 与槽 softmax 温度 0.2 属于模型读取，不能与静态 K 的概率混称。

原中心化值会将四槽共同成分消去，也改变名称槽本身。正确 AE 推理错配后若重新中心化，名称 residual 也随均值改变；这不是“只动描述、名称值不动”的严格干预。N_name 特意用非零名称值，但与 N_clean 的值几何不同。

后来没有将 `(description-name)/2` 建议升级为正式主实现：那样名称控制会成为 name−name=0，无法作为有效同容量对照。正式共享值处理改为原始槽：

$$
\begin{aligned}
\widetilde u_{a,r}&=u_{a,r}+U D u_{a,r},\\
v_{a,r}^{\mathrm{aligned}}
 &=\frac12\mathcal{N}(\widetilde u_{a,r}),\\
e_{a,r}(x)&=\mathcal{N}(f_a(x))^{\mathsf T}v_{a,r}^{\mathrm{aligned}},\\
\Delta(x,a,o)&=\eta\sum_r\alpha_{a,r}(x,o)e_{a,r}(x),\\
\eta&=0.1\tanh(g),\qquad g_{\mathrm{init}}=0.
\end{aligned}
$$

D 为 300→32，U 为 32→300；D 随机初始化、U 零初始化，使初始 value 不变。两个矩阵不可同时零初始化。value 范数 0.5、增量绝对界小于 0.05；gain 有符号，可以学到负值，不是 K 的非负可信度。

旧增量关闭只能回到 C_text，不能撤销 C_text 对输入表的改变；名称锚点增量关闭回 A_unit；共享值/SG 的新增分支关闭回冻结名称锚点。三个“回退”含义不同。query 依赖视觉/对象在代码中存在，也不能直接证明学到了强对象条件语义；现有结果需要名称与错配对照支持。

这些修正没有生成新的 AE/OE 文件；变化的是消费代码和 learned parameter，源 bank 仍应一起保存。

## 13. SG-EKR：如何用视觉筛选知识槽

来源 [SG-EKR / pilot.py](<../sgekr_support_mask_20261008/pilot.py>)；产物 [SG-EKR / support_mask.json](<../output/sgekr_review_20261008/artifacts/support_mask.json>)。输入是已缓存的 498 张 TRAIN 属性视觉特征：83 个 TRAIN 组合各 6 张图。按每组合图像 ID 排序后交替分为 3/3 两折。

在此之前，[初次 TRAIN 语义支持 / probe.py](<../ekr_train_support_probe_20261008/probe.py>) 已做过一次无 optimizer 的 TRAIN 几何诊断：正确四槽均值相对错配的宏 margin +0.026839，但相对名称 −0.004576；one-vs-rest 有 34/48 个 AE 正支持槽。它不同于后续同对象、对象平衡、两折保守筛选的 30/48 个槽，不能把 34 和 30 混报为同一个阈值的结果。

对属性 a、描述槽 r、折 f，在每个可比较的对象 o 内，计算正确属性图片与同对象其他属性图片的平均相似度差，再对对象等权平均。定义 $\hat v_x$、$\hat u_{a,r}$ 为单位向量，$X^+_{a,o,f}$ 与 $X^-_{a,o,f}$ 分别为正/负图片集合：

$$
\begin{aligned}
D_{a,r,o,f}
 &=\frac12\left[
   \operatorname{mean}_{x\in X^+_{a,o,f}}
     \hat v_x^{\mathsf T}\hat u_{a,r}
   \right.\\[-2pt]
 &\hspace{30mm}\left.
   -\operatorname{mean}_{x\in X^-_{a,o,f}}
     \hat v_x^{\mathsf T}\hat u_{a,r}
   \right],\\
S_{a,r,f}&=\operatorname{mean}_{o\in\mathcal{O}_{a,f}}D_{a,r,o,f},\\
m^{\mathrm{SG}}_{a,r}
 &=\mathbf{1}\!\left[\min_{f\in\{0,1\}}S_{a,r,f}\ge0.005\right].
\end{aligned}
$$

**1/2 必须保留**：实际代码是 `vectors=unit_numpy(units)/2`，不能用完整余弦差配原阈值。名称 slot 0 永远保留；48 个 AE 槽中保留 30 个。掩码 JSON 还保留两折分数、比较对象 ID、输入缓存 SHA。

它是既有描述的 TRAIN 视觉支持选择，不是描述语义真值标注。相似度高可能来自非主体区域或类间共性；两折只是降低当前 TRAIN 小样本不稳定性，不是独立跨组合验证。错配组沿用 recipient 的支持 mask，不能把 donor 的 mask 一起移动制造不同对照。

实际使用 [SG-EKR / core.py](<../sgekr_support_mask_20261008/core.py>) 固定 value=单位槽/2，只训练 query/key/null/gain。所有源句仍在 bank 中，mask 只限制本路径可被注意力选择的槽。

筛选没有重写 AE/OE。SG 的已有结果也不能被解释成“删掉了 18 条无用知识就一定提高性能”。

## 14. B/F：另一套论文方法重建知识与逻辑 bank

### 14.1 来源和与 UT-v1/v2 的区别

[B/F / knowledge.json](<../logiczsl_cpf_bf_20261008/assets/ut-zap50k/knowledge.json>) 的 provenance 明确声明：`paper_method_reconstructed_ut_v1_20261008`、`assistant_reconstructed_requires_user_review`、`official_author_data=false`；论文引导的 assistant 语义重建，不是历史 GPT-4 输出。构建范围是 UT primitive 标签，没有根据 VAL/TEST 指标生成新知识。

其 OI 是 5 个父类，而旧 v1/v2 OI 是 4 个。B/F 的 AE/OE 文本也与 v2 不同；完整内容见附录 C，不要误把两套源字典互换。

| bank 父类 ID | 父类短语 | 子类（数据集原标签） |
| --- | --- | --- |
| 0 | boots | Boots.Ankle<br>Boots.Knee.High<br>Boots.Mid-Calf |
| 1 | closed low shoes | Shoes.Boat.Shoes<br>Shoes.Flats<br>Shoes.Loafers<br>Shoes.Oxfords<br>Shoes.Sneakers.and.Athletic.Shoes |
| 2 | raised heel shoes | Shoes.Heels |
| 3 | slip on comfort footwear | Shoes.Clogs.and.Mules<br>Slippers |
| 4 | strapped open footwear | Sandals |

parent 顺序由校验器排序后冻结，`object_parent` 为：

```text
[0, 0, 0, 4, 1, 3, 1, 2, 1, 1, 1, 3]
```

它不是 knowledge.json 原始 JSON 插入顺序。父类全覆盖，每对象仅一个父类；Sandals、Heels 的父类只有单成员，语义抽象与约束强度需保留这一限制。

### 14.2 逻辑节点怎么编码

[B/F / build_bank.py](<../logiczsl_cpf_bf_20261008/build_bank.py>) 读取 knowledge、labels、GloVe，将 5 个父类短语、36 条 OE、48 条 AE 分别编码。正则 `[a-z]+`，每个短语取 in-vocabulary GloVe 均值后单位化。每条覆盖率必须至少 0.8，且非空/非零/finite，否则停止；不悄悄退回名称向量。

$$
\begin{aligned}
e(t)
 &=\mathcal{N}\!\left(
    \frac1{\lvert T^{\mathrm{IV}}(t)\rvert}
    \sum_{w\in T^{\mathrm{IV}}(t)}g(w)\right),\\
\operatorname{coverage}(t)
 &=\frac{\lvert T^{\mathrm{IV}}(t)\rvert}{\lvert T(t)\rvert}
   \ge0.8.
\end{aligned}
$$

此处每条描述是独立节点，**不掺名称形成四槽平均**。同样使用 GloVe 不意味着 B/F bank 与 EKR bank 等价。

| B/F 文件 / 键 | 来源 / 形状 | 实际作用 |
| --- | --- | --- |
| `knowledge.json` | provenance＋oi＋oe＋ae | 这套逻辑知识的原文与组结构 |
| `bank.pt['parent']` | 5×300，父类短语 | 对象视觉分支的父类 logits |
| `bank.pt['oe']` | 12×3×300，每条 OE 独立单位化 | 对象视觉分支的 3 槽 OE logits |
| `bank.pt['ae']` | 16×3×300，每条 AE 独立单位化 | 属性视觉分支的 3 槽 AE logits |
| `coverage.json` | 每短语 token、缺失与覆盖率 | 编码审计；最初本地真实 GloVe 检查为 100% 覆盖 |
| `manifest.json` | labels、parents、object_parent、源/编码器/bank SHA | 保证节点与类身份不被换轴或换版本 |

可核实的固定资产副本：[B/F 固定逻辑 bank / manifest.json](<../reproduce_0601_20261010/assets/logic_bank/manifest.json>)、[B/F 固定逻辑 bank / bank.pt](<../reproduce_0601_20261010/assets/logic_bank/bank.pt>)。这套构建器不生成组合 compatibility prior。

### 14.3 知识如何成为逻辑约束

[B/F / bf_logic.py](<../logiczsl_cpf_bf_20261008/model/bf_logic.py>) 直接用 CPF 原生 classifier，把对象视觉表示打到 parent/OE，把属性视觉表示打到 AE；没有新增可训练知识 head。知识向量固定，但梯度经 logits 回传至视觉分支与原 CPF。

对象、属性、父类各按 class 轴 softmax。OE 形状 batch×object×3、AE 为 batch×attribute×3：softmax 也是 **dim=1 的类轴**，不是在 3 条描述之间做 softmax。这与“注意力选择三句”不同。

设对象所属父类为 h(o)，广义均值聚合为：

$$
\mathcal{M}_q(v)
 =\left(\frac1B\sum_{b=1}^{B}v_b^q\right)^{1/q},
 \qquad q=3.
$$

各类 / 槽先沿 batch 聚合违反度，再对其余维平均：

$$
\begin{aligned}
L_{\mathrm{OI}}^{\uparrow}
 &=\operatorname{mean}_{o}
   \mathcal{M}_q\!\left(p_o(1-p_{h(o)})\right),\\
L_{\mathrm{OI}}^{\downarrow}
 &=\operatorname{mean}_{h}
   \mathcal{M}_q\!\left(p_h(1-\max_{o:h(o)=h}p_o)\right),\\
L_{\mathrm{OE}}
 &=\operatorname{mean}_{o,r}
   \mathcal{M}_q\!\left(p_o(1-p^{\mathrm{OE}}_{o,r})\right),\\
L_{\mathrm{AE}}
 &=\operatorname{mean}_{a,r}
   \mathcal{M}_q\!\left(p_a(1-p^{\mathrm{AE}}_{a,r})\right).
\end{aligned}
$$

$$
\begin{aligned}
L_{\mathrm{OI}}&=L_{\mathrm{OI}}^{\uparrow}+L_{\mathrm{OI}}^{\downarrow},\\
L_{\mathrm{total}}
 &=L_{\mathrm{CPF}}
   +0.01\left(L_{\mathrm{OI}}+L_{\mathrm{OE}}+L_{\mathrm{AE}}\right).
\end{aligned}
$$

原语 → 其描述，而不是把 OE/AE 蕴含方向写反。β 只乘一次。零违反度时使用等价的向量范数聚合保持有限梯度。没有额外知识节点 CE，也没有让逻辑节点直接改变推理分数。实际实现 [B/F / bf_logic_loss.py](<../reproduce_0601_20261010/reference_source/models/bf_logic_loss.py>)。

此前作者代码副本审计发现：逻辑分支注释、部分入口仍返回 CE 而非计算出的总损失、知识字段命名不一致，以及原 JSON 路径占位符。B/F 因此重新建立实际节点 logits、显式总 loss 和梯度检查，而非仅复制源代码或放置知识文件就宣称知识已训练。这里是对本地副本的归档审计，不代表本次重新验证作者线上仓库的最新状态。

### 14.4 B 和 F 使用的知识并不完全相同

| 实验 | Text/Prior 输入 | Logic 输入 | 关键边界 |
| --- | --- | --- | --- |
| B | 纯 CPF 名称底座，不启用 Text K/Prior | 新 5 父类 B/F bank | 检验加入论文式逻辑正则 |
| F | 旧 UT-v1 `W_a_K/W_o_K`＋template P，λ=0.1 | 与 B 同一新 B/F bank | F=旧 E＋Logic，不是 UT-v2 C_text＋Logic |

`--allow-reconstructed` / `bf_allow_reconstructed` 是明确承认使用重建资产的探索性质，不会把知识状态改为官方或人工审核完成。B/F 的完整训练已回传，当前仍无充分新增收益；训练完成与语义知识验收是独立状态。

## 15. 条件属性头 / generic：已有知识怎样成为视觉相关输入

[条件属性头 / head.py](<../conditional_attribute_pilot_20261009/head.py>) 使用同一 UT-v2 bank，把名称和三句均值分开：

$$
\begin{aligned}
e_c^{\mathrm{name}}&=u_{c,0},\\
e_c^{\mathrm{desc}}&=\frac13\sum_{r=1}^{3}u_{c,r}.
\end{aligned}
$$

desc 均值没有再单位化，也不包括名称槽。这不同于 `W_c_K=normalize(mean(name+3descs))`，不能将两者互换。

| 控制 / 文件 | 实际作用 | 不代表什么 |
| --- | --- | --- |
| P_name / G_name | 新分支 desc 输入替换为名称向量；公用 C_text 底座仍在 | 整个系统完全不含知识 |
| P_K / G_K | 名称＋正确 OE/AE 三句均值 | 新外部知识被重新生成 |
| P_perm / G_perm | 名称不动，仅 desc 均值换 donor | 完全删除知识或随机图 |
| P_generic / generic controls | 相同输入投影，普通拼接 MLP | 架构胜出即正确知识胜出 |
| `head.pt` / best checkpoint | learned head 参数 | 能独自恢复所有外部 buffers 与协议 |
| StageA `train.pt` / `val.pt` | 冻结 C_text 的视觉表示、原分数、标签 | GloVe bank 或新的描述库 |
| `controls.json` / protocol | 实际输入置换、配方、身份与 SHA | 语义正确性标签 |

小头含视觉投影、对象 query、名称/描述投影及 LayerNorm。固定 bank 不学习，投影和组合方式学习；知识被“视觉化使用”与“更新外部知识文件”是不同操作。

这些输入注册为 `persistent=False` buffers：checkpoint 本身可能不包含它们。要恢复一次实验，必须保留 bank、置换协议、底座 checkpoint 及其 source manifest，不能只留 head 权重。

25% 是 generic 新条件分布与原条件分布的固定概率读出比例，不是 OE/AE 权重、不代表 K 中 25% 边为真，更不是把 25% 知识加入图结构。原训练损失仍为 unmixed conditional CE；评价混合并没有重新训练知识。

## 16. CLIP 与 TRAIN 审查：同一原文的另一条处理链

### 16.1 构建来源没有变，只改变编码器和 prompt

[CLIP-r2 / ae_dict.json](<../clip_knowledge_probe_20261010_r2/assets/ae_dict.json>)、[CLIP-r2 / oe_dict.json](<../clip_knowledge_probe_20261010_r2/assets/oe_dict.json>) 与 UT-v2 字典 byte-for-byte 相同；TRAIN 审查包中的两份原文也相同。本次核查已验证实际文件 SHA，不能把 CLIP 探针写成“生成了新版视觉知识”。

[CLIP-r2 / protocol.json](<../clip_knowledge_probe_20261010_r2/protocol.json>) 固定 OpenAI CLIP ViT-L/14，224px，768 维；源码 commit=`a1d071733d7111c9c014f024669f959182114e33`；权重 SHA：

```text
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
```

这里是归档协议记录的编码器来源，本文没有重新下载或查询最新版 CLIP。CLIP 自然语言名称表显式冻结，例如 `Synthetic → synthetic material`；它不同于原 CPF 的别名 token。

[CLIP-r2 / prompts.py](<../clip_knowledge_probe_20261010_r2/prompts.py>) 的模板为：

```text
名称：A photo of {object} made of {attribute}.
加 AE：It has {AE}.
加 OE：Object shape: {OE}.
```

每个候选采用 AE 3 条×OE 3 条的 9 个等权单元。T_name 的名称重复 9 次；AE-only 各句重复 3 次，确保单元数一致。7 个变体为 name、正确 OE+AE、两套错配 OE+AE、正确 AE-only、两套错配 AE-only；名称从不随描述错配改变。

所有 12×16 候选都生成；跨变体 deduplicate 后共 7,041 个唯一文本，不能按 `7×192×9` 当成唯一 prompt 数。

### 16.2 逐句单位化 → 均值 → 图文分数 → q

令 $t_{i,j,r}$ 为单位化文本编码，$\hat v_x$ 为单位化图片编码：

$$
\begin{aligned}
\bar t_{i,j}&=\frac19\sum_{r=1}^{9}t_{i,j,r},\\
c_{i,j}(x)&=\hat v_x^{\mathsf T}\bar t_{i,j},\\
q(a_j\mid o_i,x)
 &=\operatorname{softmax}_{j}\!\left(c_{i,j}(x)/0.05\right).
\end{aligned}
$$

平均文本向量不再单位化，因而 c 等价于逐句余弦的平均；重新单位化均值会改变已冻结配方。q 是依赖图片的条件概率，不是固定 K_oa feasibility，也不证明教师正确。

| CLIP 结果文件（ZIP 内） | 构建来源 | 作用 |
| --- | --- | --- |
| `prompts.json` | 固定模板＋名称表＋v2 AE/OE＋置换 | 保存每一个实际文本位置，可回溯内容 |
| `text_features.npz` | 冻结 CLIP 文本编码；按 9 槽均值 | 每变体 12×16×768 表；不含所有逐槽向量 |
| `text_audit.json` | 去重、tokenize、长度审计 | 7,041 唯一文本；禁止静默截断 |
| `features/train/*.npz` / `features/val/*.npz` | 冻结 CLIP 图片编码 | 单位 768 维图片向量；每 shard 有身份与 SHA receipt |
| `train_records.json` / `val_records.json` | 元数据、图片路径/身份记录 | cache 与真实样本匹配 |
| `teacher/<phase>_<variant>.npz` | 图文分数与固定温度 softmax | 保存 cosine 与 q，用于机制诊断，不是已训练教师 |
| `teacher/<phase>_truth.npz` | 原 metadata 与 seen/unseen 身份 | 指标与对照，不是构建源知识的输入 |
| `split_audit.json` / `environment.json` | 输入身份与编码运行 | 记录正常 seen overlap、图片不重叠、模型环境 |

归档入口 [CLIP-r2 结果 ZIP](<../output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip>)。原探针 v1 错误断言 train/val pair 完全互斥；r2 修正为 VAL 15 个 seen＋15 个 unseen，仍验证 TRAIN/VAL 图像 ID 与图像字节身份无交集。该修复没有改知识或利用错误期间的指标选句。

### 16.3 TRAIN-only 审查新增了什么文件

原 CLIP 结果只有 9 槽均值，不足以分析单句，因此 [TRAIN 审查 / run.py](<../output/train_knowledge_audit_results_20261010/extracted/source/run.py>) 在同一冻结编码器下重编码 7,041 个唯一文本，保存 `unit_text_features.npz` 和 [TRAIN 审查 / unit_text_receipt.json](<../output/train_knowledge_audit_results_20261010/extracted/result/unit_text_receipt.json>)，再恢复每变体的 12×16×9×768 单槽表。

它复用原 CLIP TRAIN 图片特征，没有调用图像编码器；只解码 96 张固定 TRAIN 图片用于拼图。[TRAIN 审查 / input_reads.json](<../output/train_knowledge_audit_results_20261010/extracted/result/input_reads.json>) 和 [TRAIN 审查 / complete.json](<../output/train_knowledge_audit_results_20261010/extracted/result/complete.json>) 记录：VAL/TEST 文件读取空列表、optimizer step=0、weights fitted=false。

| 新文件 | 来源 / 用途 | 当前地位 |
| --- | --- | --- |
| `unit_text_features.npz` | 7,041 个单位文本向量 | 派生文本缓存，不是新 AE/OE |
| `unit_text_receipt.json` | 文本清单、batch 与文件 hash | 单槽 cache 身份 |
| `sentence_train_report.json` | 9 个 OE+AE 位置、3 个 AE-only 位置，各带正确/两错配控制 | 36 个 TRAIN 位置对照，不是新 prompt 训练 |
| TRAIN 拼图 / `review.html` | 固定 96 张 TRAIN 样本 | 质性适用性审阅，不是材质专家新真值 |
| `human_review_blank.json` | 空白人工可见性/区分性评分表 | 目前没有已填专家验收记录 |
| `train_input_manifest.json` | 原探针 TRAIN 输入白名单与 SHA | 证明读取边界，不代表可合法声称盲试验 |

[TRAIN 审查 / human_review_blank.json](<../output/train_knowledge_audit_results_20261010/extracted/result/human_review_blank.json>) 中评分仍为 null；文件包含 review instructions 是表单说明，不是本次用户要求执行的新指令。

## 17. 原 0601 新 G/H 实现中的知识资产作用

[0601 G/H / full_feature.py](<../reproduce_0601_20261010/source/models/full_feature.py>) 已有实际消费 pair block 的代码；[0601 G/H 执行包说明](<../reproduce_0601_20261010/README.md>) 明确这是后续执行包。本文以当前可验证的已回传历史为界，不将执行包、CPU tests 或 smoke 当成 G/H 完整训练结果。

新实现使用旧 v1 cache 中 `W_a_K`、`W_o_K` 与全部 `W_ao_K/obj_XXXXX.pt`，并保留原 CPF composition score 作 residual 底座。pair block 不只是继续生成 prior，还进入可学习 pair-bias MLP：

$$
\begin{aligned}
z_{i,j}^{\mathcal{K}}(x)
 &=\mathcal{N}(f_o(v_o(x)))^{\mathsf T}
   \mathcal{N}(g_o(w_{o_i}^{\mathcal{K}}))\\
 &\quad+\mathcal{N}(f_a(v_a(x)))^{\mathsf T}
   \mathcal{N}(g_a(w_{a_j}^{\mathcal{K}}))\\
 &\quad+\operatorname{MLP}_{\mathrm{pair}}(b_{i,j}).
\end{aligned}
$$

它是新 Feature 消费路径，不是重建 pair-level LLM 知识；pair 内容仍为全覆盖三模板。四个 300→512 投影＋pair MLP 学习如何使用这些固定向量。

| 受控资产变体 | 当前实现含义 | 不能声称的结论 |
| --- | --- | --- |
| `H_no_pair` | 不加 pair MLP 分支，保留 primitive Text | 所有外部知识均被删除 |
| `H_shuffle` | 固定交换 OE/AE，Text/Feature 与 Logic 使用一致交换身份 | 使用随机知识图 |
| `H_random_prior` | seed=603354 的 12×16 随机 P，标准化尺度与真实 P 一致 | 已验证随机 prior 更好 |
| `H_uniform_prior` | 常数 K 经 centered 后为零 | 只是给每个候选任意固定偏移 |
| `H_noise10/30/50` | 固定 slot 删除，仅重聚合 Text/Feature primitive K | 全部知识树/Logic/Prior/pair 模板都测了噪声 |

这些是实现与待运行对照的定义，不是本档案新增的实验发现。只有下载到真实运行产物、核验实际 source/config/manifest 后，才能升级它们的完成状态。

## 18. 哪些文件进入了哪些实验：最终对应表

| 已有路径 / 实验 | 实际知识输入 | OI 是否实际使用 | 是否依赖视觉读取知识 | 需要保留的资产 |
| --- | --- | --- | --- | --- |
| A 原 CPF | 原名称 raw 表 / 原名称初始化 | 否 | CPF 原生视觉分类 | 名称规则、编码器、底座配置与权重 |
| C Text K | `W_a_K`、`W_o_K` | 只做源 schema 校验 | 通过 CPF 原生模块 | 源 AE/OE、cache、读取代码 |
| D Calibration | 名称表＋centered P | 否 | P 本身静态 | K/P、cache manifest、λ |
| E Text＋Calibration | primitive K＋P | 只校验 | CPF＋静态 prior | 两条路径全部资产 |
| UT-v2 A_unit/C_text/C_shuffle | `bank.unit/correct/shuffle` | 无前向 OI | CPF 原生模块 | 四槽 bank、置换、源字典 |
| 名称锚点 / 对齐 EKR | 名称槽、描述槽/中心化残差 | 无前向 OI | learned query / value 或路由 | bank＋路由状态＋底座身份 |
| SG-EKR | v2 bank＋TRAIN mask | 无前向 OI | TRAIN 视觉选择＋模型读取 | mask、support cache 身份、bank |
| B | 原名称＋独立 logic bank | 是，父子逻辑 | 原对象/属性视觉分支 grounding | BF knowledge、bank、q/β、adapter |
| F | 旧 v1 Text/Prior＋同一 BF logic bank | 是 | CPF＋grounding；P 静态 | 必须同时留两套知识版本 |
| Conditional / Generic | C_text 底座＋名称/描述均值/错配 | 否 | 视觉 query / learned head | StageA、bank、controls、checkpoint |
| Generic＋25% | 同上，固定概率读出 | 否 | 同上 | 额外保留 α 与读出实现 |
| CLIP 七组探针 | v2 原文＋固定自然语言名称＋prompt | 否 | 冻结图文余弦 | 原文字典、prompt、模型/缓存 hash |
| TRAIN 逐句审查 | 同一 prompt 的单槽特征 | 否 | 已有 TRAIN 图文证据 | 单槽缓存、白名单、空白/已填审查状态 |
| 新 G/H 实现（训练状态未核实） | v1 primitive K＋pair blocks＋P；H 另加 BF logic | H 是 | 新 Feature 与 grounding | 旧 cache＋BF bank＋full head 源码 |

知识向量作为 buffer 固定，不意味着模型不使用视觉；存在固定 OI 文件，也不意味着整个方案采用 GNN 或“固定语义图”。当前多数已运行路径是固定文本表＋视觉可学习消费，不是图传播。

## 19. 现有数据对知识资产质量能说明什么

这部分只解释知识资产与处理链，不以文档整理替代新的实验。

| 已有证据 | 对知识 / 处理方式的含义 | 仍不能推出什么 |
| --- | --- | --- |
| v1/v2 OOV 低、回退为 0 | 编码输入齐全，不能归因于大量空描述 | 所有语义与图像标签正确匹配 |
| C_text 有相对底座的收益 | 整套名称＋描述聚合与 CPF 可以有工程价值 | 收益全来自正确描述语义，而非单位化/均值变化 |
| 错配有时相近或更好 | 需要证明正确语义的独有增量，不能只看 K 组涨点 | 知识本身永远无用或随机图会必然更好 |
| B/F 已运行但未显示充分增量 | 独立逻辑层级 / 节点并未自动成为有效监督 | LogiCZSL 原数据和原视觉框架无效 |
| generic＋25% 有工程线索、generic semantic gate 未过 | 架构/温和读出可能有价值；正确内容归因仍不足 | 已证明正确知识有不可替代性 |
| CLIP OE+AE / AE-only gate 均失败 | 当前原文＋固定 prompt＋等权均值不能支持扩大教师训练 | 更强视觉编码一定无益或所有句子都无信号 |
| AE 第三条 TRAIN 局部优势 | 平均配方遮蔽了一部分内容信号，值得作为有限候选记录 | 已有可迁移教师或可以逐类别 oracle 选句 |

[CLIP-r2 独立复核](<../output/clip_knowledge_probe_results_20261010_r2/CLIP知识证据探针_r2_独立复核与下一步判断_20261010.md>)：OE+AE 的 VAL unseen pair macro 相对名称 +0.3859 pp，但 micro −12.9653 pp，且 macro 低于两套错配；AE-only 也未过门槛。它们是 GT-object 属性机制诊断，不是最终部署 CW/OW AUC。

[TRAIN 知识落地复核](<../output/train_knowledge_audit_results_20261010/TRAIN知识落地审查_独立复核与路线判断_20261010.md>)：名称 TRAIN micro/macro=8.7921/22.3766%；AE-only 三句均值为 11.2662/22.6919%；仅各类第三条 AE 为 25.1935/23.4280%。后者相对名称净增 3,772 张，其中 Leather 净增 3,293 张，收益高度集中，且仅为 TRAIN 位置审查。

“第三条”是数组位置，不是统一的纯视觉概念类型；有表面、面板，也有鞋底/领口等区域线索。没有理由仅根据 TRAIN 最高位置就改写所有源字典、称其为 v3，或跳过组合迁移验证。平均分数也不等于单句准确率的平均，不能简单总结成“平均必然有害”。

目前最稳妥的资产结论：保留已有有用 Text 底座与全部失败/错配证据；未验收的逻辑知识、视觉支持、CLIP 原文和单句候选保持各自状态，不伪造“统一高质量知识库”。

## 20. 构建、处理与归档的核查规则

### 20.1 重新构建前应固定什么

1. 明确重建哪一套：v1 template cache、v2 EKR bank、B/F logic bank 或 CLIP prompt，不可用同名 `bank.pt` 混装。
2. 固定完整标签顺序、实际 split SHA、名称 alias 源码、原文字典与审核状态。
3. 固定编码器实际文件 SHA；GloVe 300 维与 CLIP 768 维不是可直接互换的张量。
4. 记录 tokenizer、OOV/fallback、句子/集合归一化、均值是否再单位化、矩阵轴和 prior 变换。
5. 输出新版本目录，不覆盖旧来源、cache、bank 或已跑 run。
6. 保存原文、实际代码、manifest、quality report、控制置换、source-to-output hash 链；外部 buffer 与 checkpoint 成套保存。
7. 若将来真的调用 LLM，额外保存实际模型 ID、日期、完整 prompt、raw response、清洗规则、每次人为修改；未调用时不补造这些证据。

### 20.2 已有入口与推荐的使用边界

| 工作 | 实际入口 | 执行约束 |
| --- | --- | --- |
| 重建初期语义草稿 | [generate.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/generate.py>) | 会写源文件，只应在新 draft 目录；当前不需要重跑 |
| lexical 候选 | [build_text_prior.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/build_text_prior.py>) | 会更新源目录 manifest，不应在冻结 trusted/v2 上直接运行 |
| 源目录结构检查 | [validate_server.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/validate_server.py>) | 核对 exact server primitive 轴；不是知识质量认证 |
| 确定性 GloVe cache | [冻结 GloVe / build_knowledge.py](<../reproduce_0601_20261010/reference_source/tools/build_knowledge.py>) | 源只读，新输出目录；保留真实配置/编码器身份 |
| EKR 四槽 bank | [EKR / bank.py](<../ut_semantic_v2_20261006/transition/bank.py>) | 源 hash 必须与 C cache 一致，重新聚合误差≤1e−6 |
| v2 准备 | [UT-v2 / v2_prepare.py](<../ut_semantic_v2_20261006/transition/v2_prepare.py>) | 强制 OOV/fallback、名称和输入身份守卫 |
| BF 节点 bank | [B/F / build_bank.py](<../logiczsl_cpf_bf_20261008/build_bank.py>) | 明确认可重建性质，覆盖不足停止，无 split-derived prior |
| CLIP 编码与探针 | [CLIP-r2 / probe.py](<../clip_knowledge_probe_20261010_r2/probe.py>) | 固定 protocol；不是全模型训练或 prompt 搜索 |
| TRAIN 逐句审查 | [TRAIN 审查 / run.py](<../output/train_knowledge_audit_results_20261010/extracted/source/run.py>) | TRAIN 输入白名单；不据此偷偷改 VAL 门槛或源字典 |

本文不给会覆盖历史知识的“一键重建”命令：用户本次要求的是单独整理档案。上表是实际来源与入口索引，不代表本轮又执行了这些构建器。

### 20.3 这次整理验证了什么

本档案生成时重新检查：所有文内本地文件链接存在；三数据集 v1 与 UT-v2 源 manifest 的知识 SHA 正确；v2 cache builder SHA 与本地冻结代码一致；cache 两个 manifest 副本一致；EKR bank 的输出与源知识 hash 链一致；CLIP/TRAIN 的 AE/OE 与 v2 字节相同；B/F source 与 bank hash 正确。旧全部实验报告与所有读取的证据保持不变。

边界：未重新读取服务器实时文件系统，未执行原编码器 GPU 前向，未加载并逐元素回算全部知识张量，也未重新审查所有图像或专家材料真值。历史报告中的数值依其现有核验来源引用，不把这次 manifest 核查称为完整 GPU 复现。

## 附录 A. 当前 UT-v2 全部 AE 原文与固定标签顺序

原文来源 [UT-v2 / ae_dict.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/ae_dict.json>)；ID 为 0-based，直接对应 bank 的属性行。描述 1/2/3 对应 bank slot 1/2/3，不是 slot 0。

| ID | 原始标签 | 描述 1 | 描述 2 | 描述 3 |
| --- | --- | --- | --- | --- |
| 0 | Canvas | woven canvas fabric | matte textile upper panels | fabric panels with stitched edges |
| 1 | Cotton | cotton textile fibers | cloth upper panels | woven or knitted cotton fabric |
| 2 | Faux.Fur | fluffy synthetic fur | raised fur like pile | fur trim around footwear openings |
| 3 | Faux.Leather | imitation leather | synthetic leather upper panels | leather like surface finish |
| 4 | Full.grain.leather | natural full grain hide | smooth or textured leather surface | stitched leather upper panels |
| 5 | Hair.Calf | short haired calfskin | hair bearing leather surface | plain or patterned hide panels |
| 6 | Leather | genuine leather | leather upper panels and straps | smooth or textured leather finish |
| 7 | Nubuck | sanded leather | fine napped leather surface | matte brushed hide |
| 8 | Nylon | nylon textile fibers | synthetic fabric panels | fabric straps and padded panels |
| 9 | Patent.Leather | glossy coated leather | bright specular surface highlights | smooth finished upper panels |
| 10 | Rubber | flexible rubber material | molded rubber components | rubber soles or smooth outer shells |
| 11 | Satin | smooth satin fabric | lustrous fabric surface | fabric folds and gathered edges |
| 12 | Sheepskin | sheepskin hide | fleece at collars and openings | matte hide panels |
| 13 | Suede | brushed suede leather | napped leather surface | suede upper panels and straps |
| 14 | Synthetic | man made material | synthetic fabric or polymer components | textile and molded upper surfaces |
| 15 | Wool | wool textile fibers | fuzzy natural fibers | felted or knitted fabric surface |

## 附录 B. 当前 UT-v2 全部 OE 原文与固定标签顺序

原文来源 [UT-v2 / oe_dict.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/oe_dict.json>)；对应对象行。

| ID | 原始标签 | 描述 1 | 描述 2 | 描述 3 |
| --- | --- | --- | --- | --- |
| 0 | Boots.Ankle | shaft ending at the ankle | boot upper covering the foot and ankle | low boot collar |
| 1 | Boots.Knee.High | shaft reaching the knee | tall boot shaft | closed full foot upper |
| 2 | Boots.Mid-Calf | shaft ending at mid calf | medium height boot leg | closed boot construction |
| 3 | Sandals | open toe construction | exposed foot with straps | upper openings around the foot |
| 4 | Shoes.Boat.Shoes | moccasin style stitched toe | low profile lace up upper | laces threaded through side eyelets |
| 5 | Shoes.Clogs.and.Mules | low cut upper with a broad toe box | upper covering the front of the foot | slip on clog and mule silhouette |
| 6 | Shoes.Flats | very low or no heel | shallow low cut upper | low sole and enclosed toe |
| 7 | Shoes.Heels | raised heel or wedge | arched sole profile | dress shoe silhouette |
| 8 | Shoes.Loafers | laceless slip on upper | upper covering the instep | low cut collar around the foot |
| 9 | Shoes.Oxfords | laced or slip on low shoe upper | enclosed toe and low shoe collar | low heel or flat sole profile |
| 10 | Shoes.Sneakers.and.Athletic.Shoes | flat or treaded athletic sole | laced or slip on sporty upper | low or high athletic shoe collar |
| 11 | Slippers | indoor slipper footwear | slip on or strap fastened opening | low heel sole profile |

旧/v2 OI 的 4 个父类为 `boot`、`casual shoe`、`dress shoe`、`open or indoor footwear`，分组见 [UT-v2 / oi_dict.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/oi_dict.json>)。它们不是附录 C 的 5 父类 bank。

## 附录 C. B/F 独立重建的全部 AE/OE 原文

以下来自 [B/F / knowledge.json](<../logiczsl_cpf_bf_20261008/assets/ut-zap50k/knowledge.json>)，不是 UT-v2。父类对应第 14 节的排序后 IDs。

### C.1 属性节点 AE

| ID | 原始标签 | 描述 1 | 描述 2 | 描述 3 |
| --- | --- | --- | --- | --- |
| 0 | Canvas | coarse woven cloth | sturdy fabric with visible threads | textile surface with a tight weave |
| 1 | Cotton | soft cotton cloth | woven natural fibre fabric | cotton textile surface |
| 2 | Faux.Fur | artificial fur pile | synthetic fluffy hair surface | imitation fur texture |
| 3 | Faux.Leather | artificial leather surface | synthetic leather coating | smooth imitation leather |
| 4 | Full.grain.leather | natural leather grain | leather hide with visible surface texture | thick leather retaining its grain |
| 5 | Hair.Calf | short animal hair on leather | hair covered hide surface | calf hide with visible hair |
| 6 | Leather | animal hide surface | smooth leather texture | natural leather with fine grain |
| 7 | Nubuck | finely brushed leather surface | short soft nap on leather | matte leather with a velvet texture |
| 8 | Nylon | smooth synthetic fibre fabric | light woven nylon textile | fine synthetic cloth texture |
| 9 | Patent.Leather | glossy coated leather | shiny reflective leather surface | smooth leather with a bright finish |
| 10 | Rubber | flexible rubber surface | moulded elastic material | smooth or textured rubber coating |
| 11 | Satin | smooth lustrous fabric | silky textile with a soft sheen | fine woven cloth with a shiny face |
| 12 | Sheepskin | sheep hide with wool lining | soft hide with a fleece surface | sheepskin leather and wool texture |
| 13 | Suede | soft brushed leather | matte leather with a visible nap | velvet like leather texture |
| 14 | Synthetic | man made upper material | artificial fibre surface | synthetic fabric or coating |
| 15 | Wool | wool fibre textile | soft fuzzy woven fabric | dense natural wool texture |

### C.2 对象节点 OE

| ID | 原始标签 | 描述 1 | 描述 2 | 描述 3 |
| --- | --- | --- | --- | --- |
| 0 | Boots.Ankle | short shaft around the ankle | enclosed foot with low boot collar | boot upper ending near the ankle |
| 1 | Boots.Knee.High | tall shaft reaching the knee | long boot upper covering the lower leg | high collar close to the knee |
| 2 | Boots.Mid-Calf | boot shaft ending around the middle calf | medium height collar above the ankle | boot upper below the knee |
| 3 | Sandals | open upper with straps | exposed toes and sides of the foot | straps attached to an open sole |
| 4 | Shoes.Boat.Shoes | low shoe with laces around the sides | casual shoe with stitched toe seam | boat style shoe with short front lacing |
| 5 | Shoes.Clogs.and.Mules | slip on shoe with an open back | closed front without a full heel collar | backless upper covering the front of the foot |
| 6 | Shoes.Flats | low heel close to the ground | shallow collar exposing the top of the foot | flat sole with a light low upper |
| 7 | Shoes.Heels | raised heel under the back of the foot | heel noticeably higher than the front sole | elevated rear support with a sloping foot bed |
| 8 | Shoes.Loafers | slip on upper without front laces | low enclosed shoe with a broad front panel | casual low collar with a stitched toe seam |
| 9 | Shoes.Oxfords | front laces on a low enclosed shoe | formal shoe with a structured leather upper | closed lacing over the front of the foot |
| 10 | Shoes.Sneakers.and.Athletic.Shoes | thick cushioned sole for movement | sport shoe with a padded collar | athletic upper with laces and flexible panels |
| 11 | Slippers | soft slip on indoor footwear | relaxed upper with a soft lining | easy entry shoe with light indoor sole |

## 附录 D. SG-EKR 的实际保留槽

| 属性 ID | 标签 | 保留 slot（0 为名称） | 被排除的 AE slot |
| --- | --- | --- | --- |
| 0 | Canvas | 0, 1 | 2, 3 |
| 1 | Cotton | 0, 1 | 2, 3 |
| 2 | Faux.Fur | 0, 1, 2, 3 | 无 |
| 3 | Faux.Leather | 0, 1, 2, 3 | 无 |
| 4 | Full.grain.leather | 0, 1, 2, 3 | 无 |
| 5 | Hair.Calf | 0, 1, 2 | 3 |
| 6 | Leather | 0, 1, 2, 3 | 无 |
| 7 | Nubuck | 0 | 1, 2, 3 |
| 8 | Nylon | 0, 1 | 2, 3 |
| 9 | Patent.Leather | 0, 1, 2, 3 | 无 |
| 10 | Rubber | 0, 1, 2, 3 | 无 |
| 11 | Satin | 0, 1 | 2, 3 |
| 12 | Sheepskin | 0, 1 | 2, 3 |
| 13 | Suede | 0 | 1, 2, 3 |
| 14 | Synthetic | 0, 1, 2, 3 | 无 |
| 15 | Wool | 0, 1, 3 | 2 |

这里“排除”仅指 SG mask 关闭对应槽，未从源 AE JSON 删除文字；名称 slot 0 全保留。

## 附录 E. 关键版本指纹

| 关键资产 | SHA256（本地实际文件） |
| --- | --- |
| [冻结 GloVe / build_knowledge.py](<../reproduce_0601_20261010/reference_source/tools/build_knowledge.py>) | `5997b07bee331e9a17c169b11a25c2d2eb9bb27e8cc1b4e3aad10cd5bb6b8687` |
| [generate.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/generate.py>) | `b91fe61dc1bd958a67e7514ca5b4367193ce8c7ec8595875e9dc9bbbdedd25cb` |
| [build_text_prior.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/build_text_prior.py>) | `0aae7f7166d1e9ad43619d63f68c8bfad3df1c8bb8c13e7cd28d6f51a01c33ac` |
| [UT-v2 / manifest.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/manifest.json>) | `6919219f8d99d9f3107fea2a80e31ea25b8a8f54ccaa899c0c917a351414466d` |
| [UT-v2 EKR bank / bank.pt](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/bank.pt>) | `94259cca60bdc132d26a6c26a799d823826152d4a9dcf60fe8bcc93e0cca0525` |
| [UT-v2 EKR bank / manifest.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/manifest.json>) | `ed17757749ffe48fb0d3e2303d7032ea6fce12a27eabbdbb6cdfc87fc4a8a71e` |
| [B/F / knowledge.json](<../logiczsl_cpf_bf_20261008/assets/ut-zap50k/knowledge.json>) | `549bfe4550a99c266eb6bb5df04202aba7657ab285e444213de5af59a0cd1538` |
| [B/F 固定逻辑 bank / bank.pt](<../reproduce_0601_20261010/assets/logic_bank/bank.pt>) | `a8fe1e4fe4629264603569ef6457f19ae737b08bad34a4267c7d39f3bfce7428` |
| [SG-EKR / support_mask.json](<../output/sgekr_review_20261008/artifacts/support_mask.json>) | `fe52b3bfa7d55557786ced9857c1384c6877e191226289f9444b22f7c93f4086` |
| [CLIP-r2 / protocol.json](<../clip_knowledge_probe_20261010_r2/protocol.json>) | `6eb8cdcebab91a168c4c72769ca1e04a72573d330d1ef90fc39c08d267f01a0b` |
| [TRAIN 审查 / train_input_manifest.json](<../output/train_knowledge_audit_results_20261010/extracted/source/train_input_manifest.json>) | `6b73239db9e6bb60f6fb24e70bcd09ebe6e6d6025b2eef7fcb6b1811a0ee5ab5` |

完整本次读取文件的绝对路径、byte size、SHA、核验结果另存于同名 `.sources.json`。这些 hash 用于身份追踪，不替代质量/性能判断。bank schema 的版本字符串可能沿用历史命名；识别源版本应同时看 knowledge_inputs 和文件 hash，不能只看 `description_bank_transition_v1` 字样。

## 附录 F. 全部本地来源与相关知识文件索引

下表是实际文件路径，可逐个打开。本地 cache 轻量副本与服务器大张量的可用性边界已在第 10 节说明。

| 本地证据文件（点击打开） | 来源 / 用途 |
| --- | --- |
| [最初 0601 方案](<../archive_inputs/referenced_sources/0601origin/ZSCL实验设计0601.md>) | 设计来源，不是已执行事实 |
| [全部实验总档案](<../output/截至20261010_CZSL全部实验_公式修正_决策与数据分析总档案.md>) | 已有实验与决策的配套报告，保持不变 |
| [LogiCZSL 原论文 PDF](<../archive_inputs/referenced_sources/CZSL/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf>) | 此前论文方法核对的原件；本次只索引与校验文件身份，不重新解析 PDF |
| [本地作者代码副本](<../archive_inputs/referenced_sources/CZSL/CZSL_code/LogiCZSL_copy.tar.gz>) | 此前逻辑公式/接口审计来源；不包含原作者历史知识 JSON |
| 本地 GloVe 6B 原始 ZIP（原存储位置：`C:/Users/llenovo/Desktop/CZSL/CZSL_code/downloads/glove.6B.zip`） | 可用词向量源包；实际编码成员身份以完整 glove.6B.300d.txt SHA 为准 |
| [WordNet 原始许可](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/WORDNET_LICENSE.txt>) | 初始知识生成包附带的 WordNet 许可 |
| [generate.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/generate.py>) | 初期三数据集知识包的实际脚本 |
| [build_text_prior.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/build_text_prior.py>) | 初期三数据集知识包的实际脚本 |
| [validate_server.py](<../archive_inputs/desktop_experiments/20261004归档/knowledge/kcc_logic_3datasets_reconstructed_v1/scripts/validate_server.py>) | 初期三数据集知识包的实际脚本 |
| [最早 UT / generation_manifest.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/generation_manifest.json>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / build_compatibility.py](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/build_compatibility.py>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / compatibility.csv](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/compatibility.csv>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / README.md](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/README.md>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / oi_dict.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/oi_dict.json>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / oe_dict.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/oe_dict.json>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / ae_dict.json](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/ae_dict.json>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / validate_knowledge.py](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/validate_knowledge.py>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [最早 UT / knowledge-files.sha256](<../archive_inputs/desktop_experiments/CPF_all_results（20261005）/czsl_all_completed_20261005_000428_2772161/files/reconstructed_knowledge/ut-zap50k/knowledge-files.sha256>) | 2026-09-29 非官方语义重建及旧候选掩码 |
| [v1 ut-zap50k / attributes.txt](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/attributes.txt>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / objects.txt](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/objects.txt>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / oi_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/oi_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / oe_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/oe_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / ae_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/ae_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / entry_provenance.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/entry_provenance.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / review_status.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/review_status.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / manifest.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/manifest.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / K_oa_text_lexical.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/K_oa_text_lexical.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 ut-zap50k / K_oa_text_report.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/ut-zap50k/K_oa_text_report.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / attributes.txt](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/attributes.txt>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / objects.txt](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/objects.txt>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / oi_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/oi_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / oe_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/oe_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / ae_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/ae_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / entry_provenance.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/entry_provenance.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / review_status.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/review_status.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / manifest.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/manifest.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / K_oa_text_lexical.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/K_oa_text_lexical.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 mit-states / K_oa_text_report.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/mit-states/K_oa_text_report.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / attributes.txt](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/attributes.txt>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / objects.txt](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/objects.txt>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / oi_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/oi_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / oe_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/oe_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / ae_dict.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/ae_dict.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / entry_provenance.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/entry_provenance.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / review_status.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/review_status.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / manifest.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/manifest.json>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / K_oa_text_lexical.csv](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/K_oa_text_lexical.csv>) | 已核对早期 cache 输入的源知识副本 |
| [v1 cgqa / K_oa_text_report.json](<../output/week5_evidence_20261005_verified/evidence/knowledge_source/cgqa/K_oa_text_report.json>) | 已核对早期 cache 输入的源知识副本 |
| [UT-v2 / attributes.txt](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/attributes.txt>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / objects.txt](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/objects.txt>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / oi_dict.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/oi_dict.json>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / oe_dict.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/oe_dict.json>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / ae_dict.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/ae_dict.json>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / review_status.csv](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/review_status.csv>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / manifest.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/manifest.json>) | TRAIN 图片审查后冻结的源知识 |
| [UT-v2 / item_review.json](<../ut_semantic_v2_20261006/knowledge/ut-zap50k/item_review.json>) | TRAIN 图片审查后冻结的源知识 |
| [TRAIN 审图 / manifest.json](<../ut_semantic_v2_20261006/review_evidence/manifest.json>) | v2 知识改写的图片抽样与来源证据 |
| [TRAIN 审图 / selected_train_images.json](<../ut_semantic_v2_20261006/review_evidence/selected_train_images.json>) | v2 知识改写的图片抽样与来源证据 |
| [TRAIN 审图 / original_descriptions.md](<../ut_semantic_v2_20261006/review_evidence/original_descriptions.md>) | v2 知识改写的图片抽样与来源证据 |
| [UT-v2 cache / manifest.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/cache/manifest.json>) | 服务器构建后回传的缓存元数据；此目录没有所有缓存张量 |
| [UT-v2 cache / labels.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/cache/labels.json>) | 服务器构建后回传的缓存元数据；此目录没有所有缓存张量 |
| [UT-v2 cache / report.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/cache/report.json>) | 服务器构建后回传的缓存元数据；此目录没有所有缓存张量 |
| [UT-v2 cache / cache_manifest.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/cache/cache_manifest.json>) | 服务器构建后回传的缓存元数据；此目录没有所有缓存张量 |
| [UT-v2 EKR bank / bank.pt](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/bank.pt>) | 服务器构建后回传的描述槽资产 |
| [UT-v2 EKR bank / manifest.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/manifest.json>) | 服务器构建后回传的描述槽资产 |
| [UT-v2 EKR bank / labels.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/labels.json>) | 服务器构建后回传的描述槽资产 |
| [UT-v2 EKR bank / descriptions.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/descriptions.json>) | 服务器构建后回传的描述槽资产 |
| [UT-v2 EKR bank / report.json](<../output/BF_conditional_preflight_20261009/extracted/conditional/bank/report.json>) | 服务器构建后回传的描述槽资产 |
| [冻结 GloVe / build_knowledge.py](<../reproduce_0601_20261010/reference_source/tools/build_knowledge.py>) | 原缓存构建脚本的本地冻结副本 |
| [CPF 名称别名 / word_embedding_utils.py](<../reproduce_0601_20261010/reference_source/models/word_embedding_utils.py>) | 名称编码来源 |
| [原 Text/Prior 读取 / kcc_cache.py](<../reproduce_0601_20261010/reference_source/models/kcc_cache.py>) | 实际模型消费及校验 |
| [CPF Text/Prior 前向 / kcc_cpf.py](<../reproduce_0601_20261010/reference_source/models/kcc_cpf.py>) | 固定知识表在文本组合、文字引导和分类中的具体消费 |
| [CLIP-r2 / metrics.py](<../clip_knowledge_probe_20261010_r2/metrics.py>) | 图文分数沿属性轴转换为条件教师概率 |
| [EKR / bank.py](<../ut_semantic_v2_20261006/transition/bank.py>) | 描述槽、均值、残差、固定错配的构建 |
| [旧 EKR / router.py](<../ut_semantic_v2_20261006/transition/router.py>) | 中心化 key/value 与对象条件查询的读取 |
| [名称锚点 / name_router.py](<../name_anchor_ekr_20261007/transition/name_router.py>) | 名称控制的非零值与仅名称对象上下文 |
| [共享值对齐 / core.py](<../ekr_v2_anchor_alignment_20261007/core.py>) | 原始槽的低秩 value 处理，不重写源知识 |
| [初次 TRAIN 语义支持 / probe.py](<../ekr_train_support_probe_20261008/probe.py>) | 固定语义几何的初步诊断，不是 SG 两折 mask |
| [SG-EKR / core.py](<../sgekr_support_mask_20261008/core.py>) | 固定半单位值、recipient 掩码与视觉查询 |
| [UT-v2 / v2_prepare.py](<../ut_semantic_v2_20261006/transition/v2_prepare.py>) | v2 质量门槛与构建串联 |
| [B/F / knowledge.json](<../logiczsl_cpf_bf_20261008/assets/ut-zap50k/knowledge.json>) | 另一套论文方法重建源知识 |
| [B/F / build_bank.py](<../logiczsl_cpf_bf_20261008/build_bank.py>) | 固定逻辑节点编码 |
| [B/F / common.py](<../logiczsl_cpf_bf_20261008/common.py>) | OI/OE/AE 结构校验 |
| [B/F / bf_logic.py](<../logiczsl_cpf_bf_20261008/model/bf_logic.py>) | 原生 CPF 视觉分支到知识节点的 grounding |
| [B/F / bf_logic_loss.py](<../reproduce_0601_20261010/reference_source/models/bf_logic_loss.py>) | 论文三项逻辑约束的实际实现 |
| [B/F 固定逻辑 bank / manifest.json](<../reproduce_0601_20261010/assets/logic_bank/manifest.json>) | 后续执行包携带的同源 B/F 逻辑资产 |
| [B/F 固定逻辑 bank / bank.pt](<../reproduce_0601_20261010/assets/logic_bank/bank.pt>) | 后续执行包携带的同源 B/F 逻辑资产 |
| [LogiCZSL 方法重建手册](<../output/LogiCZSL论文方法重建_CPF-BF训练完整手册_20261008.md>) | 论文核对、原代码审计与适配说明；部分状态是发布时状态 |
| [SG-EKR / pilot.py](<../sgekr_support_mask_20261008/pilot.py>) | TRAIN 视觉支持掩码构建 |
| [SG-EKR / support_mask.json](<../output/sgekr_review_20261008/artifacts/support_mask.json>) | 已运行的两折视觉支持选择结果 |
| [条件属性头 / head.py](<../conditional_attribute_pilot_20261009/head.py>) | 均值描述与名称/错配控制的实际处理 |
| [StageA / prepare.py](<../conditional_attribute_stageA_20261009/prepare.py>) | 冻结 CPF 视觉/基准分数缓存，不是源知识 |
| [CLIP-r2 / protocol.json](<../clip_knowledge_probe_20261010_r2/protocol.json>) | 固定模型、知识、名称、置换和数据身份 |
| [CLIP-r2 / prompts.py](<../clip_knowledge_probe_20261010_r2/prompts.py>) | 七组等权九槽 prompt 构建 |
| [CLIP-r2 / probe.py](<../clip_knowledge_probe_20261010_r2/probe.py>) | 冻结图文编码、均值和教师概率计算 |
| [CLIP-r2 / ae_dict.json](<../clip_knowledge_probe_20261010_r2/assets/ae_dict.json>) | 复制 v2 原文，不是新改写知识 |
| [CLIP-r2 / oe_dict.json](<../clip_knowledge_probe_20261010_r2/assets/oe_dict.json>) | 复制 v2 原文，不是新改写知识 |
| [CLIP-r2 结果 ZIP](<../output/clip_knowledge_probe_results_20261010_r2/czsl_clip_knowledge_probe_20261010_r2.zip>) | 包含图文缓存、prompts、教师输出和审计；本报告不展开大数组 |
| [CLIP-r2 独立复核](<../output/clip_knowledge_probe_results_20261010_r2/CLIP知识证据探针_r2_独立复核与下一步判断_20261010.md>) | 现有探针结论 |
| [TRAIN 审查 / input_reads.json](<../output/train_knowledge_audit_results_20261010/extracted/result/input_reads.json>) | 逐句证据、输入范围与未填人工审查表 |
| [TRAIN 审查 / complete.json](<../output/train_knowledge_audit_results_20261010/extracted/result/complete.json>) | 逐句证据、输入范围与未填人工审查表 |
| [TRAIN 审查 / human_review_blank.json](<../output/train_knowledge_audit_results_20261010/extracted/result/human_review_blank.json>) | 逐句证据、输入范围与未填人工审查表 |
| [TRAIN 审查 / sentence_train_report.json](<../output/train_knowledge_audit_results_20261010/extracted/result/sentence_train_report.json>) | 逐句证据、输入范围与未填人工审查表 |
| [TRAIN 审查 / unit_text_receipt.json](<../output/train_knowledge_audit_results_20261010/extracted/result/unit_text_receipt.json>) | 逐句证据、输入范围与未填人工审查表 |
| [TRAIN 审查 / run.py](<../output/train_knowledge_audit_results_20261010/extracted/source/run.py>) | 重编码单槽文本并分析已有 TRAIN 缓存 |
| [TRAIN 审查 / train_input_manifest.json](<../output/train_knowledge_audit_results_20261010/extracted/source/train_input_manifest.json>) | 逐文件锁定已使用的原 CLIP-r2 输入 |
| [TRAIN 知识落地复核](<../output/train_knowledge_audit_results_20261010/TRAIN知识落地审查_独立复核与路线判断_20261010.md>) | 已完成的审查结论，不是知识新版本 |
| [0601 G/H / full_feature.py](<../reproduce_0601_20261010/source/models/full_feature.py>) | 已实现的 pair block 消费路径，尚无本报告核实的训练结果 |
| [0601 G/H 执行包说明](<../reproduce_0601_20261010/README.md>) | 后续实现/待执行与已有结果分界 |
| [三数据集知识编码统计](<../output/week5_evidence_20261005_verified/tables/knowledge_summary.csv>) | 早期缓存数值与来源摘要 |

---

归档原则：原作者历史知识、非官方语义重建、编码后的固定文本表、从 TRAIN 派生的视觉支持，以及训练出的模型参数，必须分别命名、分别追溯。当前结果不支持将它们笼统合并为“已恢复并验证有效的 LogiCZSL 知识”。
