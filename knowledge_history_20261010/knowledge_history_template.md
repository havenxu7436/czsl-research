# CZSL 知识恢复、构建与处理完整档案：来源、文件与作用

整理日期：2026-10-10。范围：已保存的原方案、源知识、构建代码、服务器回传 manifest、实验资产与知识审查结果。本文独立于 {{L:full_report}}，只完整追踪“知识从哪里来、怎样成为模型输入、哪些部分实际被用过”，不重复所有训练主表。

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

方法核对、原代码问题与论文来源见 {{L:bf_manual}}。这份旧手册中“尚未训练”等文字是发布时状态；本档案和全部实验总档案记录的是后续已回传证据。

直接方法来源是 {{L:original_paper}} 的 §3.2–3.4、Fig.4 与 Eq.6/9/12/13；代码参照来源是 {{L:author_copy}}。这些原件此前已做方法核对，本次只整理和确认文件身份，不重复解析论文或运行作者代码。知识 JSON 的实际构建来源则必须分别写作“assistant 论文方法重建”“WordNet 规则草稿”或“TRAIN 审查后的修订”，不能只把来源写成论文标题。

## 2. 从原方案到实际资产：知识不是一个文件

{{L:plan}} 预期包含 OI、OE、AE、对象—属性兼容性先验，并有 Text、Calibration、Feature、Logic 四类用途。实际推进时，先完成了能对接原 CPF/GloVe 的确定性预处理，之后才增加视觉支持筛选和逻辑 grounding。

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

直接来源记录：{{L:initial_generation_manifest.json}}；构建规则：{{L:initial_build_compatibility.py}}。

早期 UT 的 OI/OE/AE 是按论文知识树结构、UT 的 16 个属性和 12 个对象标签进行的语义重建。对象共 36 条 OE，属性共 48 条 AE。最早原文文件为 {{L:initial_oi_dict.json}}、{{L:initial_oe_dict.json}}、{{L:initial_ae_dict.json}}。没有可用于宣称“逐字复现原作者”的完整历史 prompt、raw GPT response、模型版本或采样参数。

旧 {{L:initial_compatibility.csv}} 的含义必须单列。它由 train、val、test 三份 pair 文件的并集构造；旧脚本的矩阵轴为 **attribute × object，16×12**：

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

源生成器 {{L:generate.py}} 从 `compositional-split-natural` 的 train/val/test pair 清单提取属性与对象并去重排序。UT/MIT 的初始来源是本地 `compositional_split_natural.tar.gz`；C-GQA 初始来源在 manifest 中记录为更新后的 pair-list 镜像。脚本也支持 `--source-data-root` 使用精确服务器 split 文件。

后续 GloVe 构建器再次调用实际项目 `CompositionDataset.parse_split()` 检查标签顺序；不实例化图片读取流程。初始镜像的来源记录与服务器实际训练轴校验是两件事，不能省掉后者。

使用 test pair 元数据确定公开 primitive 词表，不等于读取 test 图片；但如果把 test pair 存在性写入语义兼容性先验，就超出了这个用途。构建器将 split membership 仅用于标签对齐和覆盖审计。

### 5.2 UT：复用既有人工式重建；MIT/C-GQA：WordNet 自动草稿

{{T:source_counts}}

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

UT-v1 三个源 JSON：{{L:v1_ut-zap50k_oi_dict.json}}、{{L:v1_ut-zap50k_oe_dict.json}}、{{L:v1_ut-zap50k_ae_dict.json}}。MIT/C-GQA 的全部源文件在附录文件索引中逐个给出。

每次构建都必须把“目录名字”“manifest 内容”“实际文件 hash”一起保存。目录名含 `trusted`、`reviewed`，不替代内部的真实状态字段。

## 7. GloVe 文本处理：名称、描述与 primitive K

实际冻结构建器：{{L:glove_builder}}。名称别名来源：{{L:aliases}}。编码器固定为 GloVe 6B 300d，float32，完整文本身份为：

```text
91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed
```

构建时一次扫描 GloVe，仅保留所需 token，但 SHA 对完整文本字节计算。因此“只留部分 token”不等于无法确认完整编码器来源。

本地可用原始词向量压缩包为 {{L:glove_archive}}。ZIP 的 hash 与内部 `glove.6B.300d.txt` 的 hash 是两种不同身份；本文保留包的实际文件索引/指纹，但没有重新解压并扫描全部成员。WordNet 的许可随初始生成包保存为 {{L:wordnet_license}}；本档案以源 manifest 的 WordNet 包指纹追溯，不声称当前索引已经备份了整个词典 corpus。

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

实际消费代码 {{L:cpf_consumer}} 使用固定 `W_o_K/W_a_K` 替代原名称表的读取：组合类文本由两个向量拼接后经原 `compose` 模块生成；对象文本参与原文字引导视觉选择；两表还作为对象/属性分类权重。固定输入本身不优化，原视觉、compose 等模块继续按 CPF 协议学习，因此 Text K 并非只在输出分数上加一个常数。

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

{{L:build_text_prior.py}} 以对象名称＋OE、属性名称＋AE 为文档，删除指定英文停用词，计算词级 TF-IDF 与带词边界标记的字符三元组 TF-IDF：

$$
K^{\mathrm{lexical}}_{i,j}
 =0.75\,\operatorname{cos}(v_{o_i}^{\mathrm{word}},v_{a_j}^{\mathrm{word}})
  +0.25\,\operatorname{cos}(v_{o_i}^{\mathrm{char}},v_{a_j}^{\mathrm{char}}).
$$

TF 使用 $1+\log\mathrm{count}$，IDF 使用 $1+\log((D+1)/(\mathrm{df}+1))$，每个文档向量单位化。CSV 值保留 8 位小数并限制在 [0,1]。其 tokenizer/停用词处理与 GloVe builder 不同。

只有显式选择 `--prior-source lexical`，构建器才把它作为 `K_oa_text.pt` 的来源，再使用相同 centered 转换。已完成主训练的 `prior_source=template`；本文没有找到支持“已完成 lexical K 主训练”的证据。

原方案允许 `K_oa_llm.pt` 和 `K_oa_mix.pt`，但当前没有可核实的可靠 LLM feasibility 资产及相应完整实验。不能以全零矩阵或 split 并集冒充 LLM K，再混合后报成真实语义知识。

### 9.4 已保存编码统计

{{T:encoding_stats}}

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

本地轻量证据目录通常仅回传 labels、manifest、report，不含所有 `.pt`。本次可直接打开的 v2 cache 元数据为 {{L:cache_manifest.json}}、{{L:cache_report.json}}。不能向用户承诺这些目录已经备份了所有张量或原始 GloVe；manifest 记录输出 SHA 不等于相应大文件已下载到此目录。

构建器先在同父目录临时 stage 生成，再原子改名。输出不得与源知识目录重叠；非空目标拒绝覆盖。重建同一方法应新建版本，而不是删除原 cache 后覆盖旧实验。

## 11. UT-v2：TRAIN 审查到底改变了什么

来源证据：{{L:gallery_manifest.json}}、{{L:gallery_selected_train_images.json}}、{{L:v2_item_review.json}}、{{L:v2_manifest.json}}。

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

{{T:lineage}}

OI 的 hash 不同但 JSON 对象语义相同，原因是序列化字节改变（UTF-8/LF 等），不是增加/删除了父类。OE/AE 发生描述内容变化。新源目录保留 previous source 与 knowledge SHA；名称 raw 表按原 CPF 规则应保持不变。

v2 外层准备检查：任何描述 fallback 停止；单句 OOV 占比不得超过 50%；每个轴全部描述 OOV 不得超过 10%。这些检查位于 {{L:v2_prepare}}，不是说 `bank.py` 本身永远不会产生 fallback。

## 12. EKR description bank：从聚合表恢复描述槽，不是原作者数据恢复

构建代码 {{L:bank_builder}} 从与 C cache 一致的源 AE/OE、原始名称表、同一 GloVe 重新得到四个单位槽：名称 slot 0，描述 slot 1/2/3。这里的“恢复槽”是把我们自己已有的源句子重新编码，不是从一个均值向量反演出遗失的 GPT 文本。

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

外层文件为 `bank.pt`、`labels.json`、`descriptions.json`、`report.json`、`manifest.json`。本地真实文件：{{L:bank_bank.pt}}；逐句 token/OOV 记录：{{L:bank_descriptions.json}}；数值审计：{{L:bank_report.json}}。

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

旧路径实现 {{L:ekr_router}}；名称锚点修正 {{L:name_router}}；共享值处理 {{L:aligned_values}}。null 节点不是外部知识条目：它是可学习的“不给新增证据”的选项，value=0。query/key/null 与槽 softmax 温度 0.2 属于模型读取，不能与静态 K 的概率混称。

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

来源 {{L:sg_builder}}；产物 {{L:sg_mask}}。输入是已缓存的 498 张 TRAIN 属性视觉特征：83 个 TRAIN 组合各 6 张图。按每组合图像 ID 排序后交替分为 3/3 两折。

在此之前，{{L:train_support_probe}} 已做过一次无 optimizer 的 TRAIN 几何诊断：正确四槽均值相对错配的宏 margin +0.026839，但相对名称 −0.004576；one-vs-rest 有 34/48 个 AE 正支持槽。它不同于后续同对象、对象平衡、两折保守筛选的 30/48 个槽，不能把 34 和 30 混报为同一个阈值的结果。

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

实际使用 {{L:sg_consumer}} 固定 value=单位槽/2，只训练 query/key/null/gain。所有源句仍在 bank 中，mask 只限制本路径可被注意力选择的槽。

筛选没有重写 AE/OE。SG 的已有结果也不能被解释成“删掉了 18 条无用知识就一定提高性能”。

## 14. B/F：另一套论文方法重建知识与逻辑 bank

### 14.1 来源和与 UT-v1/v2 的区别

{{L:bf_knowledge}} 的 provenance 明确声明：`paper_method_reconstructed_ut_v1_20261008`、`assistant_reconstructed_requires_user_review`、`official_author_data=false`；论文引导的 assistant 语义重建，不是历史 GPT-4 输出。构建范围是 UT primitive 标签，没有根据 VAL/TEST 指标生成新知识。

其 OI 是 5 个父类，而旧 v1/v2 OI 是 4 个。B/F 的 AE/OE 文本也与 v2 不同；完整内容见附录 C，不要误把两套源字典互换。

{{T:bf_groups}}

parent 顺序由校验器排序后冻结，`object_parent` 为：

```text
[0, 0, 0, 4, 1, 3, 1, 2, 1, 1, 1, 3]
```

它不是 knowledge.json 原始 JSON 插入顺序。父类全覆盖，每对象仅一个父类；Sandals、Heels 的父类只有单成员，语义抽象与约束强度需保留这一限制。

### 14.2 逻辑节点怎么编码

{{L:bf_builder}} 读取 knowledge、labels、GloVe，将 5 个父类短语、36 条 OE、48 条 AE 分别编码。正则 `[a-z]+`，每个短语取 in-vocabulary GloVe 均值后单位化。每条覆盖率必须至少 0.8，且非空/非零/finite，否则停止；不悄悄退回名称向量。

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

可核实的固定资产副本：{{L:bf_manifest.json}}、{{L:bf_bank.pt}}。这套构建器不生成组合 compatibility prior。

### 14.3 知识如何成为逻辑约束

{{L:bf_adapter}} 直接用 CPF 原生 classifier，把对象视觉表示打到 parent/OE，把属性视觉表示打到 AE；没有新增可训练知识 head。知识向量固定，但梯度经 logits 回传至视觉分支与原 CPF。

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

原语 → 其描述，而不是把 OE/AE 蕴含方向写反。β 只乘一次。零违反度时使用等价的向量范数聚合保持有限梯度。没有额外知识节点 CE，也没有让逻辑节点直接改变推理分数。实际实现 {{L:bf_loss}}。

此前作者代码副本审计发现：逻辑分支注释、部分入口仍返回 CE 而非计算出的总损失、知识字段命名不一致，以及原 JSON 路径占位符。B/F 因此重新建立实际节点 logits、显式总 loss 和梯度检查，而非仅复制源代码或放置知识文件就宣称知识已训练。这里是对本地副本的归档审计，不代表本次重新验证作者线上仓库的最新状态。

### 14.4 B 和 F 使用的知识并不完全相同

| 实验 | Text/Prior 输入 | Logic 输入 | 关键边界 |
| --- | --- | --- | --- |
| B | 纯 CPF 名称底座，不启用 Text K/Prior | 新 5 父类 B/F bank | 检验加入论文式逻辑正则 |
| F | 旧 UT-v1 `W_a_K/W_o_K`＋template P，λ=0.1 | 与 B 同一新 B/F bank | F=旧 E＋Logic，不是 UT-v2 C_text＋Logic |

`--allow-reconstructed` / `bf_allow_reconstructed` 是明确承认使用重建资产的探索性质，不会把知识状态改为官方或人工审核完成。B/F 的完整训练已回传，当前仍无充分新增收益；训练完成与语义知识验收是独立状态。

## 15. 条件属性头 / generic：已有知识怎样成为视觉相关输入

{{L:conditional_head}} 使用同一 UT-v2 bank，把名称和三句均值分开：

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

{{L:clip_ae_dict.json}}、{{L:clip_oe_dict.json}} 与 UT-v2 字典 byte-for-byte 相同；TRAIN 审查包中的两份原文也相同。本次核查已验证实际文件 SHA，不能把 CLIP 探针写成“生成了新版视觉知识”。

{{L:clip_protocol}} 固定 OpenAI CLIP ViT-L/14，224px，768 维；源码 commit=`a1d071733d7111c9c014f024669f959182114e33`；权重 SHA：

```text
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
```

这里是归档协议记录的编码器来源，本文没有重新下载或查询最新版 CLIP。CLIP 自然语言名称表显式冻结，例如 `Synthetic → synthetic material`；它不同于原 CPF 的别名 token。

{{L:clip_prompts}} 的模板为：

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

归档入口 {{L:clip_result}}。原探针 v1 错误断言 train/val pair 完全互斥；r2 修正为 VAL 15 个 seen＋15 个 unseen，仍验证 TRAIN/VAL 图像 ID 与图像字节身份无交集。该修复没有改知识或利用错误期间的指标选句。

### 16.3 TRAIN-only 审查新增了什么文件

原 CLIP 结果只有 9 槽均值，不足以分析单句，因此 {{L:train_run}} 在同一冻结编码器下重编码 7,041 个唯一文本，保存 `unit_text_features.npz` 和 {{L:train_unit_text_receipt.json}}，再恢复每变体的 12×16×9×768 单槽表。

它复用原 CLIP TRAIN 图片特征，没有调用图像编码器；只解码 96 张固定 TRAIN 图片用于拼图。{{L:train_input_reads.json}} 和 {{L:train_complete.json}} 记录：VAL/TEST 文件读取空列表、optimizer step=0、weights fitted=false。

| 新文件 | 来源 / 用途 | 当前地位 |
| --- | --- | --- |
| `unit_text_features.npz` | 7,041 个单位文本向量 | 派生文本缓存，不是新 AE/OE |
| `unit_text_receipt.json` | 文本清单、batch 与文件 hash | 单槽 cache 身份 |
| `sentence_train_report.json` | 9 个 OE+AE 位置、3 个 AE-only 位置，各带正确/两错配控制 | 36 个 TRAIN 位置对照，不是新 prompt 训练 |
| TRAIN 拼图 / `review.html` | 固定 96 张 TRAIN 样本 | 质性适用性审阅，不是材质专家新真值 |
| `human_review_blank.json` | 空白人工可见性/区分性评分表 | 目前没有已填专家验收记录 |
| `train_input_manifest.json` | 原探针 TRAIN 输入白名单与 SHA | 证明读取边界，不代表可合法声称盲试验 |

{{L:train_human_review_blank.json}} 中评分仍为 null；文件包含 review instructions 是表单说明，不是本次用户要求执行的新指令。

## 17. 原 0601 新 G/H 实现中的知识资产作用

{{L:full_feature}} 已有实际消费 pair block 的代码；{{L:full_readme}} 明确这是后续执行包。本文以当前可验证的已回传历史为界，不将执行包、CPU tests 或 smoke 当成 G/H 完整训练结果。

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

{{L:clip_review}}：OE+AE 的 VAL unseen pair macro 相对名称 +0.3859 pp，但 micro −12.9653 pp，且 macro 低于两套错配；AE-only 也未过门槛。它们是 GT-object 属性机制诊断，不是最终部署 CW/OW AUC。

{{L:train_review}}：名称 TRAIN micro/macro=8.7921/22.3766%；AE-only 三句均值为 11.2662/22.6919%；仅各类第三条 AE 为 25.1935/23.4280%。后者相对名称净增 3,772 张，其中 Leather 净增 3,293 张，收益高度集中，且仅为 TRAIN 位置审查。

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
| 重建初期语义草稿 | {{L:generate.py}} | 会写源文件，只应在新 draft 目录；当前不需要重跑 |
| lexical 候选 | {{L:build_text_prior.py}} | 会更新源目录 manifest，不应在冻结 trusted/v2 上直接运行 |
| 源目录结构检查 | {{L:validate_server.py}} | 核对 exact server primitive 轴；不是知识质量认证 |
| 确定性 GloVe cache | {{L:glove_builder}} | 源只读，新输出目录；保留真实配置/编码器身份 |
| EKR 四槽 bank | {{L:bank_builder}} | 源 hash 必须与 C cache 一致，重新聚合误差≤1e−6 |
| v2 准备 | {{L:v2_prepare}} | 强制 OOV/fallback、名称和输入身份守卫 |
| BF 节点 bank | {{L:bf_builder}} | 明确认可重建性质，覆盖不足停止，无 split-derived prior |
| CLIP 编码与探针 | {{L:clip_probe}} | 固定 protocol；不是全模型训练或 prompt 搜索 |
| TRAIN 逐句审查 | {{L:train_run}} | TRAIN 输入白名单；不据此偷偷改 VAL 门槛或源字典 |

本文不给会覆盖历史知识的“一键重建”命令：用户本次要求的是单独整理档案。上表是实际来源与入口索引，不代表本轮又执行了这些构建器。

### 20.3 这次整理验证了什么

本档案生成时重新检查：所有文内本地文件链接存在；三数据集 v1 与 UT-v2 源 manifest 的知识 SHA 正确；v2 cache builder SHA 与本地冻结代码一致；cache 两个 manifest 副本一致；EKR bank 的输出与源知识 hash 链一致；CLIP/TRAIN 的 AE/OE 与 v2 字节相同；B/F source 与 bank hash 正确。旧全部实验报告与所有读取的证据保持不变。

边界：未重新读取服务器实时文件系统，未执行原编码器 GPU 前向，未加载并逐元素回算全部知识张量，也未重新审查所有图像或专家材料真值。历史报告中的数值依其现有核验来源引用，不把这次 manifest 核查称为完整 GPU 复现。

## 附录 A. 当前 UT-v2 全部 AE 原文与固定标签顺序

原文来源 {{L:v2_ae_dict.json}}；ID 为 0-based，直接对应 bank 的属性行。描述 1/2/3 对应 bank slot 1/2/3，不是 slot 0。

{{T:v2_attrs}}

## 附录 B. 当前 UT-v2 全部 OE 原文与固定标签顺序

原文来源 {{L:v2_oe_dict.json}}；对应对象行。

{{T:v2_objs}}

旧/v2 OI 的 4 个父类为 `boot`、`casual shoe`、`dress shoe`、`open or indoor footwear`，分组见 {{L:v2_oi_dict.json}}。它们不是附录 C 的 5 父类 bank。

## 附录 C. B/F 独立重建的全部 AE/OE 原文

以下来自 {{L:bf_knowledge}}，不是 UT-v2。父类对应第 14 节的排序后 IDs。

### C.1 属性节点 AE

{{T:bf_attrs}}

### C.2 对象节点 OE

{{T:bf_objs}}

## 附录 D. SG-EKR 的实际保留槽

{{T:mask}}

这里“排除”仅指 SG mask 关闭对应槽，未从源 AE JSON 删除文字；名称 slot 0 全保留。

## 附录 E. 关键版本指纹

{{T:asset_hashes}}

完整本次读取文件的绝对路径、byte size、SHA、核验结果另存于同名 `.sources.json`。这些 hash 用于身份追踪，不替代质量/性能判断。bank schema 的版本字符串可能沿用历史命名；识别源版本应同时看 knowledge_inputs 和文件 hash，不能只看 `description_bank_transition_v1` 字样。

## 附录 F. 全部本地来源与相关知识文件索引

下表是实际文件路径，可逐个打开。本地 cache 轻量副本与服务器大张量的可用性边界已在第 10 节说明。

{{T:source_index}}

---

归档原则：原作者历史知识、非官方语义重建、编码后的固定文本表、从 TRAIN 派生的视觉支持，以及训练出的模型参数，必须分别命名、分别追溯。当前结果不支持将它们笼统合并为“已恢复并验证有效的 LogiCZSL 知识”。
