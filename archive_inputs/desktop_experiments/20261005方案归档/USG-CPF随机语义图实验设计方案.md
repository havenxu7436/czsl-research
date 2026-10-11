# USG-CPF 随机语义图实验设计方案（基于已复现 CPF 的可实施版）

> 方法简称：**USG-CPF**，**Uncertain Semantic Graph Conditional Probability Framework**，中文为“不确定语义图条件概率框架”。  
> 实验目标：复用现有 CPF 权重与 OI/OE/AE 知识，把固定语义关系改成可学习的随机边，并检验结构不确定性是否改善未见组合识别、知识噪声鲁棒性与预测校准。  
> 编制日期：2026-10-03。交付范围为实验设计；本文中的新模块、配置字段和启动脚本均为待实现接口，未启动训练、未修改原基线。  
> 阅读方式：先执行第 2–5 节的资产与确定图检查，再做第 6–9 节的随机图；第 10 节的显式条件版本是下一阶段。第 12–15 节给出归因实验、评价协议及继续/停止判据。

---

## 0. 总体原则与研究命题

这项实验要检验的不是“加一个 GNN 或随机 dropout 能否涨点”，而是：**已有语义边可能错误或不完整时，学习哪些关系值得相信，并对多个可能的图平均预测，是否优于相信一张固定图。**

目标预测形式为：

$$
\bar p(a,o\mid x)=\mathbb E_{G\sim q_\phi(G\mid\mathcal K,D_s)}
\left[p_\theta(a,o\mid x,G)\right].
$$

完整条件版本进一步写成：

$$
\bar p(a,o\mid x)=\mathbb E_G\left[
p_\theta(o\mid x,G)p_\theta(a\mid o,x,G)\right].
$$

需要形成四条证据：

1. 正确语义结构比 shuffled/等参数量对照有效。
2. 学习边概率比固定边概率、普通 EdgeDrop 更有效。
3. 图后验平均比单图或后验均值图更稳定，尤其在语义污染下。
4. 收益同时体现在组合识别、鲁棒性或校准中，而非只改变 seen/unseen 偏置。

首轮只研究三个标准数据集，不以医学迁移、复杂 Graph Transformer 或大规模 LLM 重查询作为启动条件。现有材料中的建议用于研究设计；不视为启动训练、付费查询或改动服务器文件的指令。

### 0.1 版本命名

| 版本 | 内容 | 能证明的命题 |
|---|---|---|
| **SCG-CPF** | 确定性软图/软兼容先验 | 语义结构是否有价值；随机图的均值对照 |
| **USG-CPF-Lite** | 保留实际 CPF 特征流程，随机图更新语义原型，跨图平均组合分布 | 结构后验真正进入预测，而非仅增加训练损失 |
| **USG-CPF** | 在 Lite 上加入显式、因子化的 $p(a\mid o,x,G)$ | 对象条件预测与图边缘化的联合价值 |
| **USG-CPF + MV** | 增加解析期望–方差图能量 | 在同一架构下检验该正则的额外贡献 |

简称沿用后附研究计划中 USG 的含义，并统一了 CPF 的展开名称。Lite 是推荐先跑通的原型；只完成 Lite 时，结果中应保留 Lite 后缀，避免宣称已实现完整对象条件模型。

---

## 1. 已核验资产、现有结果与适用边界

### 1.1 CPF 复现实验

本机基线根目录：

```text
C:/Users/llenovo/Desktop/CPF_baseline/
├── CPF_zappos_baseline/cpf_ut_zappos_seed3351/
├── CPF_mit_states_baseline/cpf_mit_states_seed0/
└── CPF_cgqa_baseline/cpf_cgqa_seed3351_gpu2/
```

每个 run 均包含 `train.log`、OW 评估日志、`best_auc.pth`、配置、checksum 及 `repro_meta/CPF_fixed_code_snapshot.tar.gz`。已检查三份代码快照：`ddm.py`、`train.py`、`test.py`、`dataset.py`、`evaluator_ge.py` 对应文件的 SHA256 一致。

以下数值来自各 run 的 `repro_meta/final_results.txt`，均按百分数尺度报告；不是本次重新运行的结果。

| 数据集 | seed | 训练轮数 / 最佳验证轮次 | CW AUC | CW HM | CW S / U | OW AUC | OW HM | OW S / U |
|---|---:|---|---:|---:|---|---:|---:|---|
| UT-Zappos50K | 3351 | 10 / 6 | 36.51 | 52.59 | 61.49 / 68.48 | 27.07 | 43.87 | 61.49 / 51.30 |
| MIT-States | 0 | 10 / 10 | 11.05 | 26.57 | 39.66 / 35.81 | 4.18 | 14.67 | 39.66 / 14.38 |
| C-GQA | 3351 | 10 / 9 | 6.81 | 21.30 | 38.42 / 20.37 | 1.58 | 7.27 | 38.42 / 4.98 |

论文中 ViT-B 的 CW AUC/HM 分别为 MIT `11.20/26.80`、UT `41.40/55.70`、C-GQA `8.20/23.90`。当前 MIT 最接近论文；UT 与 C-GQA 尚有复现差距。方法收益应优先相对**相同代码、相同预算的本地 CPF 对照**计算，不能把弥补复现差距直接记成新方法贡献。[CPF 原论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf)

注意：表中的 S/U 是原评估器 `best_seen/best_unseen`，HM 是偏置曲线上的最大 HM。S、U、HM 可能对应不同偏置；不能用这两列 S/U 重新计算表中 HM。

### 1.2 当前实际代码接口

| 项目 | 核验结果 | 实验设计中的处理 |
|---|---|---|
| 实际训练模型 | `train.py` 导入 `from models.ddm import DDM` | 修改 `models/ddm.py` 或新增包装类；不从 `ddm_v3.py` 开始 |
| 视觉主干 | ViT-B/16，多层 patch 特征 | 复用原权重，先不切换 CLIP |
| 语义输入 | 冻结 GloVe，300 维 | 图节点首先使用同一 GloVe 空间 |
| 视觉特征 | 768 维，映射到 300 维语义空间 | 图原型与现有分类器维度匹配 |
| 基础损失 | `L_com + 0.6 L_att + 0.4 L_obj` | 首轮维持相同权重 |
| 训练温度 | 余弦 logit 除以 `0.05` | 训练时保留原规则 |
| 验证分数 | 三个 `scale=False` 余弦分数相加 | 精确保留为 A0 对照 |
| 属性表示 | 一个由全局对象特征引导的属性向量 | 不能直接宣称存在 `B×N×M` 的显式条件属性头 |
| OW 候选 | `list(product(attrs, objs))` | 保留全笛卡尔积，不用 split 存在性做语义过滤 |

`ddm.py` 的关键定位：`train_forward` 约第 291 行，文本引导约第 323 行，原损失约第 418 行，`val_forward` 约第 429 行，原验证融合约第 533 行。源码 SHA256：

```text
d1ebcc4e4a40070ca05ef06520a096c82c0ab02c2388a38698ff4f966d656055
```

第一份设计中的 `B×N×M×D` 只是建议接口，不能把它当成当前代码已有功能。本文通过 Lite 与显式条件版本分开落实。

### 1.3 已生成的逻辑知识

已核验本地目录：

```text
C:/Users/llenovo/Desktop/knowledge/kcc_logic_3datasets_reconstructed_v1/
├── ut-zap50k/
├── mit-states/
├── cgqa/
└── scripts/validate_server.py
```

| 数据集 | 属性 M | 对象 N | OI 父类 | OI 边 | OE 边 | AE 边 | 待复核原语条目 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UT-Zappos50K | 16 | 12 | 4 | 12 | 36 | 48 | 28 |
| MIT-States | 115 | 245 | 19 | 245 | 735 | 345 | 360 |
| C-GQA | 413 | 674 | 23 | 674 | 2022 | 1239 | 1087 |

三类 JSON 的实际格式：

```python
oi_dict = {"parent_name": ["object_1", "object_2"]}
oe_dict = {"object_name": ["description_1", "description_2", "description_3"]}
ae_dict = {"attribute_name": ["description_1", "description_2", "description_3"]}
```

同时存在 `objects.txt`、`attributes.txt`、`entry_provenance.csv`、`review_status.csv`、`manifest.json`。三个数据集的 OI/OE/AE、lexical prior 及其报告共 15 个文件均与 manifest 声明的 SHA256 匹配。

**知识可以直接复用，但其性质需要准确记录。** 当前版本是重建知识：UT 为手工草稿，MIT/C-GQA 为 WordNet 3.0 首词义草稿，并非作者 GPT-4 原始知识。所有 review 条目当前均为 `pending`。例如 MIT 的 OI 中 `beef` 被分到 `animal`，提示词义选择需要检查。允许使用 v1 做接口、原型及“带噪重建知识”探索；正式结论应注明知识版本，优先修正明显错误并冻结新版本，不能把现有结果称为作者知识的严格复现。

### 1.4 兼容性矩阵的实际性质

已有 `K_oa_text_lexical.csv` 为 `N×M`，行对象、列属性。它由词级 TF-IDF 与字符三元组相似度生成；不是已经校准的可行性概率。

| 数据集 | 均值 | 标准差 | 零值比例 |
|---|---:|---:|---:|
| UT | 0.01611 | 0.02369 | 3.65% |
| MIT | 0.00930 | 0.01437 | 9.11% |
| C-GQA | 0.00823 | 0.01527 | 12.03% |

因此首轮随机图只使用 OI/OE/AE，令兼容性融合权重 `eta_compat=0`。lexical matrix 可作独立 text-only 对照；不能直接把零值解释为“不可能组合”，也不能作为 Bernoulli 真值概率。

### 1.5 服务器核验状态

历史运行记录确认服务器根目录为 `/data/xhw/czsl_repro`，源码为 `/data/xhw/czsl_repro/src/CPF_fixed`，数据在 `data/{ut-zap50k,mit-states,cgqa}`。知识包 README 提供了候选位置 `/data/xhw/czsl_repro/data/kcc_logic_3datasets_reconstructed_v1`；配套操作手册另使用 `knowledge/kcc0601_v1`。

本次对历史主机的 SSH 只读连接返回认证失败，故**未确认服务器当前知识目录、当前代码 hash 或 split 身份**。下面的执行步骤需在已有可用服务器会话中核对；本机资产足以完成当前设计，不要求重新生成整套知识。

---

## 2. 实验主线与最小可行目标

推荐路线：

```text
现有 CPF 权重与原评分
    ↓ P0：精确基线与知识对齐
确定图增强原型（SCG-CPF，OI/OE/AE）
    ↓ P1：证明结构有信息，而不是只有额外参数
可学习边概率 + Bernoulli 图 + 预测平均（USG-CPF-Lite）
    ↓ P2：证明后验与采样优于确定图 / EdgeDrop
显式对象条件属性头（USG-CPF）
    ↓ P3：证明条件预测与图后验的互补性
解析期望–方差正则、噪声与校准分析
```

**第一个可完成实验包**：UT 上 A0/A1/B/D/E/F/H，加无图续训、等参数量和 EdgeDrop 对照；MIT 上复核决定性比较 A1/D/H。先不要把逻辑损失、兼容 prior、显式条件头同时打开。

UT 用于低成本调通，MIT 用于主要研究判断，C-GQA 用于大规模验证。短期已有 checkpoint 上的探索只形成 pilot；最终多 seed 主表另外按相同预算运行。

---

## 3. Phase 0：资产冻结、标签对齐与冒烟检查

### 3.1 建立独立源码与实验目录

使用服务器实际 `CPF_fixed` 的副本或本地归档中的完整快照作为起点，保留已有 run。建议：

```text
/data/xhw/czsl_repro/
├── src/CPF_fixed/                     # 已复现版本
├── src/USG_CPF_v1/                    # 新副本，待建立
├── knowledge/usg_v1/                  # 本轮只读知识与缓存
└── runs/usg_v1/{dataset}/{variant}/seed_{seed}/
```

从现有 `train.log` 开头保存**实际生效配置**。归档 `config_used.yml` 有时保留未被 CLI 覆盖的默认值：例如 MIT 文件写 20 epochs / batch 32 / seed -1，实际日志为 10 / 64 / 0。新实验必须归档合并 CLI 后的配置，不能只复制 YAML 并假定它等于已跑设置。

### 3.2 服务器只读定位与校验命令

以下为 Linux/Bash，须在可用服务器会话中执行；只检查目录和文件。

```bash
ROOT=/data/xhw/czsl_repro
find "$ROOT/data" "$ROOT/knowledge" -maxdepth 6 -type f \
  \( -name oi_dict.json -o -name oe_dict.json -o -name ae_dict.json \) \
  -print 2>/dev/null

# 找到本轮实际采用的完整包后，再填写 KROOT。
KROOT="$ROOT/data/kcc_logic_3datasets_reconstructed_v1"
python "$KROOT/scripts/validate_server.py" \
  --data-root "$ROOT/data" --knowledge-root "$KROOT"
```

若包放在其他目录，只修改 `KROOT`；若仅有三份 JSON 而没有校验脚本，使用本地包的校验脚本并指定实际目录。校验通过说明文件结构、词表和 split 身份一致，不等于语义全部正确。C-GQA 若报 pair identity mismatch，按实际 split 重做词表映射与新版本缓存，不能跳过校验。

### 3.3 必须固定的标签契约

- `attrs/objs` 使用当前 `CompositionDataset` 的排序及原始名称。
- JSON 索引由原始名称映射到类别 ID；显示文本可以规范化，但不能改变类别 key。
- pair 始终使用 `(attr_idx, obj_idx)`，候选分数形状为 `B×P`。
- 图中的兼容矩阵如启用，明确为 `[object, attribute]`；用 `K[obj_idx, attr_idx]` gather。
- seen mask 只从训练 pair 列表生成。验证/测试 pair 列表可用于协议规定的候选 mask 与标签校验，不得作为“可行性监督”。
- 新模型缓存中同时保存 `attr_names`、`obj_names`、`candidate_pairs` 和 hash；加载时逐项一致性检查。

### 3.4 P0 验收

1. 关闭全部 USG 开关时，加载原 checkpoint 的同一 batch 分数与原实现一致，FP32 最大绝对差小于 `1e-6`。
2. 原评估器得到相同 CW/OW 指标，允许仅有约 `0.01` 个百分点的打印/浮点误差。
3. `P0` 单独记录，不训练新模块；检验知识 key、标签、mask、checkpoint 对应数据集。
4. 错误应在加载时抛出，不应静默使用全零图、随机原型或缺失类别 fallback。

---

## 4. Phase 1：把现有 OI/OE/AE 转成多关系候选图

### 4.1 节点与边

$$
V=V_o\cup V_a\cup V_h\cup V_{do}\cup V_{da}.
$$

| 节点/关系 | 来源 | ID 示例 / 实现要求 |
|---|---|---|
| 对象节点 | `objects.txt` | `obj:animal` |
| 属性节点 | `attributes.txt` | `attr:young` |
| 上位类节点 | `oi_dict.json` 的 key | `hier:animal`；与同名对象分开 |
| 对象描述节点 | OE 文本 | `odesc:文本hash` |
| 属性描述节点 | AE 文本 | `adesc:文本hash` |
| OI 边 | child object → parent | 有方向，另建逆向消息通道 |
| OE 边 | object → object description | 同上 |
| AE 边 | attribute → attribute description | 同上 |

每条语义关系只产生**一个随机变量** $E_e$；消息传递的正向/逆向共享该变量。否则同一条关系被错误计成两个独立 Bernoulli 变量。

描述按“节点类型 + 完整文本”去重；保留每条边的 concept、JSON 来源、review 状态。C-GQA 有 3261 条描述关联，但只有 3252 个不同描述字符串，故去重不能改变边的归属或覆盖率。三类边均保持覆盖全部原语，不根据某个 pair 是否 seen 决定描述数量。

首轮没有全属性–对象边：现有三类知识已经足够验证随机结构，能避免在 C-GQA 上突然引入 278362 条未经验证的兼容边。

### 4.2 GloVe 节点编码

原语节点直接使用 CPF 原始 embedding，保持完全一致。新增描述和父类节点使用同一 `glove.6B.300d.txt`：

$$
z_t^{(0)}=\frac{1}{|\mathcal T_t|}\sum_{w\in\mathcal T_t} \mathrm{GloVe}(w).
$$

实现顺序：

1. 保留原语的原始 key；仅对显示文本把点、下划线、连字符替换为空格，小写并分词。
2. 描述先删除前缀中重复的类名、明显模板词；清洗规则版本化，不手工按测试结果改文本。
3. 均值只计算词表内 token；记录每节点有效 token 数、OOV 比例和清洗前后文本。
4. 描述全 OOV 时回退到所属原语向量；无法编码的父类使用其 child 原语向量均值。保留 fallback 标记。
5. 不把随机生成的 OOV 向量当成正常知识；图模块关闭时仍使用原 embedding。

先做 name-only、描述均值和 shuffled-descriptions 三组。GloVe 均值是工程起点，未必充分表达长释义。若失败主要与文本编码有关，再把冻结 CLIP 描述向量经 `d_text→300` 投影作为独立消融；不同时更换视觉主干。

### 4.3 边先验：复用草稿，不伪造投票

现有知识没有可核验的五次 LLM 投票。因此不按 Beta 投票公式虚构置信度，也不要求启动时重查全部边。

首轮采用明确标记为**人为设定的关系保留先验**：

$$
p_0(G\mid\mathcal K)=\prod_{e\in E_c}\mathrm{Bern}(E_e;\pi_e^0),
\qquad \pi_e^0=0.7.
$$

三个关系先用相同 `0.7`，敏感性比较 `0.5/0.7/0.9`；不声称 0.7 是知识正确率。人工复核或重复查询后才构建 source-aware/vote-aware 版本，并独立比较。

没有 prior evidence 的边可有中性值，但“没有查询”不是否定证据。首轮后验特征中不加入 pair 的 train/val/test 身份、seen 指示或人为噪声标签。

### 4.4 预处理交付物

```text
knowledge/usg_v1/{dataset}/
├── nodes.json                  # typed ID / 文本 / 原始类别映射
├── edges.csv                   # src,dst,relation,pi0,source
├── node_embeddings.pt          # V×300，含原语 embedding
├── graph_cache.pt              # COO edge_index / type / 固定度数
├── label_map.json              # attrs / objs / pairs / seen mask hash
├── coverage_report.json        # OOV / fallback / 图覆盖 / 去重统计
├── source_manifest.json        # 原知识及脚本 SHA256
└── knowledge_notes.md           # 草稿或复核版本及生成规则
```

正式比较共享同一份只读知识缓存；人工修订后另起版本并重新生成，不在某一方法的 run 中单独修正知识。

---

## 5. Phase 2：确定图过渡与 SCG-CPF 对照

### 5.1 一层关系消息，不堆深 GNN

对关系 $r$，设 $i$ 为接收节点，$j$ 为发送节点，候选边强度先取 $w_e=1$。定义与图采样无关的固定候选度：

$$
d^0_{i,r}=\max\left(1,\sum_{e=(i,j,r)\in E_c}w_e\right).
$$

消息为：

$$
m_{i,r}(G)=\frac{1}{d^0_{i,r}}
\sum_{e=(i,j,r)}E_e w_e\,T_r z_j^{(0)},
\qquad T_r\in\mathbb R^{300\times300}.
$$

$$
h_i(G)=\mathrm{LN}\left[z_i^{(0)}+\mathrm{GELU}\left(\sum_r m_{i,r}(G)\right)\right].
$$

固定候选度可避免采样后的零度除法，并避免“只剩一条边时被重新归一化成满权重”。动态度归一化可以后续比较；各变体必须使用相同约定。

对象/属性原型使用残差：

$$
\widetilde W_o(G)=W_o^0+\rho U_oH_o(G),\qquad
\widetilde W_a(G)=W_a^0+\rho U_aH_a(G).
$$

`U_o/U_a` 零初始化，`rho=0.1`。关系变换可用 `300→64→300` 低秩分解。分类器沿用内部 `F.normalize`；**不要在残差外无条件归一化后再替换原型**，因为原 CPF 的文本加权还会使用原型幅值。`rho=0` 必须精确恢复原始输入。

### 5.2 原型进入 CPF 的具体位置

将 `train_forward/val_forward` 内的 `obj_weight/attr_weight` 替换为上述残差原型：

```python
# 设计接口；需在新模型中实现，不是当前 ddm.py 已有方法。
obj_weight, attr_weight = semantic_graph.prototypes(graph)
text_scores = choose_texts(vis_obj, obj_weight)
obj_guide_texts = text_scores @ obj_weight
# 继续沿原 CPF 的 text→patch、object→shallow-patch 流程。
```

组合文本也要一致：原 `compose_word_embeddings` 直接读取 `nn.Embedding`，应新增显式输入原型的函数，在相同 pair 顺序下执行原 composer。禁止只更新对象分类器却让注意力/组合原型继续使用旧表。

先比较三种确定图：

- hard graph：全部现有边权为 1。
- prior mean graph：边权为 `pi0=0.7`。
- learned mean graph：后验学得 `pi`，只前向均值图，不采样。

### 5.3 三个评分基准必须分开

记原未缩放余弦分数为 $c_c(x,a,o)$、$c_a(x,a)$、$c_o(x,o)$：

$$
B_{a,o}(x,G)=c_c(x,a,o,G)+c_a(x,a,G)+c_o(x,o,G).
$$

1. **A0 原 CPF**：直接输出原 `B` 给原评估器。
2. **A1 概率包装**：$p_G=\mathrm{softmax}_{c\in\mathcal C}(B/T_s)$，`T_s=1`；单图输出 `log p_G`。使用同一原权重时，与 A0 只差每张图像上的类别无关常数，应保留预测及偏置曲线。消融表中的 A1 则在这一包装上增加匹配预算的续训；另记录续训前的 `A1-init` 一致性结果。
3. **其他融合**：如把三分支训练 logit `c/0.05` 相加，或改 `lambda_a/lambda_o`，必须单独命名为 changed-fusion 对照，不能归入图模块收益。

图增强后的 Lite 就沿 A1 定义每个图内的分布，不把三头 softmax 概率直接相加后称为联合概率。

### 5.4 后附软兼容推导的可选复现

若后续已有独立核验的兼容目标 $q_{oa}$，可实现：

$$
S_{a,o}=B_{a,o}+\eta\log\pi_{oa},\qquad
p=\mathrm{softmax}(S),\qquad \pi_{oa}\in[\epsilon,1-\epsilon].
$$

与无 prior、constant、shuffled、fixed q、learned pi 比较。已有 lexical 相似度不能直接替代 $q_{oa}$。若仅做 lexical centered bias，明确报告为启发式 text prior，单独调 `eta` 并做均匀/打乱对照。

这条二部兼容图是随机图的旁支对照；**兼容 prior 无收益不等于 OI/OE/AE 图必然无收益**。二者包含不同信息，不以昂贵的全 pair 查询作为 USG-Lite 的前置门槛。

---

## 6. Phase 3：边后验学习与随机图采样

### 6.1 共享边后验网络

$$
q_\phi(G\mid\mathcal K,D_s)=\prod_{e\in E_c}\mathrm{Bern}(E_e;\pi_e).
$$

$$
k_e=[z_i^{(0)};z_j^{(0)};|z_i^{(0)}-z_j^{(0)}|;
z_i^{(0)}\odot z_j^{(0)};\mathrm{Emb}(r);\mathrm{OOV}_i;\mathrm{OOV}_j].
$$

$$
\pi_e=\sigma\left[\mathrm{logit}(\pi_e^0)+h_\phi(k_e)\right].
$$

`h_phi` 两层 MLP，hidden=64，最后一层零初始化，初始后验等于先验。用一个共享网络，避免每条边独立可训练参数导致记忆；通过 seen 图像训练梯度更新，但输入不包含 seen 标记。数值计算时将概率 clip 至 `[0.02,0.98]`。

后验在整个数据集共享，初版不依赖当前图像；这使一次采样的图可供一个 batch 共同使用，并可在推理前缓存样本原型。它是 KL 正则化的结构分布近似，不是已经证明正确的真实 Bayesian posterior。

### 6.2 训练使用 Binary Concrete 与 straight-through

$$
\widetilde E_e=\sigma\left[
\frac{\mathrm{logit}(\pi_e)+\log u_e-\log(1-u_e)}{\tau_g}\right],
\qquad u_e\sim U(0,1).
$$

推荐 hard-forward / soft-backward：

```python
def sample_edges(pi, tau, training):
    pi = pi.float().clamp(0.02, 0.98)
    if not training:
        return torch.bernoulli(pi)
    u = torch.rand_like(pi).clamp(1e-6, 1 - 1e-6)
    noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((torch.logit(pi) + noise) / tau)
    hard = (soft > 0.5).to(soft.dtype)
    return hard.detach() - soft.detach() + soft
```

这样训练与测试的前向均使用 Bernoulli 硬边，梯度通过连续松弛近似传递。梯度有偏，不能声称是无偏重参数化；pure-soft Concrete 另作消融。[Concrete 原始论文](https://arxiv.org/abs/1611.00712)

`tau_g` 从 1.0 到 0.3 线性退火；只在随机阶段计步。先验证固定 1.0，再退火。不要把软 Concrete 样本的经验均值无条件当成 `pi`，两者在有限温度下不完全相等。

### 6.3 KL 约束

按关系先均值再平均，避免 OE 数量大导致它支配总损失：

$$
\mathcal L_{KL}=\frac1{|\mathcal R|}\sum_r\frac1{|E_r|}\sum_{e\in E_r}
\left[\pi_e\log\frac{\pi_e}{\pi_e^0}+
(1-\pi_e)\log\frac{1-\pi_e}{1-\pi_e^0}\right].
$$

初始 `lambda_KL=0.01`，试 `{0.001,0.01,0.1}`。先有冻结后验的确定图阶段，再开放学习；若后验迅速塌缩，可先增大约束或限制残差，不把调低 KL 当成默认操作。

### 6.4 监测与退化处理

每个 epoch 保存各关系的 `mean(pi)`、`std(pi)`、边熵、`pi<0.1`/`pi>0.9` 比例、期望边数、孤立原语比例、残差范数比 `||delta W||/||W0||`、预测样本间 JS/MI。

| 退化 | 识别信号 | 优先处理 |
|---|---|---|
| 全关图 | 有效边数骤降；后验靠近下界 | 增大 KL，回退冻结后验对照 |
| 全开图 | 后验靠近上界；MC 分歧近零 | 检查采样/梯度；减少分类损失对后验的压力 |
| 后验完全不动 | `pi` 一直等于 `pi0` | 检查是否 detach、优化器遗漏或残差尚为零 |
| 知识压过图像 | 残差范数过大；所有图像偏向相同类 | 降低 rho，加入残差范数约束 |
| 过平滑 | 原型类间余弦显著增大 | 一层图、残差、降低能量权重 |

如果学习后验退化而固定随机图有效，只能得出采样/集成有效，不能归因于可靠性学习。

---

## 7. 训练目标与解析期望–方差正则

### 7.1 Lite 的最小目标

对每张训练图像，在训练 seen 候选集合上计算 $p_G^{train}$：

$$
\mathcal L_{CPF}(G)=\mathcal L_{com}(G)+0.6\mathcal L_{att}(G)+0.4\mathcal L_{obj}(G).
$$

$$
\mathcal L_{mix}=-\frac1B\sum_b\log\left[
\frac1{S_{train}}\sum_s p_{G_s}^{train}(c_b\mid x_b)\right].
$$

$$
\mathcal L_{Lite}=\frac1{S_{train}}\sum_s\mathcal L_{CPF}(G_s)
+\lambda_{mix}\mathcal L_{mix}+\lambda_{KL}\mathcal L_{KL}.
$$

`lambda_mix=0.5`，`S_train=2`。三分支 CE 继续使用原训练温度；mix 评分使用第 5 节的明确尺度。训练 softmax 只含 seen pair，避免把所有未见组合直接作为大规模负类。验证/测试则在各自协议允许的完整候选上归一化。

所有增加的训练项要在相应无图/确定图对照中匹配。首轮不加 logic、Brier、consistency loss；概率指标先用于评价。

`-log(mean p)` 与 `mean(-log p)` 不同。本文选前者训练图混合预测，后者保留原 CPF 的分支监督。该损失加 KL 不自动等于标准 ELBO，论文叙事需准确。

### 7.2 期望–方差项：成立条件必须明确

令无向语义边每条只计一次，$E_e$ 独立，$\bar Z$ **不依赖本次随机样本 G**，令

$$
d_e=\|\bar z_i-\bar z_j\|_2^2,\quad
\mathcal E_G(\bar Z)=\sum_{e\in E_u}E_e w_e d_e.
$$

则：

$$
\mathbb E[\mathcal E_G]=\sum_e\pi_e w_e d_e,
\qquad
\mathrm{Var}(\mathcal E_G)=\sum_e\pi_e(1-\pi_e)w_e^2d_e^2.
$$

证明分别利用线性期望，以及独立边协方差为零、`Var(Bern(pi))=pi(1-pi)`。

关键实现约束：

- 正反向消息共享随机边，能量只计一条关系，不能用双向展开列表重复累加方差。
- 不把 `Z(G)` 直接代进并声称仍是上述精确闭式；此时距离与随机边相关。
- 建议取后验均值图得到的确定表示 $\bar Z=f_{graph}(Z^0,\pi)$，再对一个假想独立 Bernoulli 图计算能量矩。它可对参数求导，但在给定参数时不依赖该图样本。
- 如果只用冻结 `Z0` 且固定 `pi`，该项为常数，不会优化 CPF；必须确保作用于可学习节点适配器/均值图表示，或明确它只训练后验。
- 对象与属性“能组合”不意味着向量应该相等。初版能量只作用于 OI/OE/AE，不对属性–对象兼容边一律做平滑。

按关系平衡的数值目标定义为：

$$
\mathcal L_{MV}=\frac1{|\mathcal R|}\sum_r\left[
\frac1{|E_r|}\sum_e\pi_e w_ed_e+
\gamma\frac1{|E_r|}\sum_e\pi_e(1-\pi_e)w_e^2d_e^2\right].
$$

第二项是**方差总和除以边数**的规模归一化惩罚，不是边平均能量的方差；若定义平均能量 $\mathcal E/|E|$，其严格方差应除以 $|E|^2$。全文统一使用上面的训练定义。

初始 `lambda_MV=1e-3`、`gamma=0.1`；先用归一化表示让距离有界。该项晚于 Lite 加入，比较 `none / mean / mean+variance`，不能要求方差项必然提升。

### 7.3 逻辑损失为后续消融

如接入 LOGICZSL 规则，应新增概念真值 adapter，产生 object/attribute/parent/description 各类分数，并逐项核对原规则及聚合方式；不能把 pair logits 任意取 sigmoid 就充当所有逻辑真值。

可比较固定规则与后验加权规则 $\sum_e\pi_e\ell_{logic,e}$。它属于后续 `+Logic` 版本；首轮图机制成立无需该损失。

---

## 8. Phase 4：图后验边缘化与推理接口

### 8.1 正确平均的是概率

对每个图内完整候选分布：

$$
p_s(c\mid x)=\mathrm{softmax}_{\mathcal C}(B_s/T_s),\qquad
\bar p(c\mid x)=\frac1S\sum_s p_s(c\mid x).
$$

```python
# logits_s: S×B×P，P 是当前 CW/OW 候选，所有样本顺序一致。
logp_s = torch.log_softmax(logits_s.float() / score_temperature, dim=-1)
logp_mc = torch.logsumexp(logp_s, dim=0) - math.log(logits_s.shape[0])
pred_pair = logp_mc.argmax(dim=-1)
```

不能用 `softmax(mean logits)` 代替 `mean softmax(logits)`。更不能把已经 softmax 的概率再放进一次 softmax 当成最终置信度。

Lite 输出 `logp_mc` 给原 evaluator；`scores[(attr,obj)]` 与现有接口一致。对 S=1、无图、`T_s=1`，它只比原 B 少一个类别无关常数，可用作一致性检查。

### 8.2 图采样与温度

- 训练 `S_train=2`，测试先 `S_test=5`，最终比较 `1/2/5/10`。
- 后验均值图作为单独速度版本；非线性网络下 `p(E[G])` 通常不等于 `E[p(G)]`。
- 验证、测试各使用固定且独立的图采样 seed bank，避免在验证 bank 上挑样本。
- 在冻结模型后可缓存每张图的原型和组合文本原型；正文报告 warm-cache 与包含缓存构建的耗时。
- 不通过“挑出表现最好的图样本”测试；所有预定样本等权平均。

### 8.3 CW/OW 与 seen bias

每张图内先应用当前协议的候选 mask，再归一化和跨图平均；不把 CW softmax 的概率直接拿到 OW 使用。

原 evaluator 对 unseen 加 bias $b$。采用统一 score convention：

$$
q_b(c\mid x)\propto \bar p(c\mid x)
\exp\{b\,\mathbb I[c\notin C_s]\}.
$$

因此 bias 加在 `logp_mc` 上。标准 S/U/HM/AUC 保留原 evaluator 的曲线协议；另外提供只在验证集确定的固定 bias 结果，区分曲线上界与可部署工作点。

概率校准温度是另一个步骤：在固定 bias 后定义

$$
q_{b,T}(c\mid x)=\mathrm{softmax}\left[(\log\bar p(c\mid x)+b\mathbb I_{unseen})/T\right].
$$

`b`、`T` 在对应 CW/OW 验证协议上分别选择；不按测试指标调参。标准 AUC 用未做额外概率校准的 score，以免把 score scale 改变误当成图收益。

---

## 9. 训练日程、计算预算与工程限制

### 9.1 pilot 日程

| 阶段 | 初始化与冻结 | 长度 | 目的 |
|---|---|---|---|
| P0 | 现有 checkpoint，图全关 | 只评估 | 固定原基线 |
| P1 | 冻结 backbone 与原 CPF 头；固定 `pi0` | 2 epochs | 只学习图残差如何进入原型 |
| P2 | 同一 P1 权重；开放图后验与 CPF 头；backbone 首先保持冻结 | 3 epochs | 确定图、EdgeDrop、随机图成对比较 |
| P2b | 验证有效后才解冻 backbone 后部 | 可选 3 epochs | 比较进一步适配；所有关键对照匹配预算 |
| P3 | 条件头加入；设置无图条件头对照 | 独立实验 | 检验完整模型 |

pilot 不需要再跑已有三个历史基线，但必须新增 CPF 同预算续训对照，排除额外训练时间。P1/P2 期间原冻结 composer 的 BatchNorm/Dropout 状态也要正确：冻结参数不等于冻结 BN 统计；被完全冻结的子模块应保持 eval 状态。

### 9.2 最终 seed 与公平比较

正式核心比较建议 seeds `{0,3351,2026}`；每个 seed 先获得 10 epoch CPF 的 validation-best warm start，然后所有变体采用相同 2+3 epoch 增量日程。历史 checkpoint 只覆盖每个数据集一个 seed，不能假装已经有三个。

同 seed、同数据顺序、同初始化、同解冻计划和更新次数比较 A1/D/F/H；标准偏差只跨模型训练 seed 统计。图采样 seed 与模型 seed 独立记录，不能将同一模型的十张图当成十次独立复现实验。

学习率以当前基线为起点：原 CPF 头 `1e-4`，backbone 解冻时 `1e-5`；图层 `1e-4`，后验 MLP `5e-5`，GloVe 保持冻结，优化器先沿用 Adam，weight decay `5e-5`。CLIP 方案中的 `3.15e-6` 不直接混入当前 ViT-B 实验。

主 checkpoint 只由验证集 AUC 选择；如果增加校准/鲁棒性模型选择，预先定义并单独报告，不能看完测试后换选择准则。

### 9.3 计算与显存

假定沿用配套手册中的单卡 RTX 3090 24 GB，服务器当前硬件需执行前确认。先 batch 32，能容纳再升至 64；关键对照使用相同有效 batch。

- 图节点最多几千个、边约几千条，使用 COO + `index_add_`，不依赖 PyG/DGL 启动。
- 每个 batch **只计算一次 ViT patch 特征**，图样本复用这些特征，避免 S 次 backbone 前向。
- 缓存 patch 特征仅适用于 backbone 冻结、预处理固定的阶段；数据增强或解冻后不能长期复用过期特征。
- C-GQA 的全组合分数为 `B×278362`；B=64 的单份 FP32 约 68 MiB。静态 `278362×300` 原型约 319 MiB。
- `B×N×M×300` 在同一设置单份 FP32 约 **19.9 GiB**，还未计梯度和其他中间量，禁止物化。
- 组合原型与分数按 pair chunk `2048/4096` 处理；softmax 必须在全候选上用全局 logsumexp 归一化，不能每个 chunk 各自 softmax。
- 推理时在线累计各图 `log p`，不保存完整 `S×B×P` 堆叠。训练先 S=2，不因图能量闭式而误称整个模型不需要采样。
- 原 composer 包含 BatchNorm。冻结/eval 时分块安全；训练时逐 chunk 会改变 BN 统计，需统一统计、冻结 BN 或在可容纳的训练 seen 原型上一次计算，并给所有对照相同处理。
- 概率、KL、logsumexp 用 FP32；若使用 AMP，记录精度并先排查非有限值。

### 9.4 建议新增文件

| 文件 | 职责 |
|---|---|
| `tools/build_usg_cache.py` | typed 节点、词表编码、图与 hash |
| `models/usg_graph.py` | 关系消息、残差原型、后验网络、采样、KL |
| `models/usg_cpf.py` | 复用 CPF 特征，Lite / conditional head / MC 输出 |
| `models/usg_losses.py` | mixture NLL、能量矩、可选规则损失 |
| `tools/eval_usg.py` | validation/test 导出、CW/OW、概率指标 |
| `tools/summarize_usg.py` | 按数据集/方法/seed 汇总和成对差值 |
| `config/usg/*.yml` | 新开关及日程，基于实际生效配置 |

原 `DDM.forward()` 的 eval 分支包了 `no_grad`；图模块训练不要借用该分支得到损失，否则梯度消失。checkpoint 加载时允许只缺新增键，但需打印并校验 missing/unexpected keys，不能任意 `strict=False` 掩盖 backbone 不匹配。

---

## 10. Phase 5：显式对象条件属性头，形成完整 USG-CPF

### 10.1 最小可行条件头：因子化，而非全对象注意力重写

Lite 已有 $v_a(x,G)\in\mathbb R^{300}$ 与图增强原型。新增 rank-$k$ 交互，`k=64`：

$$
u_b=P_xv_a(x_b,G),\quad r_o=P_o\widetilde w_o(G),
\quad t_a=P_a\widetilde w_a(G).
$$

$$
\ell_{b,o,a}=\frac{c_a(x_b,a,G)}{\tau_a}
+\kappa\frac{\langle u_b\odot r_o,t_a\rangle}{\sqrt{k}}.
$$

该交互同时依赖图像、对象和属性。只加 `MLP(w_o,w_a)` 静态偏置不能等价于它。

```python
# u: B×k，r: N×k，t: M×k；输出 B×N×M，但不构造 B×N×M×k。
interaction = torch.einsum("bk,nk,mk->bnm", u, r, t) / math.sqrt(rank)
cond_logits = attr_logits[:, None, :] + kappa * interaction
logpa_o = torch.log_softmax(cond_logits.float(), dim=-1)
logpo = torch.log_softmax(obj_logits.float(), dim=-1)
logp_joint = logpo[:, :, None] + logpa_o
```

各对象块可分块计算，先保留全属性。初始化 `kappa=0.1`；三投影共享一个 rank 维度。该头是基于当前 CPF 特征的可实现近似，不等同于研究 PDF 中逐对象重新提取 patch 注意力的完整网络；后者仅在此版成功后再比较。

### 10.2 概率与损失

$$
p_G(a,o\mid x)=p_G(o\mid x)p_G(a\mid o,x),
\qquad \sum_{o,a}p_G(a,o\mid x)=1.
$$

主预测使用这个联合分布并跨图平均。原直接组合分支保留为辅助监督：

$$
\mathcal L_{full}=\mathcal L_{joint}^{MC}
+\lambda_c\mathbb E_G\mathcal L_{com}
+0.6\mathbb E_G[-\log p_G(a^*\mid o^*,x)]
+0.4\mathbb E_G[-\log p_G(o^*\mid x)]
+\lambda_{KL}\mathcal L_{KL}.
$$

初始 `lambda_c=1`；`+MV` 为单独版本。训练的 joint MC NLL 使用训练 seen 候选限制并重新归一化后的分布，方式与第 10.3 节的候选约束相同；直接组合 CE 也只使用训练 seen pair。对象与属性辅助监督仍覆盖各自原语类别。辅助属性损失使用训练 GT object，推理不使用 GT object。这里**直接组合头不再额外叠加到最终联合分数**；若增加该融合，需要命名新的 fusion 变体，因为最终对象边缘概率将不再等于分支 $p_G(o)$。

严格对照必须新增 **CPF + 同样条件头、无图**，使用完全相同目标、rank、训练预算。完整方法与该对照的差异才可归因于语义图，不能只与原三分支融合相比。

### 10.3 候选约束与对象边缘化

完整 $A×O$ 下链式分布自然归一化。CW 应先计算联合分布，再对允许候选限制并重新归一化：

$$
p_G^{CW}(c\mid x)=\frac{\mathbb I[c\in\mathcal C_{CW}]p_G(c\mid x)}
{\sum_{c'\in\mathcal C_{CW}}p_G(c'\mid x)}.
$$

CW 限制后的对象边缘不再等于原对象分支，这属于候选约束的正常结果。

$$
\bar p(a\mid x)=\sum_o\bar p(a,o\mid x),\qquad
\bar p(o\mid x)=\sum_a\bar p(a,o\mid x).
$$

先在**同一个图样本**内相乘，再对图平均：

$$
\mathbb E_G[p_G(o)p_G(a\mid o)]
\neq \mathbb E_G[p_G(o)]\,\mathbb E_G[p_G(a\mid o)].
$$

组合识别保留所有 `(a,o)` 联合概率；对象求和用于属性边缘或选择性分析。不能把“把所有对象概率求和”误解释成组合分类时消除对象标签。

### 10.4 不优先使用 top-K 对象截断

第一版使用全部对象并按块计算；因子化 head 无需为每个对象复制 ViT 特征。若后续确需 top-K，比较 `K=1/8/16/all`，报告真值对象覆盖率、丢失质量 `1-sum p_o(topK)` 与全量结果。

将 top-K 重新归一化得到的是截断近似，不是精确边缘化。一个未定义的 `other` 类无法自动恢复未计算对象的条件属性概率；不按研究 PDF 的概念建议直接宣称已经校正。hard top-1 是新模型的 routing 消融，不应描述为现有 CPF 已经采用的算法。

---

## 11. 建议配置与启动契约

下面字段是待实现 schema；现有 `train.py` 不会因为 YAML 中出现它们就自动具备 USG 功能。

```yaml
MODEL:
  usg_enabled: true
  usg_head: lite                 # lite / conditional
  usg_graph_mode: sampled        # off / hard / prior_mean / learned_mean / sampled
  usg_posterior: learned         # fixed / learned
  usg_relations: [oi, oe, ae]
  usg_dim: 300
  usg_relation_rank: 64
  usg_posterior_hidden: 64
  usg_pi0: 0.7
  usg_pi_clip: [0.02, 0.98]
  usg_rho: 0.1
  usg_straight_through: true
  usg_tau_start: 1.0
  usg_tau_end: 0.3
  usg_samples_train: 2
  usg_samples_test: 5
  usg_score_temperature: 1.0
  usg_lambda_mix: 0.5
  usg_lambda_kl: 0.01
  usg_lambda_mv: 0.0              # 首轮关闭，后续 +MV 为 0.001
  usg_gamma_var: 0.1
  usg_eta_compat: 0.0
  usg_use_logic: false
  usg_cond_rank: 64
  usg_cond_kappa: 0.1
  usg_pair_chunk: 2048

TRAIN:
  batch_size: 32
  usg_warm_graph_epochs: 2
  usg_joint_epochs: 3
  usg_lr_graph: 0.0001
  usg_lr_posterior: 0.00005

EVAL:
  usg_model_seed: 3351
  usg_val_graph_seed: 13351
  usg_test_graph_seed: 23351
  usg_export_probabilities: true
```

建议新增统一入口 `train_usg.py` 与 `eval_usg.py`，显式接受 `--cfg/--init-cpf/--knowledge-root/--run-dir/--seed`。实现后示例调用为：

```bash
# 接口实现和 P0 验收后才可执行；以下脚本与配置目前待创建。
ROOT=/data/xhw/czsl_repro
python train_usg.py \
  --cfg config/usg/ut_zappos_lite_h.yml \
  --init-cpf "$ROOT/runs/cpf_ut_zappos_seed3351/checkpoints/ut-zappos-czsl_final_predcition_mul/best_auc.pth" \
  --knowledge-root "$ROOT/knowledge/usg_v1/ut-zap50k" \
  --run-dir "$ROOT/runs/usg_v1/ut-zap50k/H/seed_3351" --seed 3351
```

每个 run 保存生效配置、初始化 checkpoint hash、知识 hash、源码 hash、优化器分组及参数数目。不同变体通过开关生成，避免复制多份代码产生隐性差异。

---

## 12. 核心消融与需要回答的问题

### 12.1 Lite 主消融表

除 A0/A1 原 checkpoint 外，下列有训练的行使用相同续训预算、同一 Lite mixture 目标；无图续训另保留“只用原 CPF 目标”一行排除额外 loss 的影响。

| ID | 变体 | 图进预测 | pi 可学习 | 前向采样 | 测试平均 | 主要问题 |
|---|---|:---:|:---:|:---:|:---:|---|
| A0 | 原 CPF 历史 checkpoint | — | — | — | — | 历史参考 |
| A1 | 无图概率包装 + 同预算续训 | — | — | — | — | 公平基础对照 |
| B | name+description 均值原型 | 是，静态文本 | — | — | — | 是否只是描述变长 |
| C | 等参数量无边 adapter | 无邻居 | — | — | — | 是否只是参数量 |
| D | hard OI/OE/AE 图 | 是 | — | — | — | 固定关系的价值 |
| E | prior mean 图 | 是 | — | — | — | 固定软权重价值 |
| F | learned mean 图 | 是 | 是 | — | — | 边权学习价值 |
| G | 固定 pi0 的 EdgeDrop | 是 | — | 是 | MC | 普通随机正则/集成 |
| H1 | 学习 pi 的随机图 | 是 | 是 | 是 | 单图 | 随机训练作用 |
| H | USG-CPF-Lite | 是 | 是 | 是 | MC | 后验学习与预测平均 |
| I | H + 期望能量 | 是 | 是 | 是 | MC | mean 正则价值 |
| J | H + 期望与方差能量 | 是 | 是 | 是 | MC | variance 正则价值 |

C 使用同规模的节点自身适配器及分类 head，尽量匹配可训练参数与运算；同时提供“关系类型内度数保持重连图”，用相同代码与完全相同参数数目进一步归因。

H1 与 H 最好使用**同一已训练随机模型**，仅改变推断 S；同理从 H 权重额外评估 posterior mean，避免把训练差异当成 MC 效果。F 则是训练时也用 mean 图的单独模型。

### 12.2 完整条件头消融

| 变体 | 条件头 | 图后验 | 对象路径 | 目的 |
|---|---|---|---|---|
| C0 | 同 rank 因子化头 | 无图 | all | 完整版的公平基础对照 |
| C1 | 同 rank 因子化头 | learned mean | all | 条件预测 + 确定图 |
| C2 | 同 rank 因子化头 | sampled | hard top-1 | 新模型单路径消融 |
| C3 | 同 rank 因子化头 | sampled | top-K | 截断与效率 |
| C4 | 完整 USG-CPF | sampled + MC | all | 全联合预测 |

### 12.3 判定逻辑

| 研究命题 | 必要对比 |
|---|---|
| 正确语义结构有价值 | D/E/F/H 对比 A1/C/shuffled |
| 可学习结构优于普通 dropout | H 对比 G，MC 样本数相同 |
| 概率平均带来增益 | 同 H 权重，S=1/5/10 对比 mean 图 |
| 方差项有独立贡献 | J 对比 I/H |
| 显式条件建模有额外价值 | C4 对比 C0/C1，且 Lite 另报告 |
| 全量对象优于截断路径 | C4 对比 C2/C3，报告 coverage |

不要求所有新增模块都涨点。若 variance 无收益，就保留 H 或 I 为主模型；如果条件头无收益，就以 Lite 为主结论，不把失败组件留在“Full”里凑复杂度。

### 12.4 必须完成的 sanity checks

1. **Graph off / rho=0**：恢复原型、原分数；同时关闭新增 loss 恢复原训练行为。
2. **Constant compatibility prior**：若启用 prior，加入所有类相同常数后 softmax/argmax 不变。
3. **Shuffled descriptions**：在对象/属性类型内打乱，描述数量和长度分布保持。
4. **关系内度数保持重连**：同一 relation 的 endpoint swap，排除一般图平滑。
5. **pi=pi0 / no-KL / fixed pi**：明确后验是否真的学到可靠结构。
6. **MC 数值**：每个分布和为 1，无 NaN/Inf；`S=1` 与单图一致。
7. **能量矩验证**：小图枚举或大量 Bernoulli 样本与解析均值/方差吻合；对双向展开特别检查。
8. **文本 fallback**：空描述/OOV 不造成类别缺失；其是否存在不能编码 seen 身份。

---

## 13. 噪声、校准、不确定性与效率实验

### 13.1 主实验：结构噪声曲线

噪声比例 `r={0,5%,10%,20%,30%}`，在每种关系内分别抽样，使用预定噪声 seeds `{11,23,47}`：

| 类型 | 操作 | 保持条件 |
|---|---|---|
| 缺边 | 随机删除候选关系 | 节点集合不变；缺边时依赖残差 fallback |
| 错误关系 | 关系类型内替换/交换端点，排除原有边与自环 | 保持节点类型；优先度数保持 |
| 描述污染 | 同类型中替换描述归属 | 描述数、长度、词表空间一致 |
| 置信扰动 | 对 logit(pi0) 加零均值噪声 | 候选边与文本不变 |

人工替换的关系不保证必然错误，报告名称应为“合成端点污染”；仅在人工确认的子集上报告“错误边”。不能把全部原始草稿边当作已知正确真值。

分开报告两种协议：

- **训练期污染**：每个噪声图重新训练相同模型；D/G/H 使用相同污染图和先验。新边同关系赋相同 pi0，后验不能读取污染标记或干净图身份。
- **推理期扰动**：固定干净训练权重、验证偏置与温度，仅替换输入图；不重新训练或按测试噪声调参。共享后验 MLP 可计算新端点边概率；删除边直接退出候选图，不声称模型能恢复未提供的边。

第一轮先做推理期 0/10%/30%，验证趋势后再扩大训练期污染实验，避免一开始倍增训练预算。

报告 AUC/HM、固定工作点 accuracy、NLL/ECE 的曲线及 `30%-0%` 差值。不同方法同时报告绝对性能与相对跌幅；不能仅因起点较差、跌幅小就称更鲁棒。若计算噪声曲线积分，命名 `Noise-Area`，与 CZSL 的 seen–unseen AUC 区分。

### 13.2 概率校准与选择性预测

在第 8 节固定验证 bias/temperature 下，报告：

$$
\mathrm{NLL}=-\frac1n\sum_x\log q(c^*\mid x),
\quad
\mathrm{Brier}=\frac1n\sum_x\sum_c(q(c\mid x)-\mathbb I[c=c^*])^2.
$$

ECE 使用 15 个等宽置信度 bin，明确 bin 边界；报告 overall、seen、unseen，同时提供校准前/后结果。Brier 按类求和，不在不同 P 下随意再除以类别数；CW/OW 分开比较。

图结构分歧：

$$
U_{pred}(x)=H[\bar p],\qquad
U_{struct}(x)=H[\bar p]-\frac1S\sum_s H[p_s].
$$

同一候选集合、同一校准规则计算，保留采样方差。它是模型内图样本分歧的互信息估计，不保证等于真实 epistemic uncertainty；若差值出现小负数，应检查数值精度和分布归一化。

按 `U_struct` 或 predictive entropy 从低到高保留样本，绘制 risk–coverage，计算 AURC（越低越好）；与 `1-max probability` 比较。不能用拒绝后的准确率替代全样本 CZSL 主指标。

### 13.3 后验是否知道错误边

在合成污染标记或人工复核的边集合上，比较 `1-pi` 对错误关系的 AUROC/AUPRC；同时按 OI/OE/AE 分组，报告人工标签数量和一致性。原始 unreviewed JSON 不能直接当 correctness ground truth。

人工复核边用于诊断；若用于调 prior，应拆成 prior-development 与 held-out audit 两部分，避免在同一批边上调参又报告检测性能。

### 13.4 误差与效率

- Lite：组合误判、对象分类错误、属性错误；完整版本额外报告 GT-object 条件属性正确率和 object-posterior 熵分组。
- 按类别频率、OOV/fallback、关系先验或后验分位数分组，检查收益是否集中于少数类别。
- 展示三类失败案例：错误关系导致误判、正确知识被视觉证据压制、图样本高度分歧。
- 报告 trainable params、总参数、峰值显存、images/s、单图/MC 耗时、S 与 rank 的准确率–效率曲线。
- GPU 测时使用 warm-up 与同步；同设备、batch、精度和候选数。不要直接套用论文给出的 CPF fps。

---

## 14. 理论分析的可交付范围

### 14.1 可直接完成：能量矩及梯度

第 7 节的独立无向 Bernoulli 边能量矩给出一个自洽、可数值验证的命题。方差对边概率的导数：

$$
\frac{\partial\mathrm{Var}(\mathcal E_G)}{\partial\pi_e}
=(1-2\pi_e)w_e^2d_e^2.
$$

上式是固定距离 $d_e$ 时的偏导。若采用第 7 节的均值图表示，$d_e$ 还依赖后验参数，实际总梯度应包含这条链式项，由自动微分计算；不能只实现上式并替代完整梯度。因此方差惩罚可能把后验推向 0/1，而不是自动保持不确定性。KL、残差和 posterior 统计用于监测；不能把“加方差”本身解释为防塌缩定理。

### 14.2 可直接完成：固定表示下的图扰动界

无向非负图 Laplacian $L=D-W$，且 $\bar Z$ 固定：

$$
\left|\mathrm{Tr}(\bar Z^\top L'\bar Z)-\mathrm{Tr}(\bar Z^\top L\bar Z)\right|
\le \|L'-L\|_2\,\|\bar Z\|_F^2.
$$

该式约束的是能量变化，不自动约束 GNN 输出的分类误差。若还要给预测界，需明确映射对图矩阵的 Lipschitz 条件、最小度/固定度、LayerNorm epsilon、残差与权重范数；并对同一输入测量 logit/分布扰动。首轮以能量界 + 实测敏感性为交付，不强行写完整 unseen risk bound。

### 14.3 边缘化的正确解释

全量对象条件模型下，属性边缘是链式法则；图混合是 model averaging。它们在正确条件模型下有自然的 log-loss 解释，但**不保证任意近似后验上的 top-1 准确率必然优于单图**。这需要第 12–13 节的实证比较。

创新定位宜写为“语义结构分布进入条件组合预测，并配套结构扰动与校准验证”。不声称首次图 CZSL、首次 Bayesian CZSL，也不把 Bernoulli/Concrete 作为新发明。相关原语分布方法如 [Bayesian Primitive Distributing 的 OpenReview 稿件](https://openreview.net/pdf?id=phRRjC0Da6)已涉及另一种表征不确定性；其录用状态本次未核验，不在本方案中推断。投稿前另做系统查新。

---

## 15. 继续、回退与停止判据

### 15.1 Gate 0：接口可靠

P0 全关闭一致性通过，词表与 split 身份匹配，图缓存覆盖与数值正常。否则修接口，不能继续做机制归因。

### 15.2 Gate 1：语义图有信息

在 UT pilot 与 MIT 决定性比较中，D/E 至少有一个相对 A1/C/shuffled，在识别、校准或噪声性能上稳定改善。首轮 screening 可把 AUC `+0.3` 或 HM `+0.5` 个百分点、NLL 相对下降 `2%` 作为有意义的优先级信号；这些是预设探索阈值，不是统计显著性定律。

若只有 description-mean 有效而图无增益，回退文本增强方案；先检查知识语义和编码，不盲目增加图层。

### 15.3 Gate 2：结构不确定性有价值

H 相对 D/F/G 在下列三类证据中至少两类有效：

1. 三个模型 seed 的成对 AUC/HM 改善，且不依赖一个 seed。
2. 10%/30% 污染下绝对性能或下降幅度优于确定图，同时起点接近。
3. NLL/Brier/AURC 或错误边检测更好，不仅是 temperature calibration 的效果。

最终表报告 mean±std 与逐 seed 差值；三个 seed 的置信区间通常不稳定，不能强行宣称显著。图样本 seed 的收益另作 Monte Carlo 稳定性分析。

若 H≈G，只能主张随机图集成有效，需改进后验或降低方法主张；若 H≈F，mean graph 可能已经足够，选择更便宜版本。

### 15.4 Gate 3：完整条件模型值得保留

C4 相对 C0/C1 有稳定增益，且能够解释对象歧义分组的收益。若没有，保留 Lite，不进入逐对象 patch attention 或复杂 GNN。

停止继续加复杂度的情况：正确结构不优于打乱；模型全部关图；精度、校准、鲁棒性均无增益；增益完全来自多训练轮数或新评分。先完成负结果、知识诊断和轻量回退。

---

## 16. 推荐执行顺序与结果归档

### 16.1 六到八周路线（按 Gate 调整）

| 时间 | 工作 | 必须交付 |
|---|---|---|
| 第 1 周 | P0、服务器路径核验、typed 图、GloVe 编码 | asset/hash manifest、coverage/OOV、精确对照 |
| 第 2 周 | UT 确定图、描述与容量对照，MIT 关键比较 | Gate 1 表与失败诊断 |
| 第 3 周 | learned mean、EdgeDrop、USG-Lite | 后验统计、训练稳定性、Gate 2 pilot |
| 第 4 周 | MC 数量、推理期噪声与校准 | 0/10/30% 曲线、NLL/ECE/AURC、效率 |
| 第 5 周 | 显式条件头与 C0/C1/C4 | 条件头归因、全量对象结果 |
| 第 6 周 | +mean/+variance、解析矩验证 | 能量矩命题、消融与扰动测量 |
| 第 7–8 周 | 三数据集三 seed 核心表、训练期污染 | 最终逐 seed / mean±std / 失败案例 |

这是工作分解，不是保证的 GPU 耗时。正式扩大预算之前，先测 UT/MIT 单 epoch 时间，再据 C-GQA 与 S=2 的成本决定最后保留哪些变体。

### 16.2 每个 run 的最小归档

```text
run/
├── effective_config.yml
├── code_manifest.json
├── knowledge_manifest.json
├── init_checkpoint.sha256
├── train.log
├── posterior_stats.csv
├── best_val_checkpoint.pth
├── val_metrics.json
├── test_cw_metrics.json
├── test_ow_metrics.json
├── probability_metrics.json
├── noise_metrics.csv
└── efficiency.json
```

大规模 OW 不要求保存每个样本的全部 `S×P` 概率。可流式累计 NLL/Brier/熵，保存样本 ID、GT、top-k、最大概率、正确性、结构 MI；需要复核时按模型与 seed bank 重算。

### 16.3 最终结果表模板

| Dataset | Setting | Variant | Knowledge version | Seeds | S | U | HM-curve | AUC | HM-val-bias | NLL | ECE | AURC | Peak GB | images/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MIT | CW | A1 / D / G / H / C4 | v1_draft 或 reviewed_v2 | 0,3351,2026 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

`HM-curve` 与验证固定偏置的 HM 分开；所有数字填写实测值，不根据已有 PDF 的推荐方向预设正结果。

---

## 17. 材料对应关系与最终实验定位

本方案参照 [ZSCL实验设计0601.md](C:/Users/llenovo/Desktop/ZSCL实验设计0601.md) 的“原则—基线接口—知识预处理—分阶段实现—损失—消融—分析”结构，按本机实际基线与知识资产修订。

| 材料 | 本方案采用的内容 | 作出的实施修正 |
|---|---|---|
| [CPF 原论文，本地 PDF](C:/Users/llenovo/Desktop/CZSL/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.pdf) | 对象引导属性、三项监督、训练设置 | 与当前 ddm.py 的实际推理尺度区分；不混用 CLIP 版 |
| [LOGICZSL 原论文，本地 PDF](C:/Users/llenovo/Desktop/CZSL/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf) | OI/OE/AE 语义关系与规则思路 | 现有知识是重建草稿；先图进预测，规则 loss 后加 |
| [软兼容图 V1 推导](C:/Users/llenovo/Desktop/CZSL/cpf_soft_semantic_graph_v1_derivation.pdf) | 确定先验、log prior、均值对照、归因检查 | 兼容先验为独立支路；已有 lexical 值不能冒充概率 |
| [创新路线图](C:/Users/llenovo/Desktop/CZSL/czsl_innovation_roadmap.pdf) | 结构不确定性、图条件预测、噪声与理论 | 无向边只计一次，避免双向独立性与方差系数混淆 |
| [不确定语义图研究计划](C:/Users/llenovo/Desktop/CZSL/czsl_uncertain_semantic_graph_research_plan.pdf) | 图后验、Concrete、MC、条件推断 | Lite 先行；无投票不虚构置信度；用因子化头替代首轮逐对象注意力；不宣称未定义的 other 已校正 |
| [现有知识 README](C:/Users/llenovo/Desktop/knowledge/kcc_logic_3datasets_reconstructed_v1/README.md) | 来源、词表与 prior 说明 | 纳入 pending 状态、OOV、hash、split 检查 |

LOGICZSL 的固定关系训练约束提供了研究出发点；USG-CPF 的待验证机制是结构分布通过语义原型和对象条件属性交互进入预测，并用图平均表达关系不确定性。[LOGICZSL 官方论文页面](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.html)

首个目标交付不是一个预先假定有效的“大 Full Model”，而是一个能复用已有资产、准确归因并允许及时回退的 **USG-CPF-Lite**；通过 Gate 后，完整 **USG-CPF** 才承担条件组合建模的主结论。
