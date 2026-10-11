# CPF/KCC、语义归因与 EKR 路线完整实验记录

整理日期：2026-10-07  
适用范围：截至本日已有的实验、审计与分析，包含后来归档的名称锚点 EKR 阶段 B pilot，以及尚未运行的后续机制诊断。  
数学排版：行内公式统一使用 `$...$`，独立公式统一使用 `$$...$$`，并已检查分隔符成对闭合。

知识构建的逐步记录见第3.5—3.17节：原始OI/OE/AE、标签与分词、原语知识向量、组合模板、原始K、归一化与校准、v2残差库、真实统计、实现代码及服务器命令。

## 0. 结论先行

到目前为止的实验并非没有意义。它已经形成一条完整的证据链：

1. 原始 CPF 基线已在 UT-Zappos、MIT-States、C-GQA 上复现并保存结果。
2. 在 UT-Zappos 上，静态描述输入曾显著提高 AUC/HM，但固定的组合先验校准 $K$ 明显降低指标，固定 checkpoint 的后处理最终选择了 $\lambda=0$。
3. 三 seed 语义归因证明“加入文本”优于单位名称输入，但 v1 中正确描述没有优于固定错配描述；v2 经训练图审核和描述修订后，正确描述相对错配描述的 CW AUC 平均优势提高到 $+0.379$ pp，仍低于预先固定的 $+0.5$ pp 门槛。
4. v2 在 OW AUC 上存在正信号：`C_text-C_shuffle` 三 seed 均为正，平均 $+1.980$ pp；但预先声明的主门槛是 CW AUC，不能事后改指标宣布通过。
5. 逐图分析发现静态描述的影响具有明显类别差异：它改善 `Shoes.Flats`，却显著破坏 `Shoes.Boat.Shoes`，尤其是 `Leather × Boat Shoes`。这给“何时使用或拒绝外部知识”的 EKR 问题提供了直接动机。
6. 原 EKR 以 `C_text` 为冻结锚点；关闭残差仍回到 `C_text`，不能撤销描述本身已经造成的对象混淆。名称锚点改进现已完成 seed 0 的三项、各三轮 pilot：正确描述相对冻结名称仅提高 CW AUC +0.047790 pp，且落后于名称分支和错配分支。当前实现尚未显示正确描述优势，详见第17节。
7. gate 的数值计算核对通过，但它是预算筛选规则。静态文本达到 +0.5 pp 并不是动态 EKR 可行的必要条件；单 seed 的分支 pilot 失败也不能排除所有后续机制改进。第18节记录这一解释修正。

当前正式状态如下。

| 项目 | 状态 | 可以得出的结论 |
| --- | --- | --- |
| 三数据集 CPF A/C 与 UT A/C/D/E | 已完成并核验 | 文本收益主要集中于 UT；静态 $K$ 路线失败 |
| 固定 checkpoint 的 $\lambda$ 后处理 | 已完成并核验 | Dpost、Epost 均选择 $\lambda=0$ |
| UT v1 三 seed 语义归因 | 已完成并核验 | 文本有收益，正确归属没有稳定优势 |
| UT 训练图与 84 句描述审核 | 已完成 | 助手审核候选，不等同于人工专家批准 |
| UT v2 四控制 × 三 seed | 已完成并核验 | 8/9 条 gate 通过，唯一失败项是正确文本对错配文本的 CW AUC 优势不足 |
| UT v2 逐图预测分析 | 已完成并核验 | 找到 seen 倾向与 Boat Shoes 局部退化 |
| 原正式 EKR pilot | 未运行 | 不能说 EKR 已成功或失败 |
| A_unit 名称锚点 EKR | seed 0 三项、各三轮 pilot 已完成并审阅 | 正确描述未优于名称和错配对照；不自动扩大当前版本 |
| 名称锚点 EKR 机制诊断 | 工具包已交付，本地暂无服务器结果 | 待检查值向量对齐、null 响应、对象条件和 margin |
| USG 随机语义图 | 独立路线，本文未运行 | 不与本路线的 shuffle 负对照混用 |

## 1. 时间线与研究决策

| 日期 | 阶段 | 完成内容 | 研究决策 |
| --- | --- | --- | --- |
| 2026-10-05 | 阶段一 | 汇总并核验三数据集 A/C、UT A/C/D/E、20 次 $\lambda$ 后处理 | UT 文本输入值得继续；固定 $K$ 校准停止 |
| 2026-10-05 至 2026-10-06 | 阶段二 | 建立 UT 三 seed 的 `A_raw/A_unit/C_text/C_shuffle` 归因协议 | 正确语义必须优于名称控制和错配控制，才进入正式 EKR |
| 2026-10-06 | 阶段三 | v1 三 seed 完成；审计工具 v1 暴露 YACS 表示和旧 top5 格式兼容问题，v2 审计修复后技术检查通过 | 保留研究 gate=false，不因审计工具修复而改实验结论 |
| 2026-10-06 | 阶段四 | 审核 28 张训练图板、560 个面板、528 张不同训练图；修订 84 句描述 | 产生一次性 v2 描述候选，不再根据 VAL 错误继续改句子 |
| 2026-10-06 至 2026-10-07 | 阶段五 | 从头运行 v2 的 12 个训练身份、24 次 VAL CW/OW 评价 | 完整性通过；研究 gate 仍为 false |
| 2026-10-07 | 阶段六 | 下载并独立核对 v2 结果包与逐图预测包 | 确认指标、身份、配置和 gate 计算无误 |
| 2026-10-07 | 阶段七 | 按 seen/unseen、属性、对象、组合分析 3214 张 VAL 图 | 识别 Boat Shoes 对象混淆，提出名称锚点条件知识路线 |
| 2026-10-07 | 阶段八 | 名称锚点 EKR seed 0 的 N_name/N_clean/N_shuffle 各三轮完成并审阅 | pilot gate=false；正确描述没有对照优势 |
| 2026-10-07 | 当前 | 交付既有分支的无新增训练机制诊断工具，并复核 gate 的解释边界 | 根据诊断决定是否另立机制改进版本 |

## 2. 数据、环境与评价协议

### 2.1 运行环境

已记录的主要服务器环境：

- Python：`/data/xhw/miniconda3/envs/cpf/bin/python`，Python 3.8.18。
- PyTorch：1.13.1+cu117。
- torchvision：0.14.1+cu117。
- GPU：NVIDIA GeForce RTX 3090。
- UT v2 每项训练 10 epoch；batch size 64；主学习率 $10^{-4}$；encoder 学习率 $10^{-5}$；word 学习率 $2.5\times10^{-6}$；weight decay $5\times10^{-5}$。
- 每个 run 依据 VAL CW AUC 选择最佳 epoch；再用同一 checkpoint 评价 VAL CW 与 OW。

### 2.2 三个数据集和知识规模

| 数据集 | 属性数 | 对象数 | 父类数 | OE 描述数 | AE 描述数 | 当前早期实验 seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UT-Zappos | 16 | 12 | 4 | 36 | 48 | 3351 |
| MIT-States | 115 | 245 | 19 | 735 | 345 | 0 |
| C-GQA | 413 | 674 | 23 | 2022 | 1239 | 3351 |

早期知识表中的 `approved`/`bulk_trusted_unverified` 仅表示操作状态和版本来源，并不等同于逐句人工专家验证。

UT v2 的 VAL 数据：

| 项目 | 数量 |
| --- | ---: |
| VAL 图像 | 3214 |
| seen 组合图像 | 877（27.287%） |
| unseen 组合图像 | 2337（72.713%） |
| VAL 组合类型 | 30 |
| 有 gold 样本的属性 | 14/16 |
| 有 gold 样本的对象 | 12/12 |
| CW 模型分数候选 | 116 |
| CW 原生评价允许候选 | 98（train pairs ∪ val pairs） |
| OW 候选 | 192，即 $16\times12$ |

### 2.3 CW、OW、bias、HM 与 AUC

- CW：closed world，只在协议允许的组合集合内评价。
- OW：open world，在全部属性与对象笛卡尔积上评价。
- 原评估器对 unseen 候选增加 bias $b$：

$$
s_b(x,p)=s(x,p)+b\,\mathbf{1}[p\notin\mathcal P_{train}].
$$

在每个 bias 点得到 seen accuracy $S(b)$ 和 unseen accuracy $U(b)$。调和均值为：

$$
HM(b)=\frac{2S(b)U(b)}{S(b)+U(b)}.
$$

报告中的 `best_hm` 是 bias 扫描上的最大调和均值；AUC 是 $S$-$U$ 曲线下的面积。`top1_at_bias0` 只是 $b=0$ 的一个固定预测点，不能代替 AUC/HM。

本文的 `pp` 表示百分点。例如 44.0% 到 45.0% 是 $+1.0$ pp。

## 3. 原始 CPF/KCC 模型与 A/C/D/E 定义

### 3.1 GloVe 表示

本项目的文本编码器是 300 维 GloVe 静态词向量，不是 CLIP 文本编码器。句子经过原项目分词后，对词表内 token 的 GloVe 向量求均值，再做 L2 归一化。

对非零向量 $z$：

$$
\mathcal N(z)=\frac{z}{\lVert z\rVert_2}.
$$

### 3.2 CPF 评价分数

CPF 在评价时将组合、属性和对象三个 cosine 分数相加：

$$
s_{CPF}(x,a,o)=s_{comp}(x,a,o)+s_{attr}(x,a)+s_{obj}(x,o).
$$

关键实现位于原 `models/kcc_cpf.py`：

```python
obj_pred = self.classifier(vis_obj_new, obj_weight, scale=False)
attr_pred = self.classifier(attr_emb, attr_weight, scale=False)
comp_pred = self.classifier(vis_comp, concept, scale=False)

pred = comp_pred + attr_pred + obj_pred
pred = self._kcc_calibrate(pred, self.val_objs, self.val_attrs)
```

训练中的基本分类损失为：

$$
\mathcal L=\mathcal L_{comp}+0.4\mathcal L_{obj}+0.6\mathcal L_{attr},
$$

启用原有增强路径时还会加入 `mix_loss`；各变体必须保持相同训练路径才能进行配对比较。

### 3.3 A/C/D/E 的含义

| 变体 | primitive 文本向量 | 静态组合校准 | 目的 |
| --- | --- | --- | --- |
| A | 原名称向量 | 关闭 | CPF 基线 |
| C | 名称与描述聚合后的 $W_a^K,W_o^K$ | 关闭 | 检查描述文本输入 |
| D | 原名称向量 | 开启 | 检查固定 $K$ 校准 |
| E | 描述聚合向量 | 开启 | 文本与固定 $K$ 同时使用 |

当前 C 实际读取 `W_a_K[M,300]` 与 `W_o_K[N,300]`。D/E 使用 `K_oa_prior_centered[N,M]`；OI 关系和 pair blocks 没有在这一版执行图消息传播。

### 3.4 静态 $K$ 的公式

设原组合知识矩阵为 $K$。代码用总体均值和总体标准差标准化：

$$
\widetilde K=\begin{cases}
0,&\operatorname{std}(K)<10^{-8},\\
\dfrac{K-\operatorname{mean}(K)}{\operatorname{std}(K)},&\text{otherwise}.
\end{cases}
$$

静态校准分数为：

$$
s'(x,a,o)=s(x,a,o)+\lambda\widetilde K(o,a).
$$

对应的关键代码：

```python
def _kcc_calibrate(self, scores, obj_ids, attr_ids):
    if not self.kcc_calib or self.kcc_lambda == 0.0:
        return scores
    prior = self.K_oa_prior[obj_ids, attr_ids]
    return scores + self.kcc_lambda * prior.unsqueeze(0)
```

这项 $\lambda$ 是每个组合固定的全局先验强度，不依赖当前图像，因此与后面的 EKR 动态残差幅度 $\eta$ 不是同一个量。

### 3.5 知识矩阵构建总流程与符号

以下过程依据实际 `tools/build_knowledge.py`、`transition/bank.py`、缓存manifest及源字典记录整理。本地构建器SHA256为 `5997b07bee331e9a17c169b11a25c2d2eb9bb27e8cc1b4e3aad10cd5bb6b8687`，与归档v1缓存的 `builder_sha256` 一致。

设属性集合为 $\mathcal A=\{a_0,\ldots,a_{M-1}\}$，对象集合为 $\mathcal O=\{o_0,\ldots,o_{N-1}\}$，GloVe维度 $D=300$。这里名称中的下标均从0开始，与Python数组一致。

```text
冻结的属性/对象标签 + OI/OE/AE字典 + 来源/审核记录
    ↓ 标签顺序、完整覆盖、描述条数、源文件SHA检查
原CPF名称别名 + 同一份glove.6B.300d.txt
    ↓ 名称与描述采用各自实际分词规则
W_a_name_raw、W_o_name_raw；名称单位向量
    ↓ 名称单位向量与三句描述单位向量聚合
W_a_K[M,300]、W_o_K[N,300]
    ↓ 全部N×M组合的三个固定模板
W_ao_K：N个[M,300]块
    ↓ 模板向量与属性/对象知识原型的cosine
K_oa_text_raw[N,M]
    ↓ 全矩阵min-max；退化常量设为0.5
K_oa_text_template[N,M]
    ↓ prior_source选择，当前实际为template
K_oa_text[N,M]
    ↓ 全矩阵总体标准化；常量先验归零
K_oa_prior_centered[N,M]

同一份名称/描述单位向量
    ↓ 保留每条描述、中心化、固定无固定点错配
bank.pt：units、mean、residual、correct、unit、shuffle等
```

需要区分三个数值对象：`W_a_K/W_o_K` 是每个原语的300维知识表示；`K_oa_text/K_oa_prior_centered` 是每个对象×属性组合的一个标量；`bank.pt` 是保留每条描述的向量库。它们的维度、作用和模型消费路径不同。

### 3.6 第一步：冻结原始知识字典与来源

每个数据集使用以下7个必需源文件：

| 文件 | 内容 | 构建时的作用 |
| --- | --- | --- |
| `attributes.txt` | 属性原始标签，一行一个 | 固定属性轴顺序 |
| `objects.txt` | 对象原始标签，一行一个 | 固定对象轴顺序 |
| `oi_dict.json` | 父概念 → 对象成员列表 | 检查每个对象恰好归属一次 |
| `oe_dict.json` | 对象 → 三句描述 | 生成对象知识向量 |
| `ae_dict.json` | 属性 → 三句描述 | 生成属性知识向量 |
| `review_status.csv` | 每个属性/对象的来源使用状态 | 检查身份覆盖与状态 |
| `manifest.json` | 来源、数量、知识文件SHA256 | 锁定输入版本 |

`entry_provenance.csv` 提供词义选择等补充记录。早期UT是手工重构文本，28条原语来源记录标为 `manual_reconstruction_unreviewed`；MIT/C-GQA是Princeton WordNet 3.0首义项词典草稿。MIT有360条 `wordnet_auto_unreviewed`；C-GQA有1081条该类记录及6条 `label_template_only`。具体采用的query、synset、候选词义数、词头回退及父概念以来源CSV为准；复现数值时直接使用已冻结JSON，不重新检索WordNet后替换词义。

这些资产不是LOGICZSL官方知识，也没有被验证为现实组合可行性标签。早期 `approved` 表示允许构建器使用该版本；`bulk_trusted_unverified` 保留了其未经逐句人工核验的状态。

真实v1结构示例：

```json
{
  "OI示例": {"boot": ["Boots.Ankle", "Boots.Knee.High", "Boots.Mid-Calf"]},
  "OE示例": {"Boots.Ankle": ["shaft ending at the ankle", "closed toe boot silhouette", "low boot collar"]},
  "AE示例": {"Canvas": ["woven canvas fabric", "coarse textile surface", "matte cloth finish"]}
}
```

上面的三个外层键用于展示，实际对应三个独立文件。OI只保存父类成员关系，没有被本构建器转换为图邻接矩阵，也没有参与K的cosine公式。

### 3.7 第二步：严格对齐标签与矩阵坐标

构建器调用原项目 `CompositionDataset.parse_split()` 获取实际标签顺序。原函数读取train、val、test的pair元数据，将所有属性去重排序、所有对象去重排序。构建器不创建完整图像数据集，也不读取图像。

这一步确实读取了test pair文件用于既有类别词表和顺序核对，不能写成“构建从未读取test列表”。但是后面的语义公式不使用某组合属于哪个split，也不使用test图像或test指标确定描述、模板、K或lambda。矩阵覆盖全部笛卡尔积，不把split中的pair当正兼容标签，不把未列出的pair当负兼容标签。

检查顺序如下：

1. `attributes.txt`、`objects.txt` 逐行内容必须与dataset属性/对象列表完全一致。
2. AE/OE的JSON键顺序必须与对应列表一致；每个原语恰好3句非空字符串。
3. OI所有成员合并后必须覆盖全部对象且没有重复成员。
4. `review_status.csv` 恰好包含 $M+N$ 个原语身份，且状态满足构建器要求。
5. source manifest里的知识文件SHA必须与当前源字节一致；别名处理后还检查名称token序列没有碰撞。

UT的实际矩阵规格：

| 文件/字段 | 形状 | 轴顺序 |
| --- | --- | --- |
| `W_a_name_raw/W_a_name_unit/W_a_K` | `[16,300]` | 属性、词向量维度 |
| `W_o_name_raw/W_o_name_unit/W_o_K` | `[12,300]` | 对象、词向量维度 |
| `W_ao_K/obj_XXXXX.pt` | 每块`[16,300]`，共12块 | 固定对象内的属性、词向量维度 |
| 所有 `K_oa_*` | `[12,16]` | **对象、属性** |
| `bank['a']['units']/residual` | `[16,4,300]` | 属性、名称/描述槽、维度 |
| `bank['o']['units']/residual` | `[12,4,300]` | 对象、名称/描述槽、维度 |

例如 `Boots.Ankle` 的对象索引是0，`Leather` 的属性索引是6；对应值是 `K[0,6]`。`Shoes.Boat.Shoes` 的对象索引是4，因此 `Leather × Boat Shoes` 对应 `K[4,6]`。候选pair通常按 `(attribute, object)` 保存，但索引K时必须交换成 `[object_index, attribute_index]`。模型不能依赖flatten后的默认顺序猜对应关系。

### 3.8 第三步：名称别名、描述分词与GloVe加载

名称与描述没有使用同一种字符串预处理。名称沿用原CPF的别名表：

| 原标签 | 原CPF名称别名 | 名称token |
| --- | --- | --- |
| `Boots.Ankle` | `ankle_boots` | `ankle, boots` |
| `Shoes.Boat.Shoes` | `boat_shoes` | `boat, shoes` |
| `Full.grain.leather` | `thick_leather` | `thick, leather` |
| `Patent.Leather` | `shiny_leather` | `shiny, leather` |
| `Nubuck` | `grainy_leather` | `grainy, leather` |
| `Canvas` | 无替换 | `canvas` |

这些别名是原基线行为，不能在知识构建中擅自换成更自然的名称，否则A与C的编码口径会改变。构建器从实际 `word_embedding_utils.py` 的AST提取字面量别名表，避免导入该模块时引入fasttext/gensim依赖。

名称先用别名、再转小写；有下划线才按下划线拆token，并使用原token别名（如 `selfie→photo`）。没有下划线时将整个字符串当一个token。名称token缺失属于致命错误，不能静默补随机向量。

描述采用以下正则，保留数字和特定英文撇号形式；没有停用词过滤或词形还原：

```python
def description_tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
```

GloVe加载只扫描一遍 `glove.6B.300d.txt`，收集所需的名称、描述及模板填充词，保留对应300维float32向量，同时散列完整文件。描述中OOV token跳过，但记录其词频；模板填充词或名称OOV直接停止。原始构建器允许整句没有可用向量或均值近零时回退到名称，v2审核流程进一步规定任何整句fallback都停止。

当前已记录的GloVe SHA256为 `91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`。v1、v2、bank必须使用同一份编码文件。

### 3.9 第四步：生成原语知识矩阵

令 $g(w)\in\mathbb R^{300}$ 是GloVe词向量，$\mathcal T_c$ 是名称按原别名规则得到的token序列。保留重复token，均值按序列长度计算：

$$
n_c=\frac{1}{|\mathcal T_c|}\sum_{w\in\mathcal T_c}g(w),
\qquad u_{c,0}=\mathcal N(n_c).
$$

这里 $n_c$ 是raw名称向量。对第 $r$ 句描述，令 $\mathcal V_{c,r}$ 为正则分词后在GloVe词表内的token序列：

$$
v_{c,r}=\frac{1}{|\mathcal V_{c,r}|}\sum_{w\in\mathcal V_{c,r}}g(w),
\qquad u_{c,r}=\mathcal N(v_{c,r}),\quad r=1,2,3.
$$

只有 $\mathcal V_{c,r}$ 非空且向量非零时使用上式；否则原构建器令 $v_{c,r}=n_c$ 并记录fallback。v2不允许发生该回退。

先平均名称单位向量与三个描述单位向量，再对平均值归一化：

$$
\mu_c=\frac{1}{4}\sum_{r=0}^{3}u_{c,r},
\qquad w_c^K=\mathcal N(\mu_c).
$$

因此名称与每句描述在聚合前各占一个slot：名称占1/4，三句描述合计占3/4。不能把所有句子token一起平均，也不能省略逐句归一化；两者都会改变实际数值配方。

按冻结标签顺序逐行堆叠：

$$
W_a^K=\begin{bmatrix}(w_{a_0}^K)^\top\\ \vdots\\ (w_{a_{M-1}}^K)^\top\end{bmatrix}\in\mathbb R^{M\times300}.
$$

$$
W_o^K=\begin{bmatrix}(w_{o_0}^K)^\top\\ \vdots\\ (w_{o_{N-1}}^K)^\top\end{bmatrix}\in\mathbb R^{N\times300}.
$$

实现摘录（省略OOV计数和错误分支）：

```python
raw_name = torch.stack([glove[t] for t in label_tokens[label]]).mean(0)
sentence_vectors = [normalize(raw_name)]
for text in descriptions[label]:
    known = [glove[t] for t in description_tokens(text) if t in glove]
    sentence = torch.stack(known).mean(0) if known else torch.zeros(300)
    if sentence.norm() < EPS:
        sentence = raw_name
    sentence_vectors.append(normalize(sentence))
knowledge_vector = normalize(torch.stack(sentence_vectors).mean(0))
```

`W_*_name_raw.pt` 保留 $n_c$，`W_*_name_unit.pt` 保留 $u_{c,0}$，`W_*_K.pt` 保留 $w_c^K$。原模型加载时还将raw名称与自身初始化向量比较，要求 `atol=1e-6, rtol=1e-5` 内一致；知识向量的行范数检查采用 `atol=1e-5`。

### 3.10 第五步：构建全部组合的模板向量块

对每个对象 $o_i$ 与每个属性 $a_j$，采用三个固定模板：

```text
a {object} that is {attribute}
the {attribute} {object}
{attribute} appearance of {object}
```

实际计算使用名称别名token与模板填充词的向量和，填充词序列分别是 `a,that,is`、`the`、`appearance,of`。它没有将原始带点标签拼起来后再用描述正则分词。

令模板 $\ell$ 的填充词序列为 $\mathcal F_\ell$，则每个模板的单位向量为：

$$
b_{ij}^{(\ell)}=\mathcal N\left(
\frac{\sum_{w\in\mathcal T_{a_j}}g(w)+\sum_{w\in\mathcal T_{o_i}}g(w)+\sum_{w\in\mathcal F_\ell}g(w)}
{|\mathcal T_{a_j}|+|\mathcal T_{o_i}|+|\mathcal F_\ell|}
\right),\quad \ell=1,2,3.
$$

三个单位模板向量再平均并归一化：

$$
b_{ij}=\mathcal N\left(\frac{b_{ij}^{(1)}+b_{ij}^{(2)}+b_{ij}^{(3)}}{3}\right).
$$

固定对象 $o_i$ 后，所有属性的 $b_{ij}$ 堆成 `[M,300]`，写到 `W_ao_K/obj_00000.pt` 等文件；共有 $N$ 块。全矩阵对应 `[N,M,300]`，采用对象分块避免一次性保留全部组合向量。UT、MIT、C-GQA分别生成192、28175、278362个组合向量。

这一编码是词向量均值；词序本身不被建模。三个模板的区别主要来自不同填充词，不是语言模型理解了三个句子的语法。这些向量用于构建K，当前A/C/D/E并不直接将pair block输入CPF组合器。

### 3.11 第六步：由模板相似度构建原始K和归一化K

每个组合的知识原型是属性与对象知识向量相加后归一化：

$$
p_{ij}=\mathcal N(w_{o_i}^K+w_{a_j}^K).
$$

原始组合相似度矩阵：

$$
R_{ij}=b_{ij}^{\top}p_{ij},\qquad R\in\mathbb R^{N\times M}.
$$

$R$ 保存为 `K_oa_text_raw.pt`。两侧都是单位向量，因此数值是cosine；它度量两种文本表达的一致程度，不是由真实正负组合拟合的可行性概率。

令 $R_{min}=\min_{i,j}R_{ij}$、$R_{max}=\max_{i,j}R_{ij}$。构建器在全数据集矩阵上做min-max：

$$
K_{ij}^{template}=\begin{cases}
0.5,&R_{max}-R_{min}<10^{-8}\ \text{或}\ \sigma_{pop}(R)<10^{-8},\\
\dfrac{R_{ij}-R_{min}}{R_{max}-R_{min}},&\text{其他情况}.
\end{cases}
$$

这不是按对象逐行归一化，不是按属性逐列归一化，也不是softmax。$K$ 的行和/列和不要求为1。

关键实现（循环内的 `pair_embeddings`、`proto` 已按行单位化）：

```python
raw_k[oi] = (pair_embeddings * proto).sum(-1)
torch.save(pair_embeddings, stage / "W_ao_K" / ("obj_{:05d}.pt".format(oi)))

degenerate = ((raw_k.max() - raw_k.min()).item() < EPS
              or raw_k.std(unbiased=False).item() < EPS)
template_k = (torch.full_like(raw_k, 0.5) if degenerate else
              (raw_k - raw_k.min()) / (raw_k.max() - raw_k.min()))
k = template_k
```

`K_oa_text_template.pt` 保存模板候选；`K_oa_text.pt` 保存由 `prior_source` 实际选择的候选。本项目已核验的早期三数据集及UT v2均为 `prior_source=template`。

### 3.12 第七步：构建中心化校准矩阵及候选索引

令选中的 $K$ 包含 $NM$ 个值，全局均值和总体标准差定义为：

$$
\begin{aligned}
\bar K&=\frac{1}{NM}\sum_{i,j}K_{ij},\\
\sigma_K&=\sqrt{\frac{1}{NM}\sum_{i,j}(K_{ij}-\bar K)^2}.
\end{aligned}
$$

校准矩阵保存为 `K_oa_prior_centered.pt`：

$$
P_{ij}=\begin{cases}0,&\sigma_K<10^{-8},\\(K_{ij}-\bar K)/\sigma_K,&\text{其他情况}.\end{cases}
$$

代码使用 `std(unbiased=False)`。非退化情况下 $P$ 的总体均值约0、标准差约1；它允许负数。常量K先设为0.5再标准化得到全零，因此不能对所有组合加入同样的非零偏置。

```python
def centered_prior(k, eps=1e-8):
    k = k.float()
    if not torch.isfinite(k).all():
        raise ValueError("non-finite K")
    std = k.std(unbiased=False)
    if std < eps:
        return torch.zeros_like(k)
    return (k - k.mean()) / std

prior = P[obj_ids, attr_ids]  # 一维候选pair顺序由这两个索引明确给出
calibrated = scores + kcc_lambda * prior.unsqueeze(0)
```

在非退化且忽略浮点舍入时，min-max是正仿射变换，因此先min-max再总体标准化，与直接对 $R$ 做总体标准化相同。min-max让保存的候选范围易读，没有把cosine变成概率。

静态校准在训练/推理候选分数中复用同一个 $P$，不依赖当前图像。后续v2配对实验及名称锚点EKR关闭校准，$\lambda=0$；缓存生成了K不意味着该轮模型使用了K。

### 3.13 真实数值、版本差异与一个复算示例

下表来自已核验v1 `knowledge_summary.csv` 及UT v2缓存 `report.json`。K统计指实际选中的 `[N,M]` 模板min-max矩阵，OOV百分比按描述token计数：

| 知识版本 | K形状 | K范围 | K均值 | K总体标准差 | 描述OOV | 整句fallback |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| UT v1 | `[12,16]` | `[0,1]` | 0.508405 | 0.244008 | 0.361011% | 0 |
| MIT v1 | `[245,115]` | `[0,1]` | 0.722081 | 0.120815 | 0.259794% | 0 |
| C-GQA v1 | `[674,413]` | `[0,1]` | 0.756356 | 0.092073 | 0.337334% | 0 |
| UT v2 | `[12,16]` | `[0,1]` | 0.537236 | 0.201683 | 0.283286% | 0 |

UT v2的原始 $R$：min=0.7536016703、max=0.9328895807、mean=0.8499217033、总体std=0.0361592844。其标准化 $P$：min=−2.6637682915、max=2.2945129871、mean约 $-2.98\times10^{-8}$、总体std=1。

v2真实top记录中 `Boots.Ankle × Leather` 的 $K[0,6]=0.9589079022$。用已存统计量复算：

$$
R[0,6]\approx0.7536016703+0.9589079022\times(0.9328895807-0.7536016703)
\approx0.9255222644.
$$

$$
P[0,6]\approx\frac{0.9589079022-0.5372362137}{0.2016827911}
\approx2.0907668.
$$

假设启用 $\lambda=0.1$，这个组合会对每张图增加约0.2090767分；这是配方演示，并非v2实际运行，因为v2关闭了K。该示例同时说明“0到1的K”经过标准化后可产生更大的正负分数扰动。

模板高相似组合不能自动解释为现实中更常见或更可行。K失败的可能原因包括文本模板偏差、GloVe表示局限、分数尺度及图像无关的固定偏置；已有lambda负结果不能单独定位其中哪一项。

### 3.14 可选lexical K及本轮没有使用的内容

源目录还有 `K_oa_text_lexical.csv`。其来源报告记录：

$$
K_{ij}^{lexical}=0.75\,\cos(z_{o_i}^{word},z_{a_j}^{word})
+0.25\,\cos(z_{o_i}^{char3},z_{a_j}^{char3}),
$$

其中word与character-trigram的TF-IDF特征来自原语标签及OE/AE描述。它是另一个词面相似度候选，不是本轮GloVe模板K。UT归档lexical候选的范围为 `[0,0.1545676294]`、均值0.0161076362、总体std=0.0236923875；不要将这些统计混入上表模板K。

只有指定 `--prior-source lexical` 才会读取该CSV。读取时要求表头为 `object` 加完整属性列表、每行对象与固定顺序一致、形状正确、值有限且位于 `[0,1]`。选中后还会做相同总体标准化。当前所有已列结果均使用template，没有完成lexical替换实验，也没有LLM可行性评分、OI图传播或随机语义图构建。

### 3.15 第八步：v2描述字典与EKR残差库的构建

UT v2仅修订AE/OE描述，标签与OI语义保持不变。28类各3句，共84句；30句保留、54句改写。文件统一UTF-8/LF；因此即使标签或OI内容不变，重新序列化后的字节SHA也可能不同，版本lineage同时保存旧SHA与新SHA。

v2沿用上述GloVe构建器和同一个GloVe文件，生成独立cache。原raw名称与v1要求逐元素完全一致，不能通过重命名或换编码器制造文本收益。v2的manifest记录 `assistant_reviewed_train_images` 与 `human_reviewed=false`。

静态W聚合会丢弃每条描述身份，EKR因此进一步保存4个slot：slot 0是名称，slot 1—3是描述。对任意原语 $c$：

$$
U_c=[u_{c,0},u_{c,1},u_{c,2},u_{c,3}],\qquad
\mu_c=\frac14\sum_{r=0}^{3}u_{c,r},\qquad d_{c,r}=u_{c,r}-\mu_c.
$$

`units` 存 $U_c$、`mean` 存未归一化 $\mu_c$、`residual` 存 $d_{c,r}$、`correct` 存 $\mathcal N(\mu_c)$、`unit` 存名称slot。`mask` 标记有效槽，当前v2全部4槽有效。构建器支持不同条数时的padding和masked mean，但本轮没有变长描述。

中心化后 $\sum_r d_{c,r}=0$。不要对每个残差重新L2单位化，那会改变中心化性质及小差异的尺度；路由器后续投影键再归一化是另一操作。

正确聚合须与该版本cache的 `W_*_K.pt` 最大绝对误差不超过 $10^{-6}$，检查后直接保留cache旧张量原值。v2报告的两个轴聚合误差均为0。

错配负对照保留接收类自己的名称，用其他同轴原语的三句描述替换：

$$
w_c^{shuffle}=\mathcal N\left(\frac{u_{c,0}+u_{\pi(c),1}+u_{\pi(c),2}+u_{\pi(c),3}}{4}\right),
\qquad \pi(c)\ne c.
$$

属性使用固定seed 1729，对象使用1730。无固定点映射由随机顺序接环生成，固定后存档；三个训练seed共用该映射。名称不置换，属性与对象不跨轴交换。这里没有采样图结构。

```python
def derangement(n, seed):
    order = np.random.RandomState(seed).permutation(n)
    result = np.empty(n, dtype=np.int64)
    result[order] = np.roll(order, 1)
    assert not np.any(result == np.arange(n))
    return result.tolist()

mean = (units * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
residual = (units - mean[:, None]) * mask[..., None]
for i, donor in enumerate(permutation):
    shuffled[i, 0] = units[i, 0]
    shuffled[i, 1:4] = units[donor, 1:4]
shuffle = normalize(shuffled.mean(1))  # 当前每类恰好4个有效槽
```

v2质量检查禁止任何整句fallback；单句OOV比例大于50%或一个轴的描述总OOV比例大于10%均停止。实际属性OOV为0/181，对象为1/172，合计1/353。属性/对象残差和最大绝对值分别为 $7.45\times10^{-8}$、$5.96\times10^{-8}$，符合float32中心化误差范围；残差范数中位数分别为0.547679、0.617751。

### 3.16 第九步：落盘、来源冻结、完整性与模型消费

构建器先在输出父目录下建立临时stage，完整生成、散列后再rename为最终目录。输出不能覆盖非空缓存，不能位于知识源内部，也不能包含知识源。原知识只读；v1/v2和后续探索各自使用独立cache。

```text
cache/<dataset>/
  labels.json
  W_a_name_raw.pt  W_o_name_raw.pt
  W_a_name_unit.pt W_o_name_unit.pt
  W_a_K.pt        W_o_K.pt
  W_ao_K/obj_00000.pt ... obj_<N-1>.pt
  K_oa_text_raw.pt K_oa_text_template.pt
  K_oa_text.pt     K_oa_prior_centered.pt
  report.json     knowledge_coverage_report.json
  manifest.json   cache_manifest.json

bank/<dataset>/
  bank.pt  labels.json  descriptions.json  report.json  manifest.json
```

cache manifest记录全部标签、维度、dtype、模板、prior_source、编码/OOV配方、builder SHA、输入SHA和输出SHA。输入包含7份必需知识源、配置、dataset代码、名称初始化代码、split元数据和完整GloVe文件；选择lexical时再加入lexical CSV。`entry_provenance.csv` 未被早期模板cache独立冻结为语义输入，作为补充来源记录保存。

cache verify核验：两个manifest内容相同；输出及可访问的原输入SHA一致；W/K形状与标签轴一致；值有限；W行范数约1；重新计算 `centered_prior(K)` 与保存P逐元素一致；全一常量K标准化后严格全零。报告另保存原始/归一化/中心化K的min、max、mean、总体std、分位数、零比例、top/bottom组合和按split的覆盖统计。

bank manifest关联cache manifest SHA、7份知识源SHA、GloVe SHA、固定错配seed与输出SHA。加载bank必须同时核验标签顺序和文件哈希。

模型消费关系：A使用原名称；C用固定W知识向量进入原CPF网络；D在原名称分数上加P；E同时用W与P。W/P在模型里登记为buffer，知识构建本身没有optimizer或反向传播。v2的C_text/C_shuffle使用bank相应聚合表，名称锚点EKR使用名称anchor及描述残差。模型存在缓存文件不等于所有文件都被forward使用。

### 3.17 服务器构建与核验命令

以下命令用于记录和复现构建流程，不需要重新训练。原结果cache继续保留；如果要重建，只向新的时间戳目录输出。明确指定CPF解释器，因此不依赖终端当前显示base还是cpf。

**重建v1 UT模板cache的操作配方**（服务器Bash）：

```bash
(
  set -euo pipefail
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  ROOT=/data/xhw/czsl_repro
  PROJECT="$ROOT/src/KCC0601_v1"
  KNOWLEDGE_SOURCE="$ROOT/knowledge/kcc0601_v1/trusted_v1/ut-zap50k"
  BUILD_STAMP=$(date +%Y%m%d_%H%M%S)
  NEW_CACHE="/var/tmp/czsl_xhw_knowledge_rebuild_${BUILD_STAMP}/cache/ut-zap50k"

  "$CPF_PY" "$PROJECT/tools/build_knowledge.py" \
    --project-root "$PROJECT" \
    --cfg "$PROJECT/config/ut-zappos.yml" \
    --data-root "$ROOT/data/ut-zap50k" \
    --knowledge-dir "$KNOWLEDGE_SOURCE" \
    --glove-file "$ROOT/data/glove/glove.6B.300d.txt" \
    --output-dir "$NEW_CACHE" \
    --encoder glove --descriptions-per-class 3 --pair-mode template \
    --prior-source template --knowledge-status bulk_trusted_unverified --seed 91001

  "$CPF_PY" "$PROJECT/tools/build_knowledge.py" \
    --verify-only --output-dir "$NEW_CACHE"
)
```

这段选择原v1源字典，不能拿其重建结果替换v2。重建路径/config路径不同会改变manifest输入路径或SHA；PyTorch序列化字节也不保证跨环境完全一致。复现比较应同时保留来源身份并核对张量数值，不为追求相同SHA覆盖原归档。

**核验服务器既有v2 cache和bank**（服务器Bash）：

```bash
(
  set -euo pipefail
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  ROOT=/data/xhw/czsl_repro
  UT_V2_WORK=/var/tmp/czsl_xhw_ut_v2_20261006
  cd "$UT_V2_WORK/src"

  "$CPF_PY" tools/build_knowledge.py \
    --verify-only --output-dir "$UT_V2_WORK/cache/ut-zap50k"

  "$CPF_PY" -m transition.v2_prepare \
    --root "$ROOT" --work "$UT_V2_WORK" --verify-only
)
```

第二条核验是v2准备协议的完整检查，包含资产、cache、bank、固定donor及配对配置，正常结束打印 `V2 CPU INTEGRITY PASSED`。它不会重新训练。完整重建v2通过发布包 `prepare.sh` 进入：原构建器 `builder.build(...)` 调用生成cache，`build_bank(...)` 生成描述库，`table_integrity(...)` 核对v1名称不变、v2聚合与错配，最后冻结 `prepared.json`。CLI原构建器的knowledge-status选项只接受早期两个状态，因此不要仅更换命令中的状态字符串来代替v2准备入口。

**读取实际配方与统计**（服务器Bash）：

```bash
CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
UT_V2_WORK=/var/tmp/czsl_xhw_ut_v2_20261006
"$CPF_PY" -m json.tool "$UT_V2_WORK/cache/ut-zap50k/manifest.json"
"$CPF_PY" -m json.tool "$UT_V2_WORK/cache/ut-zap50k/report.json"
"$CPF_PY" -m json.tool "$UT_V2_WORK/bank/ut-zap50k/report.json"
```

`seed=91001` 记录构建环境种子，模板K计算没有随机抽图、随机图或随机向量；`1729/1730` 才用于固定错配映射，训练seed又是第三类独立身份。

## 4. 阶段一：三数据集基线与 UT A/C/D/E

### 4.1 完成规模

- 独立训练 run：8 个。UT 为 A/C/D/E，MIT 与 C-GQA 为 A/C。
- 每个 checkpoint 评价 CW 与 OW，因此形成 16 份主 metrics。
- 固定 checkpoint 的后处理：Dpost/Epost 各 5 个 $\lambda$、各 CW/OW，共 20 份诊断 metrics；这些不是 20 个新训练 run。
- 核验：16 份主 metrics、20 份诊断 metrics、11 个 checkpoint SHA、7 个知识源文件、18 个主要缓存张量均通过相应完整性检查。

### 4.2 UT A/C/D/E 结果

单位均为百分数。

| 设置 | 变体 | AUC | HM | best-seen | best-unseen | 最佳 epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CW | A | 39.470 | 54.841 | 67.959 | 66.410 | 6 |
| OW | A | 25.492 | 37.269 | 67.959 | 53.487 | 6 |
| CW | C | 44.262 | 57.894 | 68.871 | 71.502 | 6 |
| OW | C | 32.079 | 46.367 | 68.871 | 59.435 | 6 |
| CW | D | 32.464 | 51.795 | 65.564 | 56.611 | 6 |
| OW | D | 19.107 | 31.700 | 65.564 | 45.272 | 6 |
| CW | E | 37.239 | 53.897 | 66.135 | 64.741 | 6 |
| OW | E | 23.972 | 36.262 | 66.135 | 53.830 | 6 |

配对差值：

| 设置 | 对比 | $\Delta$AUC | $\Delta$HM | 解释 |
| --- | --- | ---: | ---: | --- |
| CW | C − A | +4.791 | +3.053 | 描述输入有明显正收益 |
| OW | C − A | +6.587 | +9.097 | 描述输入在 UT OW 上同样明显改善 |
| CW | D − A | −7.006 | −3.046 | 固定 $K$ 校准显著破坏结果 |
| OW | D − A | −6.385 | −5.569 | 固定 $K$ 在 OW 也退化 |
| CW | E − C | −7.022 | −3.997 | 在文本模型上再加固定 $K$ 仍退化 |
| OW | E − C | −8.107 | −10.105 | OW 退化更明显 |

结论：文本输入 C 值得继续研究；静态 $K$ 不是可靠增益来源。

### 4.3 三数据集 A/C 结果

| 数据集 | seed | 设置 | A AUC | C AUC | $\Delta$AUC | $\Delta$HM |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| UT-Zappos | 3351 | CW | 39.470 | 44.262 | +4.791 | +3.053 |
| UT-Zappos | 3351 | OW | 25.492 | 32.079 | +6.587 | +9.097 |
| MIT-States | 0 | CW | 13.707 | 13.392 | −0.315 | −0.217 |
| MIT-States | 0 | OW | 4.602 | 4.592 | −0.010 | +0.077 |
| C-GQA | 3351 | CW | 8.689 | 8.742 | +0.053 | +0.385 |
| C-GQA | 3351 | OW | 7.574 | 7.825 | +0.251 | +0.677 |

六个 AUC 对比中 C−A 为正 4 个、为负 2 个。只有 UT 的提升较大；MIT 基本持平略降，C-GQA 提升很小。每个数据集此时只有一对 seed，不能声称跨 seed 稳定或统计显著。

### 4.4 历史 CPF test 参考

这些是独立归档的历史 test 结果，不参与当前 validation 选参，也不与当前 VAL 增益直接相减。

| 数据集 | seed | 设置 | AUC | HM | Seen | Unseen |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| UT-Zappos | 3351 | CW | 36.510 | 52.590 | 61.490 | 68.480 |
| UT-Zappos | 3351 | OW | 27.070 | 43.870 | 61.490 | 51.300 |
| MIT-States | 0 | CW | 11.050 | 26.570 | 39.660 | 35.810 |
| MIT-States | 0 | OW | 4.180 | 14.670 | 39.660 | 14.380 |
| C-GQA | 3351 | CW | 6.810 | 21.300 | 38.420 | 20.370 |
| C-GQA | 3351 | OW | 1.580 | 7.270 | 38.420 | 4.980 |

## 5. 阶段二：固定 checkpoint 的 $\lambda$ 后处理

候选网格为 $\{0,0.025,0.05,0.1,0.2\}$。完整结果如下。

| 诊断 | $\lambda$ | CW AUC | CW HM | OW AUC | OW HM | OW unseen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dpost | 0 | 39.470 | 54.841 | 25.492 | 37.269 | 53.487 |
| Dpost | 0.025 | 38.399 | 55.024 | 24.449 | 36.775 | 52.931 |
| Dpost | 0.05 | 36.775 | 54.211 | 22.719 | 35.208 | 50.492 |
| Dpost | 0.1 | 31.572 | 51.037 | 18.348 | 30.782 | 43.902 |
| Dpost | 0.2 | 20.402 | 40.479 | 11.897 | 25.168 | 33.248 |
| Epost | 0 | 44.262 | 57.894 | 32.079 | 46.367 | 59.435 |
| Epost | 0.025 | 43.963 | 57.209 | 31.882 | 45.589 | 60.419 |
| Epost | 0.05 | 42.577 | 56.584 | 30.529 | 43.931 | 59.949 |
| Epost | 0.1 | 38.187 | 54.816 | 25.842 | 39.163 | 55.413 |
| Epost | 0.2 | 26.807 | 46.666 | 16.993 | 31.480 | 43.603 |

按预定主指标 VAL CW AUC，Dpost 与 Epost 都选择 $\lambda=0$。这只说明当前知识版本、当前网格、当前 checkpoint 和当前约束下不采用正强度固定校准；它不证明所有外部知识都无效。

## 6. 阶段三：UT v1 三 seed 语义归因

### 6.1 为什么加入 `A_unit` 与 `C_shuffle`

早期的 A/C 比较只能回答“换成文本聚合向量后是否提高”，无法区分：

- 名称向量被单位化带来的变化；
- 加入领域相关文本带来的普通正则或输入变化；
- 正确类别描述的语义归属是否真正重要。

因此定义四组：

| 组别 | 输入 | 归因作用 |
| --- | --- | --- |
| `A_raw` | 原始名称向量 | 原 CPF 参考 |
| `A_unit` | 单位化名称向量 | 排除名称归一化效应 |
| `C_text` | 本类名称 + 本类三句描述 | 正确文本处理组 |
| `C_shuffle` | 本类名称 + 固定错误类别三句描述 | 错配文本负对照 |

### 6.2 v1 运行结果

三 seed 为 0、3351、2026。主指标为 VAL CW AUC。

| seed | `C_text-A_unit` CW AUC | `C_text-C_shuffle` CW AUC |
| ---: | ---: | ---: |
| 0 | +0.508647 | −2.088948 |
| 3351 | +4.662171 | +1.019615 |
| 2026 | +2.125442 | +0.269668 |
| 平均 | +2.432087 | −0.266555 |

三个 seed 中，`C_text` 全部优于 `A_unit`；但相对 `C_shuffle` 两次小幅提高、一次明显下降，平均为负。因此 v1 支持“文本输入有作用”，不支持“正确语义归属是稳定收益来源”。

### 6.3 审计修复与结果边界

首次审计脚本 v1 出现两类工具兼容问题：

1. YACS 日志打印将 `1e-05`、`None` 表示为字符串，而 `cfg.dump()` 保存为浮点数和空值；直接比较造成假不一致。
2. 旧 A/C 的 NPZ 保存 `top5_indices_at_bias0`，新格式保存 `top1_at_bias0`；v1 工具没有兼容旧字段。

修正版审计工具统一使用 YACS 表示，并从旧 top5 第一列恢复 bias=0 top1。修复后所有 run、配置、checkpoint、bank、固定置换和预测配对检查通过。这个修复没有改变任何 AUC/HM，也没有把 gate 变为 true。

## 7. 阶段四：训练图审核与 v2 描述

### 7.1 审核规模与边界

- 16 个属性图板 + 12 个对象图板 = 28 张图板。
- 共 560 个面板、528 张不同训练原图，分辨率 136×136。
- 共 84 句描述；保留 30 句，改写 54 句。
- 没有读取 test pair 列表，没有查看 test 图像，也没有依据 VAL 错误类别定向改句子。
- 两个原导出路径失效，但都存在可读的同文件名候选；审计报告保留了这两个异常，没有复制或替换训练数据。
- 审阅者是 Codex assistant；`human_reviewed=false` 与 `compatible_with_formal_results=false` 被保留，不能写成人工专家批准。

主要修订包括：删除过窄的 `narrow/fitted/slim`、对象固定为 leather、严格 closed lacing、统一 lace-up、thin/flexible、whole-shoe waterproof 等断言；将不可直接观察的触感、性能、材质真伪与可见外观区分开。

### 7.2 v2 文本公式

设类别 $c$ 的原始名称向量为 $n_c$，三句描述为 $s_{c,j}$，$j=1,2,3$。名称单位向量为：

$$
u_{c,0}=\mathcal N(n_c).
$$

描述句向量为：

$$
u_{c,j}=\mathcal N\left(
\frac{\sum_{w\in V(s_{c,j})}g(w)}{|V(s_{c,j})|}
\right),\qquad j=1,2,3.
$$

正确文本输入：

$$
e_c^{text}=\mathcal N\left(\frac{u_{c,0}+u_{c,1}+u_{c,2}+u_{c,3}}{4}\right).
$$

固定错配文本输入：

$$
e_c^{shuffle}=\mathcal N\left(
\frac{u_{c,0}+u_{\pi(c),1}+u_{\pi(c),2}+u_{\pi(c),3}}{4}
\right),
$$

其中属性和对象分别使用无固定点置换，$\pi(c)\ne c$；三个训练 seed 使用同一个置换。

为后续 EKR 保存中心化描述残差：

$$
\mu_c=\frac14\sum_{j=0}^{3}u_{c,j},\qquad
r_{c,j}=u_{c,j}-\mu_c,
$$

所以：

$$
\sum_{j=0}^{3}r_{c,j}=0.
$$

v2 配对实验本身只更换 CPF 的 primitive 输入表；残差只保存到 bank，尚未用于训练 EKR。静态 $K$ 始终关闭，$\lambda=0$。

### 7.3 bank 构建的关键实现

`transition/bank.py` 的核心逻辑：

```python
mean = (units * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
residual = (units - mean[:, None]) * mask[..., None]

shuffled[i, 0] = units[i, 0]          # 保留本类名称
shuffled[i, 1:donor_n] = units[donor, 1:donor_n]

table["correct"] = normalize(mean)
table["shuffle"] = normalize(shuffled_mean)
```

构建器还逐元素验证 v1 `C_text` 聚合能够复现旧 C 缓存，误差要求不超过 $10^{-6}$；句子 fallback 在 v2 中视为错误并停止。

## 8. 阶段五：UT v2 四控制 × 三 seed 完整结果

### 8.1 运行与完整性

- 4 组 × 3 seed = 12 个独立训练身份。
- 每项 10 epoch，共 24 次 VAL 评价（CW 与 OW）。
- seed 0 和 3351 的四组最佳 epoch 均为 6；seed 2026 四组最佳 epoch 均为 4。
- 每次 3214 张 VAL 图；CW 116 个分数候选，OW 192 个候选。
- 结果 ZIP：109 个文件，CRC、路径、48 个 collection 输入 SHA、报告 fingerprint、12 份 effective config、checkpoint 对应、知识源与置换均通过核对。
- 服务器训练曾因 `/data` 分区 100% 满而在一项中途停止；释放空间后重新执行队列。已完成的 run 被验证/跳过，部分 run 被保留并要求新版本，最终 12 个身份均有完整记录。最终结果不是把残缺输出混入汇总。

### 8.2 每个 seed 的完整指标

单位为百分数。`Umax` 是 bias 扫描中的 `best_unseen`。

| seed | 组别 | CW AUC | CW HM | OW AUC | OW HM | OW Umax | 最佳 epoch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | A_raw | 42.250 | 55.070 | 30.450 | 44.510 | 57.380 | 6 |
| 0 | A_unit | 42.820 | 55.870 | 29.460 | 43.820 | 55.670 | 6 |
| 0 | C_text | 45.440 | 58.080 | 34.940 | 48.520 | 63.930 | 6 |
| 0 | C_shuffle | 44.640 | 58.720 | 30.710 | 45.940 | 53.790 | 6 |
| 3351 | A_raw | 39.470 | 54.840 | 25.490 | 37.270 | 53.490 | 6 |
| 3351 | A_unit | 39.600 | 55.080 | 23.880 | 35.690 | 51.010 | 6 |
| 3351 | C_text | 44.000 | 58.170 | 28.660 | 40.610 | 58.320 | 6 |
| 3351 | C_shuffle | 42.980 | 56.830 | 28.250 | 42.800 | 52.800 | 6 |
| 2026 | A_raw | 41.830 | 57.660 | 31.060 | 46.010 | 56.830 | 4 |
| 2026 | A_unit | 42.000 | 57.330 | 31.050 | 46.160 | 57.470 | 4 |
| 2026 | C_text | 44.170 | 58.560 | 33.190 | 48.370 | 59.560 | 4 |
| 2026 | C_shuffle | 44.860 | 61.440 | 31.880 | 48.450 | 55.070 | 4 |

### 8.3 三 seed 均值与标准差

括号为三个训练 seed 的样本标准差，不是置信区间。

| 组别 | CW AUC | CW HM | OW AUC | OW HM | OW Umax |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_raw | 41.182 (1.497) | 55.857 (1.562) | 29.001 (3.053) | 42.595 (4.673) | 55.898 (2.106) |
| A_unit | 41.473 (1.673) | 56.095 (1.141) | 28.129 (3.767) | 41.892 (5.497) | 54.714 (3.335) |
| C_text | 44.540 (0.785) | 58.270 (0.252) | 32.261 (3.242) | 45.835 (4.523) | 60.605 (2.944) |
| C_shuffle | 44.161 (1.025) | 58.998 (2.317) | 30.280 (1.855) | 45.731 (2.833) | 53.887 (1.137) |

### 8.4 配对差值

| 比较 | CW AUC | CW HM | OW AUC | OW HM | OW Umax |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_unit − A_raw | +0.291 | +0.238 | −0.871 | −0.704 | −1.184 |
| C_text − A_unit | +3.067 | +2.176 | +4.131 | +3.943 | +5.891 |
| C_text − C_shuffle | +0.379 | −0.727 | +1.980 | +0.104 | +6.718 |
| C_text − A_raw | +3.358 | +2.414 | +3.260 | +3.240 | +4.707 |
| C_shuffle − A_unit | +2.688 | +2.903 | +2.151 | +3.839 | −0.827 |

关键逐 seed 差值：

| seed | C_text − A_unit CW AUC | C_text − C_shuffle CW AUC | C_text − C_shuffle CW HM | C_text − C_shuffle OW AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +2.622 | +0.801 | −0.639 | +4.227 |
| 3351 | +4.404 | +1.019 | +1.340 | +0.407 |
| 2026 | +2.176 | −0.683 | −2.883 | +1.308 |

### 8.5 预定 gate

| 条件 | 实际结果 | 状态 |
| --- | --- | --- |
| 三 seed 完整 | 0、3351、2026 | 通过 |
| 平均 C_text−A_unit CW AUC ≥ +0.5 pp | +3.067 | 通过 |
| 平均 C_text−C_shuffle CW AUC ≥ +0.5 pp | +0.379 | **未通过** |
| C_text−A_unit 至少 2/3 seed 为正 | 3/3 | 通过 |
| C_text−C_shuffle 至少 2/3 seed 为正 | 2/3 | 通过 |
| C_text 平均 CW AUC 不低于 A_raw | +3.358 | 通过 |
| 相对 A_unit：CW HM ≥ −0.5 pp | +2.176 | 通过 |
| 相对 A_unit：OW AUC ≥ −0.5 pp | +4.131 | 通过 |
| 相对 A_unit：OW Umax ≥ −2 pp | +5.891 | 通过 |

九项中仅一项失败，但它正是用于判断“正确语义是否优于错配文本”的核心归因条件。距离阈值约 0.121 pp，不能事后降低门槛或增加 seed 直到通过。最终 `proceed=false`。

## 9. 阶段六：v2 逐图预测分析

### 9.1 独立核对

- 预测 ZIP SHA256：`50b11e85cd2e02d8db5013e2ff71548b8d00bf8c591287b94b3eb66abcf9d38a`。
- 关联结果 ZIP SHA256：`31285ef2b7d1fdea65049b322246d623f8a3073cf373342d05e67abe0fa7e022`。
- 共通过 434 项文件、身份、标签、顺序、形状、范围和计数检查。
- 24 次预测使用相同 3214 张 VAL 图；gold 标签与冻结原始 VAL 元数据逐项一致。
- OW 逐图重算的 pair/attribute/object 无 bias 准确率与原 metrics 一致。
- 三个 seed 是同一批图上的三次训练结果，不能当成 9642 张独立图做显著性检验。

### 9.2 bias=0 的总体结果

| 组别 | 全部 pair | seen pair | unseen pair | 预测为 seen 的比例 | 属性准确率 | 对象准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_raw | 22.827 | 66.857 | 6.304 | 93.207 | 33.396 | 78.635 |
| A_unit | 23.398 | 66.781 | 7.117 | 92.398 | 33.769 | 78.988 |
| C_text | 22.734 | 67.617 | 5.891 | 93.435 | 33.810 | 78.210 |
| C_shuffle | 22.703 | 67.199 | 6.005 | 93.124 | 33.748 | 78.490 |

`C_text-A_unit`：

- 全部 pair：$-0.664$ pp。
- seen pair：$+0.836$ pp，三个 seed 均为正。
- unseen pair：$-1.227$ pp，三个 seed 均为负。
- 预测落到 seen 组合的比例：$+1.037$ pp。
- 属性准确率：$+0.041$ pp。
- 对象准确率：$-0.778$ pp。

这表明 v2 文本的 AUC/HM 增益依赖 bias 扫描和排序变化；它没有让 $b=0$ 的逐图 top1 普遍更准。

### 9.3 按对象的完整 OW bias=0 差值

下面是 `C_text-A_unit`。`正 seed` 表示三个 seed 中该组 pair 差值严格为正的数量。

| gold 对象 | n | pair 差值 | 正 seed |
| --- | ---: | ---: | ---: |
| Shoes.Boat.Shoes | 316 | −9.177 | 0/3 |
| Shoes.Oxfords | 381 | −0.962 | 1/3 |
| Shoes.Heels | 73 | −1.826 | 0/3 |
| Boots.Knee.High | 3 | 0.000 | 0/3 |
| Shoes.Loafers | 127 | 0.000 | 0/3 |
| Shoes.Sneakers.and.Athletic.Shoes | 34 | 0.000 | 0/3 |
| Sandals | 1133 | +0.029 | 1/3 |
| Boots.Mid-Calf | 255 | +0.261 | 2/3 |
| Slippers | 32 | +2.083 | 2/3 |
| Boots.Ankle | 357 | +0.280 | 2/3 |
| Shoes.Clogs.and.Mules | 195 | +1.368 | 2/3 |
| Shoes.Flats | 308 | +2.381 | 3/3 |

Boat Shoes 的 316 张图上，对象准确率平均下降约 11.287 pp，而属性准确率只下降约 0.105 pp，主要问题是对象混淆。

### 9.4 按 30 个 VAL 组合的完整 `C_text-A_unit` 差值

| gold 组合 | n | split | pair 差值 | 正 seed |
| --- | ---: | --- | ---: | ---: |
| Synthetic / Sandals | 767 | unseen | +0.130 | 1/3 |
| Leather / Shoes.Flats | 308 | seen | +2.381 | 3/3 |
| Leather / Shoes.Boat.Shoes | 263 | unseen | −11.280 | 0/3 |
| Full.grain.leather / Boots.Ankle | 259 | unseen | 0.000 | 0/3 |
| Leather / Shoes.Oxfords | 216 | seen | +0.617 | 3/3 |
| Rubber / Sandals | 172 | unseen | 0.000 | 0/3 |
| Suede / Shoes.Oxfords | 165 | unseen | −3.030 | 0/3 |
| Full.grain.leather / Shoes.Loafers | 124 | unseen | 0.000 | 0/3 |
| Faux.Leather / Boots.Mid-Calf | 115 | unseen | 0.000 | 0/3 |
| Patent.Leather / Sandals | 115 | unseen | −0.580 | 0/3 |
| Leather / Shoes.Clogs.and.Mules | 108 | seen | −0.926 | 1/3 |
| Rubber / Boots.Ankle | 89 | unseen | +1.124 | 2/3 |
| Suede / Shoes.Clogs.and.Mules | 83 | unseen | +4.418 | 2/3 |
| Nubuck / Sandals | 79 | unseen | 0.000 | 0/3 |
| Rubber / Boots.Mid-Calf | 77 | seen | +1.299 | 2/3 |
| Satin / Shoes.Heels | 73 | seen | −1.826 | 0/3 |
| Canvas / Shoes.Boat.Shoes | 53 | unseen | +1.258 | 1/3 |
| Full.grain.leather / Boots.Mid-Calf | 37 | seen | 0.000 | 0/3 |
| Nylon / Slippers | 25 | unseen | +1.333 | 1/3 |
| Rubber / Shoes.Sneakers.and.Athletic.Shoes | 15 | seen | 0.000 | 0/3 |
| Wool / Shoes.Sneakers.and.Athletic.Shoes | 15 | unseen | 0.000 | 0/3 |
| Sheepskin / Boots.Mid-Calf | 13 | seen | −2.564 | 1/3 |
| Faux.Fur / Boots.Mid-Calf | 13 | unseen | 0.000 | 0/3 |
| Nubuck / Boots.Ankle | 9 | seen | 0.000 | 0/3 |
| Sheepskin / Slippers | 7 | seen | +4.762 | 1/3 |
| Nylon / Shoes.Sneakers.and.Athletic.Shoes | 4 | seen | 0.000 | 0/3 |
| Faux.Leather / Shoes.Loafers | 3 | seen | 0.000 | 0/3 |
| Nylon / Boots.Knee.High | 3 | seen | 0.000 | 0/3 |
| Wool / Shoes.Clogs.and.Mules | 2 | seen | 0.000 | 0/3 |
| Nubuck / Shoes.Clogs.and.Mules | 2 | seen | 0.000 | 0/3 |

支持数很小的组合只能描述，不能作为稳定收益证据。

### 9.5 Boat Shoes 的具体错误流向

`Leather × Boat Shoes` 有 263 张图，平均 pair 准确率下降 11.280 pp，相当于每个 seed 少答对约 29.667 张。三个 seed 合计有 97 个“名称正确、文本错误”事件和 8 个反向纠正事件。97 个新错误主要流向：

| 新预测 | 事件数 |
| --- | ---: |
| Leather × Loafers | 36 |
| Leather × Oxfords | 28 |
| Leather × Flats | 15 |
| Canvas × Sneakers | 8 |
| Leather × Sneakers | 7 |
| Suede × Loafers | 2 |
| Suede × Oxfords | 1 |

大部分事件仍预测为 Leather，错误集中在对象类别。该结果支持“描述造成对象语义混淆”作为待验证解释，但现有数据没有导出对象向量和完整分数，不能把它写成已经证明的因果机制。

### 9.6 互补性

| 三 seed 合计预测事件 | C_text 对 A_unit | C_text 对 C_shuffle |
| --- | ---: | ---: |
| 错误 → 正确 | 109 | 134 |
| 正确 → 错误 | 173 | 131 |
| 平均完全相同预测比例 | 92.149% | 91.029% |
| 同一图三 seed 都被纠正 | 0 | 0 |
| 同一图三 seed 都被弄错 | 5 | 0 |

使用 VAL gold 做不可部署的 oracle，只要 A_unit 或 C_text 任一预测正确就算对，OW pair 准确率为 24.528%，比 A_unit 高 1.130 pp。这只能说明两套 top1 存在互补，不能当成可学习门控的预计收益。

### 9.7 CW top1 文件的限制

保存的 CW `top1_at_bias0` 是在原生 closed mask 之前直接 argmax；原生 CW 评价只允许 98 个候选，而分数表含 116 个候选。两者总体准确率仅差约 0.03 至 0.04 pp，但逐图 CW 文件不是严格的原生 masked 预测。因此类别诊断主表使用已与评价器一致核对的 OW top1。

## 10. 原 EKR 的原理、实现与当前问题

### 10.1 原 EKR 数学定义

原方案冻结 `C_text` 基模型，保存每个属性的名称和描述单位向量 $t_{ar}$：

$$
\mu_a=\frac1{R_a}\sum_r t_{ar},\qquad d_{ar}=t_{ar}-\mu_a.
$$

复用属性视觉向量：

$$
u(x)=\mathcal N(attr\_emb(x)).
$$

每条描述的中心化视觉证据：

$$
e_{ar}(x)=\frac12u(x)^\top d_{ar}
=\frac12\left[
\cos(u,t_{ar})-\frac1{R_a}\sum_j\cos(u,t_{aj})
\right].
$$

图像与候选对象共同形成低秩查询：

$$
q(x,o)=\mathcal N\left((Q_xu(x))\odot\sigma(Q_oW_o^K)\right),
\qquad
k_{ar}=\mathcal N(Q_kd_{ar}).
$$

描述 logits 与 null logit：

$$
\ell_{ar}(x,o)=\frac{q(x,o)^\top k_{ar}}{\tau},
$$

$$
\ell_{a0}(x,o)=\frac{q(x,o)^\top\mathcal N(k_0)}{\tau}+b_0,
\qquad e_{a0}=0.
$$

用 softmax 权重聚合知识残差：

$$
\alpha_{ar}=\operatorname{softmax}_r(\ell_{ar}),
\qquad
\Delta(x,a,o)=\sum_{r=0}^{R_a}\alpha_{ar}(x,o)e_{ar}(x).
$$

最终分数：

$$
s_{EKR}(x,a,o)=s_C(x,a,o)+\eta\Delta(x,a,o),
\qquad
\eta=\eta_{max}\tanh(g).
$$

首版设置 $\eta_{max}=0.1$、$g=0$、$\tau=0.2$、低秩维度 32。$g=0$ 使初始残差严格为零；训练后 $\eta$ 可正可负。

### 10.2 Router 关键代码

```python
query = normalize(qx[:, None, :] * sigmoid(qo(object_source))[None, :, :])
logits = einsum("bnk,mrk->bnmr", query, keys) / temperature

null = einsum("bnk,k->bn", query, normalize(null_key))
logits = cat([null, logits], dim=-1)
weights = logits.softmax(-1)

delta = (weights[..., 1:] * evidence[:, None]).sum(-1)
residual = output * eta
```

工程实现按对象分块，不显式创建完整 `B×N×M×D` 张量；padding 以负无穷 mask，null 始终可选。

### 10.3 BranchModel 关键代码

```python
with torch.no_grad():
    values = self.base.raw_forward(batch)
base, feature = values["pred"], values["attr_feature"]

residual = self.router(feature, attr_ids, obj_ids, force_off=force_off)
return base + residual
```

原实现的 `base` 是冻结的 `C_text` checkpoint。`force_off=True` 或 $\eta=0$ 只关闭新增 residual，返回的仍然是 `C_text`。

### 10.4 为什么现有 null 不能自动修复 Boat Shoes

null 只表示“不增加新的 EKR 残差”。如果对象混淆已经存在于基础 `C_text` 分数中，选择 null 后仍然保留该错误：

$$
\Delta=0\quad\Longrightarrow\quad s_{EKR}=s_C,
$$

而不是退回 `A_unit`。所以原 EKR 是否能弥补 Boat Shoes 退化仍需实验，不能从 null 的存在直接推出。

## 11. 当前最值得继续的方向：名称锚点条件知识

### 11.1 最小改动

把冻结锚点从 `C_text` 改成 `A_unit`，保留原有中心化知识条目、对象条件查询、null、低秩路由和有界残差：

$$
s_{N\text{-}EKR}(x,a,o)
=s_{A\_unit}(x,a,o)+\eta\Delta_{EKR}(x,a,o).
$$

当强制关闭知识分支时：

$$
\eta=0\quad\Longrightarrow\quad s_{N\text{-}EKR}=s_{A\_unit}.
$$

这使“是否使用描述知识”成为真正可验证的增量问题，避免把静态描述造成的损失固化在锚点里。

### 11.2 名称锚点实现

工作区 `name_anchor_ekr_20261007/transition/name_router.py` 已表达两个关键控制：

```python
# 对象条件只用对象名称，不让 OE 描述泄漏进名称控制
self.object_source = bank['o']['unit'].float().clone()

if mode == 'name_only':
    # 重复名称保持四个 slot 和相同参数规模
    names = bank['a']['unit'][:, None, :].expand(-1, 4, -1).clone()
    self.key_source = names
    self.value_source = names / 2
```

`name_smoke.py` 设计为检查：

- 初始输出与真实 A_unit checkpoint 最大误差不超过 $10^{-6}$；
- 冻结 anchor 不接收梯度且更新前后参数摘要不变；
- router 能产生有限且非零的梯度；
- 更新 router 后 `force_off` 仍恢复 A_unit，误差不超过 $10^{-6}$。

当前状态已更新为：**完整 release 与服务器安装记录存在，真实 GPU smoke 记录通过，seed 0 的三项三轮训练和 CW/OW 评价已完成，本地审阅通过记录与配对复算。正确描述未通过 pilot 筛选，尚未补另两个 seed。** 本地审阅包不含三项新权重，因此不声称已独立 GPU 重放新模型。详细数据见第17节。

### 11.3 建议的最小实验矩阵

| 组别 | 锚点/知识 | 目的 |
| --- | --- | --- |
| N0 | 冻结 A_unit，残差关闭 | 精确名称锚点 |
| R0 | 名称槽，等参数 router | 排除增加参数和路由器容量的收益 |
| R3 | 原始描述注意力 | 检查普通多描述选择 |
| R5 | 中心化描述 + 对象条件 + null | 核心条件知识模型 |
| R7 | 固定错配描述残差 | 检查正确知识归属 |
| R5-no-object | 去掉对象条件 | 验证对象条件是否必要 |
| R5-no-null | 去掉拒绝项 | 验证 null 是否有保护作用 |

上表是此前建议的矩阵。实际第一轮采用 N_name/N_clean/N_shuffle 三项，对应名称容量控制、中心化条件描述、固定错配描述；N_clean 没有显示对照优势，当前版本不自动追加 3351/2026。普通描述注意力及对象/null 的重新训练消融尚未完成，不能当作已有证据。

## 12. 下一步执行顺序

1. 已完成：18项 A_unit/C_text/C_shuffle、三 seed、CW/OW 的冻结分数回放；名称锚点 release、GPU smoke 和 seed 0 三项 pilot。不能重复列为待完成。
2. 当前应做：运行已交付的 `name_ekr_mechanism_diag_20261007_v1`，从旧权重导出分支分数、槽注意力、残差与 margin；没有新增训练。
3. 查看固定 TRAIN 子集与 VAL 的差异，以及描述错配、对象上下文错配、推理去 null 的响应。TRAIN gold 只用于诊断与训练监督，不进入部署预测。
4. 若诊断提示描述值与冻结视觉特征未对齐，再研究低秩可学习值映射或局部属性模块适配；若 null 对错配无响应，再研究 TRAIN 内的知识不匹配监督。每次只改变明确的机制假设，并提供同容量名称/错配控制。
5. 新版本先另立有限预算探索协议，记录旧两个 gate 未通过及开发 VAL 已被反复使用；不能把修改阈值当成修复方法，也不能把新结果写成旧 gate 通过。
6. 新版本出现配对优势后，再做多 seed、多错配映射的确认与跨数据集验证。由于 MIT/C-GQA 的早期文本收益很小，不能默认 UT 结果会推广。
7. 全程保持 USG 随机语义图为独立路线；描述错配与污染对照没有图采样或图传播，不是随机图方法。

## 13. 已支持、尚未支持与应避免的表述

### 13.1 已支持

- UT 的描述输入相对名称控制提高了 bias 扫描下的 AUC/HM。
- v2 中正确描述相对错配描述在 OW AUC 上有三 seed 正向信号。
- 静态描述在不同对象/组合上的效果不均匀。
- Boat Shoes 的主要退化表现为对象混淆，Flats 存在稳定局部改善。
- 现有结果足以提出“条件使用或拒绝知识”的研究问题。

### 13.2 尚未支持

- 正确描述语义已在预定 CW 门槛下得到稳定归因证明。
- EKR 已经提高性能或修复 Boat Shoes。
- null 自动恢复名称模型。
- 三个训练 seed 构成统计显著性证明。
- UT 上的观察能自动推广到 MIT/C-GQA。
- GloVe 描述路由等同于显式逻辑推理或概率分解。

### 13.3 应避免

- 删除 seed 2026 或不利类别。
- 在看到结果后降低 $+0.5$ pp 门槛。
- 根据 VAL Boat Shoes 错误再次修改描述后把结果当盲测。
- 把 bias=0 top1、best-HM、AUC 混为同一个预测点。
- 把 `C_shuffle` 称为 USG 随机语义图。
- 把名称锚点 EKR 已完成的负结果 pilot 写成“新方法已获收益”。

## 14. 工程问题与可复现性记录

### 14.1 磁盘空间

一次训练因 `/data` 分区 100% 满触发 `OSError: [Errno 28] No space left on device`。当时 inode 仅使用约 12%，根因是字节容量，不是 inode 耗尽。`/data/xhw` 约 642 GB，其中多个旧训练目录占较大空间；pip cache 约 5.1 GB。处理后保留已有 checkpoints，用队列的完成检测避免重复训练。

### 14.2 Python/pip 异常

基础 Python 3.14 的 pip vendor 子模块曾出现“文件存在但 import 找不到”的异常，同时 `/proc/self/maps` 显示 `/etc/ld.so.preload` 注入的 `libudev-compat.so` 与 `udev-settle-ng.so`。这属于系统/环境异常，不是 CPF 模型逻辑问题；后续实验固定使用 `cpf` 环境 Python 3.8.18。不要在服务器上用损坏的 base pip 覆盖 `cpf` 环境。

### 14.3 训练图路径

早期导出脚本把两张图片指向旧目录名称；审计发现候选目录中存在同文件名，但源配置根目录之外的路径不能自动替换。最终图板审核报告保留两个异常并未改动训练数据。

### 14.4 审计工具版本

v1 审计的 YACS/旧 NPZ 兼容问题已由 v2 工具修复。旧审计目录保留，不删除；最终研究结论采用修复后审计与 v2 结果包的独立复算。

## 15. 证据文件索引

主要结果与分析：

- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/阶段实验报告_自动草稿.md`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/tables/all_val_metrics.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/tables/lambda_analysis.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/tables/ac_comparison.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/tables/knowledge_structure.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/tables/knowledge_summary.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/week5_evidence_20261005_verified/knowledge_schema.json`
- `C:/Users/llenovo/Desktop/UT_description_review/UT_v2_results_20261007_153654/files/cache/ut-zap50k/report.json`
- `C:/Users/llenovo/Desktop/UT_description_review/UT_v2_results_20261007_153654/files/bank/ut-zap50k/report.json`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/ut_semantic_v2_20261006/UT审核完成报告.md`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ut_v2_results_audit_20261007/UT_v2结果审阅与下一步_20261007.md`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ut_v2_results_audit_20261007/artifact_audit.json`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ut_v2_prediction_analysis_20261007/UT_v2逐图预测分析与EKR判断_20261007.md`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ut_v2_prediction_analysis_20261007/grouped_paired_deltas.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ut_v2_prediction_analysis_20261007/changed_predictions.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/ut_v2_prediction_analysis_20261007/prediction_analysis.json`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/name_ekr_pilot_review_20261007/名称锚点EKR_阶段B审阅与下一步判断_20261007.md`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/name_ekr_pilot_review_20261007/final_metrics.csv`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/output/name_ekr_pilot_review_20261007/training_audit.json`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/name_ekr_mechanism_diag_20261007_v1/README.md`

关键实现：

- `C:/Users/llenovo/Documents/ChatGPT/CZSL/tmp/ekr_source_fixture/models/kcc_cache.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/tmp/ekr_source_fixture/tools/build_knowledge.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/tmp/ekr_source_fixture/models/word_embedding_utils.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/tmp/ekr_source_fixture/models/kcc_cpf.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/ut_semantic_v2_20261006/transition/bank.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/ut_semantic_v2_20261006/transition/router.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/ut_semantic_v2_20261006/transition/runtime.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/ut_semantic_v2_20261006/transition/evaluate.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/name_anchor_ekr_20261007/transition/name_router.py`
- `C:/Users/llenovo/Documents/ChatGPT/CZSL/name_anchor_ekr_20261007/transition/name_smoke.py`

## 16. 最终研究判断

现有实验已经排除了两个看似直接、实际不可靠的方向：固定组合先验 $K$，以及“只要加入静态描述就能证明正确语义贡献”。它同时发现了可以继续研究的真实问题：外部知识既能帮助，也能按对象和组合系统性地伤害预测。

名称锚点已完成一次小预算验证，但正确描述没有显示增量优势。当前更值得投入的是有明确问题的机制诊断：冻结视觉特征是否能利用固定描述差异值、对象条件是否真的影响路由、null 是否对不匹配知识响应。若结果支持具体缺陷，保留 CPF、GloVe、v2 描述库、evaluator 与数据划分，另立可学习对齐/可靠性监督的机制改进版本；若没有相应证据，不继续机械增加当前版本预算。

## 17. 补充：名称锚点 EKR 阶段 B 真实结果

来源：`name_ekr_review_20261007_194443.zip` 及其本地审阅报告。三项均为 seed 0、三轮、同一个冻结 A_unit、28,834 个可训练分支参数。这里只使用已审阅的记录，不声称本地独立重放了新权重。

| 模型 | CW AUC（%） | 相对 N0（pp） | OW AUC（%） | 相对 N0（pp） | 最佳 epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| N0 冻结 A_unit | 42.8186 | 0 | 29.4649 | 0 | 原权重 |
| N_name 名称分支 | 43.1232 | +0.3046 | 29.5805 | +0.1156 | 3 |
| N_clean 正确描述 | 42.8664 | +0.0478 | 29.5018 | +0.0369 | 2 |
| N_shuffle 错配描述 | 42.9992 | +0.1806 | 29.5008 | +0.0359 | 3 |

pilot 要求 N_clean 对 N0/N_name/N_shuffle 的 CW AUC 均至少提高 +0.2 pp。实际差值依次为 +0.047790、−0.256764、−0.132762 pp。保护指标全部满足，但三个主要收益条件失败。即使把收益阈值降到零，也不能令 N_clean 优于两个分支对照。

已记录的机制现象：

- N_clean 训练损失从1.415845降至1.397870，不能解释为没有训练。
- N_clean 最佳轮次的全局幅度为 −0.056094；分支允许有符号幅度，负号本身不是错误。
- N_clean 的 CW null 平均注意力仅0.059593%，错配分支约0.082420%；这没有证明拒绝无效知识的能力。平均权重不等于选择 null 的图像比例。
- N_clean 平均绝对 CW 分数增量约0.002026；影响正确/竞争类别 margin 的能力尚待诊断，不能仅凭数值小就断言需要增大幅度。
- OW bias=0 下只改变11个预测：0个纠正、2个破坏，对象预测没有改变；Boat Shoes 组也没有预测变化。这只涉及一个偏置点，不替代 AUC。
- `selected_state.json` 的 adopted=true 使用更弱的单分支条件，与要求超过三个对照的 pilot proceed=false 不冲突。
- 旧 decision.json 在失败时仍输出 next=`same-budget other seeds` 是说明文本缺陷；真正执行决定来自 proceed=false。不是 gate 数值误算。

单 seed、三轮和单一错配映射不足以排除更长训练或其他 seed 的不同结果；本轮也没有给这些扩大预算提供正向对照证据。无新增训练的机制诊断工具已交付，尚无本地服务器诊断结果。

## 18. 补充：gate 是否可能错过机会，以及创新的判断边界

### 18.1 数值与研究含义分开

静态 v2 的 C_text−C_shuffle 三 seed CW AUC 差值为 +0.800834、+1.019395、−0.683492 pp。均值 +0.378912 pp，样本标准差约0.926536 pp；距离 +0.5 pp 规则仅差0.121088 pp。现有样本不足以可靠区分真实平均效应在0.379附近还是高于0.5，更不能由规则未过推出真实效应为零。

这些数字核对无误。0.5 pp 没有来自功效分析、最小可检测效应或统计显著性阈值，它是预先设定的预算规则。把它解释为科学上的可行/不可行分界会过度推断。多条条件全部满足的 AND 规则与单 seed pilot 均可能漏掉有潜力的实现；现有实验不能估计这种漏判概率。

### 18.2 静态成功不是动态选择的必要条件

静态平均描述会损失图像/对象相关的信息；动态选择改变了模型函数和训练机制。因此，静态文本不能稳定超过错配文本，不构成动态 EKR 不可行的证明。此前把静态 gate 与所有后续 EKR 可行性绑定得过紧，应修正为：它限制原预设推进流程的预算，不能否决另立假设的机制探索。

保留旧 false 与完整负结果；新版本可以事先声明预算、主要指标、对照与分析规则。追加固定预算、完整报告的新研究是允许的；只追加到门槛通过、事后挑 seed 或隐藏旧负结果不可支持可靠结论。新训练 seed 和新错配映射也不能替代对独立数据泛化的验证。

### 18.3 目前仍值得研究的候选贡献

候选问题是“如何对齐并有选择地使用外部描述，使知识匹配时提供有用增量、不匹配时限制损害”。必须证明学习到的是知识的任务用途，而不是简单增加参数或改变分数校准。优先依据机制诊断决定可学习值对齐、局部属性适配或 TRAIN 内知识不匹配监督；这些尚是方案，不能给出成功率或可靠收益保证。

条件属性学习已有 [CANet](https://arxiv.org/abs/2305.17940)，语义变体选择已有 [EVA](https://arxiv.org/abs/2506.20986)，条件语义重排序亦有 [CLEAR](https://arxiv.org/abs/2609.15649)。本次查阅这些作者论文的摘要仅用于提醒最近邻方向；未构成穷尽的新颖性检索。单独添加注意力、对象条件、残差或 null 不足以宣称新方法，具体贡献仍需与相关方法和普通适配器对照。

当前结论：原 gate 计算正确、早先对它的否决性解释过强；当前名称锚点实现也没有显示正确描述优势。继续研究有可能形成创新，但应围绕诊断支持的机制改进，避免单靠降低 gate 或重复训练来寻找成功结果。USG 随机语义图继续作为另一条路线。
