# EKR-CPF：视觉与对象条件下的知识增量选择实验设计

日期：2026-10-04。工作名称：**Evidence-conditioned Knowledge Residual CPF**。中文：条件知识增量选择。

> 推荐用途：在现有 ViT-B/16 + GloVe 的 CPF/C 知识文本基线上，验证描述差异是否能提供额外的、图像与对象相关的判别证据。
>
> 本文交付实验设计、数学定义、代码接入位置、对照矩阵和记录协议。新模块、配置字段与接口均待实现；没有启动训练，也没有修改原 CPF 或服务器环境。
>
> 本路线不构建知识图，不学习关系边概率，不采样图，不做跨图 MC 平均。USG 文件中的随机语义图是另一个研究方案。

## 0. 推荐结论与完成标准

**先从 C 的文本表示出发，加一个可以退回 C 的小型条件知识分支。不要先恢复固定 K 偏置，不要同时换成 CLIP，也不要直接实现完整 USG。**

核心问题是：同一原语的多条描述经过平均后，一部分描述差异被消掉；这些差异是否能在不同图像、不同候选对象上补充有用证据？

例如，同一个状态可能有颜色、表面纹理、局部形变等描述。平均表示保留总体语义，但当前图像未必呈现每个描述方面。新分支只决定“此次补充哪种差异”，同时保留“不补充”的选项。例子仅用于解释假设，不代表现有 GloVe 描述已经能区分这些方面。

本方案有两层完成标准：

- **性能层**：在相同知识、同一个 C 起点及匹配预算下，EKR 相对 C 与普通适配器有稳定增益。
- **机制层**：增益依赖正确描述内容，优于普通描述注意力；去掉对象条件、中心化或空选项后，相关指标按预设假设变化。否则只能确认工程效果，不能确认研究贡献。

不存在可事先保证的涨点幅度。设计中的阈值是控制投入的工程规则，不是预估结果，也不是统计显著性的替代。

## 1. 现有证据与不可混用的结果

### 1.1 当前 MVP 验证结果

| UT-Zappos / validation | A | C | D | E |
|---|---:|---:|---:|---:|
| CW AUC% | 39.470 | 44.262 | 32.464 | 37.239 |
| CW HM% | 54.841 | 57.894 | 51.795 | 53.897 |
| OW AUC% | 25.492 | 32.079 | 19.107 | 23.972 |
| OW HM% | 37.269 | 46.367 | 31.700 | 36.262 |

A 是无新知识文本/校准的基线；C 是知识文本版本；D 是固定 K 校准版本；E 是文本与校准组合版本。这里是当前 10 epoch、单 seed 的 MVP 结果。不能与历史基线文件中的 test 数字直接相减。

后处理 Dpost/Epost 已在既定网格和约束下选到 lambda=0。新方案将 `kcc_calib=False` 固定为主协议，不因新模块名称重新加入同一个固定 K。

### 1.2 当前知识状态

缓存来自重构知识，标记为 `bulk_trusted_unverified`。UT 的语义条目较少，MIT/C-GQA 的人工审核尚未由现有结构检查证明完成。

保留此标记与来源。不得把 CSV 批量改为 approved、OOV 低、矩阵形状正确解释成所有描述正确。新增论文结论只适用于该知识版本；若改知识内容，使用新 manifest 和新缓存版本。

### 1.3 当前资产路径

本机参考源：

```text
C:/Users/llenovo/Desktop/CPF_baseline/CPF_zappos_baseline/
  cpf_ut_zappos_seed3351/repro_meta/CPF_fixed_code_snapshot.tar.gz
```

服务器已有路径：

```text
ROOT=/data/xhw/czsl_repro
当前代码：$ROOT/src/KCC0601_v1
当前实验：$ROOT/experiments/kcc0601_glove_mvp_v1
已有缓存：$ROOT/knowledge/kcc0601_v1/cache_trusted_v1
CPF Python：/data/xhw/miniconda3/envs/cpf/bin/python
```

新设计建议用独立目录：

```text
$ROOT/src/EKR0601_v1
$ROOT/experiments/ekr0601_v1
$ROOT/knowledge/ekr0601_v1/description_bank_v1
```

这些新目录尚未创建。复制来源应是包含当前 C 的 KCC 代码版本，不是直接从远程 CPF 仓库重拉一份未知版本。

## 2. 创新位置与最近邻方法

以下是检索后的边界，不能把已有组件重新包装为首次提出。

| 工作 | 与本方案有关的已有内容 | EKR 必须回答的额外问题 |
|---|---|---|
| CPF | 文本描述、对象指导的视觉关注 | 平均知识之外的描述差异是否仍有增量用途？ |
| CANet | 对象和图像条件属性表示 | 条件输入本身不新；外部描述的增量证据是否有特定价值？ |
| EVA | 专家适配、语义变体对齐 | EKR 是否优于同输入的普通语义变体选择，而非仅再做注意力？ |
| ClusPro | 多个原语原型发现与更新 | 本方案不以“多个原型”作为贡献，且不在初期引入视觉聚类 |
| LGPro | 语言描述初始化并约束视觉原型 | 不以“语言指导原型”作为贡献；须体现平均分量与差异分量的作用区别 |

来源：[CPF](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_A_Conditional_Probability_Framework_for_Compositional_Zero-shot_Learning_ICCV_2025_paper.html)、[CANet](https://arxiv.org/abs/2305.17940)、[EVA](https://arxiv.org/abs/2506.20986)、[ClusPro](https://arxiv.org/abs/2502.06501)、[LGPro 作者稿](https://openreview.net/pdf/9f7909c7a01f43fca8a0f9ec0f7258e97882e802.pdf)。

拟检验的贡献应写成：**对外部描述中的平均语义与可变差异分量分开处理，并在图像/对象条件下选择差异证据，同时检验无效知识下的退回行为。**

这是候选研究定位，不是已经确认的领域空白。中心化、残差、注意力、空选项、知识扰动分别都是常见技术；是否形成足够的整体贡献，要靠最近邻检索、强基线与机制实验。

## 3. 与 USG 的严格区分

| 项目 | EKR 本方案 | 用户的 USG 方案 |
|---|---|---|
| 学习对象 | 描述差异的条件使用权重 | OI/OE/AE 候选关系的边保留概率 |
| 条件 | 当前图像与候选对象 | Lite 中主要是数据集级知识图分布 |
| 结构 | 原语描述库，无邻接矩阵 | 有类型的语义图 |
| 随机性 | 主方法确定性推理 | Bernoulli/Concrete 图采样 |
| 主要运算 | 描述中心化、条件选择、小幅分数残差 | 图消息传播、图分布学习、跨图预测平均 |
| 不确定性含义 | 可选择零补充；权重是任务用途，不是事实正确率 | 边分布表示任务相关结构不确定性；也不自动等于事实正确率 |
| 第一证据 | 超过 C 和普通描述注意力 | 超过确定图、学习均值图、EdgeDrop 等 |

训练期的随机替换描述只是一项可选污染对照/鲁棒性训练，不能因此把 EKR 叫作随机语义图。也不要在 EKR 第一版中把父类节点传播、采样边和图 KL 搭进来。

USG 的具体定义以用户提供的 [USG-CPF 随机语义图方案](C:/Users/llenovo/Desktop/USG-CPF随机语义图实验设计方案.md) 为准。

## 4. 原型数学定义

### 4.1 完全保留 C 的基础分数

设 C 原有评估分数为：

$$
s_C(x,a,o)=s_{comp}(x,a,o)+s_{attr}(x,a)+s_{obj}(x,o).
$$

这对应实际 CPF 代码的三个未缩放 cosine 分数相加。第一阶段冻结 C 的所有参数，保持其已有知识文本、视觉特征、组合器、候选集与 evaluator。

不把 C 的评分改写成新的概率乘积，也不声称本原型已经实现显式的 $p(o|x)p(a|o,x,K)$ 条件头。本方案的“条件”特指补充知识的路由依赖 $(x,o)$。

### 4.2 描述库与中心化

对属性 $a$，保存类别名称及该属性已有的全部描述向量：

$$
T_a=\{t_{a1},\ldots,t_{aR_a}\},\qquad t_{ar}\in\mathbb R^{300},\quad \|t_{ar}\|_2=1.
$$

这里 $R_a$ 包括名称向量。完全复用当前 C 构建器的 token 化、GloVe、OOV fallback 和归一化顺序；不要另换一个近似的字符串处理器。

定义：

$$
\mu_a=\frac{1}{R_a}\sum_r t_{ar},\qquad d_{ar}=t_{ar}-\mu_a.
$$

当前 C 的知识初始化若为上述单位向量平均后归一化，应有 $W_a^K=normalize(\mu_a)$。构建时与旧缓存逐元素比较；若实现存在不同聚合规则，应先定位差异，不能静默重建 C。

**中心化后不要再把每个 $d_{ar}$ 单独单位归一化**，否则会破坏 $\sum_r d_{ar}=0$，并放大接近零的差异噪声。

不需要把 $d_{ar}$ 对类别名称做正交投影作为第一版默认。那会丢弃可能有用的信息；如要研究，另立消融版本。

### 4.3 当前图像的差异证据

复用 CPF 的属性视觉向量 `attr_emb`，设：

$$
u(x)=normalize(attr\_emb(x)),\qquad e_{ar}(x)=\tfrac12 u(x)^\top d_{ar}.
$$

这等价于该描述的视觉 cosine 匹配减去该属性全部描述的平均视觉匹配，再固定除以 2：

$$
e_{ar}(x)=\tfrac12\left[cos(u,t_{ar})-\frac1{R_a}\sum_j cos(u,t_{aj})\right].
$$

因此，描述被均匀使用时，差异证据平均值为零。它是在评分空间移除同属性的平均描述贡献，不是因果识别，也不是已证实的“纯属性信息”。

固定除以 2 是让差异 values 与 R0/R3 的 raw cosine values 同属 `[-1,1]`，使相同幅度上限代表相同最坏分数扰动范围；它不是需要搜索的新超参数。

### 4.4 图像与候选对象条件路由

设候选对象的 C 知识表示为 $W_o^K$。用三个无 bias 的线性层，默认秩 $r=32$：

$$
q(x,o)=normalize\left((Q_xu(x))\odot sigmoid(Q_oW_o^K)\right),
\qquad k_{ar}=normalize(Q_kd_{ar}).
$$

真实描述的路由 logits：

$$
\ell_{ar}(x,o)=q(x,o)^\top k_{ar}/\tau_{route},\qquad \tau_{route}=0.2.
$$

加一个始终可选的空描述：

$$
e_{a0}=0,\qquad \ell_{a0}=q(x,o)^\top normalize(k_0)/\tau_{route}+b_0.
$$

$k_0$ 与 $b_0$ 共享于所有属性；不设置可自由记忆训练 pair 的 $N\times M$ 参数表。对有效描述与空选项做 softmax：

$$
\alpha_{ar}(x,o)=softmax_r(\ell_{ar}),\qquad
\Delta(x,a,o)=\sum_{r=0}^{R_a}\alpha_{ar}(x,o)e_{ar}(x).
$$

训练阶段仅使用真实 train 标签。评估对候选对象分别计算，不用 GT 对象，也不只取一个预测对象。共享路由器可以用于未见组合，但这只是结构上允许迁移，不保证迁移成功。

normalize 使用 `eps=1e-6`；投影矩阵采用固定 seed 的常规随机初始化，空 key 用小幅非零随机值初始化。只归一化路由 keys，不归一化证据 values $d_{ar}$。这样既保留差异证据的零均值性质，也避免归一化 query 后再除以 $\sqrt r$ 使初始权重过于接近均匀。路由温度固定，不参与首轮搜索。

### 4.5 最终分数、幅度与初始化

$$
s_{EKR}(x,a,o)=s_C(x,a,o)+\eta\,\Delta(x,a,o),
\qquad \eta=\eta_{max}\tanh(g).
$$

默认 $\eta_{max}=0.1$，初始化 $g=0$；允许正负增量，记录最终 $\eta$ 的符号与数值。该参数是训练的残差幅度，不是先前固定 $K_{oa}$ 的 lambda。

由于真实 $t$ 为单位向量、$\|\mu\|\le1$，有 $|e|\le1$，softmax 是凸组合，从而 $|s_{EKR}-s_C|\le0.1$。这是单个原始 cosine 分数的扰动界，**不保证准确率不下降**；即使小扰动也可能改变接近的候选排名。

初始化 $g=0$ 时输出严格退回 C。第一次更新主要推动 $g$，随后路由器获得梯度；诊断必须检查它没有长期停在零。推理还允许全局选择 `residual_off`，即 $\eta=0$。若 validation 最终选择 off，新增模块没有被采用，不能把 C 的结果算作 EKR 成功。

空选项软权重通常小于 1，并不保证对每个噪声样本完全关闭分支。**严格相等的退回仅在残差关闭、全部差异为零，或只剩空选项时成立。**

### 4.6 边界情况

- 只有名称、没有描述：$R_a=1$，所有差异为零，该属性不增加分数。
- 多条描述相同：中心化后为零，不允许归一化后凭空放大。
- padding：不参与均值，logits 必须 mask 为负无穷；空选项始终有效。
- 个别描述 OOV fallback 与名称相同：保留来源并报告，不把重复 fallback 当作多条独立证据。
- 全部描述不可用：分支退回 C；不能改名称 embedding 或把全类分数写成 NaN。
- GloVe 是词袋式表示，难以表达语序和否定。不能把路由输出解释成逻辑规则满足概率。

## 5. 训练协议

### 5.1 第一阶段：冻结 C，只训练分支

加载当前已完成的 C checkpoint。C 的全部参数 `requires_grad=False`，且保持 `eval()`，避免 BN/dropout 或模型状态改变；新路由器保持 `train()`。

通过共享特征提取路径取得 `attr_emb` 和原有 `s_C`；base 特征可以在 `no_grad()` 中取得，分支运算与 loss 在外面保留梯度。禁止把新分支放进整个 `val_forward()` 的 `torch.no_grad()` 区域。

主损失为 train seen-pair 集合上的交叉熵：

$$
L_{task}=CE\left(s_{EKR}(x,\mathcal C_{train})/\tau,y_{pair}\right).
$$

$\tau$ 读取 C 配置中的 cosine temperature，当前基线通常为 0.05，不能训练用缩放分数、评估却把增量多缩放一次。三个基础 raw cosine 分数和新增 $\eta\Delta$ 应处于相同单位，再统一用于新增联合 CE。

该联合 CE 是新分支的训练目标，**不是原 CPF 的三支损失**。因此 R0 普通适配器、R1 静态权重、R2 无对象条件、R3 普通注意力都用同一个联合 CE 与同一候选集，避免把损失变化归给知识机制。

由于 C 冻结，无需再对常数的原 CPF loss 做反向传播。保存原 C 验证输出作为 anchor，训练前后的 C 权重 checksum 应一致。

### 5.2 可选第二阶段：有限知识污染下的退回训练

先验证无污染版 R5。只有它超过 C 与普通适配器后，才增加 R6；不要一开始就同时引入多项 loss。

训练时以 20% 的 batch 概率生成描述错配视图：随机挑选 10% 的属性，将其补充描述换成其他属性的描述；名称、标签和 C 的平均知识输入保持原样，只有新增分支的描述库被替换。重新计算替换库的均值与差异。

这称为“描述归属错配”，不能保证每条替换句都事实错误。若要测试明确矛盾描述，需要人工审核的小型集合并单独报告。

在错配视图增加：

$$
L_{fallback}=KL\left(stopgrad(p_C^\tau(x))\;\|\;p_{EKR,corrupt}^\tau(x)\right),
\qquad L=L_{task,clean}+\beta L_{fallback},\quad \beta=0.05.
$$

教师与学生使用相同 train seen-pair mask、相同温度与 `batchmean` 归约。此处是分支退回正则，并非让模型模仿 C 的所有错误作为主任务。污染强度不再做大网格搜索。

可能失败方式：模型通过一直关闭分支满足正则。必须同时检查 clean validation 的增益、$\eta$、空选项使用率和残差幅度。R0/R3 加相同污染正则的版本是最终鲁棒性归因对照，否则收益可能只是普通正则化。

### 5.3 最终阶段：匹配预算微调

冻结主干的原型成功后，才进行端到端微调。相同 C 起点、相同数据顺序/增强预算、相同 epochs、相同 optimizer/scheduler，对照：

1. C-budget：C 本身继续训练。
2. R0：C + 普通条件适配器。
3. R3：C + 普通描述注意力。
4. EKR：C + 条件差异分支。

用预先定义的相同联合 CE；如保留原 CPF 的 `L_comp+0.4L_obj+0.6L_attr`，四组均加入，系数相同。只冻结阶段有增益，不能证明端到端阶段也有增益；两种训练制度分别报告。

## 6. 实际代码应修改什么

### 6.1 已核验的入口

参考特征流程是本机 baseline tar 中的 `CPF_fixed/models/ddm.py`。同时已核对现有 MVP 安装器：它生成 `models/kcc_cpf.py`，让 `train.py/test.py` 导入 `from models.kcc_cpf import DDM`，并由 `models/kcc_cache.py` 提供知识 buffers。服务器若有后续改动，以第 13 节实际 import 检查为准。

**EKR 应接在实际使用的 KCC/C 模型上，不能直接跳过 `kcc_cpf.py` 调用原 `ddm.py` 的评分。** 否则会失去 C 的知识文本，以及 KCC 已修正的验证候选顺序。下面行号只用于定位原始特征流程；当前 KCC 对应方法的行号会变化。

| 位置 | 快照参考行 | 接入用途 |
|---|---:|---|
| `DDM.__init__` | 34 起 | 注册描述库 buffers 和新路由器；保留 C 模块 |
| `train_forward` | 291 起 | 抽取共有特征，不重复运行 ViT |
| `vis_obj_new` | 336 附近 | 已有对象视觉特征；首版不必新增对象描述分支 |
| `attr_emb = self.attrc_embedder(fused_vis)` | train 388；val 516 | 复用的属性视觉向量，形状 `[B,300]` |
| `val_forward` 的三个 cosine 分支 | 519–528 | 取得基础 raw 分数 |
| `pred = comp_pred + attr_pred + obj_pred` | 533 | 在保持原顺序的基础上添加 `eta * delta` |
| `forward` 的 `torch.no_grad()` | 548–549 | 不能覆盖新增分支的训练计算 |
| `CosineClassifier` | 553–568 | train 的 temperature 缩放与 val raw cosine 单位 |

不要改成 `ddm_v3.py`，除非当前 `train.py` 的实际 import 明确使用它。

当前 C 使用 `_kcc_attr_embeddings()` / `_kcc_obj_embeddings()` 返回 `W_a_K` / `W_o_K`，不是把它们写入原名称 embedding 表。保持 `kcc_text=True, kcc_calib=False, kcc_lambda=0`，并保留原 cache manifest 验证。KCC 的验证顺序由 `val_attrs/val_objs/val_pairs` 提供，EKR 应沿用这些数组而非原快照的 `self.pairs` 顺序。

### 6.2 建议新增模块与接口

下列是**待实现接口**，不能直接当作已有脚本运行。

```text
models/ekr.py
  CenteredDescriptionRouter
  forward(attr_feature, object_knowledge, pair_indices, description_bank)
    -> delta_for_pairs, diagnostics

models/ekr_cpf.py
  在现有 models.kcc_cpf.DDM 上接入分支；保持 C 的 cache 与候选顺序

tools/build_description_bank.py
  复用当前 C 构建器；生成独立描述库并核验旧聚合结果

tools/audit_ekr.py
  baseline 相等、padding、梯度、pair 顺序、分块一致性检查

tools/eval_ekr.py
  沿用现有 evaluator；额外记录分支诊断

tools/summarize_ekr.py
  汇总分 seed 的 C/对照/EKR 指标与配对差值
```

基模型应提供不改原预测行为的接口，例如 `extract_features_and_raw_scores()`。它要共享现有实现，避免分别复制 train/val 特征路径后发生不一致。

接口输入的 pair 顺序严格由 dataset/evaluator 提供；收集结果时使用 `delta[:, pair_object, pair_attribute]`。不要依赖 JSON、CSV、目录枚举的自然排序。

### 6.3 新描述缓存

旧缓存只有聚合向量时，不足以恢复每条描述。应读取原知识 JSON/CSV，用同一个构建规则生成：

```text
description_bank_v1/<dataset>/
  labels.json                     # 原 attrs/objs 顺序
  attr_text_units.pt               # [M,R_max,300]，名称+原有描述
  attr_text_mask.pt                # [M,R_max]
  attr_text_mean.pt                # [M,300]，padding 不计入
  attr_text_residuals.pt           # [M,R_max,300]
  description_ids.json            # 每个槽对应的原始文本与来源ID
  bank_report.json                 # OOV、重复、差异范数、旧缓存比对
  manifest.json                   # hashes、构建规则、知识审核标记
```

$R_{max}$ 根据源文件实际描述数确定。若通常为名称加三条描述，则 $R_{max}=4$；不能未经核验截断更多描述，否则“相同知识输入”的前提失效。

单独记录类别名称向量、描述向量和未归一化均值，方便核验。不覆盖 `cache_trusted_v1`，不改动已有 K 文件。

### 6.4 checkpoint 兼容

- 原 C 权重应严格按原有模块加载；新增参数初始化并记录缺失 key 列表。
- 只允许预期的 `ekr.*` 新 key 缺失；不能用一个无过滤的 `strict=False` 掩盖 backbone/组合器丢权重。
- bank 可作为不持久化 buffer，从 manifest 对应路径加载；checkpoint 保存 bank hash 与 label hash。
- `residual_off` 下，在完全相同的 batch、transform、候选顺序与 dtype 下，`max_abs(score_new-score_C)<1e-6`。
- 保存增量 checkpoint 和用于训练的 C checkpoint hash；避免把分支结果写回原 C 的 `best_auc.pth`。

## 7. 计算与内存

首先计算图像证据 `e:[B,M,R]`。按对象块计算路由权重，避免创建 `[B,N,M,R,300]`。

推荐 `object_chunk=16`，rank=32。C-GQA 的 $N=674,M=413$，假设 $B=32,R=4$：

- 单对象块路由 logits `[32,16,413,5]` 约 4.0 MiB，包含空选项，按 float32 估计。
- 全部 pair 标量增量 `[32,674,413]` 约 34.0 MiB；也可按 evaluator 的 pair 列表收集，减少无用组合输出。
- 实际峰值还包括 softmax、中间值、autograd 与基础模型；上述数字不是 GPU 总显存承诺。

共有三个 `300->32` 矩阵，约 28,800 个权重，另加空 key、bias 与幅度参数。首版额外 trainable params 约 2.9 万；实际数以实现统计为准。

每张图像只运行一次基础 ViT。冻结阶段可省掉主干反向计算，但前向和数据加载仍存在，因此不能按参数比例宣称训练时间缩短多少。用前 100 个有效 batch 测量秒/步、峰值显存和每次验证耗时，再估算总预算。

## 8. 必须保留的对照矩阵

所有冻结阶段版本从同一个 C checkpoint 开始。不要沿用 A/C/D/E 字母，以免与旧实验混淆。

| ID | 版本 | 输入/机制 | 回答的问题 |
|---|---|---|---|
| C0 | 原 C checkpoint | 平均知识，无分支 | 新增增益的 anchor |
| R0 | 普通条件适配器 | 同样的图像/对象 query、key 投影与空选项；只读取 C 的聚合属性向量；小幅重评分 | 额外参数与新增联合 CE 能否解释收益？ |
| R1 | 静态描述选择 | 描述差异不变，但路由 query 是共享的学习常量 | 是否确实需要当前图像/对象的条件选择？ |
| R2 | 图像条件、无对象条件 | 使用 $q(x)$，取消候选对象输入 | 对象条件是否带来额外作用？ |
| R3 | 普通描述注意力 | 同样的 query、容量、描述库与空选项；读取 raw 描述而非差异 | 中心化差异机制是否优于普通融合？ |
| R4 | EKR 无空选项 | 去掉 null，其他一致 | 空选项对 clean/污染知识是否有用？ |
| R5 | EKR 主版本 | 中心化差异、图像/对象路由、空选项；beta=0 | 首轮主假设 |
| R6 | EKR + 退回训练 | R5 加第 5.2 节正则 | 额外鲁棒性是否有价值？ |
| R7 | 描述归属打乱后训练 | 原 C 保持；新库描述跨属性置换，训练预算与 R5 一致 | 语义内容是否优于随机 codebook/容量效应？ |

R0 的输出定义为同范围的 `eta * (1-alpha_null) * cosine(u,W_a^K)`，使用与 R5 相同的三个投影层/空 key/幅度参数。它已经拥有 C 的聚合知识，缺少的是单条描述差异。记录精确参数与 FLOPs，不把不同计算次数冒充完全相等的计算预算。

R3 同时使用 raw 描述作 key/value，是常规注意力对照。若 R5 优于 R3，再增加一个“keys 仍中心化、仅 values 改为 raw”的单因素版本，分离路由 key 的改变与证据中心化的改变。

R1/R2/R4 是功能消融，参数可能较少；它们不能替代 R0/R3 的容量对照。附加 trainable 参数全部要实际统计。

均匀描述权重的差异分支应解析上为零，可作为 sanity check；它不能作为“同容量学习机制”的性能基线。廉价 `max_r e_ar` 或固定 top-1 选择可作为额外无训练对照，排除最大匹配本身解释收益。

### 8.1 最小执行集

第一轮只做：C0 复核，以及 R0/R3/R5 三次分支训练。相同起点、seed、数据顺序与 3 epoch 初筛预算；如果 3 epoch 无趋势而训练尚未稳定，最多延长这同一组到 5 epoch，不能只延长 R5。

初筛之后有价值才跑 R1/R2/R4/R7；不要第一轮把整个矩阵乘三个数据集和三个 seed。

## 9. 数据与配置协议

### 9.1 数据使用

保留标准 train/val/test 图像划分。训练只用 seen train 图像与标签。描述可以覆盖预先公布的原语词表，但构造分支时不输入 pair 的 seen/unseen 标志、val/test 频率或 test 错误。

validation：CW 使用既有 evaluator 定义的合法验证候选；OW 使用完整属性×对象候选，不增加新的、通过 test 得到的 feasibility mask。

只根据 validation 选择训练 checkpoint 和预先限定的配置。test 留到路线与配置冻结后；不能看一次 test 后再改 router、文本、幅度上限或 loss。

### 9.2 起点与 seed

已完成的首个 UT pilot 用当前 C checkpoint，seed=3351。MIT 首轮沿用 seed=0，C-GQA 首轮沿用 seed=3351，方便承接已有计划。

正式配对验证预设 seeds `{3351, 0, 2026}`。每个数据集的每个 seed 使用对应的 C 起点，并为 R0/R3/EKR 保持同一数据顺序。若只从一个 C checkpoint 训练三次分支，应明确这是“固定 anchor 的分支随机性”，不能冒充三个独立 C 重训的稳定性。

若正式从 A 训练到 C 再到 EKR，各组 stage 的预算和 checkpoint 选择规则必须一致，单独报告训练制度。

### 9.3 首版超参数

| 参数 | 默认 | 首轮处理 |
|---|---:|---|
| backbone/text | 原 ViT-B/16 / GloVe 300d | 固定 |
| rank | 32 | 固定，不同时搜索多个 rank |
| route temperature | 0.2 | 与 CPF classifier temperature 分开，固定 |
| optimizer | AdamW | 只训练新分支 |
| new branch lr | 1e-3 | 若训练明显发散，所有对照共同改为 3e-4 并记协议修订 |
| weight decay | 1e-4 | 固定 |
| eta_max | 0.1 | 固定；不把它当 lambda 大网格搜索 |
| raw gain g | 0 | 保证初始输出等于 C |
| null bias b0 | 0 | 不预先将空权重推到接近 1 |
| batch size | 尽量等于现有 C；显存不足统一缩小 | 各对照保持一致 |
| object chunk | 16 | 计算实现参数，不参与效果选择 |
| frozen pilot epochs | 3，最多同组延长到 5 | 各训练版本匹配 |
| confirmation epochs | 10 | 优先承接 MVP 预算 |
| beta | R5=0；R6=0.05 | 两个明确版本，不做 loss 系数拼盘 |
| validation 选择 | 预先声明 CW AUC 主指标 | 使用第 10 节约束 |

这是待实现的配置含义，不能直接往现有 YACS 命令行塞入未知 `EKR.*` 字段。需要先在配置定义中注册，再生成和验证 YAML。

## 10. 评价与继续/停止规则

### 10.1 配置和 checkpoint 选择

使用 validation CW AUC 为主指标，并相对同起点 C 要求：

- CW HM 不降低超过 0.5 pp。
- OW AUC 不降低超过 0.5 pp。
- OW best-unseen 不降低超过 2 pp。

如果没有启用分支的候选满足约束，选择 residual_off，明确记录本次未采用 EKR。近似持平优先较小残差幅度/更简单版本，规则在结果出来前冻结。

该选择不保证 OW 最优；若决定把 OW AUC 改成主要目标，应另立协议，在新实验前修订，不能见到结果后切主指标。

### 10.2 投入分阶段

| 阶段 | 工作 | 继续条件 | 不满足时 |
|---|---|---|---|
| P0 | 复核 C；MIT/C-GQA 的 A/C 对照；描述差异检查 | 至少目标数据集文本收益/描述差异值得验证 | 对失败数据集查知识质量；不直接扩展复杂分支 |
| P1 | bank、特征接口、精确退回与梯度检查 | 全部实现 sanity checks 通过 | 修实现，不进入效果讨论 |
| P2 | UT 的 C0/R0/R3/R5，3–5 epoch 冻结 pilot | R5 有超过 C 与强对照的趋势，且不违反主要约束 | 若分支已收敛并无优势，停止扩展；若还未稳定，只共同延长一次 |
| P3 | 增加消融、MIT/C-GQA 首 seed 确认 | 至少两个数据集显示相同方向；语义与条件作用能被对照识别 | 缩小适用范围或改研究命题，不宣称普遍有效 |
| P4 | 三 seed 配对、匹配预算微调 | 主指标配对增益稳定，超出强对照，主要副指标可接受 | 按工程改进/负结果记录；不为追点无限调参 |
| P5 | 协议冻结后最终 test、论文图表 | 完整证据与来源记录 | 不用 test 补救模型选择 |

P2 可把 **+0.5 pp CW AUC** 作为“值得确认”的实际投入阈值，而非成功证明。P4 的目标可以预设为至少两个数据集平均 **+1 pp CW AUC**，但实际报告全部数据，包括不达标的数据；这些数字不是收益预测。

三个 seed 太少，不能仅凭均值为正就宣称统计显著。报告配对差值、标准差、每 seed 结果；若计算均值置信区间，使用小样本合适的 t 区间，明确其假设。图像 bootstrap 若用于 AUC/HM，应在每次重采样后完整重算指标，不能把候选 pair 当作独立样本。

### 10.3 什么算机制失败

| 观察 | 可支持的结论 | 不能支持的结论 |
|---|---|---|
| R0 与 R5 相同 | 普通条件适配已足够 | 知识差异机制有效 |
| R3 与 R5 相同 | 普通描述注意力解释增益 | 中心化有独立贡献 |
| R1 与 R5 相同 | 静态重加权可能足够 | 图像/对象条件是关键 |
| R2 与 R5 相同 | 图像条件已有作用，对象条件未证明 | 完整对象条件机制有效 |
| R7 与 R5 相同 | 随机语义码/容量可能解释结果 | 正确知识内容是关键 |
| residual_off 被选中 | 新分支没有被采用 | C−A 可以重新算作 EKR 增益 |
| 噪声下更稳但 clean 无增益 | 可能有鲁棒性贡献 | 识别精度已经提高 |
| 只 UT 有增益 | 数据集相关效果 | 三数据集通用方法 |

## 11. 诊断与知识质量复核

### 11.1 开训前描述诊断

每个数据集输出：

- 每类描述数、重复比例、名称 fallback 比例、OOV token 比例。
- $\|d_{ar}\|$ 的 min/median/max 和近零比例。
- `max_abs(sum_r residual)`，应在浮点误差范围内接近零。
- `normalize(mean_text)` 与旧 `W_a_K` 的最大绝对误差。
- train 内真实属性与易混淆属性的 $e_{ar}$ 分布；如果差异对视觉毫无区分，不默认加模块能补救。

先按 OOV/重复/训练支持度分层抽取至多 64 个原语的全部描述，核对名称归属、视觉相关性、是否过泛。固定随机抽样 seed，保存全部抽样结果，不能只挑看起来合理的例子。发现明显错误时生成新知识版本，并同步重建所有相关对照的知识输入。

### 11.2 训练与 validation 诊断

记录每个 epoch：

1. 原始 $g$、实际 $\eta$、router 梯度范数、空选项平均权重。
2. `$mean(abs(eta*delta))`、p95、最大值、零值比例。
3. 与 C 的 top-1 变化比例；correct→wrong 与 wrong→correct 数量。
4. 对象正确/错误子集的属性增量表现，按标签只做诊断，不把 GT 对象输入推理。
5. seen 与 validation-unseen 分组效果、原语 train 频率分组效果。
6. 同一图像、不同候选对象下的权重变化；权重几乎恒定时，承认路由退化。

注意：softmax 权重不是知识正确概率，null 权重不是校准过的“不确定性”，训练分支增量也不是因果效果。

### 11.3 污染实验

冻结训练模型后，固定扰动 seeds `{11,22,33}`，对分支 bank 做三档归属错配：0%、10%、30%，每次保持 C 输入不变。记录 R0/R3/R5/R6 的绝对指标和相对自身 clean 的下降。

这个实验回答“补充知识被污染时是否更容易退回原 C”，不代表整个知识系统都被污染。若另做 C 的知识也被污染的实验，必须重新计算 C 的输入与基线结果，独立成表。

不把描述错配、删除描述、事实矛盾混成一个 noise 指标。训练期污染与仅推理期污染分开报告。

## 12. 结果记录与比较

### 12.1 每个 run 的产物

```text
experiments/ekr0601_v1/
  protocol/
    protocol_v1.md
    environment.txt
    code_manifest.sha256
    knowledge_manifest.json
    selection_rule.json
  configs/
  runs/<dataset>/<variant>/seed_<seed>/
    effective_config.yml
    source_checkpoint.sha256
    train.log
    epoch_metrics.jsonl
    branch_checkpoint.pth
    val_cw.json
    val_ow.json
    selected_state.json
    diagnostics.json
  tables/
    runs.csv
    paired_deltas.csv
    ablations.csv
    noise_results.csv
  figures/
```

验证 JSON 使用相同 evaluator 字段。后续 test 结果放 `test_cw.json/test_ow.json`，禁止写入 val 文件或反向影响 `selected_state.json`。

### 12.2 `runs.csv` 字段

```csv
run_id,dataset,variant,seed,anchor_seed,phase,setting,training_regime,epochs,selected_epoch,base_checkpoint_hash,code_hash,knowledge_hash,bank_hash,label_hash,protocol_version,auc_pct,hm_pct,best_seen_pct,best_unseen_pct,hm_seen_pct,hm_unseen_pct,eta,mean_null_weight,mean_abs_residual,new_trainable_params,gpu_peak_mib,train_seconds,eval_seconds,status
```

`phase` 取 `val/test`，`setting` 取 `cw/ow`。同一 run 的 CW/OW 各一行。字段 `_pct` 固定百分数；evaluator 若输出 `[0,1]`，只在转换层乘 100 一次。

best-seen/best-unseen 是曲线各自最优值，不一定来自 HM 最优点；因此同时记录 `hm_seen_pct/hm_unseen_pct`，不要拿不同 operating points 的两个最大值自行重算 HM。

### 12.3 配对比较

`paired_deltas.csv` 每行由相同 dataset、seed、anchor、phase、setting、training_regime 配对。记录：

```csv
dataset,seed,anchor_seed,phase,setting,training_regime,comparison,delta_auc_pp,delta_hm_pp,delta_best_unseen_pp
```

重点比较：`R5-C0`、`R5-R0`、`R5-R3`；最终微调比较 `EKR-C_budget`、`EKR-R0`、`EKR-R3`。`C-A` 是已有知识输入效果，另列，不计入新增机制贡献。

主表按数据集分别报告平均值±标准差与逐 seed 配对差值。不能把三个数据集的不同尺度 AUC 简单平均后当作唯一结论。

### 12.4 推荐图表

1. 每数据集的 C/R0/R3/EKR CW、OW AUC 与 HM，含 seed 散点。
2. 同 seed 的 EKR−C、EKR−R3 配对差值图。
3. 描述错配比例与指标下降曲线，clean 绝对值同时给出。
4. 对象/属性混淆案例：原图、C 排名、EKR 排名、描述差异、null 权重；同时展示成功与失败，不只挑成功。

## 13. 当前即可运行的服务器只读检查

下面只检查已存在的代码与环境，不依赖尚未实现的 EKR 脚本。完整块在服务器 Bash 中执行；不要粘到本机 PowerShell。

```bash
(
  set -eu
  ROOT=/data/xhw/czsl_repro
  PROJ="$ROOT/src/KCC0601_v1"
  CPF_PY=/data/xhw/miniconda3/envs/cpf/bin/python
  test -x "$CPF_PY"
  test -f "$PROJ/models/ddm.py"
  test -f "$PROJ/models/kcc_cpf.py"
  test -f "$PROJ/models/kcc_cache.py"
  cd "$PROJ"

  "$CPF_PY" -c 'import sys,torch,torchvision; print(sys.executable); print(torch.__version__, torchvision.__version__); print("CUDA",torch.cuda.is_available())'

  if command -v rg >/dev/null 2>&1; then
    rg -n 'from models|def train_forward|def val_forward|attr_emb =|pred =|no_grad|kcc_|val_attrs|val_objs' train.py models/ddm.py models/kcc_cpf.py models/kcc_cache.py
  else
    grep -nE 'from models|def train_forward|def val_forward|attr_emb =|pred =|no_grad|kcc_|val_attrs|val_objs' train.py models/ddm.py models/kcc_cpf.py models/kcc_cache.py
  fi

  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
)
```

这个检查不等于选择了可用 GPU，也不证明环境已恢复；以真实返回值为准。无需为只读检查启动 tmux。后续实现完成的长训练应放 tmux，并通过 run ID 独立保存日志。

**本文没有提供伪装成可执行的 `train_ekr.py` 启动命令。** 在第 6 节接口、YAML 注册、P1 sanity checks 完成后，再生成对应代码版本的命令流程；不能假设当前 KCC 已支持新增参数。

## 14. 实现后的验收清单

这些检查直接影响结论是否可信，需在效果实验前完成。

- [ ] bank 与当前 C 的文本聚合结果一致，标签顺序一致，来源可追溯。
- [ ] 初始化/关闭分支时逐元素复现 C，而不是只比较最终 AUC 近似相同。
- [ ] padding 不计均值，空选项有效，描述缺失不会产生 NaN。
- [ ] 均匀有效描述权重的差异输出接近零。
- [ ] 图像/对象 chunk 与不分块的小型参考实现得到相同分数。
- [ ] train/val 的增量分数在相同 raw cosine 单位下工作。
- [ ] 冻结阶段 C 参数与 buffer 没被改；新参数有梯度并产生更新。
- [ ] 未使用 GT 对象推理，未设置训练 pair 专属参数表。
- [ ] R0/R3 使用相同新增 loss、预算和起点，参数量实测。
- [ ] validation/test、CW/OW、fraction/percent 字段明确分离。
- [ ] 所有训练和打乱版本都有独立配置、hash、run ID 与日志。

## 15. 论文应如何表述

如果验证成功，可以围绕“平均外部知识之外的条件差异证据”和“无效补充知识下的退回能力”组织论文，而不是说又添加了文本模块。

可以成立的候选结果叙述是：同一知识库和 backbone 下，差异分支优于聚合知识 C 与普通语义注意力；正确描述、对象条件与中心化具有可辨识的作用；在描述归属扰动下更稳，且推理成本可量化。

还不能提前成立的说法包括：“首次条件知识学习”“知识权重等于事实正确概率”“证明因果解耦”“保证不降低精度”“已优于最新 SOTA”。SOTA 比较必须核对 backbone、训练数据、知识来源与评估协议，不能把 ViT-B/GloVe 与更强 CLIP 的原文结果混成公平同配置实验。

如果最终仅 R0 有益，报告普通适配的结果；如果只 R6 噪声鲁棒性有益，聚焦受扰知识的稳定性；如果 residual_off 最好，停止该路线并保留负结果。**模块复杂度不是创新证据；新增机制超过强对照才是继续投入的依据。**

