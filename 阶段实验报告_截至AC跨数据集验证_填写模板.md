# CPF 基线复现与知识文本/兼容先验的阶段实验报告

报告日期：____。截止时间：____。作者：____。代码版本：____。知识版本：____。

> 当前模板中 UT MVP 数字来自此前截图和记录，保留为待原始 JSON 核验的转录值。MIT/C-GQA 本轮结果未读取，留空。自动汇总器会从真实文件填表；正文结论应在核验后完成。

## 1. 阶段摘要

本阶段以 ViT-B/16 + GloVe 的 CPF 为基础，比较知识文本输入与固定兼容先验。已完成/待完成项目：____。

UT-Zappos 单 seed 的初步 validation 结果显示，文本版本 C 相对 A 提升 CW/OW AUC 约4.791/6.587 pp；当前固定K、lambda=0.1的D/E下降。对固定A/C checkpoint的既定lambda网格，采用规则选择0。跨MIT/C-GQA的新A/C结果为：____。

以上属于单 seed 的探索/工程验证。当前尚未证明跨 seed稳定性、条件知识新机制、随机语义图或统计显著性。

## 2. 研究问题与变体

| 变体 | 新知识文本 | 固定K校准 | 作用 |
|---|---|---|---|
| A | 否 | 否 | 同预算基线 |
| C | 是 | 否 | 文本路径 |
| D | 否 | 是，lambda=0.1 | 当前固定K路径 |
| E | 是 | 是，lambda=0.1 | 两路径组合 |
| Dpost | A checkpoint | 仅推理期网格 | 校准后处理诊断 |
| Epost | C checkpoint | 仅推理期网格 | 文本存在时的后处理诊断 |

当前实现不包含原0601所有变体，也不包含EKR/USG新方案的训练结果。

## 3. 资产与评价协议

| 项目 | 当前协议 / 待补充 |
|---|---|
| 数据集 | UT-Zappos50K、MIT-States、C-GQA |
| split | compositional-split-natural，hash见附录 |
| seeds | UT=3351；MIT=0；C-GQA=3351 |
| 视觉 / 文本 | 原CPF ViT-B/16 / GloVe 300d |
| 当前MVP预算 | 每个run 10 epoch；各数据集A/C配对 |
| backbone训练范围、lr、batch、optimizer | 从effective/requested config填入：____ |
| checkpoint选择 | 当前run按validation AUC选best_auc；最佳epoch逐run报告 |
| CW/OW | 同一最佳checkpoint，按既有evaluator分别评价 |
| 主结果phase | 当前MVP为val；历史baseline test独立附表 |
| score_mode | legacy；保留原组合/属性/对象评分协议 |
| test选型 | 本轮不使用test结果选择lambda/模型 |

应补充各数据集实际train/val图像数、原语数、CW/OW候选数及文件来源。当前val图像/候选数可从metrics读取；train图像数从原配置/日志或数据元信息记录，缺失时不猜测。

## 4. 历史CPF复现结果：单独的test参考表

以下来自本机已归档的三份 `repro_meta/final_results.txt`；不用于与本轮validation计算增益。

| 数据集 | seed | 设置 | AUC% | HM% | Seen% | Unseen% |
|---|---:|---|---:|---:|---:|---:|
| UT-Zappos | 3351 | test CW | 36.51 | 52.59 | 61.49 | 68.48 |
| UT-Zappos | 3351 | test OW | 27.07 | 43.87 | 61.49 | 51.30 |
| MIT-States | 0 | test CW | 11.05 | 26.57 | 39.66 | 35.81 |
| MIT-States | 0 | test OW | 4.18 | 14.67 | 39.66 | 14.38 |
| C-GQA | 3351 | test CW | 6.81 | 21.30 | 38.42 | 20.37 |
| C-GQA | 3351 | test OW | 1.58 | 7.27 | 38.42 | 4.98 |

复现差距与协议说明：____。如引用归档文件中的论文参考数字，应另核对原论文表格与配置，不能直接当作同协议SOTA比较。

## 5. 知识输入与质量

知识来源：重构OI/OE/AE材料；当前状态 `bulk_trusted_unverified`。C实际消费聚合对象/属性知识向量；当前MVP没有通过随机图或完整逻辑损失使用所有关系。

### 5.1 源文件组织与数据格式

```text
<实际训练知识源目录>/
├── ut-zap50k/
├── mit-states/
└── cgqa/
    ├── oi_dict.json                    # parent -> [object, ...]
    ├── oe_dict.json                    # object -> [description1,2,3]
    ├── ae_dict.json                    # attribute -> [description1,2,3]
    ├── attributes.txt / objects.txt    # 固定类别顺序
    ├── entry_provenance.csv            # 原语来源与词义选择
    ├── review_status.csv               # 每原语一行的审核/信任状态
    ├── manifest.json                   # 来源、数量与 hash
    └── K_oa_text_lexical.csv / K_oa_text_report.json
```

三个数据集子目录同构。实际训练源目录：____；由 cache `input_sha256` 追溯，来源核验状态见 `knowledge_structure.csv`。run 中 `ut-zappos` 对应知识目录 `ut-zap50k`。

真实 UT draft 片段，说明 schema（不代表服务器训练版本已人工审核）：

```json
{
  "OI示例": {"boot": ["Boots.Ankle", "Boots.Knee.High", "Boots.Mid-Calf"]},
  "OE示例": {"Boots.Ankle": ["shaft ending at the ankle", "closed toe boot silhouette", "low boot collar"]},
  "AE示例": {"Canvas": ["woven canvas fabric", "coarse textile surface", "matte cloth finish"]}
}
```

OI/OE/AE是三个独立JSON文件；上面将片段并列展示。OI每个对象恰好归属一个父概念；OE/AE每个原语各三条描述，键及顺序与标签文件、cache、dataset保持一致。

| 数据集 | M属性 | N对象 | OI父概念 | OI成员 | OE描述 | AE描述 | provenance/review行数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| UT | 16 | 12 | 4 | 12 | 36 | 48 | 28 |
| MIT | 115 | 245 | 19 | 245 | 735 | 345 | 360 |
| C-GQA | 413 | 674 | 23 | 674 | 2022 | 1239 | 1087 |

以上数量来自本机实际draft，最终与服务器冻结源文件再核对。描述条数不是独立图节点数。

### 5.2 来源、审核字段与版本

UT来源为手工重构；MIT/C-GQA源manifest记录WordNet 3.0首义自动草稿，非LOGICZSL官方知识。实际训练来源与任何修订说明：____。

`entry_provenance.csv` 字段：`kind,label,query,synset,sense_count,used_head_fallback,parent,quality,review_reason`。`review_status.csv` 字段：`kind,label,status,reviewer,date,note`。每种原语各一行；approved只表示当前操作状态，不自动证明逐条人工核验。

本机原始draft有28/360/1087行pending；服务器bulk-trusted副本与本机draft分别记录。训练源文件hash、cache hash、approved/pending计数、审核记录可信程度：____。没有冻结输入hash的补充provenance记录注明“当前补充材料”。

### 5.3 缓存结构与模型消费

```text
cache_trusted_v1/<dataset>/
  labels.json                          # attrs/objs 的固定顺序
  W_o_name_raw.pt / W_a_name_raw.pt     # baseline 名称向量
  W_o_name_unit.pt / W_a_name_unit.pt
  W_o_K.pt [N,300] / W_a_K.pt [M,300]   # 当前C/E的原语知识输入
  W_ao_K/obj_00000.pt ... [M,300]       # N块，当前MVP不直接消费
  K_oa_text_raw.pt / K_oa_text_template.pt
  K_oa_text.pt [N,M]                   # 实际 prior_source
  K_oa_prior_centered.pt [N,M]          # D/E及后处理的校准项
  report.json / knowledge_coverage_report.json
  manifest.json / cache_manifest.json
```

文本配方：沿用baseline名称GloVe初始化；描述按regex分词并平均词表内向量，逐句L2归一化；名称单位向量与三条描述单位向量平均后再归一化。记录OOV和无可用词时的名称回退。缓存保持GloVe 300维。

K配方：所有N×M pair使用相同三个template，与知识原语向量和做cosine，全局min-max；退化常量设0.5；`P=(K−mean)/population_std`，std<1e-8时P=0。具体template、`prior_source`、builder hash：____。lexical CSV是不同候选，不将其统计混入template K。

矩阵轴始终为`[object,attribute]`，取值为`P[obj,attr]`。A沿用名称向量，C读取W_o_K/W_a_K，D使用校准，E二者都有。当前OI尚未用于图传播，也没有随机语义图训练，pair block亦未直接输入当前MVP。具体字段和完整目录树见[知识结构说明](C:/Users/llenovo/Documents/ChatGPT/CZSL/cpf_stage_report_tools_20261004_v2/知识文件结构与模型使用说明.md)。

报告汇总器记录的是manifest形状合同，不重新用torch核验张量；cache verify与smoke记录：____。

### 5.4 知识质量统计

| 数据集 | attrs | objs | 描述OOV | K形状[object,attribute] | K mean/std | 来源hash |
|---|---:|---:|---|---|---|---|
| UT | 16 | 12 | ____ | [12,16] | ____ | ____ |
| MIT | 115 | 245 | ____ | [245,115] | ____ | ____ |
| C-GQA | 413 | 674 | ____ | [674,413] | ____ | ____ |

构建规则、fallback、归一化、K模板、常量检查及top/bottom审查记录：____。

语义审核完成程度：____。明确区分结构检查、批量信任和人工审核。若仅bulk trust，不写成“全部知识已人工核验”。

## 6. UT A/C/D/E消融

下列为此前记录的validation转录值；提交前用 `all_val_metrics.csv` 与原始JSON替换/核对。

| 设置 | 变体 | AUC% | HM% | best-seen% | best-unseen% |
|---|---|---:|---:|---:|---:|
| CW | A | 39.470 | 54.841 | 67.959 | 66.410 |
| CW | C | 44.262 | 57.894 | 68.871 | 71.502 |
| CW | D | 32.464 | 51.795 | 65.564 | 56.161 |
| CW | E | 37.239 | 53.897 | 66.135 | 64.741 |
| OW | A | 25.492 | 37.269 | 67.959 | 53.487 |
| OW | C | 32.079 | 46.367 | 68.871 | 59.435 |
| OW | D | 19.107 | 31.700 | 65.564 | 45.272 |
| OW | E | 23.972 | 36.262 | 66.135 | 53.830 |

主要差值（使用未四舍五入的数据）：

| 效果 | CW ΔAUC pp | CW ΔHM pp | OW ΔAUC pp | OW ΔHM pp |
|---|---:|---:|---:|---:|
| C−A | +4.791 | +3.053 | +6.587 | +9.097 |
| D−A | -7.006 | -3.046 | -6.385 | -5.569 |
| E−C | -7.022 | -3.997 | -8.107 | -10.105 |
| E−D | +4.775 | +2.101 | +4.865 | +4.562 |
| E−A | -2.231 | -0.944 | -1.520 | -1.008 |
| E−C−D+A | -0.016 | -0.952 | -1.722 | -4.536 |

分析：____。区分文本效果、当前校准效果、组合总效果。交互项仅为当前模型指标的描述性差值，不宣称已识别因果交互。

## 7. 固定checkpoint的lambda诊断

网格：`{0,0.025,0.05,0.1,0.2}`。主指标：val CW AUC。保护约束与tie-break沿用已冻结规则。

| 分支 | anchor | 选择lambda | 未采用正lambda的主要原因 |
|---|---|---:|---|
| Dpost | A | 0（待原文件核验） | 从lambda_analysis逐候选填写：____ |
| Epost | C | 0（待原文件核验） | 从lambda_analysis逐候选填写：____ |

附完整十行候选表，包括CW/OW AUC/HM、OW unseen、各项delta、eligible与失败约束。

lambda=0复现检查：____。本次结果支持“当前网格与采用规则下未选中正强度校准”，不等于所有非零候选的所有指标都更差。

## 8. 三数据集A/C跨数据集主结果

| 数据集 | seed | 设置 | A AUC% | C AUC% | ΔAUC pp | A HM% | C HM% | ΔHM pp | 配对核验 |
|---|---:|---|---|---|---|---|---|---|---|
| UT | 3351 | val CW | 39.470 | 44.262 | +4.791 | 54.841 | 57.894 | +3.053 | 待原文件核验 |
| UT | 3351 | val OW | 25.492 | 32.079 | +6.587 | 37.269 | 46.367 | +9.097 | 待原文件核验 |
| MIT | 0 | val CW | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| MIT | 0 | val OW | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| C-GQA | 3351 | val CW | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |
| C-GQA | 3351 | val OW | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 | 待完成 |

附best-seen/best-unseen及HM点S/U完整表。没有原始文件时不把待完成写成0。

分析逐项填：CW是否正向？OW是否正向？HM与AUC是否一致？seen/unseen是否存在代价？失败设置是什么？____。

## 9. 过程与错误分析

### 9.1 学习过程

各run最佳epoch、是否预算末端、val曲线趋势、loss趋势：____。绘图源来自rounded_train_log，正式结果用JSON。

### 9.2 Bias=0的逐图配对

| 数据集/设置 | n | 两者正确 | A错C对 | A对C错 | 两者错误 | 净修正 | 固定bias0组合准确率变化 |
|---|---:|---:|---:|---:|---:|---:|---|
| ____ | ____ | ____ | ____ | ____ | ____ | ____ | ____ |

分别报告all/seen/val-unseen，原语准确率和每属性样本量。此表不对应best-HM bias。

成功案例与失败案例：各选同样数量，附image_id和预测，不只展示帮助最大的样本。机制解释写为观察或假设：____。

## 10. 异常、限制与下一步

发生过的环境导入异常、任务中断、恢复记录、受影响run：____。失败或不完整任务不计作正式完成；维护前后源代码、环境和知识版本是否一致应核对。

局限至少包括：各数据集单seed；validation同时用于checkpoint/后处理选型；重构知识未必完成语义审核；仅部分模型变体；当前尚无新路线结果。

结论分三栏：

| 已观察到的结果 | 尚未证明的解释 | 下一步验证 |
|---|---|---|
| ____ | ____ | ____ |

下一步决定：____。决策依据必须引用本报告中的表/图，不用test指标挑方案。

## 附录：可复现证据

报告目录：____。提交版文件：____。原始run路径：____。8个run的config、checkpoint与source/data/knowledge hash：见____。

附：完整主指标CSV、ACDE差值CSV、lambda候选与选择JSON、smoke记录、配置与实际命令、环境记录、收敛曲线、错误案例与归档校验清单。

