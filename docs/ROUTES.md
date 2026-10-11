# 两条 0601 路线与共同研究前史

本页截至 2026-10-11。两条路线共享 CPF、知识来源和前期研究背景；实现版本与正式结果分别登记。

| 对照项 | residual-0601 | formula-aligned-0601 r2 |
|---|---|---|
| 主实现 | `reproduce_0601_20261010/`；服务器实际源码另保存在结果包 `extracted/source/` | `faithful_0601_20261011/`；完整交付包在 `archive_inputs/formula_aligned_bundle/` |
| G/H 属性条件分数 | 保留 native CPF 分数并加知识残差 | 直接使用 factorized head 原始输出作为条件分数 |
| 联合评分/属性 CE | 以旧版本实际源码及公式审计为准 | train/infer 共用 `z_obj + conditional + lambda * prior`；CE 读取 GT-object 条件分数 |
| 逻辑 grounding | 旧 adapter 接口 | object softmax、attribute object-axis logsumexp 后 softmax、pair/description sigmoid |
| 已有研究结果 | UT、seed3351、12组×10轮，48份 VAL/TEST CW/OW评价 | 当前归档仅有源码与 CPU/交付核验；没有该版本正式训练结果 |
| 计划预算 | 原已执行 10 轮协议；另有只追加三项描述删除的扩展包 | 3 seeds×14组×30轮，共42次新训练计划；计划不计已完成 |
| 仍需说明的接口 | native CPF + residual，不声称逐公式复刻 | `ddm.py` 视觉底座、共享向量广播、GloVe300、CLS 投影等均属已声明接口选择；不是 CLIP 描述编码器复现 |

旧版本用 VAL CW AUC 选模，之后以同一 checkpoint 评价 VAL/TEST CW/OW。负结果原样保留。主公式对齐版修正的是实现与原设计的对应关系；它的性能只有实际训练后才能填写。

先读 [旧 residual 结果分析](residual_review_20261011.md)和 [逐公式审计](../output/0601逐公式复刻审计_20261011.md)，再读 [对齐版 README](../archive_inputs/formula_aligned_bundle/README.md)、[公式与代码索引](../archive_inputs/formula_aligned_bundle/docs/公式与代码对应索引.md)、[依赖与未覆盖项目](../archive_inputs/formula_aligned_bundle/docs/依赖与未覆盖项目.md)。

后续以 `residual-0601`、`formula-aligned-0601` 标记结果目录、记录和指标行。若实现、输入资产或预算改变，创建新的 protocol/run，不将历史成绩重新贴到新公式名下。
