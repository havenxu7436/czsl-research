# 08 · 配置与超参数策略

| 方案 | gate_input | ρ | β | η | native／exact最佳epoch |
|---|---|---|---|---|---|
| [G0静态](round_g/G0.json) | none | 0.15 | 0.075 | 配置0.25，静态分支不使用 | 8/8/9；8/8/6 |
| [G1动态](round_g/G1.json) | local | 0.15 | 0.075 | 0.25 | 8/8/9；8/8/6 |

每列轮次按seed3351/3352/3353顺序。六次训练各十轮、3600次更新；两组逐seed轮次对齐。完整配置与两套指标见[experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. **β取中间值。** 在已有E1 β=0.1与F1 β=0.05、相同ρ=0.15下取β=0.075，测量属性泛化与组合排序折中。
2. **精确解释系数。** pair系数1.5不变，support系数为9/86≈0.104651163。
3. **加入同配置静态训练。** G0/G1同β、ρ、rank、读出、初始化规则与优化日程；G0冻结Bvis且orth为0，G1使用视觉门控与对应正交项。
4. **固定剩余超参数。** η=0.25、τγ=0.1、CP分组LR、batch64和十轮日程沿用F；三GPU各负责一个seed的两组队列。
5. **沿用选权重规则。** native主结果取首次严格最大VAL CW native AUC；exact路径独立保存。各条件使用各自VAL native bias并固定到TEST。
6. **同seed配对。** 先读G1−E1/F1，再读G1−G0，随后读G1 dynamic−mean及分组排序诊断。

```math
\mathcal{L}_{\mathrm{task}}(0.075)=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结OpenAI ViT-L/14，224px、768维 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.1 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 梯度裁剪／动态正交项／workers | 1 / 0.001 / 4 |

内部variant=V1；由gate_input决定是否启用视觉门控，中心化模式保留。CP组G0为Au/Av/Wc，G1增加Bvis；cp_lr_scale=0.3，Wc零初始化，prim/cent分支关闭。

## 配置、资产与对照索引

- [G0/G1实际配置](round_g/)、[F1](round_f/F1.json)、[E1](round_e/E1.json)、[D1](round_d/D1.json)、[默认配置](default.json)。
- [六项训练清单](experiment_inventory.json)、[六项均值评估清单](intervention_inventory.json)。
- [G前20份参考指标](references/pre_g_complete.json)、[标签资产](assets/)、[已有阶段与外部输入](../external_inputs.json)。
- 缓存manifest SHA256为 `bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

后续H1门控输入表示设计见[数据分析](../analysis/README.md)，本阶段实际配置为G0/G1。
