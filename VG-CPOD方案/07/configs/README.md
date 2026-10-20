# 07 · 配置与超参数策略

| 方案 | 读出 | 门控 | ρ | β | η | 新增seed／native最佳epoch |
|---|---|---|---|---|---|---|
| [F1](round_f/F1.json) | 全局 g | 物体查询输入、模式中心化 | 0.15 | **0.05** | 0.25 | 3351/8、3352/8、3353/6 |

F1 与 [E1](round_e/E1.json) 的完整配置仅 `task_aux_beta` 与 `experiment_id` 不同。三次训练各 10 epochs、3600 次更新；native／exact 均选中 8／8／6 轮。实际全字段与两套指标见[experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. E1 按既定 VAL CW 三 seed 均值规则作为基座，固定 ρ=0.15；以 β=0.05 测量支持集权重下调的效果。
2. 精确保持 pair 项系数 1.5；support 系数从 3/22 降到 1/14，为原来的 52.38%。
3. 固定模型结构、η=0.25、τγ=0.1、rank32、CP分组学习率与10轮日程，直接与 E1 同 seed 配对。
4. seed3351／3352／3353 分配到物理 GPU1／2／3，每个 seed 从对应原始初始化训练。
5. 主结果按首次严格最大 VAL CW native AUC 选权重；exact 权重单独保存。每种选择完成一次 TRAIN 全集均值替换，条件分别用各自 VAL bias。
6. 对 F1−E1 的 AUC、HM、条件属性、支持集质量联合分析，后续建议 β=0.075 的 G1 与同配置静态 G0；本阶段实际配置为 F1、β=0.05。

```math
\mathcal{L}_{\mathrm{task}}^{\mathrm{F1}}
=\frac32\mathcal{L}_{\mathrm{pair}}+\frac1{14}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结 OpenAI ViT-L/14，224px，768维 |
| ρ／η／τγ／β | 0.15 / 0.25 / 0.1 / 0.05 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 梯度裁剪／正交项／workers | 1 / 0.001 / 4 |

CP 参数组为 Bvis、Au、Av、Wc，`cp_lr_scale=0.3`，Wc 从零初始化。内部 `variant=V1`；通过 `gate_input=local`、`gate_center_modes=true` 启用视觉模式。prim 与 cent 分支关闭，兼容字段保留于原配置。

## 配置、资产与对照索引

- [F1实际配置](round_f/F1.json)、[E1直接对照](round_e/E1.json)、[D1配置](round_d/D1.json)、[C配置](round_c/)、[A配置](round_a/)、[B配置](round_b/)、[默认配置](default.json)。
- [三项训练清单](experiment_inventory.json)、[六项均值评估清单](intervention_inventory.json)。
- [F前17份参考指标](references/pre_f_complete.json)、[E前参考](references/pre_e_complete.json)、[A组三seed参考](references/round_a_confirm.json)、[标签资产](assets/)。
- [外部输入与已有阶段](../external_inputs.json)。缓存 manifest SHA256 为 `bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

后续 β=0.075 的系数与对照策略见[数据分析](../analysis/README.md)。
