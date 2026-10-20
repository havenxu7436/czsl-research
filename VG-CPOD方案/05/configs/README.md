# 05 · 配置与超参数策略

| 方案 | 读出 | 门控 | ρ | η | 本阶段新增 seed／最佳epoch |
|---|---|---|---|---|---|
| [D1](round_d/D1.json) | 全局 g | 物体查询输入，模式中心化 | 0.125 | 0.25 | 3351/10、3352/8、3353/9 |
| [C2补充](round_c/C2.json) | 全局 g | 物体查询输入，模式中心化 | 0.1 | 0.25 | 3352/8、3353/6 |
| [C1补充](round_c/C1.json) | 全局 g | 静态全 1 | 0.125 | 0 | 3352/8 |

C1/C2 的 seed3351 保存在 [04](../../04/results/README.md)。D1 与 C2 的完整配置只在 `rho` 和 `experiment_id` 不同。六次新增训练的全部字段、指标、运行路径与实际源代码阶段见[experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. 组合 04 已完成的 C1 幅度设置与 C2 中心化机制，D1 取 ρ=0.125。
2. 固定 A5 基座：β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32、10 轮。
3. D1、C2 与 A5/A6 使用同 seed 3351/3352/3353；C1 额外补 3352，形成第二组完整四格。
4. 均值替换对六个已选 checkpoint 分别统计 TRAIN 全集，并在各条件自己的 VAL 上选工作点。
5. best 继续采用首次严格最大 VAL CW native AUC，保存 best/last；双 native/exact 保存为后续建议。

| 固定项目 | 设置 |
|---|---|
| CLIP | 冻结 ViT-L/14，224px，768 维 |
| β／rank／adapter rank | 0.1 / 32 / 8 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器 LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| κ／τv／τγ | 20 / 0.07 / 0.1 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |

D1 的 Bvis、Au、Av、Wc 同属 CP 参数组；Wc 从零初始化。原子属性和伙伴辅助分支关闭。内部 `variant=V1`，实际动态行为由显式配置启用。

## 配置、资产与评估索引

- [D1配置](round_d/D1.json)、[C配置](round_c/)、[A/B兼容配置](round_a/)、[默认配置](default.json)。
- [六次新增训练清单](experiment_inventory.json)、[六个checkpoint干预清单](intervention_inventory.json)。
- [A组三seed参考](references/round_a_confirm.json)、[B组首seed参考](references/round_b_pilot.json)、[C组首seed参考](references/round_c_pilot.json)。
- [标签和命名资产](assets/)、[外部输入及跨阶段引用](../external_inputs.json)。

下一轮计划 E1 取 ρ=0.15，其他参数保持 D1，直接与现成三 seed 的 ρ=0.1、0.125 配对。本目录保存本轮实际训练配置。
