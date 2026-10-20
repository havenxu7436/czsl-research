# 06 · 配置与超参数策略

| 方案 | 读出 | 门控 | ρ | η | 本阶段新增 seed／native最佳epoch |
|---|---|---|---|---|---|
| [E1](round_e/E1.json) | 全局 g | 物体查询输入、模式中心化 | 0.15 | 0.25 | 3351/8、3352/8、3353/6 |

E1 与 [D1](round_d/D1.json) 的完整配置仅 `rho` 与 `experiment_id` 不同。三次训练各 10 epochs、3600 次更新。两种选权重均选中 8／8／6 轮，实际全字段配置及两套指标见[experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. 将 D1 的修正强度 ρ=0.125 提高至相邻的 0.15，结合已有 C2=0.1 形成三点比较。
2. 保持 A5／D1 基座：β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32、10 轮；η=0.25、τγ=0.1 保持原设置。
3. 使用 seed3351／3352／3353，直接与 A5／C2／D1 同 seed 配对。物理 GPU1／2／3 各执行一个 seed。
4. 主结果按首次严格最大 VAL CW native AUC 选权重；新增 exact 峰值并行保存与评价。三档 ρ 表统一采用 native 选择。
5. 对两套选择各做 TRAIN 全集均值替换，各条件用自身 VAL native bias。每个 seed 形成两项评估，共六项。
6. 完成本轮后按原 VAL CW 均值规则保留 E1 作为下一轮基座。下一项建议仅将 β=0.1 改为 0.05，直接测量支持集项；F1 尚属后续计划。

| 固定项目 | 设置 |
|---|---|
| CLIP | 冻结 OpenAI ViT-L/14，224px，768 维 |
| ρ／η／τγ | 0.15 / 0.25 / 0.1 |
| β／rank／adapter rank | 0.1 / 32 / 8 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器 LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| κ／τv | 20 / 0.07 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项 | 1 / 0.001 |
| eval batch／workers | 64 / 4 |

CP 参数组为 Bvis、Au、Av、Wc，学习率比例 `cp_lr_scale=0.3`；Wc 从零初始化。内部 `variant=V1`，通过 `gate_input=local`、`gate_center_modes=true` 启用动态中心化。原子属性和伙伴辅助分支关闭，配置保留的对应兼容字段不产生额外训练项。

## 配置、资产与评估索引

- [E1实际配置](round_e/E1.json)、[D1对照配置](round_d/D1.json)、[C兼容配置](round_c/)、[A/B兼容配置](round_a/)、[默认配置](default.json)。
- [三次训练清单](experiment_inventory.json)、[六项均值评估清单](intervention_inventory.json)。
- [E前14份参考指标](references/pre_e_complete.json)、[A组三seed参考](references/round_a_confirm.json)、[C组首seed参考](references/round_c_pilot.json)。
- [标签和命名资产](assets/)、[外部输入及跨阶段引用](../external_inputs.json)。特征缓存 manifest SHA256 为 `bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

本目录保留本轮实际训练配置；后续 β 调整的系数推导见[数据分析](../analysis/README.md)。
