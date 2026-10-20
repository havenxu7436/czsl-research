# 04 · 配置与超参数策略

| 组 | 修正读出 | 门控 | ρ | η | seed / 最佳epoch |
|---|---|---|---|---|---|
| [C1](round_c/C1.json) | 全局 g | 静态全 1 | 0.125 | 0 | 3351 / 8 |
| [C2](round_c/C2.json) | 全局 g | 局部输入，模式中心化 | 0.1 | 0.25 | 3351 / 8 |
| [B3补seed](round_b/B3.json) | 全局 g | 原 B 弱门控 | 0.1 | 0.25 | 3352 / 8 |

B3 配置随 C 包保留；本次 B3/3352 实际由 [03 原 B 源码](../../03/code/README.md)训练，配置字节与该源包一致。

## 选取策略

1. **固定 A5 已确认的基座。** β=0.1、CP LR=3e-4、适配器 LR=1e-3、rank32、10 轮保持一致。
2. **C1 用上一轮门控均值确定幅度。** B3 的 γ 均值约 1.241，对应 ρ≈0.1241；采用 ρ=0.125，形成静态 1.25 倍对照。
3. **C2 保持弱调制范围。** η=0.25、τγ=0.1 沿用 B3；中心化后乘 η/2，使模式均值为 1，系数位于 (0.75,1.25)。
4. **原 B3 补 seed3352。** 与 A5/3352 配对，检查原小幅组合增量；C1/C2 均先以 seed3351 与现有模型比较。
5. **按 VAL 选权重与工作点。** best 为首次严格最大 VAL CW AUC_native，TEST 固定相应 VAL bias；保存四种设置的完整曲线与预测。

| 固定项目 | 取值 |
|---|---|
| CLIP | 冻结 ViT-L/14，768 维，224px |
| β / rank / adapter rank | 0.1 / 32 / 8 |
| epochs / batch / 每轮更新 | 10 / 64 / 360 |
| 适配器 LR / CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦调度 |
| κ / τv / τγ | 20 / 0.07 / 0.1 |
| AdamW β₁、β₂ / ε / weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪阈值 / 动态正交项 | 1.0 / 0.001 |

C2 与 B3 的 Bvis 随 CP 参数组更新；C1 冻结 Bvis。Wc 初始化为零，同 seed 使用相同原初始化。原子属性与伙伴辅助分支关闭。

## 配置与资产索引

- [三次已完成运行的完整配置与指标](experiment_inventory.json)，分别记录 case、seed、最佳 epoch 与实际源码阶段。
- [C1/C2配置](round_c/)、[B组兼容配置](round_b/)、[A组兼容配置](round_a/)、[默认配置](default.json)。
- [A组三seed标量参考](references/round_a_confirm.json)、[原B组seed3351参考](references/round_b_pilot.json)、[原V0/V1参考](references/original_v1.json)。
- [标签、文本命名及数据索引](assets/)、[外部缓存与跨阶段输入](../external_inputs.json)。

C1/C2 内部 `variant` 均为 V1，实际计算由 `gate_input`、`gate_center_modes` 与 ρ 决定。三个已完成运行的超参数选择依据及配对关系见[主 README 第5节](../README.md#5-超参数选取策略与实际实验矩阵)。
