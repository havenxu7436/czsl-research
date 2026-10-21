# 09 · 配置与超参数策略

| 方案 | gate_input | ε | ρ | β | η／τγ | native／exact最佳epoch |
|---|---|---|---|---|---|---|
| [H1](round_h/H1.json) | train_residual | 1e-12 | 0.15 | 0.075 | 0.25／0.1 | 5/8/6；5/8/6 |

轮次按 seed3351/3352/3353 顺序。三次训练各十轮、3600 次更新。完整实际配置与两套指标见 [experiment_inventory.json](experiment_inventory.json)。

## 选取策略

1. **保持已有目标权重。** β=0.075，pair 系数为 3/2、support 系数为 9/86；ρ=0.15，CP 全局 g 读出保留。
2. **变化集中于门控表示。** 将 G1 的原查询特征改为 TRAIN 去均值后的归一化残差，η=0.25、τγ=0.1、模式中心化不变。
3. **固定统计来源。** 每个物体查询由全部 22,998 张 TRAIN 图像求均值，三个 seed 共用同一 [12,768] 资产；标签不参与统计。
4. **保持初始化与优化日程。** 三 seed 从原始初始化开始，rank32、adapter rank8、batch64、十轮、CP LR scale0.3、λorth0.001。
5. **分开保存选权重路径。** 主表使用首次最大 VAL CW native AUC；exact 路径独立保存，各条件使用自己的 VAL native bias。
6. **按同 seed 比较。** H1−G1、H1−G0 与 H1 dynamic−mean 分别衡量输入变化、整体训练效果及图像动态的推理贡献。

```math
\mathcal{L}_{\mathrm{task}}(0.075)=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结 OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，余弦 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／动态正交项／workers | 1 / 0.001 / 4 |

内部 variant=V1，通过 `gate_input=train_residual` 开启门控。CP 参数组为 Au/Av/Wc/Bvis；`gate_fo_mean` 为固定 buffer。prim／cent 分支关闭。

## 模板、运行配置与统计资产

- [H1模板](round_h/H1.json) 是原 ZIP 的配置，保留实验超参数；[runtime/H1.json](runtime/H1.json) 为原服务器实际运行配置，新增均值文件路径及 SHA256。
- [TRAIN均值及身份](../results/raw_export/roundH_20261020/feature_statistics/)与 [原输出运行配置](../results/raw_export/roundH_20261020/configs/H1.json)随结果保存。三 seed checkpoint 记录相同均值和哈希。
- 新的复现运行由调度器在指定输出目录生成自己的均值资产及运行配置。原运行配置保留服务器当时的路径。
- [三项训练](experiment_inventory.json)、[六项均值评估](intervention_inventory.json)、[26份历史参考](references/pre_h_complete.json)、[标签资产](assets/)、[外部输入](../external_inputs.json)。
- 直接对照为 [G1](round_g/G1.json)，静态对照为 [G0](round_g/G0.json)。缓存 manifest SHA256：`bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

下一轮 I1/I2 的输入与温度建议见[数据分析](../analysis/README.md)；本阶段实际训练配置为 H1。
