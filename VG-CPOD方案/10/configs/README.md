# 10 · 配置与超参数策略

| 方案 | gate_input | τγ | η／ρ／β | native／exact最佳epoch |
|---|---|---|---|---|
| [I1](round_i/I1.json) | train_centered | 0.1 | 0.25／0.15／0.075 | 8/8/6；10/8/6 |
| [I2](round_i/I2.json) | train_residual | 0.56 | 0.25／0.15／0.075 | 8/8/6；8/8/6 |

轮次按seed3351/3352/3353顺序。六次训练各十轮、3600次更新；实际配置与两套指标见[experiment_inventory.json](experiment_inventory.json)。

## 1. 选取策略

1. **保留幅值作为主变量。** I1相对H1仅改`gate_input`及实验编号，温度仍为0.1。
2. **由TRAIN尺度设置对照。** I2相对H1仅改温度及编号；残差代表RMS约0.17835，0.1/0.17835约为0.56。该温度在本轮训练前固定。
3. **复用已有目标与优化。** β=0.075、ρ=0.15、η=0.25、全局g读出、rank32和CP LR scale0.3延续H1。
4. **共享固定统计。** 六模型共用从H组复制的TRAIN均值；均值在全部22,998张图像和12个查询上计算，并作为buffer保存。
5. **按同seed、同选择轨道配对。** native为主结果，exact独立保存；各条件的VAL bias单独选取后用于TEST。

```math
\mathcal{L}_{\mathrm{task}}(0.075)
=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

| 固定项目 | 设置 |
|---|---|
| CLIP／输入 | 冻结OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6，原余弦日程 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |
| 模式中心化／残差ε | 开启 / 1e-12 |

内部variant=V1，CP优化组为Au/Av/Wc/Bvis；prim／cent关闭。可训练参数均为122,880，固定均值buffer为[12,768]。

## 2. 模板、运行配置与统计身份

- [模板](round_i/)来自原ZIP；[runtime](runtime/)为服务器实际运行配置，补充统计文件路径及SHA256。
- [TRAIN均值与身份](../results/raw_export/roundI_20261021/feature_statistics/)、[服务器原运行配置](../results/raw_export/roundI_20261021/configs/)。I/H统计NPZ逐字相同，六个模型checkpoint中的均值逐元素相同。
- [六项训练清单](experiment_inventory.json)、[十二项均值评估](intervention_inventory.json)、[历史参考](references/pre_i_complete.json)、[标签资产](assets/)、[外部输入](../external_inputs.json)。
- 对照配置：[H1](round_h/H1.json)、[G0](round_g/G0.json)、[G1](round_g/G1.json)。缓存manifest SHA256：`bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

下一轮J1的结构建议及固定参数见[数据分析](../analysis/README.md)；本阶段已完成模型为I1/I2。
