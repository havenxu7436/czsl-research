# 11 · 配置与超参数策略

| 方案 | 读出模式 | τγ／η／ρ／β | CP LR | native／exact最佳epoch |
|---|---|---|---|---|
| [J1](round_j/J1.json) | separate | 0.56／0.25／0.15／0.075 | Ws/Wd均3e-4→3e-6 | 5/6/6；5/6/6 |

轮次按seed3351/3352/3353顺序。三次训练各十轮、3600次更新，完整配置和两套结果见[experiment_inventory.json](experiment_inventory.json)。

## 1. 本轮选取策略

1. 以I2作直接对照，固定归一化TRAIN残差、温度0.56、η=0.25，比较读出参数共享方式。
2. 固定β=0.075、ρ=0.15、rank32、adapter rank8、全局g读出与原损失。
3. 两路读出从零初始化，属性／物体因子和适配器沿用对应seed原始初始化。
4. 本轮Ws/Wd均使用原CP组LR，不新增学习率分组；记录各自修正RMS，观察实际学习幅值。
5. native按VAL CW native AUC选权重，exact独立保存；各权重、条件分别使用自己的VAL bias。

```math
\mathcal{L}_{\mathrm{task}}(0.075)=
\frac{3}{2}\mathcal{L}_{\mathrm{pair}}+\frac{9}{86}\mathcal{L}_{\mathrm{support}}.
```

| 固定项 | 设置 |
|---|---|
| CLIP／输入 | 冻结OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 适配器LR／CP LR | 1e-3→1e-5 / 3e-4→3e-6 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |
| 模式中心化／残差ε | 开启 / 1e-12 |
| 可训练参数／均值buffer | 147,456 / [12,768] |

内部variant=V1，通过配置开启门控；prim/cent项为零。CP优化组为Au/Av/Bvis/Wc/Wd，`cp_lr_scale=0.3`。

## 2. 模板、运行配置与统计身份

- [模板](round_j/J1.json)来自发布ZIP；[runtime](runtime/J1.json)为服务器实际配置，补充TRAIN均值文件路径与SHA256。
- [TRAIN均值与身份](../results/raw_export/roundJ_20261021/feature_statistics/)包含全部22,998张TRAIN图像、12个查询的固定统计，与[10 I组](../../10/configs/README.md)逐字相同。
- [三训练清单](experiment_inventory.json)、[六项均值评估](intervention_inventory.json)、[历史参考](references/pre_j_complete.json)、[文本资产](assets/)、[外部输入](../external_inputs.json)。
- 对照模板：[I2](round_i/I2.json)、[G0](round_g/G0.json)、[G1](round_g/G1.json)。特征manifest SHA256为`bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

## 3. 后续配置建议

数据分析建议下一轮将Wd独立分组，使用原CP LR的0.3/0.1倍，即9e-5/3e-5。依据是checkpoint中的Adam步长与动态图像修正幅值；本阶段已完成配置为J1的双路相同CP学习率。推导和实验安排见[数据分析](../analysis/README.md)。
