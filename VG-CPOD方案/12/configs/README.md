# 12 · 配置与超参数策略

| 实验 | Wd倍率 | Wd首步→末步LR | native最佳epoch | exact最佳epoch |
|---|---|---|---|---|
| J1历史对照 | 1 | 3e-4→3e-6 | 5/6/6 | 5/6/6 |
| [K1](round_k/K1.json) | 0.3 | 9e-5→9e-7 | 5/8/6 | 5/6/6 |
| [K2](round_k/K2.json) | 0.1 | 3e-5→3e-7 | 5/8/6 | 5/6/6 |

轮次按seed3351/3352/3353顺序。六训练各十轮、3600次更新，实际配置和两套结果见[experiment_inventory.json](experiment_inventory.json)。

## 1. 选择依据

J1中动态梯度的一阶矩较小，但Adam归一化后实际步长接近静态读出。K组固定两档倍率，直接改变Wd的更新速度；其余表示、目标、初始化及训练预算沿用J1，三seed完整比较。

```math
\alpha_{d,t}=m_d\alpha_{\mathrm{CP},t},\qquad
\alpha_{\mathrm{CP},t}=0.3\alpha_t,\qquad
m_d\in\{0.3,0.1\}.
```

学习率同时作用于自适应项和AdamW解耦衰减，weight decay系数保持1e-4。K1的dynamic_cp lr_scale为0.09，K2为0.03。

| 固定项 | 设置 |
|---|---|
| CLIP／输入 | 冻结OpenAI ViT-L/14，224px、768维 |
| rank／adapter rank／κ／τv | 32 / 8 / 20 / 0.07 |
| ρ／β／η／τγ | 0.15 / 0.075 / 0.25 / 0.56 |
| gate_input／读出模式 | train_residual / separate |
| epochs／batch／每轮更新 | 10 / 64 / 360 |
| 后编码适配矩阵LR／普通CP LR | 1e-3→1e-5 / 3e-4→3e-6 |
| AdamW β₁、β₂／ε／weight decay | 0.9、0.999 / 1e-8 / 1e-4 |
| 总梯度裁剪／λorth／eval batch／workers | 1 / 0.001 / 64 / 4 |
| 模式中心化／残差ε／可训练参数 | 开启 / 1e-12 / 147,456 |

## 2. 模板、运行配置与数据身份

- [round_k模板](round_k/)、[服务器runtime](runtime/)、[六训练清单](experiment_inventory.json)、[十二项均值评估](intervention_inventory.json)。
- [TRAIN查询均值](../results/raw_export/roundK_20261021/feature_statistics/)来自全部22,998张TRAIN图像，形状[12,768]，与J组逐字相同。
- [38项历史运行](references/pre_k_complete.json)、[文本资产](assets/)、[外部输入](../external_inputs.json)。
- 对照模板：[J1](round_j/J1.json)、[G0](round_g/G0.json)、[I2](round_i/I2.json)。缓存manifest SHA256：`bb8b7a8eb8c4e77a14ef65dd5c626aac559880d7c8d642dcc1e9a2a843831d2a`。

## 3. 根据结果作出的决策

两档倍率实现了预期的更新缩放，TEST整体水平未超过G0。下一步结束Wd学习率插值，转向CLIP局部判别证据学习；后续正式矩阵的固定配方、算子与创新定位见[数据分析](../analysis/README.md)。本目录保存本轮已执行配置。
