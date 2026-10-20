# 03 · 配置与超参数策略

| 组 | 修正读出 | 门控输入 | η / 系数范围 | 已完成 seed |
| --- | --- | --- | --- | --- |
| [B1](round_b/B1.json) | 归一化的 0.5 全局 + 0.5 物体查询特征 | 关闭 | 全 1 | 3351 |
| [B2](round_b/B2.json) | 全局 g | 物体查询特征 | 1 / (0,2) | 3351 |
| [B3](round_b/B3.json) | 全局 g | 物体查询特征 | 0.25 / (0.75,1.25) | 3351 |

## 选取策略

02 的三 seed 结果确定 A5 作为训练基座。本轮固定 β、学习率、rank、轮数和 batch，将比较集中于读出路径及 η：B1 的等权混合对应 α=0.5；B2 使用原 η=1；B3 将 η 降为 0.25，保持全局读出和相同局部输入。

三组均为 seed3351、10 epochs，CP LR=3e-4→3e-6，适配器 LR=1e-3→1e-5，β=0.1，rank32，adapter rank8，batch64，ρ=0.1，κ=20，τγ=0.1，weight decay=1e-4。B2/B3 的 Bvis 使用 CP 参数组及 λorth=0.001。

## 配置与资产索引

- [完整已完成运行配置与指标](experiment_inventory.json)。
- [默认配置](default.json)与[02兼容配置](round_a/)保留为原运行包成员；03 的实际运行入口是 round_b/B1、B2、B3。
- [A1/A5/A6三seed标量参考](references/round_a_confirm.json)与[原V0/V1参考](references/original_v1.json)。
- [标签、文本命名与数据索引](assets/)；图像特征复用已有冻结 CLIP 缓存。

三组内部均采用 V1 标识，由 `readout_feature`、`gate_input` 与 `gate_eta` 决定实际计算。原子属性及伙伴分支关闭，配置中的对应历史权重不参与本轮目标。完整固定项与选择依据见[主README第5节](../README.md#5-超参数选取策略与实际实验矩阵)。
