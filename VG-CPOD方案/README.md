# VG-CPOD方案

独立CLIP实验路线：冻结CLIP图文双编码器上的视觉引导类别低秩算子。

**[完整公式及核心代码v1](v1/VG-CPOD公式及代码v1.md)** · [数学与实验设计](v1/数学与实验设计_v1.md) 

| 编号 | 实验 | 入口 |
|---|---|---|
| 01 | CLIP ViT-L/14；UT-Zappos natural split | [版本索引](v1/README.md) · [结果](v1/results/ut-zap50k-3seeds-20261018/README.md) |
| 02 | 支持损失重加权与 CP 分组学习率（A 组）；12 次训练，A1/A5/A6 三 seed | [修订说明](02/README.md) · [代码](02/code/README.md) · [配置](02/configs/README.md) · [结果](02/results/README.md) · [数据分析](02/analysis/README.md) |
