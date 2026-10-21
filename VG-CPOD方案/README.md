# VG-CPOD方案

独立CLIP实验路线：冻结CLIP图文双编码器上的视觉引导类别低秩算子。

**[完整公式及核心代码v1](v1/VG-CPOD公式及代码v1.md)** · [数学与实验设计](v1/数学与实验设计_v1.md) 

| 编号 | 实验 | 入口 |
|---|---|---|
| 01 | CLIP ViT-L/14；UT-Zappos natural split | [版本索引](v1/README.md) · [结果](v1/results/ut-zap50k-3seeds-20261018/README.md) |
| 02 | 支持损失重加权与 CP 分组学习率（A 组）；12 次训练，A1/A5/A6 三 seed | [修订说明](02/README.md) · [代码](02/code/README.md) · [配置](02/configs/README.md) · [结果](02/results/README.md) · [数据分析](02/analysis/README.md) |
| 03 | 读出解耦与局部弱门控（B 组）；B1/B2/B3 各 seed3351，3 次训练 | [修订说明](03/README.md) · [代码](03/code/README.md) · [配置](03/configs/README.md) · [结果](03/results/README.md) · [数据分析](03/analysis/README.md) |
| 04 | 静态幅度对照与模式中心化门控（C 组）；C1/C2 各 seed3351，另纳入 B3/3352 | [修订说明](04/README.md) · [代码](04/code/README.md) · [配置](04/configs/README.md) · [结果](04/results/README.md) · [数据分析](04/analysis/README.md) |
| 05 | 中心化 CP 幅度联动与均值门控评估（D 组）；D1 三 seed，C2 两个补 seed、C1/3352，六 checkpoint 干预 | [修订说明](05/README.md) · [代码](05/code/README.md) · [配置](05/configs/README.md) · [结果](05/results/README.md) · [数据分析](05/analysis/README.md) |
| 06 | 中心化 CP 强度扩展与双准则选权重（E 组）；E1 三 seed、两套最优权重及六项均值评估 | [修订说明](06/README.md) · [代码](06/code/README.md) · [配置](06/configs/README.md) · [结果](06/results/README.md) · [数据分析](06/analysis/README.md) |
| 07 | 支持集权重下调与跨组合排序诊断（F 组）；F1 三 seed、加权损失日志、双准则选权重及六项均值评估 | [修订说明](07/README.md) · [代码](07/code/README.md) · [配置](07/configs/README.md) · [结果](07/results/README.md) · [数据分析](07/analysis/README.md) |
| 08 | 支持集权重折中与门控训练对照（G 组）；G0/G1各三seed、β=0.075、双准则选择与六项均值评估 | [修订说明](08/README.md) · [代码](08/code/README.md) · [配置](08/configs/README.md) · [结果](08/results/README.md) · [数据分析](08/analysis/README.md) |
| 09 | 训练均值中心化与归一化残差门控（H 组）；H1三seed、共享TRAIN均值、双准则选择与六项均值评估 | [修订说明](09/README.md) · [代码](09/code/README.md) · [配置](09/configs/README.md) · [结果](09/results/README.md) · [数据分析](09/analysis/README.md) |
| 10 | 残差幅值保留与门控温度对照（I 组）；I1/I2各三seed、双准则选择与十二项均值评估 | [修订说明](10/README.md) · [代码](10/code/README.md) · [配置](10/configs/README.md) · [结果](10/results/README.md) · [数据分析](10/analysis/README.md) |
| 11 | 静态与动态CP独立读出（J 组）；J1三seed、双准则选择与六项均值评估 | [修订说明](11/README.md) · [代码](11/code/README.md) · [配置](11/configs/README.md) · [结果](11/results/README.md) · [数据分析](11/analysis/README.md) |
| 12 | 动态读出学习率分组与更新尺度对照（K 组）；K1/K2各三seed、双准则及十二项均值评估 | [修订说明](12/README.md) · [代码](12/code/README.md) · [配置](12/configs/README.md) · [结果](12/results/README.md) · [数据分析](12/analysis/README.md) |
