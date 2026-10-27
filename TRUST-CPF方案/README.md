# TRUST-CPF 方案

CLIP / CPF 上的可迁移关系效用监督、关系概率分配与 CZSL 研究。01–10 按实际执行顺序归档。

| 编号 | 实验 | 已完成内容 | 入口 |
| --- | --- | --- | --- |
| 01 | 初始 CLIP 基底 | 21 组主实验、知识打乱、效用消融、600 条噪声评估 | [方案与结果](v1/README.md) |
| 02 | 强 CPF 基底完整 TRUST | 30 份识别/读出、12 份效用审计、1,800 条噪声评估 | [阶段首页](02/README.md) · [分析](02/analysis/README.md) |
| 03 | 模型响应条件化 | 3 次新增适配、6 份复用对照 | [阶段首页](03/README.md) · [分析](03/analysis/README.md) |
| 04 | Compact 紧凑效用监督 | 6 次新增适配、3 份复用 Drop | [阶段首页](04/README.md) · [分析](04/analysis/README.md) |
| 05 | Align 半密度对齐 | 9 份识别、5 次实际适配、216 行识别、312 行 Q5 风险 | [阶段首页](05/README.md) · [分析](05/analysis/README.md) |
| 06 | Competition 三目标 / 10 epoch | 15 个新教师、15 份识别、6 次实际适配、312 行 Q5 风险、165 行曲线 | [阶段首页](06/README.md) · [分析](06/analysis/README.md) |
| 07 | R5 相对效用拟合 / 10 epoch | 27 阶段、15 份结果标签、1 次新适配；168 行 Q5 风险 | [阶段首页](07/README.md) · [分析](07/analysis/README.md) |
| 08 | State 第一批 / 10 epoch | 3 条 Uniform 重训、1512 行识别、600 行 Q5、756 份预测 | [阶段首页](08/README.md) · [分析](08/analysis/README.md) |
| 09 | Stage 第二批 / 10 epoch | 15 个教师追加训练、972 行识别、744 行 Q5、486 份预测 | [阶段首页](09/README.md) · [分析](09/analysis/README.md) |
| 10 | A/B/C 幅度、混合与情境效用 / 10 epoch | 36 阶段、936 行识别、3360 行 Q4、3504 行 Q5、1056 行方向曲线 | [阶段首页](10/README.md) · [分析](10/analysis/README.md) |

各阶段保存方案、公式、实现、配置、知识指纹、发布包、权重、测量、预测、审计、日志与复算入口。相同概率下复用的 Uniform、跨阶段复用的历史对照均明确标注。

最新 10 阶段已完成 A/B/C 整批实验。三 bank 平均 Uniform 为 36.051 / 26.638，Matched 为 36.148 / 26.744，C 为 36.147 / 26.747（CW / OW）。C 在三个 seed 均超过 Uniform、关系置换均值和情境打乱，平均性能与 Matched 接近；A/B 没有形成增量。[最新完整分析](10/analysis/README.md)。

历史六路线安排保留于 [05/PLAN.md](05/PLAN.md)，v6 修订与本次两批设计见 [08/PLAN.md](08/PLAN.md)。08、09 和 10 均已完成；v8 公式及本轮三条机制见 [10/PLAN.md](10/PLAN.md)，10 分析末尾的五组独立重复为下一批提案。修订前 Competition 3 epoch 包见 [历史版本](06/references/pre_10epoch/README.md)。

[全部实验路线](../README.md) 
