# TRUST-CPF 方案

Teacher-supervised Relation Utility for Semantic Transfer（Trust）实验归档.

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
| 11 | G₀ 三数据集完整独立流水线 / 10 epoch | 165任务、450段训练、1,620行识别、378行Q5；三数据集×三seed独立拟合 | [阶段首页](11/README.md) · [分析](11/analysis/README.md) |
