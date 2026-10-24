# TRUST-CPF 方案

CLIP / CPF 上的可迁移关系效用监督、关系概率分配与 CZSL 研究。01–06 按实际执行顺序归档。

| 编号 | 实验 | 已完成内容 | 入口 |
| --- | --- | --- | --- |
| 01 | 初始 CLIP 基底 | 21 组主实验、知识打乱、效用消融、600 条噪声评估 | [方案与结果](v1/README.md) |
| 02 | 强 CPF 基底完整 TRUST | 30 份识别/读出、12 份效用审计、1,800 条噪声评估 | [阶段首页](02/README.md) · [分析](02/analysis/README.md) |
| 03 | 模型响应条件化 | 3 次新增适配、6 份复用对照 | [阶段首页](03/README.md) · [分析](03/analysis/README.md) |
| 04 | Compact 紧凑效用监督 | 6 次新增适配、3 份复用 Drop | [阶段首页](04/README.md) · [分析](04/analysis/README.md) |
| 05 | Align 半密度对齐 | 9 份识别、5 次实际适配、216 行识别、312 行 Q5 风险 | [阶段首页](05/README.md) · [分析](05/analysis/README.md) |
| 06 | Competition 三目标 / 10 epoch | 15 个新教师、15 份识别、6 次实际适配、312 行 Q5 风险、165 行曲线 | [阶段首页](06/README.md) · [分析](06/analysis/README.md) |

各阶段保存方案、公式、实现、配置、知识指纹、发布包、权重、测量、预测、审计、日志与复算入口。相同概率下复用的 Uniform、跨阶段复用的历史对照均明确标注。

最新主 bank 三 seed TEST AUC：Uniform 35.934 / 26.532，Competition 35.950 / 26.493（CW / OW）。独立 Q5 上测量效用能够指导更好的关系分配，预测和部署迁移仍是当前主要研究问题。[最新完整分析](06/analysis/README.md) 给出逐 seed、多个图 bank、全部训练曲线和机制复算。

六路线后续安排保留于 [05/PLAN.md](05/PLAN.md)。Align 与 Competition 已完成；06 分析中的 R5 相对拟合仍为下一批提案。修订前 Competition 3 epoch 发布包作为 [历史版本](06/references/pre_10epoch/README.md) 保存，未计为独立已完成实验。

[全部实验路线](../README.md) · [05 复算](05/analysis/REPRODUCE.md) · [06 复算](06/analysis/REPRODUCE.md)

[GitHub 网页排版检查](06/analysis/repository_qa/github/checks.json) · [公式与链接检查](06/analysis/repository_qa/markdown_math_checks.json) · [全仓库校验](06/analysis/repository_qa/repository_verify.json)
