# TRUST-CPF 方案

CLIP / CPF 上的可迁移关系效用监督、结构选择与 CZSL 研究。强基底三轮均已完成，按执行顺序归档。

| 编号 | 实验 | 完成内容 | 入口 |
| --- | --- | --- | --- |
| 01 / v1 | 初始 CLIP 基底 | 21 组主实验、知识打乱、两项效用消融审计、600 条噪声评估 | [方案与结果](v1/README.md) |
| 02 | 强 CPF 基底完整 TRUST | 30 份识别/读出、12 份效用审计、1,800 条噪声评估 | [阶段首页](02/README.md) · [完整分析](02/analysis/README.md) |
| 03 | 模型响应条件化 | 新增 3 次适配，复用 6 份对照；共 9 份识别结果 | [阶段首页](03/README.md) · [完整分析](03/analysis/README.md) |
| 04 | Compact 紧凑效用监督 | 新增 6 次适配，复用 3 份 Drop；共 9 份识别结果 | [阶段首页](04/README.md) · [完整分析](04/analysis/README.md) |

02–04 保存全部方案、公式与代码、配置、初始权重、教师与测量、模型权重、预测、审计、日志、分析和原始发布包。48 份识别/读出记录中有 9 份跨阶段复用对照；读出和冻结基底评价另有明确标注。

当前证据支持继续研究低维效用监督及其结构部署方式。04 分析末尾的类型内校准与半密度对齐是下一轮提案，尚未计入已完成实验。

[弱基底完整结果](v1/analysis/server-results-20261022/README.md) · [全仓库索引](../README.md)

[本次全仓库校验](04/analysis/repository_qa/repository_verify.json) · [公式语法检查](04/analysis/repository_qa/math_checks.json) · [Compact 分析重算核对](04/analysis/repository_qa/analysis_reproduction.json)
