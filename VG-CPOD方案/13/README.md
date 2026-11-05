# 13 · G0条件知识残差与语义附件对照

> **精简归档说明：** 九组训练的汇总证据、公式、核心代码与配置仍在；下文所述预测/checkpoint 是历史实验规模记录，原始文件已从 GitHub 精简版移除。

**L组，VG-CPOD v1；实验2026-10-22完成，2026-10-23归档。** 冻结OpenAI CLIP ViT-L/14，UT-Zappos natural split。L0/L1/L2各三个seed、每组10 epochs，共9次训练、90个模型epoch；实验产生18份VAL/TEST预测、18个best/last checkpoint与3个G0初始权重，精简版仅保留汇总证据。

[修订实验方案](REVISED_PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [所有代码修订](code/README.md) · [配置](configs/README.md) · [完整结果](results/README.md) · [完整数据分析](analysis/README.md) · [下一步方向](NEXT_DIRECTION.md)

## 实验身份与结论

| 新编号 | 原始运行variant | TEST CW AUC，均值±样本SD | TEST OW AUC，均值±样本SD |
| --- | --- | --- | --- |
| L0：Base | base | 34.4384 ± 0.6282 | 25.1400 ± 0.9063 |
| L1：Knowledge | knowledge | 35.0555 ± 0.7869 | 25.7402 ± 0.8549 |
| L2：Shuffle | shuffle | 35.3067 ± 0.7471 | 25.9153 ± 0.5651 |

Knowledge相对Base提高0.6170 CW、0.6001 OW；相对同容量Shuffle低0.2512 CW、0.1751 OW，三个seed的CW均落后。收益主要体现在减少未见样本被判成同物体已见搭配；多候选未见内部准确率从81.2176%降至80.5484%。本轮没有建立正确语义附件的额外优势。

## 目录结构

| 分区 | 内容 |
| --- | --- |
| [code/](code/README.md) | 最终执行源码、续训与评价、代码修改清单、重命名差异补丁 |
| [configs/](configs/README.md) | 训练配置、96条关系、打乱附件、G0缓存指纹与9组清单 |
| [results/](results/README.md) | 实验规模、结果表与原始产物状态说明 |
| [analysis/](analysis/README.md) | 完整结果、错误分解、类别明细、参数作用量与可移植重算脚本 |
| [references/](references/README.md) | 代码来源与历史提案索引 |

## 归档边界

九组原始运行的名称与哈希属于历史完整归档；当前树保留统一的 13/L 组身份、模型定义、损失、训练设置和评价结论，不包含可执行包或原始运行目录。

下一步R0/R1/R2是独立的可训练视觉研究提案，代码尚未实现；它不计入本轮已完成实验。

## 校验记录

历史全仓库 QA、GitHub 网页核验与复算 JSON 已从精简版移除；公式、表格与结论保留在本页和分析 Markdown。
