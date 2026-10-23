# 13 · G0条件知识残差与语义附件对照

**L组，VG-CPOD v1；实验2026-10-22完成，2026-10-23归档。** 冻结OpenAI CLIP ViT-L/14，UT-Zappos natural split。L0/L1/L2各三个seed、每组10 epochs，共9次训练、90个模型epoch；全部18份VAL/TEST预测、18个best/last checkpoint与3个G0初始权重已归档。

[修订实验方案](REVISED_PLAN.md) · [公式与核心实现](FORMULAS_AND_CODE.md) · [所有代码修订](code/README.md) · [配置](configs/README.md) · [完整结果](results/README.md) · [完整数据分析](analysis/README.md) · [下一步方向](NEXT_DIRECTION.md) · [服务器命令](SERVER_COMMANDS.md)

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
| [results/](results/README.md) | 原包、181个服务器原始成员、全部权重、预测、日志、完成记录与源文件映射 |
| [analysis/](analysis/README.md) | 完整结果、错误分解、类别明细、参数作用量与可移植重算脚本 |
| [references/](references/README.md) | 代码来源、历史提案、原始报告与迁移前文档 |

## 从仓库恢复可执行包

```bash
git lfs pull --include="VG-CPOD方案/13/**"
python tools/register_vgcpod_round_l.py --check
python tools/reproduce.py materialize --line VG-CPOD方案 --stage 13 --dest "$HOME/experiments/vgcpod-L-replay"
cd "$HOME/experiments/vgcpod-L-replay/vgcpod_v1_roundL"
python -m unittest discover -s tests -v
```

恢复目录须为新的绝对路径；正式复训使用现有CLIP缓存。当前已完成的九组原始运行仍保留原始名称与哈希，统一的仓库身份为13/L组；重命名执行包只改变打包标签和阅读文档，模型、损失、训练与评价运算保持相同。

下一步R0/R1/R2是独立的可训练视觉研究提案，代码尚未实现；它不计入本轮已完成实验。

## 校验记录

[全仓库校验](analysis/repository_qa/repository_verify.json) · [GitHub网页公式与表格核验](analysis/repository_qa/github/checks.json) · [复算与单元验证](analysis/confirmed/reproduction_checks.json)
