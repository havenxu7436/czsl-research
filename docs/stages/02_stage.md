# 02. 早期 MVP、跨数据集验证与固定 prior 校准

状态：已训练＋无训练后处理。

## 修改动机与实际工作

UT A/C/D/E 与 MIT、C-GQA A/C；固定 A/C checkpoint 扫 λ，以分离文本和静态 prior。

## 结果与决策

UT 文本有信号；后处理选择 λ=0。20 次评价不计20次新训练。

## 原始证据入口

- [cpf_stage_report_tools_20261004_v1](<../../cpf_stage_report_tools_20261004_v1>)
- [cpf_stage_report_tools_20261004_v2](<../../cpf_stage_report_tools_20261004_v2>)
- [output/week5_evidence_20261005](<../../output/week5_evidence_20261005>)
- [output/week5_evidence_20261005_verified](<../../output/week5_evidence_20261005_verified>)
- [第四步_UT结果冻结_K校准诊断与跨数据集验证.md](<../../第四步_UT结果冻结_K校准诊断与跨数据集验证.md>)
- [第四步补充_lambda0后扩展AC操作.md](<../../第四步补充_lambda0后扩展AC操作.md>)

完整公式、对照、全部数值及比较限制请结合 [实验总档案](../history_20261010.md) 与 [路线对照](../ROUTES.md) 查阅；第16—18阶段的最新状态以各自结果/交付文件为准。
