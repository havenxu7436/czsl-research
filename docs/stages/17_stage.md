# 17. formula-aligned-0601：逐公式修正r2

状态：代码/交付/CPU验证；未收集正式训练结果。

## 修改动机与实际工作

按公式审计修正G/H原始head评分、GT-object CE、逻辑grounding、prior及train/infer共用接口。

## 结果与决策

保留接口选择与未覆盖项；42次×30轮为新训练计划，不继承residual旧表。

## 原始证据入口

- [faithful_0601_20261011](<../../faithful_0601_20261011>)
- [review_0601_formula_20261011](<../../review_0601_formula_20261011>)
- [desktop_bundle_0601_20261011](<../../desktop_bundle_0601_20261011>)
- [archive_inputs/formula_aligned_bundle](<../../archive_inputs/formula_aligned_bundle>)
- [output/0601逐公式复刻审计_20261011.md](<../../output/0601逐公式复刻审计_20261011.md>)
- [output/0601主公式复刻_实践命令与时间预估_20261011.md](<../../output/0601主公式复刻_实践命令与时间预估_20261011.md>)

完整公式、对照、全部数值及比较限制请结合 [实验总档案](../history_20261010.md) 与 [路线对照](../ROUTES.md) 查阅；第16—18阶段的最新状态以各自结果/交付文件为准。
