# 12. 条件属性 pilot r1→r2与固定混合回放

状态：r1工程失败；r2五组10轮＋回放。

## 修改动机与实际工作

改为GT-object条件属性目标，保持对象边缘；修复Torch1.13安全checkpoint兼容，另做固定25%混合回放。

## 结果与决策

已见拟合与unseen迁移分开；固定读出回放不是新训练；保留失败r1。

## 原始证据入口

- [conditional_attribute_pilot_20261009](<../../conditional_attribute_pilot_20261009>)
- [pilot_review_20261009](<../../pilot_review_20261009>)
- [output/conditional_attribute_pilot_results_20261009_r2](<../../output/conditional_attribute_pilot_results_20261009_r2>)

完整公式、对照、全部数值及比较限制请结合 [实验总档案](../history_20261010.md) 与 [路线对照](../ROUTES.md) 查阅；第16—18阶段的最新状态以各自结果/交付文件为准。
