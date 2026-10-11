# 14. 冻结 CLIP 探针 v1→r2

状态：v1检查失败；r2七配方、0optimizer。

## 修改动机与实际工作

修正广义seen/unseen划分assert，检查图像隔离；以冻结CLIP判断知识教师信号。

## 结果与决策

OE+AE与AE-only继续gate均False；七种配方不算七个训练模型。

## 原始证据入口

- [clip_knowledge_probe_20261010](<../../clip_knowledge_probe_20261010>)
- [clip_knowledge_probe_20261010_r2](<../../clip_knowledge_probe_20261010_r2>)
- [clip_probe_review_20261010](<../../clip_probe_review_20261010>)
- [output/clip_knowledge_probe_results_20261010_r2](<../../output/clip_knowledge_probe_results_20261010_r2>)

完整公式、对照、全部数值及比较限制请结合 [实验总档案](../history_20261010.md) 与 [路线对照](../ROUTES.md) 查阅；第16—18阶段的最新状态以各自结果/交付文件为准。
