# 09 · EKR v2：3→10 epoch 续训

状态：四组续训已完成。

## 实验设置

- 沿用第 07 阶段四组的模型、知识、optimizer 与 RNG 状态。
- 导入 epoch 3 完整断点，新增 epoch 4–10，共 28 个分支 epoch。
- 选模候选覆盖 epoch 0–10；评价 VAL CW/OW。

## 流程

1. 导入断点并回放 epoch 3 分数。
2. 从 epoch 4 继续训练至 epoch 10。
3. 按完整曲线选模，汇总 3 轮与 10 轮读数。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-09
python tools/reproduce.py materialize --line 前期实验 --stage 09 --dest "$REPLAY"
bash "$REPLAY/ekr_v2_budget10_20261008/run.sh" prepare
bash "$REPLAY/ekr_v2_budget10_20261008/run.sh" import
bash "$REPLAY/ekr_v2_budget10_20261008/run.sh" resume
bash "$REPLAY/ekr_v2_budget10_20261008/run.sh" report
```

## 结果与文件

名称组在 epoch 6 获得增量；正确描述组仍选中 epoch 0。

- [执行代码](code/ekr_v2_budget10_20261008/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/artifacts/reports/results.csv](<results/artifacts/reports/results.csv>)
