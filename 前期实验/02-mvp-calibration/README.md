# 02 · MVP 对照与固定 prior 校准

状态：训练与后处理结果已归档。

## 实验设置

- UT-Zappos：A/C/D/E；MIT-States、C-GQA：A/C。
- 后处理使用固定 A/C checkpoint 扫描 prior 系数 λ，共 20 次评价。

## 流程

1. 按对应组配置执行训练并保存 checkpoint。
2. 在固定权重上评价不同 λ 的 CW/OW 指标。
3. 汇总指标、epoch 曲线与 bias=0 逐图结果。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-02
python tools/reproduce.py materialize --line 前期实验 --stage 02 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/cpf_stage_report_tools_20261004_v2/collect_stage_report.py" --help
```

训练配置与命令参数见 [inputs.json](inputs.json) 对应源码的参数接口。

## 结果与文件

UT 文本组呈现指标增量；后处理选定 λ=0。

- [执行代码](code/cpf_stage_report_tools_20261004_v2/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/tables/epoch_curves.csv](<results/tables/epoch_curves.csv>)
