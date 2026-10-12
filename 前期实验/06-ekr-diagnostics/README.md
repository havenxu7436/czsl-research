# 06 · EKR 冻结机制诊断

状态：冻结权重诊断已完成。

## 实验设置

- 输入：seed 0 的 N_name、N_clean、N_shuffle 与共同 A_unit。
- 视图：TRAIN pair 平衡子集、VAL CW、VAL OW。
- 干预：描述置换、对象上下文置换、null 槽移除；参数固定。

## 流程

1. 缓存基础分数与属性特征。
2. 计算原分支及输入干预后的分数、注意力和 margin。
3. 核对回放指标并汇总机制统计。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-06
python tools/reproduce.py materialize --line 前期实验 --stage 06 --dest "$REPLAY"
bash "$REPLAY/name_ekr_mechanism_diag_20261007_v1/run.sh"
```

## 结果与文件

对象上下文与 null 槽干预的分数变化较小；逐图统计见 mechanism_findings.json。

- [执行代码](code/name_ekr_mechanism_diag_20261007_v1/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/artifacts/integrity_checks.json](<results/artifacts/integrity_checks.json>)
- [results/artifacts/mechanism_findings.json](<results/artifacts/mechanism_findings.json>)
