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

## 结果与文件

对象上下文与 null 槽干预的分数变化较小；逐图统计见 mechanism_findings.json。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/artifacts/integrity_checks.json（精简版未保留）
- results/artifacts/mechanism_findings.json（精简版未保留）
