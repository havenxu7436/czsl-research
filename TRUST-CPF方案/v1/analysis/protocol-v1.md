# 核心指标与分析协议

## 识别主表

七组×三seed分别整理VAL/TEST的CW/OW：`AUC_native`、`AUC_exact`、oracle-HM，以及固定VAL bias的Seen/Unseen/HM；同时记录NLL、ECE、GT-object属性准确率/NLL、三类错误数量和跨图预测方差。对三个seed计算均值与样本标准差；在相同seed下比较`robust−r0`、`robust−drop`、`robust−free`和`robust−mean`。

## 机制与成本

| 研究问题 | 核心指标 | 原始产物 |
|---|---|---|
| 配对效用可否跨组合预测 | Q5 Spearman、RMSE、balanced direction accuracy、AUROC；可靠方向子集覆盖比例 | `utility/audit.json` |
| 预测方差是否有意义 | 90%预测区间覆盖率；任务波动、支持摘要变异与模型分歧 | `utility/audit.json`、`structure.json` |
| 语义与视觉证据分别有何作用 | full对semantic；类型常数/边查表对照；20%未测边子集 | 各work的`utility/audit.json` |
| 知识内容是否有效 | clean对独立`shuffled`完整队列 | `runs/*/metrics.json` |
| 风险结构对知识变化的反应 | OI/OE/AE各5扰动seed的错配/删除曲线，RAUC | `runs/*/noise_*.json`、`robustness_summary.json` |
| 图选择与容量 | 各类型期望边数、预算乘子、风险项；capacity组识别 | `structure.json`、`runs/*/metrics.json` |
| 计算成本 | 五折教师、测量、效用拟合、参考教师与适配的耗时及head forwards | `history.json`、测量文件、效用选择记录 |

主识别权重按VAL CW `AUC_native`选择；校准bias按VAL exact-HM选择，温度按VAL NLL选择。知识扰动沿用干净VAL工作点。`report`生成`summary.csv`与`paired_comparisons.json`。

## 本次归档状态

本机数学、合成流程和CUDA验证见[results](../results/README.md)。正式UT结果由[服务器队列](../服务器实践命令.md)生成；后续结果按日期建独立批次目录，保留配置、protocol、逐epoch日志、选中权重、指标与原始导出映射。
