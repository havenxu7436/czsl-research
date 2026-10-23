# TRUST-CPF 分析

## 2026-10-22 · 完整服务器结果

**[完整结果分析](server-results-20261022/README.md)** · [全部数据与权重](../results/server-results-20261022/README.md) · [重算入口](server-results-20261022/REPRODUCE.md)


| 模型 | TEST CW AUC | TEST OW AUC |
| --- | --- | --- |
| `r0` | 2.020 ± 2.232 | 2.010 ± 2.215 |
| `capacity` | 13.902 ± 0.885 | 11.240 ± 1.065 |
| `dense` | 11.169 ± 2.488 | 9.214 ± 2.309 |
| `drop` | 10.915 ± 1.545 | 9.047 ± 1.257 |
| `free` | 10.896 ± 1.519 | 9.011 ± 1.319 |
| `mean` | 10.080 ± 2.102 | 8.249 ± 1.692 |
| `robust` | 9.901 ± 1.890 | 8.172 ± 1.515 |

![主实验](server-results-20261022/confirmed/figures/01_main_auc.png)

![完整 OW 噪声曲线](server-results-20261022/confirmed/figures/03_noise_ow_curves.png)

完整 OW 噪声记录从日志恢复，原 JSON 的覆盖情况及同样本未测边比较均在完整分析中说明。效用审计与误差分解图保存在 [figures](server-results-20261022/confirmed/figures/)，全部逐 seed 表格保存在 [confirmed](server-results-20261022/confirmed/)。

## 原实验定义

[核心指标与分析协议 v1](protocol-v1.md)保留训练前的指标定义；[本机与仓库验证](../results/README.md)保留既有数学、CUDA 和公式记录。
