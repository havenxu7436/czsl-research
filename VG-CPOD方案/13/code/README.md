# L组代码与全部修订

[执行入口](vgcpod_v1_roundL/run.py) · [模型](vgcpod_v1_roundL/ckcpf/model.py) · [训练与续训](vgcpod_v1_roundL/ckcpf/runtime.py) · [评价](vgcpod_v1_roundL/ckcpf/evaluation.py) · [AUC算法](vgcpod_v1_roundL/ckcpf/metrics.py) · [数据与缓存轴](vgcpod_v1_roundL/ckcpf/data.py)

| 修订内容 | 最终实现与目的 |
| --- | --- |
| 完整保留G0静态CP | 在条件属性项增加真实／打乱原型的差分残差，恢复强共同基座 |
| 固定Bo/Co/Au/Av | 同时保留物体评分和primitive坐标，三组共同续训Bc/Cc/Wc |
| 分组学习率 | 已有条件头2e-4、CP读出6e-5；新增消息3e-4、alpha1e-3 |
| 按类型平均96条关系 | OI12/OE36/AE48；Shuffle保持目标、类型、度数和source多重集 |
| 初始残差为零 | alpha=0、右消息矩阵为0；训练起点还原对应G0 |
| 续训与完成标记 | 优化器/调度/随机状态恢复，最终评价和预测导出后写DONE及metrics哈希 |
| 完整预测与校准 | 固定VAL选定权重、bias、温度，保存CW/OW和全部样本轴 |
| 三卡队列与跨平台打包 | GPU1/2/3各负责一个seed；Python UTF-8解压和哈希核验 |
| 归档重命名 | 仅将pack_results的固定旧包名改为实际目录名，科学运算未改动 |

最后一项的精确补丁见[source_changes.patch](source_changes.patch)；逐文件来源与差异见[server_source_identity.json](../results/server_source_identity.json)。[原始服务器执行源码](../results/raw_export/trustcpf_conditional_v2/ckcpf/)和归档ZIP均保持原字节。

[五项模型/损失/评价测试](vgcpod_v1_roundL/tests/test_experiment.py) · [原G0对照](vgcpod_v1_roundL/tests/reference_g0.py) · [完整公式及核心实现](../FORMULAS_AND_CODE.md)。分析重算脚本位于[analysis/](../analysis/README.md)。
