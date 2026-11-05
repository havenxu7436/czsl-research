# 配置与知识

| 文件 | 用途 |
|---|---|
| default.json（精简版未保留） | 768维CLIP、batch64、教师10epoch、适配3epoch、三效用模型、16支持摘要/推理图 |
| semantic.json（精简版未保留） | 纯语义效用输入 |
| unmeasured.json（精简版未保留） | 每类型留出20%边，检验未测关系迁移 |
| experiment_matrix.json（精简版未保留） | 七组×三seed主矩阵与附加队列 |
| labels.json（精简版未保留） | 属性/对象轴及组合划分 |
| names.json（精简版未保留） | CLIP名称映射 |
| path_aliases.json（精简版未保留） | UT图像路径映射 |
| historical_inputs.json（精简版未保留） | 原始metadata与split哈希 |
| assets/knowledge/（精简版未保留） | reviewed-v2 OI/OE/AE，12/36/48条候选关系 |

主矩阵的每个seed先完成五折教师与效用学习。`shuffled`另建work完成全流程；噪声队列在选中权重上执行关系错配和删除。全部原始JSON与发布包保持一致。

## 2026-10-22 实际运行与下一轮规格

- 22个终端模型运行清单（精简版未保留）保存选中epoch、指标和checkpoint位置。
- 工作目录最终配置快照（精简版未保留）按原字节保存；各运行的实际训练配置随对应checkpoint保存。
