# 配置与知识

| 文件 | 用途 |
|---|---|
| [default.json](trustcpf_clip_v1/configs/default.json) | 768维CLIP、batch64、教师10epoch、适配3epoch、三效用模型、16支持摘要/推理图 |
| [semantic.json](trustcpf_clip_v1/configs/semantic.json) | 纯语义效用输入 |
| [unmeasured.json](trustcpf_clip_v1/configs/unmeasured.json) | 每类型留出20%边，检验未测关系迁移 |
| [experiment_matrix.json](experiment_matrix.json) | 七组×三seed主矩阵与附加队列 |
| [labels.json](trustcpf_clip_v1/assets/labels.json) | 属性/对象轴及组合划分 |
| [names.json](trustcpf_clip_v1/assets/names.json) | CLIP名称映射 |
| [path_aliases.json](trustcpf_clip_v1/assets/path_aliases.json) | UT图像路径映射 |
| [historical_inputs.json](trustcpf_clip_v1/assets/historical_inputs.json) | 原始metadata与split哈希 |
| [assets/knowledge/](trustcpf_clip_v1/assets/knowledge/) | reviewed-v2 OI/OE/AE，12/36/48条候选关系 |

主矩阵的每个seed先完成五折教师与效用学习。`shuffled`另建work完成全流程；噪声队列在选中权重上执行关系错配和删除。全部原始JSON与发布包保持一致。
