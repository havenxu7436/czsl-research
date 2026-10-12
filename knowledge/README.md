# 知识资产与版本

知识按数据集、类型与版本组织。标签轴顺序、缓存维度与数据划分记录在对应 manifest 中。

| 分区 | 内容 |
|---|---|
| `ut-zap50k/`、`mit-states/`、`cgqa/` | 各数据集的描述、层级、向量与缓存 |
| `definitions/legacy-v1/` | 早期 OI/OE/AE、标签与来源记录 |
| `definitions/reviewed-v2/` | TRAIN 审查后的 AE/OE 与逐句记录 |
| `text-cache/legacy-v1/` | v1 GloVe 名称、描述、pair/prior 缓存 |
| `text-cache/reviewed-v2/` | v2 向量与缓存 |
| `logic/reconstructed-v1/` | 重建层级、逻辑知识与 bank |
| `conditional-cache/stage11/` | 条件属性阶段的冻结输入 |

各数据集的实际文件见 [registry.json](registry.json)，其中记录 SHA256、大小、来源别名与归档阶段。
相同内容合并存储；同名异内容使用版本目录与哈希后缀区分。

## 实验对应关系

第 04–09 阶段使用 reviewed-v2 相关资产；B/F 与两条 0601 路线使用各自 protocol 指定的资产。
阶段 `inputs.json` 索引知识文件，protocol/manifest 记录实际输入与顺序。

## 加载流程

1. 根据阶段 `inputs.json` 定位资产。
2. `python tools/reproduce.py verify` 检查文件哈希。
3. `materialize` 恢复原包内文件名与路径。
4. 对应执行包读取 manifest、构建或加载 cache/bank。

固定输入路径与工程目录的对应关系见 [环境说明](../environment/README.md)。
