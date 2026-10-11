# 知识资产与版本

本目录保存原始知识与缓存的真实文件，不用同名文件覆盖不同版本。

| 分区 | 内容 | 使用边界 |
|---|---|---|
| `ut-zap50k/`、`mit-states/`、`cgqa/` | 对应数据集的描述、层级、向量及缓存 | 标签顺序、维度与划分不能跨数据集复用 |
| `definitions/legacy-v1/` | 早期 OI/OE/AE、标签与来源记录 | 不是原作者遗失知识的恢复 |
| `definitions/reviewed-v2/` | 经 TRAIN 审查后冻结的描述 | 不与 v1 指标混用；不能使用 VAL/TEST 修订描述 |
| `text-cache/legacy-v1/` | 原 v1 GloVe 名称、描述、pair/prior 缓存 | 用原 manifest 校验输入和轴顺序 |
| `text-cache/reviewed-v2/` | v2 相关向量与缓存 | 按对应阶段 protocol 配对 |
| `logic/reconstructed-v1/` | 重建层级/逻辑知识及 bank | 不宣称 LLM feasibility 或 CLIP 描述编码已恢复 |
| `conditional-cache/stage11/` | 条件属性阶段的冻结输入 | 不是另一种描述来源，不能替代名称/描述字典 |

并非每个数据集都包含上表所有类型。具体实际文件见 [registry.json](registry.json)：每个资产有 SHA256、大小、来源别名和出现阶段。内容相同的副本只保存一份；不同内容即使文件名相同也用版本目录和哈希后缀区分。

`registry.json` 的阶段记录的是资产在归档中出现的位置，不等于所有消费它的阶段。后续 EKR 复用第 04 阶段知识，B/F 和两条 0601 路线仍复用各自协议冻结的早期资产；实际消费关系以各阶段 `inputs.json`、源码和原始 protocol/manifest 为准。

## 使用方法

1. 找到目标实验的 `inputs.json`，核对来源与前序依赖。
2. 用 `python tools/reproduce.py verify` 校验当前文件。
3. 用 `materialize` 恢复原包内文件名与位置；不要把带哈希后缀的展示文件直接塞到模型目录。
4. 再运行对应包的缓存/输入预检。`input_sha256` 中的原绝对路径仍须可达，详见 [环境说明](../environment/README.md)。不能为通过检查而删除哈希约束。

若在新路径重建知识或缓存，应保留新的构建输入与 manifest，作为独立复现记录；不能静默替换旧文件，再把历史性能当作新版本结果。
