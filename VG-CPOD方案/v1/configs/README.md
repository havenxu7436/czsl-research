# v1 配置

- [冻结包默认配置](vgcpod_v1/configs/default.json)：恢复运行时的 `configs/default.json`。
- [9组派生配置预览](variants/)：默认参数分别设置 V0、V1、V1L、V2–V7，未经过训练。
- [完整矩阵索引](experiment_matrix.json)：pilot、full、敏感性及选模口径。
- [标签、名称与输入身份](vgcpod_v1/assets/)：冻结 CLIP 名称映射与历史输入哈希。

正式脚本读取恢复包的默认配置并显式传入 `--variant`、`--seed`。`run.py` 的 `--variant` 默认值会覆盖配置中的同名字段；手动使用预览配置时同时指定该参数。配置预览不替代正式运行生成的 `protocol.json`。
