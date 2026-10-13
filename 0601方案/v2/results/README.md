# v2 · 验证记录

状态：实现与配置已归档，正式训练结果未产生。不存在可汇报的本版AUC/HM。

| 检查 | 结果 | 范围 |
|---|---|---|
| 本地 CPU 测试 | 34项通过 | 概率代数、梯度、synthetic模型/资产/配置与后分析接线 |
| 独立解压后的 CPU 测试 | 34项通过 | 同一ZIP的独立恢复执行检查 |
| 仓库分区恢复后的执行 | 7046项复制校验、原包123项manifest通过；34项CPU测试通过 | `materialize --line 0601方案 --stage v2` 实际恢复后的检查 |
| 原包完整性 | SHA256固定 | 原部署包不因仓库文档排版而重打包 |
| Python3.8语法、JSON/YAML、Bash语法 | 通过 | 不是torch1.13 CUDA运行保证 |
| 桌面/GitHub公式内容与KaTeX | 两版均28行内+17块公式，零解析错误 | 使用已安装VS Code内置KaTeX0.16.21；不是实际界面/截图检查完成声明 |
| 真实TRAIN图像/CUDA smoke、正式训练 | 尚未执行 | 服务器实际运行后才能更新状态 |

- [本地记录](revised_0601_20261013/local_revision_test.json)
- [独立解压记录](独立解压CPU检查.json)
- [仓库materialize后的CPU记录](materialized_repo_cpu_check.json)
- [原release manifest](revised_0601_20261013/release_manifest.json)
- [原部署包](deployment/0601_unified_revision_20261013.zip)
- [公式内容、分隔符和排版一致性](math_format_check.json)
- [实际VS Code KaTeX解析记录](katex_render_check.json)

桌面版使用 `$...$` / `$$...$$`，GitHub版沿用 `$`加反引号的行内形式和 fenced `math`。长公式用 `aligned` 在原式分组边界换行，不改变数学内容。用户按Escape中止了computer-use界面检查，因此实际VS Code/GitHub窗口的视觉复核未完成；未继续操作窗口，不将解析通过冒充视觉复核完成。

后续三seed结果与旧r2分开归档、分开统计；不得将旧30/10/10epoch结果或synthetic测试用数据写成本版训练结果。
