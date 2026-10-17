# v2 · 结果与验证记录

状态：[seed3351正式训练结果](seed3351-10epoch-20261015/README.md)已归档：32组×10 epoch、128份评价。使用20261014修正版，原始score、标签、日志、配置和checkpoint身份记录均保留。

## 结果与文件

| seed | 每组实际 epoch | 训练组数 | 评价份数 | 结果入口 |
|---|---:|---:|---:|---|
| 3351 | 10 | 32 | 128 | [指标与训练记录](seed3351-10epoch-20261015/README.md) |

- [seed3351：128份完整评价指标](seed3351-10epoch-20261015/完整评价指标.md)
- [seed3351：32组逐轮训练记录](seed3351-10epoch-20261015/训练记录.md)
- [seed3351：原实验设计命名对照](seed3351-10epoch-20261015/实验命名对照.md)
- [seed3351：归档、来源与校验](seed3351-10epoch-20261015/归档与校验.md)

以上仅按02公式对齐版格式展示原始记录，未进行数据分析。

## 验证记录

以下34项CPU、123项release manifest和7046项恢复记录属于**原20261013发布包的历史验证**。修正版另存，随包CPU记录为40项通过；本次导入检查的是其文件身份及服务器导出证据，没有重新训练。

| 检查 | 结果 | 范围 |
|---|---|---|
| 本地 CPU 测试 | 34项通过 | 概率代数、梯度、synthetic模型/资产/配置与后分析接线 |
| 独立解压后的 CPU 测试 | 34项通过 | 同一ZIP的独立恢复执行检查 |
| 仓库分区恢复后的执行 | 7046项复制校验、原包123项manifest通过；34项CPU测试通过 | `materialize --line 0601方案 --stage v2` 实际恢复后的检查 |
| 原包完整性 | SHA256固定 | 原部署包不因仓库文档排版而重打包 |
| Python3.8语法、JSON/YAML、Bash语法 | 通过 | 不是torch1.13 CUDA运行保证 |
| 桌面/GitHub公式内容与KaTeX | 两版均28行内+17块公式，零解析错误 | 使用已安装VS Code内置KaTeX0.16.21；不是实际界面/截图检查完成声明 |
| 20261014修正版真实TRAIN图像/CUDA smoke、正式训练 | seed3351的32组通过smoke和optimizer/resume acceptance，均完成10轮 | 依据本次导出的smoke.json、train.log与train_complete.json；不是导入时新启动训练 |
| 本批导出完整性 | 1766份成员全部映射；1765项导出manifest及125项修正版release manifest通过 | 另核对344项冻结输入、50份执行source和128份同checkpoint评价，见本批audit.json |

- [本地记录](revised_0601_20261013/local_revision_test.json)
- [独立解压记录](独立解压CPU检查.json)
- [仓库materialize后的CPU记录](materialized_repo_cpu_check.json)
- [原release manifest](revised_0601_20261013/release_manifest.json)
- [原部署包](deployment/0601_unified_revision_20261013.zip)
- [公式内容、分隔符和排版一致性](math_format_check.json)
- [实际VS Code KaTeX解析记录](katex_render_check.json)
- [seed3351数据范围、完整文件入口与恢复方法](seed3351-10epoch-20261015/README.md)
- [本批完整性与训练证据核对](seed3351-10epoch-20261015/audit.json)

桌面版使用 `$...$` / `$$...$$`，GitHub版沿用 `$`加反引号的行内形式和 fenced `math`。长公式用 `aligned` 在原式分组边界换行，不改变数学内容。用户按Escape中止了computer-use界面检查，因此实际VS Code/GitHub窗口的视觉复核未完成；未继续操作窗口，不将解析通过冒充视觉复核完成。

本批结果与旧r2分开归档、分开统计；不得将旧30/10/10epoch结果或synthetic测试用数据写成本版训练结果。本次只归档原始数据，未重算指标或生成posthoc结论。
