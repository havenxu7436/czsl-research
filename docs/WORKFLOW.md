# 后续实验如何接着记录

每次有实质修改，以 `docs/stages/` 的阶段卡为例创建新记录。依次填写：上一阶段证据 → 本次假设 → 改动的公式/代码 → 对照与配置 → 实际执行状态 → 全部结果 → 分析及下一步。至少明确数据集、seed、预算、VAL/TEST、CW/OW、选模规则、checkpoint 身份、知识版本和是否复用旧 run。

失败、中断、负结果、未训练方案都保留，状态使用“已训练 / 诊断回放 / 工程失败 / 仅交付代码 / 计划未执行”。同一 run 的续训不算另一独立 seed；复用结果不当作新训练。指标值应带单位，差分用百分点 pp，标注原始 CSV/JSON 路径。

建议把新实验写入 `output/<日期>_<路线>_<实验名>/`，包括 `protocol.json`、配置、`metrics.csv`、日志、`analysis.md` 和来源哈希。大型训练权重仍保留服务器并登记位置和身份。

```powershell
# 写完一个完整、可追溯的修改或实验记录后
git status
git add docs output/<本次结果目录> <本次代码目录>
git diff --cached --stat
git commit -m "experiment: describe the actual change and result"
git push
```

需要独立开发新想法时：

```powershell
git switch -c codex/<实验名>
# 修改、实验、记录，核对提交文件
git add <具体路径>
git commit -m "experiment: ..."
git push -u origin HEAD
```

归档完整性检查：

```powershell
python archive_tools/archive.py verify
```

该检查验证本次冻结的文件清单和新导航链接，不启动训练。以后故意修改已冻结来源时，保留当前归档标签，并对新版本建立新的 manifest；不要静默改旧结果。

历史脚本中可能含原电脑/服务器绝对路径；新生成的 `docs/` 导航已转换为仓库相对链接。旧源码、原报告和哈希清单保持原字节，运行历史脚本前须根据其协议检查外部输入。
