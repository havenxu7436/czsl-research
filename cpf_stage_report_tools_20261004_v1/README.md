# CPF 阶段报告工具 v1

只读取现有 MVP 和 lambda 后处理，输出新报告目录。支持 Python 3.8+。

- `collect_stage_report.py`：标准库汇总；PyYAML用于完整A/C配置核验。
- `analyze_bias0_errors.py`：numpy，逐图bias=0对齐与修正/新增错误分析。
- `plot_stage_report.py`：matplotlib，单seed图不绘制误差棒。
- `historical_baseline_reference.json` 与 `reference_history/`：本机已归档test结果，独立附表。

运行流程见工作区的《截至AC跨数据集验证_实验报告记录与分析流程.md》。不导入torch，不训练，不重评，不复制checkpoint或大型scores.dat，不覆盖已有输出。

任何 `COMPLETE: False`、`not_comparable`、`metadata_only` 或 issues 条目均需解释；不能为了通过检查修改原数据。

