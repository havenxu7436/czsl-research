# CPF 阶段报告工具 v2

只读取现有 MVP 和 lambda 后处理，输出新报告目录。支持 Python 3.8+。

- `collect_stage_report.py`：标准库汇总；PyYAML用于完整A/C配置核验。
- `analyze_bias0_errors.py`：numpy，逐图bias=0对齐与修正/新增错误分析。
- `plot_stage_report.py`：matplotlib，单seed图不绘制误差棒。
- `knowledge_evidence.py`：从cache冻结输入hash追溯OI/OE/AE、标签、manifest与审核记录，复制小文件并输出结构/schema。
- `知识文件结构与模型使用说明.md`：真实本机draft示例、JSON/CSV字段、目录树、缓存形状及当前模型消费路径。
- `historical_baseline_reference.json` 与 `reference_history/`：本机已归档test结果，独立附表。

运行流程见工作区的《截至AC跨数据集验证_实验报告记录与分析流程.md》。不导入torch，不训练，不重评，不复制checkpoint或大型scores.dat，不覆盖已有输出。

任何 `COMPLETE: False`、`not_comparable`、`metadata_only` 或 issues 条目均需解释；不能为了通过检查修改原数据。

默认从`--cache-root`下三份cache manifest的`input_sha256`定位实际源目录。源目录搬迁时可加`--knowledge-root /真实源文件父目录`；仍须逐文件匹配冻结hash。输出`tables/knowledge_structure.csv`、`knowledge_schema.json`及`evidence/knowledge_source/`。来源CSV未冻结在builder输入时标为补充记录，不把bulk approved解释成人工审核。工具不加载PT文件，张量形状说明为manifest合同。

