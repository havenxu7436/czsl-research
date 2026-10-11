# 02 · 早期 MVP、跨数据集验证与固定 prior 校准

状态：已训练＋无训练后处理。

## 修改与结论

UT A/C/D/E 与 MIT、C-GQA A/C；固定 A/C checkpoint 扫 λ，以分离文本和静态 prior。

UT 文本有信号；后处理选择 λ=0。20 次评价不计20次新训练。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 02 --dest /srv/czsl-reproduce-02
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-02
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

执行包：[cpf_stage_report_tools_20261004_v2](<code/cpf_stage_report_tools_20261004_v2>)。

运行时入口：`/srv/czsl-reproduce-02/cpf_stage_report_tools_20261004_v2/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../01-baseline/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[K_oa_text_lexical.csv](<../../knowledge/cgqa/definitions/legacy-v1/K_oa_text_lexical.csv>)、[K_oa_text_report.json](<../../knowledge/cgqa/definitions/legacy-v1/K_oa_text_report.json>)、[ae_dict.json](<../../knowledge/cgqa/definitions/legacy-v1/ae_dict.json>)、[cache_manifest.json](<../../knowledge/cgqa/definitions/legacy-v1/cache_manifest.json>)、[labels.json](<../../knowledge/cgqa/definitions/legacy-v1/labels.json>)、[oe_dict.json](<../../knowledge/cgqa/definitions/legacy-v1/oe_dict.json>)、[oi_dict.json](<../../knowledge/cgqa/definitions/legacy-v1/oi_dict.json>)、[report.json](<../../knowledge/cgqa/definitions/legacy-v1/report.json>)；完整清单见 inputs.json。

配置文件：[config_eval_effective.yml](<configs/evidence/lambda_post/Dpost_l0.025_cw/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/evidence/lambda_post/Dpost_l0.025_ow/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/evidence/lambda_post/Dpost_l0.05_cw/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/evidence/lambda_post/Dpost_l0.05_ow/config_eval_effective.yml>)；已生成的运行配置和 protocol 也在结果中，完整文件见 inputs.json。

## 证据入口

以下链接指向归档原文件；训练、回放和诊断范围以上述阶段状态为准。

- [results/tables/epoch_curves.csv](<results/tables/epoch_curves.csv>)

<details>
<summary>模型定义、对照与协议边界</summary>

### CPF 阶段报告工具 v2

只读取现有 MVP 和 lambda 后处理，输出新报告目录。支持 Python 3.8+。

- `collect_stage_report.py`：标准库汇总；PyYAML用于完整A/C配置核验。
- `analyze_bias0_errors.py`：numpy，逐图bias=0对齐与修正/新增错误分析。
- `plot_stage_report.py`：matplotlib，单seed图不绘制误差棒。
- `knowledge_evidence.py`：从cache冻结输入hash追溯OI/OE/AE、标签、manifest与审核记录，复制小文件并输出结构/schema。
- `historical_baseline_reference.json` 与 `reference_history/`：已归档 TEST 结果，独立附表。

运行参数见 runbook.json 和对应脚本的 `--help`。不导入torch，不训练，不重评，不复制checkpoint或大型scores.dat，不覆盖已有输出。

任何 `COMPLETE: False`、`not_comparable`、`metadata_only` 或 issues 条目均需解释；不能为了通过检查修改原数据。

默认从`--cache-root`下三份cache manifest的`input_sha256`定位实际源目录。源目录搬迁时可加`--knowledge-root /真实源文件父目录`；仍须逐文件匹配冻结hash。输出`tables/knowledge_structure.csv`、`knowledge_schema.json`及`evidence/knowledge_source/`。来源CSV未冻结在builder输入时标为补充记录，不把bulk approved解释成人工审核。工具不加载PT文件，张量形状说明为manifest合同。

</details>
