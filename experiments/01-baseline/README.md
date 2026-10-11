# 01 · 原方案、三数据集基线与知识重建

状态：已归档历史实验。

## 修改与结论

以 CPF 为视觉底座，恢复 OI/OE/AE、名称、GloVe 与模板组合资产；先建立基线和输入身份。

历史 TEST 与后续 VAL 分开；知识重建不等于恢复原作者遗失资产。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 01 --dest /srv/czsl-reproduce-01
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-01
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

[基础源码](../../src/)、本阶段 `configs/` 和 `inputs.json`。不同 CPF/LogiCZSL 快照不可互换。

知识版本：[K_oa_text_lexical.csv](<../../knowledge/cgqa/definitions/legacy-v1/K_oa_text_lexical.csv>)、[K_oa_text_report.json](<../../knowledge/cgqa/definitions/legacy-v1/K_oa_text_report.json>)、[ae_dict.json](<../../knowledge/cgqa/definitions/legacy-v1/ae_dict.json>)、[attributes.txt](<../../knowledge/cgqa/definitions/legacy-v1/attributes.txt>)、[bulk_acceptance_record.json](<../../knowledge/cgqa/definitions/legacy-v1/bulk_acceptance_record.json>)、[cache_manifest.json](<../../knowledge/cgqa/definitions/legacy-v1/cache_manifest.json>)、[entry_provenance.csv](<../../knowledge/cgqa/definitions/legacy-v1/entry_provenance.csv>)、[high_risk_review.csv](<../../knowledge/cgqa/definitions/legacy-v1/high_risk_review.csv>)；完整清单见 inputs.json。

配置文件：[config_used.yml](<configs/CPF_baseline/CPF_cgqa_baseline/cpf_cgqa_seed3351_gpu2/config_used.yml>)、[conda_environment.yml](<configs/CPF_baseline/CPF_cgqa_baseline/cpf_cgqa_seed3351_gpu2/repro_meta/conda_environment.yml>)、[config_used.yml](<configs/CPF_baseline/CPF_cgqa_baseline/cpf_cgqa_seed3351_gpu2/repro_meta/config_used.yml>)、[config_used.yml](<configs/CPF_baseline/CPF_mit_states_baseline/cpf_mit_states_seed0/config_used.yml>)；已生成的运行配置和 protocol 也在结果中，完整文件见 inputs.json。
