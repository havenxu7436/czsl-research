# 11 · B/F梯度复核与条件属性 StageA

状态：诊断/缓存/丢弃式检查；非正式pilot。

## 修改与结论

检查真实TRAIN图像的逻辑梯度，构建C_text缓存并检查条件head接线。

弱且部分冲突的逻辑信号促成直接条件目标；StageA自检步数不当正式训练。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 11 --dest /srv/czsl-reproduce-11
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-11
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

前序权重和冻结路径核对完成后，StageA 进行梯度、输入与缓存预检，不计入新训练量：

```bash
WORK=/srv/czsl-runs/conditional-preflight-new
"$CPF_PY" "$REPLAY/conditional_attribute_stageA_20261009/runner.py" \
  --work "$WORK" --export "$WORK.zip"
```

执行包：[conditional_attribute_stageA_20261009](<code/conditional_attribute_stageA_20261009>)。

运行时入口：`/srv/czsl-reproduce-11/conditional_attribute_stageA_20261009/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../10-cpf-logiczsl/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[assets_sha256.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/assets_sha256.json>)、[bank.pt](<../../knowledge/ut-zap50k/conditional-cache/stage11/bank.pt>)、[bank_manifest.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/bank_manifest.json>)、[cache_manifest.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/cache_manifest.json>)、[candidates.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/candidates.json>)、[candidates__20fd4730a80c.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/candidates__20fd4730a80c.json>)、[code_manifest.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/code_manifest.json>)、[config_index.json](<../../knowledge/ut-zap50k/conditional-cache/stage11/config_index.json>)；完整清单见 inputs.json。

配置文件：[config_protocol.yml](<configs/extracted/bf/runs/A_ref/config_protocol.yml>)、[config_requested.yml](<configs/extracted/bf/runs/A_ref/config_requested.yml>)、[config_eval_effective.yml](<configs/extracted/bf/runs/A_ref/eval_val_cw/config_eval_effective.yml>)、[config_eval_effective.yml](<configs/extracted/bf/runs/A_ref/eval_val_ow/config_eval_effective.yml>)；已生成的运行配置和 protocol 也在结果中，完整文件见 inputs.json。
