# 15 · TRAIN逐句审查、知识来源与研究回溯

状态：36逐句视图＋图片审查；0新训练。

## 修改与结论

检查均值是否隐藏局部信号，保留研究目标与实现完成度的解释修正。

AE第三条有局部信号但净收益集中Leather；不把TRAIN审查当unseen泛化证据。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 15 --dest /srv/czsl-reproduce-15
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-15
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

先检查原 CLIP 探针来源与逐句输入，再创建新的 TRAIN 审查输出。
发布包与后续工作源码分别保留；执行使用恢复后的冻结发布包，避免混用 `local_test_report` 与 manifest：

```bash
WORK=/var/tmp/czsl_train_knowledge_audit_20261010
PROBE_WORK="${CZSL_CLIP_PROBE_WORK:?Set the verified stage-14 work directory}"
PACKAGE="$REPLAY/recovered_containers/train_knowledge_audit_release_20261010/train_knowledge_audit_20261010"
"$CPF_PY" "$PACKAGE/run.py" \
  --probe-work "$PROBE_WORK" --check-inputs-only
"$CPF_PY" "$PACKAGE/run.py" \
  --probe-work "$PROBE_WORK" \
  --work "$WORK" --export "$WORK.zip"
```

输出目录名由 protocol 固定；使用隔离环境，已有结果不得覆盖。

执行包：[train_knowledge_audit_20261010](<code/train_knowledge_audit_20261010>)。

冻结发布包入口见 `inputs.json` 的 `runtime_package_path`。历史说明与校验文件在独立运行目录恢复，不在当前版本重复堆放。

前序阶段：[README.md](<../14-clip-probe/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[ae_dict.json](<../../knowledge/ut-zap50k/definitions/reviewed-v2/ae_dict.json>)、[oe_dict.json](<../../knowledge/ut-zap50k/definitions/reviewed-v2/oe_dict.json>)。
