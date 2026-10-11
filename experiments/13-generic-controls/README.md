# 13 · Generic同结构知识对照＋25%混合

状态：三项新训练＋G_K复用。

## 修改与结论

控制容量和读出，只改名称、正确和两置换内容，区分工程增益与正确知识贡献。

四组CW均提升，内容归因未成立；不能把复用G_K列为第四项新训练。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 13 --dest /srv/czsl-reproduce-13
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-13
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

复用第 12 阶段正确知识组，三项对照另行训练；须先核对对应完整父批次与缓存：

```bash
WORK=/srv/czsl-runs/generic-controls-new
"$CPF_PY" "$REPLAY/generic_mixture_controls_20261009/run_controls.py" \
  --work "$WORK" --export "$WORK.zip"
```

执行包：[generic_mixture_controls_20261009](<code/generic_mixture_controls_20261009>)。

运行时入口：`/srv/czsl-reproduce-13/generic_mixture_controls_20261009/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../12-conditional-pilot/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[labels.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/labels.json>)。

原始结果：[summary.json](<results/extracted/controls/mechanics/attempt_0000/summary.json>)、[summary.json](<results/extracted/controls/summary.json>)。
