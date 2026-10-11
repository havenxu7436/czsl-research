# 12 · 条件属性 pilot r1→r2与固定混合回放

状态：r1工程失败；r2五组10轮＋回放。

## 修改与结论

改为GT-object条件属性目标，保持对象边缘；修复Torch1.13安全checkpoint兼容，另做固定25%混合回放。

已见拟合与unseen迁移分开；固定读出回放不是新训练；保留失败r1。

## 复现顺序

1. 按 [统一环境与外部输入说明](../../environment/README.md) 准备数据、GloVe、预训练模型。
2. 在仓库根目录校验并生成独立运行目录；此步骤不训练、不覆盖已有目录。

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 12 --dest /srv/czsl-reproduce-12
```

3. 运行目录保留历史包内相对位置及哈希。基础工程位于 `project/`，执行包位于运行目录根部。按统一环境说明处理冻结路径和外部输入；路径迁移不能绕过原哈希检查。
4. 优先使用包的 `--root`、`--work` 或环境变量配置路径，先预检/prepare，再执行本阶段训练或诊断。检查结果的 dataset、seed、VAL/TEST、CW/OW、选模与复用身份。
5. [inputs.json](inputs.json) 登记本阶段资产、配置与结果；[runbook.json](runbook.json) 提供去重的实验命令模板及来源；模板路径变量和包内默认路径均须核对。

## 命令变量

以下命令从独立运行目录读取文件。先设置匹配的 Python 与已验证的工程根目录；
历史包内部的默认路径仍须按环境说明处理，不能只替换命令中的变量。

```bash
REPLAY=/srv/czsl-reproduce-12
export CPF_PY="${CZSL_PYTHON:?Set the matching CPF Python executable}"
export PROJECT="${CZSL_PROJECT_ROOT:?Set a verified compatible project root}"
```

## 关键入口

原 r1 失败记录与 r2 结果分开。执行前核对 StageA 来源、知识哈希和输入路径：

```bash
WORK=/srv/czsl-runs/conditional-pilot-new
"$CPF_PY" "$REPLAY/conditional_attribute_pilot_20261009/pilot.py" \
  --work "$WORK" --export "$WORK.zip"
```

执行包：[conditional_attribute_pilot_20261009](<code/conditional_attribute_pilot_20261009>)。

运行时入口：`/srv/czsl-reproduce-12/conditional_attribute_pilot_20261009/`。其原始 README/哈希校验所需说明会从不可变归档标签恢复到运行目录，不在当前仓库重复堆放。

前序阶段：[README.md](<../11-conditional-preflight/README.md>)。阶段依赖不等同于每一步都必须重新训练；续训/回放需要对应原权重。

知识版本：[labels.json](<../../knowledge/ut-zap50k/definitions/legacy-v1/labels.json>)。

原始结果：[summary.json](<results/extracted/pilot/mechanics/attempt_0000/summary.json>)、[summary.json](<results/extracted/pilot/summary.json>)。
