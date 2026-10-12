# 12 · 条件属性 pilot r2

状态：五组×10 epoch 与固定混合评价已完成。

## 实验设置

- 目标：GT-object 条件属性交叉熵；对象边缘沿用原模型。
- 模型输入：第 11 阶段固定缓存；读出包含固定 25% 混合。

## 流程

1. 读取 StageA 缓存与冻结配置。
2. 完成五组条件属性 head 训练。
3. 计算原读出与固定 25% 混合读出的指标。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-12
python tools/reproduce.py materialize --line 前期实验 --stage 12 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/conditional_attribute_pilot_20261009/pilot.py" \
  --work "$WORK" --export "$WORK.zip"
```

## 结果与文件

训练与 seen/unseen 指标、混合读数保存在 summary.json。

- [执行代码](code/conditional_attribute_pilot_20261009/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/extracted/pilot/mechanics/attempt_0000/summary.json](<results/extracted/pilot/mechanics/attempt_0000/summary.json>)
- [results/extracted/pilot/summary.json](<results/extracted/pilot/summary.json>)
