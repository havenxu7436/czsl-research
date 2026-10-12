# 11 · 条件属性 StageA 与缓存

状态：梯度检查与缓存构建已完成。

## 实验设置

- 输入：CPF B/F 逻辑分支与 C_text 模型。
- 检查对象：TRAIN 图像梯度、条件属性 head 与分数接口。

## 流程

1. 计算 B/F 逻辑梯度与任务梯度统计。
2. 构建固定图像特征与 C_text 分数缓存。
3. 检查条件属性 head 的输出、梯度和输入顺序。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-11
python tools/reproduce.py materialize --line 前期实验 --stage 11 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/conditional_attribute_stageA_20261009/runner.py" \
  --work "$WORK" --export "$WORK.zip"
```

## 结果与文件

StageA 的缓存与检查记录作为条件属性实验输入。

- [执行代码](code/conditional_attribute_stageA_20261009/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
