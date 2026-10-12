# 13 · Generic 知识对照与 25% 混合

状态：三组训练与一组权重复用已完成。

## 实验设置

- 同结构 Generic head；内容对照包括名称、正确知识与两份固定置换。
- 三项对照新训练；G_K 复用第 12 阶段权重；固定 25% 推理混合。

## 流程

1. 核对缓存、初始化与匹配训练顺序。
2. 训练三项对照并读取 G_K 权重。
3. 计算四组 CW/OW 指标与内容配对差。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-13
python tools/reproduce.py materialize --line 前期实验 --stage 13 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/generic_mixture_controls_20261009/run_controls.py" \
  --work "$WORK" --export "$WORK.zip"
```

## 结果与文件

四组 CW 指标均有增量；正确知识相对内容对照的归因比较见原始汇总。

- [执行代码](code/generic_mixture_controls_20261009/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
- [results/extracted/controls/mechanics/attempt_0000/summary.json](<results/extracted/controls/mechanics/attempt_0000/summary.json>)
- [results/extracted/controls/summary.json](<results/extracted/controls/summary.json>)
