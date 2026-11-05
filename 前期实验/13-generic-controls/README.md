# 13 · Generic 知识对照与 25% 混合

状态：三组训练与一组权重复用已完成。

## 实验设置

- 同结构 Generic head；内容对照包括名称、正确知识与两份固定置换。
- 三项对照新训练；G_K 复用第 12 阶段权重；固定 25% 推理混合。

## 流程

1. 核对缓存、初始化与匹配训练顺序。
2. 训练三项对照并读取 G_K 权重。
3. 计算四组 CW/OW 指标与内容配对差。

## 结果与文件

四组 CW 指标均有增量；正确知识相对内容对照的归因比较见原始汇总。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
- results/extracted/controls/mechanics/attempt_0000/summary.json（精简版未保留）
- results/extracted/controls/summary.json（精简版未保留）
