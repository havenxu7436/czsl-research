# 10 · 代码与修订

原I包源码位于vgcpod_v1_roundI（精简版未保留）。配置、资产、原说明和发布验证按平行结构保存；历史打包流程恢复原布局。相对[09 H组](../../09/code/README.md)的完整差异见源码补丁（精简版未保留）。

## 1. 原始残差分支与共享均值

model.py（精简版未保留）在原`train_residual`路径旁新增`train_centered`：

```python
if self.gate_input == 'train_residual':
    gate_input = F.normalize(fo - self.gate_fo_mean[None], dim=-1,
                             eps=c.get('gate_residual_eps', 1e-12))
elif self.gate_input == 'train_centered':
    gate_input = fo - self.gate_fo_mean[None]
```

两种分支都使用固定`gate_fo_mean`buffer；后续sigmoid、η=0.25模式中心化和全局g的CP读出相同。I2只通过配置把`tau_gamma`设为0.56。

engine.py（精简版未保留）的模型构造条件扩展为接受`train_residual`和`train_centered`，两种输入都调用原均值加载函数。gate_features.py（精简版未保留）仅将H1专用提示改为TRAIN中心化通用提示，统计与加载计算沿用H版本；mean_gate_eval.py（精简版未保留）逐字沿用H组。checkpoint保存并恢复固定buffer，native／exact评价直接读取保存的均值。

## 2. 三GPU队列、双选择和诊断

- round_i.py（精简版未保留）：优先逐字复用H组的同缓存TRAIN均值；生成I1/I2运行配置。GPU1/2/3分别执行seed3351/3352/3353，每队列依次训练I1、I2，再执行四项均值评估。
- tmux启动（精简版未保留）：保存激活环境Python、会话、日志和退出码。完成任务直接复用，未完成训练从最后完整epoch继续原十轮日程。
- `summarize()`：native／exact分别比较I1/I2−H1/G0/G1、I1−I2，汇总支持损失、逐组合变化和dynamic−mean。
- ranking_diagnostics.py（精简版未保留）：纳入I组的OW预测，按all/seen/unseen统计物体、条件属性、支持概率及未见候选排序。

| 内容 | 仓库入口 |
|---|---|
| 两项配置模板 | I1（精简版未保留） · I2（精简版未保留） |
| 原服务器运行配置 | runtime（精简版未保留） |
| 29项历史native及H1/G0/G1 exact参考 | pre_i_complete.json（精简版未保留） |
| 源码差异与服务器身份 | 补丁（精简版未保留） · 身份核对（精简版未保留） |
| 原包布局恢复 | package_members.json（精简版未保留） |

## 3. 发布验证与仓库复算

- 48项测试（精简版未保留）、结构化摘要（精简版未保留）、六训练十二评估流程（精简版未保留）。
- 四项I组测试（精简版未保留）：配置及参数数量、显式门控／CP和有限差分梯度、代表尺度等价与幅值响应、I1续训和固定均值恢复。
- 原H1路径在新旧核心上的参数、AdamW和Torch RNG逐元素一致；CPU合成流程重复启动跳过六项训练和十二项评估。
- 正式结果复算（精简版未保留）、原分析源码（精简版未保留）、仓库验证（精简版未保留）。

正式训练使用Python3.8.18、PyTorch1.13.1+cu117、NumPy1.24.4与RTX3090。发布流程验证使用CPU合成缓存；本阶段训练结果来自服务器CLIP缓存。CLIP许可见CLIP_LICENSE（精简版未保留）。
