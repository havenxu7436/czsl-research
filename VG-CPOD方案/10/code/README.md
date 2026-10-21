# 10 · 代码与修订

原I包源码位于[vgcpod_v1_roundI](vgcpod_v1_roundI/)。配置、资产、原说明和发布验证按平行结构保存；materialize恢复原布局。相对[09 H组](../../09/code/README.md)的完整差异见[源码补丁](source_changes.patch)。

## 1. 原始残差分支与共享均值

[model.py](vgcpod_v1_roundI/vgcpod/model.py)在原`train_residual`路径旁新增`train_centered`：

```python
if self.gate_input == 'train_residual':
    gate_input = F.normalize(fo - self.gate_fo_mean[None], dim=-1,
                             eps=c.get('gate_residual_eps', 1e-12))
elif self.gate_input == 'train_centered':
    gate_input = fo - self.gate_fo_mean[None]
```

两种分支都使用固定`gate_fo_mean`buffer；后续sigmoid、η=0.25模式中心化和全局g的CP读出相同。I2只通过配置把`tau_gamma`设为0.56。

[engine.py](vgcpod_v1_roundI/vgcpod/engine.py)的模型构造条件扩展为接受`train_residual`和`train_centered`，两种输入都调用原均值加载函数。[gate_features.py](vgcpod_v1_roundI/vgcpod/gate_features.py)仅将H1专用提示改为TRAIN中心化通用提示，统计与加载计算沿用H版本；[mean_gate_eval.py](vgcpod_v1_roundI/mean_gate_eval.py)逐字沿用H组。checkpoint保存并恢复固定buffer，native／exact评价直接读取保存的均值。

## 2. 三GPU队列、双选择和诊断

- [round_i.py](vgcpod_v1_roundI/round_i.py)：优先逐字复用H组的同缓存TRAIN均值；生成I1/I2运行配置。GPU1/2/3分别执行seed3351/3352/3353，每队列依次训练I1、I2，再执行四项均值评估。
- [tmux启动](vgcpod_v1_roundI/scripts/start_round_i_tmux.sh)：保存激活环境Python、会话、日志和退出码。完成任务直接复用，未完成训练从最后完整epoch继续原十轮日程。
- `summarize()`：native／exact分别比较I1/I2−H1/G0/G1、I1−I2，汇总支持损失、逐组合变化和dynamic−mean。
- [ranking_diagnostics.py](vgcpod_v1_roundI/ranking_diagnostics.py)：纳入I组的OW预测，按all/seen/unseen统计物体、条件属性、支持概率及未见候选排序。

| 内容 | 仓库入口 |
|---|---|
| 两项配置模板 | [I1](../configs/round_i/I1.json) · [I2](../configs/round_i/I2.json) |
| 原服务器运行配置 | [runtime](../configs/runtime/) |
| 29项历史native及H1/G0/G1 exact参考 | [pre_i_complete.json](../configs/references/pre_i_complete.json) |
| 源码差异与服务器身份 | [补丁](source_changes.patch) · [身份核对](../results/server_source_identity.json) |
| 原包布局恢复 | [package_members.json](../package_members.json) |

## 3. 发布验证与仓库复算

- [48项测试](../results/package_validation/I/validation/unit_tests.txt)、[结构化摘要](../results/package_validation/I/validation/unit_test_summary.json)、[六训练十二评估流程](../results/package_validation/I/validation/result.json)。
- [四项I组测试](vgcpod_v1_roundI/tests/test_round_i.py)：配置及参数数量、显式门控／CP和有限差分梯度、代表尺度等价与幅值响应、I1续训和固定均值恢复。
- 原H1路径在新旧核心上的参数、AdamW和Torch RNG逐元素一致；CPU合成流程重复启动跳过六项训练和十二项评估。
- [正式结果复算](../analysis/recompute.py)、[原分析源码](original_analysis/)、[仓库验证](../analysis/repository_validation.json)。

正式训练使用Python3.8.18、PyTorch1.13.1+cu117、NumPy1.24.4与RTX3090。发布流程验证使用CPU合成缓存；本阶段训练结果来自服务器CLIP缓存。CLIP许可见[CLIP_LICENSE](vgcpod_v1_roundI/vendor/CLIP_LICENSE)。
