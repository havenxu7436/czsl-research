# 05 · 代码与修订

原 D 训练包源码位于 [vgcpod_v1_roundD](vgcpod_v1_roundD/)，补充轮启动器位于 [vgcpod_next_v1_20261020](vgcpod_next_v1_20261020/)。配置、资产、原说明和验证记录分区保存，通过仓库 materialize 恢复原布局。C1/C2 使用 [04 原 C 源码](../../04/code/README.md)。

| 入口 | 具体内容 |
|---|---|
| [模型与损失](vgcpod_v1_roundD/vgcpod/model.py) | 与 C 版本相同，沿用中心化门控和 β 归一化目标 |
| [训练与评价](vgcpod_v1_roundD/vgcpod/engine.py) | 与 C 版本相同，保留原 checkpoint 规则、优化器、日志和预测导出 |
| [D1配置](../configs/round_d/D1.json) | 相对 C2 只改变 rho 和 experiment_id |
| [均值门控评估](vgcpod_v1_roundD/mean_gate_eval.py) | TRAIN 全集均值／方差、动态与固定均值条件、独立 VAL bias |
| [首轮D调度](vgcpod_v1_roundD/round_d.py) | 两张卡调用原 C 补 C2，一张卡训练 D1；训练后进行均值替换 |
| [补充轮调度](vgcpod_next_v1_20261020/next_round.py) | D1 两个新 seed、C1/3352，自动生成同 seed 交互差 |
| [首轮tmux](vgcpod_v1_roundD/scripts/start_round_d_tmux.sh)／[补充轮tmux](vgcpod_next_v1_20261020/start_tmux.sh) | 当前环境绝对 Python 路径、日志、退出码、任务复用与续训 |
| [D组测试](vgcpod_v1_roundD/tests/test_round_d.py) | 四项新增检查，累计 30 项测试 |

## 已实施的代码改动

- `train_gate_moments` 对每个物体、模式计算 TRAIN 图像加权均值和总体方差，使用 float64 累加，不使用图像标签。
- `mean_gate_model` 复用原基础分数和物体概率，用固定 `[O,R]` 均值系数重算条件 logits、条件概率和联合概率。
- `mean_gate_eval.py` 从指定的原训练源目录导入模型，原运行保持原输出，新增预测和曲线写到独立 mean_gate 目录。
- `round_d.py` 增加 C/D 跨源码调度、三 seed 汇总和动态／均值差异汇总；同 seed、同缓存进行配对。
- `next_round.py` 复用原 D 的训练／评估函数和原 C CLI，各任务独占物理 GPU；汇总两个完整四格的交互差。

核心均值替换对应代码：

```python
gamma = model(g, fa, fo)['gamma'].double()
total += gamma.sum(0)
count += len(gamma)
mean_gamma = total / count

modes = (g[:, None, :] @ model.Wc) * mean_gamma[None] * v[None]
cond = base + rho * kappa * torch.einsum('bor,ir->boi', modes, u)
log_a = cond.log_softmax(-1)
joint = log_o[:, :, None] + log_a
```

原实现包含初始化、设备与 dtype 转换，详见源文件。两条件各自在自身 VAL 上选 bias；TEST 使用对应固定值。

## 源码、验证与复算

- [相对04补丁](source_changes.patch)包含新配置、评估、测试和调度。D 的 `vgcpod/*.py` 与 `run.py` 和 04 相同，核对结果见[服务器源码身份](../results/server_source_identity.json)。
- [30项测试日志](../results/package_validation/D/validation/unit_tests.txt)、[首轮CPU流程](../results/package_validation/D/validation/result.json)、[首轮三进程日志](../results/package_validation/D/validation/cli_output.txt)。
- [补充轮CPU验证](../results/package_validation/next/validation.json)、[重复启动日志](../results/package_validation/next/validation_cli.txt)：三项训练和两项均值评估，重复启动时分别跳过。
- [原分析代码](original_analysis/)、[仓库复算](../analysis/recompute.py)、[bias复算](../analysis/recompute_bias.py)。
- [D包成员映射](../package_members.json)、[补充轮成员映射](../next_package_members.json)、[完整原包](../results/archives/)。

CLIP 许可保留于 [CLIP_LICENSE](vgcpod_v1_roundD/vendor/CLIP_LICENSE)。服务器使用 Python 3.8.18、PyTorch 1.13.1+cu117、RTX 3090；原本地数学和调度验证使用 CPU 合成特征缓存。E1 ρ=0.15 及双 native/exact 最优权重保存为下一轮计划。
