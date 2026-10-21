# 11 · 代码与修订

原运行源码位于[vgcpod_v1_roundJ](vgcpod_v1_roundJ/)，配置、资产、说明与验证按平行目录保存；[映射](../package_members.json)供materialize恢复原包。相对[10 I组](../../10/code/README.md)的全部差异见[source_changes.patch](source_changes.patch)。

## 1. 两路独立读出

[model.py](vgcpod_v1_roundJ/vgcpod/model.py)保留Wc作为静态Ws，新增零初始化Wd；原有参数的随机初始化序列保留。新分支的核心计算为：

```python
static_modes = (readout @ self.Wc) * v[None, :, :]
dynamic_modes = (readout @ self.Wd) * (gamma - 1) * v[None, :, :]
static = rho * kappa * torch.einsum('bor,ir->boi', static_modes, u)
dynamic = rho * kappa * torch.einsum('bor,ir->boi', dynamic_modes, u)
cond = base + static + dynamic
```

`cp_readout_mode="separate"`启用上述分支。I2配置沿用共享分支；J1输出额外保存`cp_static/cp_dynamic`。Au/Av共享、矩阵Wd新增24,576个参数，两路联合训练。

## 2. 均值评估与幅值日志

[mean_gate_eval.py](vgcpod_v1_roundJ/mean_gate_eval.py)先使用当前模型的静态输出，再通过Wd、TRAIN均值门控减1和原因子计算动态输出。这样两路矩阵都参与均值条件，最后重算条件softmax与联合分布。

[engine.py](vgcpod_v1_roundJ/vgcpod/engine.py)记录两路原始MSE和沿属性维中心化MSE，按图像数汇总后计算epoch RMS。日志是训练epoch各batch前向权重的统计。J1仍使用原adapter与CP两个AdamW参数组，Wc/Wd属于相同CP组；LR均为3e-4→3e-6。

## 3. 调度、配置与诊断

| 入口 | 内容 |
|---|---|
| [round_j.py](vgcpod_v1_roundJ/round_j.py) | 从I组复用TRAIN均值；三卡三个seed，完成训练及native/exact六项干预 |
| [tmux脚本](vgcpod_v1_roundJ/scripts/start_round_j_tmux.sh) | 当前Python、会话、日志与退出码；中断后从完整epoch继续 |
| [J1模板](../configs/round_j/J1.json) | 在I2基础上新增separate模式及实验编号 |
| [35项历史快照](../configs/references/pre_j_complete.json) | native及G0/G1/H1/I1/I2 exact指标 |
| [ranking_diagnostics.py](vgcpod_v1_roundJ/ranking_diagnostics.py) | J1−I2/G0/G1/H1/I1与dynamic−mean预测级比较 |
| [readout_training_summary](../results/raw_export/roundJ_20261021/readout_training_summary.csv) | native/exact选中epoch和末轮两路修正统计 |

## 4. 验证与正式结果复算

- [55项原发布测试](../results/package_validation/J/validation/unit_tests.txt)、[结构化摘要](../results/package_validation/J/validation/unit_test_summary.json)、[三训练六评估流程](../results/package_validation/J/validation/result.json)。
- [七项J组测试](vgcpod_v1_roundJ/tests/test_round_j.py)：参数计数与分组、I2恢复恒等式、显式低秩算子、五组参数有限差分、独立矩阵均值评估、首步双路学习和续训。
- 旧I2配置在新旧训练器中的参数、AdamW及Torch RNG逐元素相同；CPU流程重复启动跳过三训练和六评估。
- [正式数据复算](../analysis/recompute.py)、[原分析脚本](original_analysis/)、[仓库验证](../analysis/repository_validation.json)、[服务器源码身份](../results/server_source_identity.json)。

正式训练的软件与GPU身份保存在各protocol.json；发布验证使用CPU合成缓存。CLIP许可见[CLIP_LICENSE](vgcpod_v1_roundJ/vendor/CLIP_LICENSE)。公式推导与当前结果见[主说明](../README.md)，完整解释见[数据分析](../analysis/README.md)。
