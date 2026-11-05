# 11 · 代码与修订

原运行源码位于vgcpod_v1_roundJ（精简版未保留），配置、资产、说明与验证按平行目录保存；映射（精简版未保留）供历史打包流程恢复原包。相对[10 I组](../../10/code/README.md)的全部差异见source_changes.patch（精简版未保留）。

## 1. 两路独立读出

model.py（精简版未保留）保留Wc作为静态Ws，新增零初始化Wd；原有参数的随机初始化序列保留。新分支的核心计算为：

```python
static_modes = (readout @ self.Wc) * v[None, :, :]
dynamic_modes = (readout @ self.Wd) * (gamma - 1) * v[None, :, :]
static = rho * kappa * torch.einsum('bor,ir->boi', static_modes, u)
dynamic = rho * kappa * torch.einsum('bor,ir->boi', dynamic_modes, u)
cond = base + static + dynamic
```

`cp_readout_mode="separate"`启用上述分支。I2配置沿用共享分支；J1输出额外保存`cp_static/cp_dynamic`。Au/Av共享、矩阵Wd新增24,576个参数，两路联合训练。

## 2. 均值评估与幅值日志

mean_gate_eval.py（精简版未保留）先使用当前模型的静态输出，再通过Wd、TRAIN均值门控减1和原因子计算动态输出。这样两路矩阵都参与均值条件，最后重算条件softmax与联合分布。

engine.py（精简版未保留）记录两路原始MSE和沿属性维中心化MSE，按图像数汇总后计算epoch RMS。日志是训练epoch各batch前向权重的统计。J1仍使用原adapter与CP两个AdamW参数组，Wc/Wd属于相同CP组；LR均为3e-4→3e-6。

## 3. 调度、配置与诊断

| 入口 | 内容 |
|---|---|
| round_j.py（精简版未保留） | 从I组复用TRAIN均值；三卡三个seed，完成训练及native/exact六项干预 |
| tmux脚本（精简版未保留） | 当前Python、会话、日志与退出码；中断后从完整epoch继续 |
| J1模板（精简版未保留） | 在I2基础上新增separate模式及实验编号 |
| 35项历史快照（精简版未保留） | native及G0/G1/H1/I1/I2 exact指标 |
| ranking_diagnostics.py（精简版未保留） | J1−I2/G0/G1/H1/I1与dynamic−mean预测级比较 |
| readout_training_summary（精简版未保留） | native/exact选中epoch和末轮两路修正统计 |

## 4. 验证与正式结果复算

- 55项原发布测试（精简版未保留）、结构化摘要（精简版未保留）、三训练六评估流程（精简版未保留）。
- 七项J组测试（精简版未保留）：参数计数与分组、I2恢复恒等式、显式低秩算子、五组参数有限差分、独立矩阵均值评估、首步双路学习和续训。
- 旧I2配置在新旧训练器中的参数、AdamW及Torch RNG逐元素相同；CPU流程重复启动跳过三训练和六评估。
- 正式数据复算（精简版未保留）、原分析脚本（精简版未保留）、仓库验证（精简版未保留）、服务器源码身份（精简版未保留）。

正式训练的软件与GPU身份保存在各protocol.json；发布验证使用CPU合成缓存。CLIP许可见CLIP_LICENSE（精简版未保留）。公式推导与当前结果见[主说明](../README.md)，完整解释见[数据分析](../analysis/README.md)。
