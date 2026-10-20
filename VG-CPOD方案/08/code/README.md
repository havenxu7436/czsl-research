# 08 · 代码与修订

原 G 包源码位于 [vgcpod_v1_roundG](vgcpod_v1_roundG/)。配置、资产、原说明和验证记录按平行结构分区保存，materialize 恢复原始布局。本轮相对 [07 F组](../../07/code/README.md)的实际改动见[源码补丁](source_changes.patch)。

| 入口 | 改动与用途 |
|---|---|
| [G0/G1配置](../configs/round_g/) | β=0.075；gate_input=none/local |
| [round_g.py](vgcpod_v1_roundG/round_g.py) | `worker()` 每卡顺序 G0→G1；`launch()` 三个队列并行；`summarize()` 输出主指标、双选择、β与静态配对及诊断 |
| [ranking_diagnostics.py](vgcpod_v1_roundG/ranking_diagnostics.py) | `diagnose()` 从完整OW预测计算排序量；`summarize_entries()` 输出分组、逐组合及同seed比较 |
| [tmux启动](vgcpod_v1_roundG/scripts/start_round_g_tmux.sh) | 记录激活环境 Python、G会话和退出码 |
| [G组测试](vgcpod_v1_roundG/tests/test_round_g.py) | 配置范围、静态极限、β=0.075目标梯度、已知概率样例；累计39项 |
| [模型](vgcpod_v1_roundG/vgcpod/model.py)／[训练器](vgcpod_v1_roundG/vgcpod/engine.py) | 与 F 包逐字节一致，通过已有配置分支实现本轮目标 |
| [均值评估器](vgcpod_v1_roundG/mean_gate_eval.py) | 沿用 F 包；本轮只对 G1 的两种权重执行均值替换 |

## 1. 静态和动态训练分支

G0/G1 都训练适配器和全局读出 CP。`gate_input=none` 时 Bvis 冻结、gamma 全为1、orth为0；local 时使用中心化视觉门控并训练 Bvis。两组 task 相同：

```math
\mathcal{L}_{\mathrm{task}}(0.075)=\frac32\mathcal{L}_{\mathrm{pair}}+\frac9{86}\mathcal{L}_{\mathrm{support}}.
```

G1 保留 λorth=0.001。原模型和训练运算顺序保持源包字节身份，实际配置与训练参数数目见各运行的 protocol。

## 2. 三 GPU 双任务队列

- seed3351/3352/3353 使用物理GPU1/2/3。每个队列顺序训练G0→G1，然后做G1 native/exact均值评估；共六次训练、六项评估。
- 已完成任务直接复用，未完成任务从完整epoch继续。`round_g_status.json` 按三条seed队列记录两次训练及两项评估。
- 每个模型独立保存 `best_native.pt`、兼容别名 `best.pt`、`best_exact.pt` 和 `last.pt`，两条选择分别导出。
- `round_g_paired_*` 首先比较G1−G0、G1−E1/F1；`beta_native_*` 比较同ρ下β=0.05/0.075/0.1。

## 3. 保存预测上的排序诊断

`diagnose()` 使用全部OW组合的 log 概率恢复物体边际，使用导出的GT-object条件logits计算属性准确率和NLL。按split区分TRAIN、CW与CW集合外候选，统计概率质量、最高分超越关系和未见候选内top-1；另按组合记录正确数与NLL。

`summarize_entries()` 核对图像与标签顺序，按selection、condition、phase、group汇总。G1−G0、G1−E1/F1/D1/A5和dynamic−mean保存同seed差值、均值及样本SD。诊断不改变训练或预测。

## 4. 原发布验证与仓库复算

- [39项测试](../results/package_validation/G/validation/unittest.txt)、[六训练六评估流程](../results/package_validation/G/validation/result.json)、[流程日志](../results/package_validation/G/validation/cli_output.txt)。
- 原F训练器设置为相同G0/G1配置时，native/exact/last参数、AdamW状态和Torch RNG逐元素一致；重复启动跳过全部已完成任务。
- [真实E1/F1诊断复核](../results/package_validation/G/validation/real_prediction_diagnostics.json)、[代码范围](../results/package_validation/G/validation/source_scope.json)、[服务器身份](../results/server_source_identity.json)。
- [原包76成员映射](../package_members.json)、[原分析源码](original_analysis/)、[仓库复算入口](../analysis/recompute.py)、[归档验证](../analysis/repository_validation.json)。

服务器使用Python3.8.18、PyTorch1.13.1+cu117、RTX3090；发布数学及调度测试使用CPU合成特征，正式训练结果来自服务器。CLIP许可见[CLIP_LICENSE](vgcpod_v1_roundG/vendor/CLIP_LICENSE)。
