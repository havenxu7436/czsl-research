# 09 · 代码与修订

原 H 包源码位于 [vgcpod_v1_roundH](vgcpod_v1_roundH/)。配置、资产、原说明和验证记录按平行结构分区保存，materialize 恢复原始布局。相对 [08 G组](../../08/code/README.md)的完整差异见[源码补丁](source_changes.patch)。

## 1. TRAIN 均值统计与固定 buffer

[gate_features.py](vgcpod_v1_roundH/vgcpod/gate_features.py) 的 `prepare_gate_statistics()` 只读取 TRAIN 的 `fo.npy`，以 float64 分块求和，对每张图像的全部物体查询求均值；`train_object_query_mean.npz` 保存 `mean_fo_float64`、float32 `mean_fo` 与统计身份，JSON 保存样本数、均值范数、查询词表和 SHA256。

`load_gate_mean()` 读取运行配置中的均值路径与哈希。`VGCPOD(..., gate_mean=...)` 将均值注册为 `gate_fo_mean` buffer；该张量不进入优化器，并随 checkpoint 保存。三个 seed 使用同一统计文件，forward 使用 float32。

## 2. 门控输入与 checkpoint 评价

[model.py](vgcpod_v1_roundH/vgcpod/model.py) 新增的核心路径为：

```python
gate_input = F.normalize(
    fo - self.gate_fo_mean[None], dim=-1,
    eps=c.get('gate_residual_eps', 1e-12))
```

随后继续执行原 sigmoid、η=0.25 的模式中心化和全局 g 的 CP 读出。联合概率、objective、训练步、AdamW、学习率日程及裁剪沿用 G1。

[engine.py](vgcpod_v1_roundH/vgcpod/engine.py) 的 `make_model()` 在训练构建时加载均值；`evaluate_run()` 将 checkpoint 中的 `gate_fo_mean` 传回构造器。[mean_gate_eval.py](vgcpod_v1_roundH/mean_gate_eval.py) 同样从权重恢复均值，再统计训练后门控矩并替换成均值条件。原包测试中移除外部均值文件后，native／exact 评价与原导出一致。

## 3. 三 GPU 调度与配对汇总

- [round_h.py](vgcpod_v1_roundH/round_h.py)：`prepare_runtime_config()` 先生成共享均值和运行配置，`launch()` 再开启三个 seed 队列；每队列训练 H1，依次执行 native、exact 均值评估。
- [tmux启动](vgcpod_v1_roundH/scripts/start_round_h_tmux.sh)：固定激活环境 Python，记录主日志、退出码与会话；完成任务复用，未完成训练从最后完整 epoch 继续。
- `summarize()` 分别比较 native H1−G0/G1 和 exact H1−G0/G1；同时保存同配置主表、支持损失日志、逐组合差值与 dynamic−mean。
- [ranking_diagnostics.py](vgcpod_v1_roundH/ranking_diagnostics.py)：扩展 H1 native／exact 的完整 OW 概率诊断，按 all/seen/unseen 分组，记录物体／属性、概率质量与未见候选排序。

| 原运行入口 | 仓库入口 |
|---|---|
| H1 模板 | [configs/round_h/H1.json](../configs/round_h/H1.json) |
| 原服务器运行配置 | [configs/runtime/H1.json](../configs/runtime/H1.json) |
| H前26个历史 native 指标及G0/G1 exact结果 | [pre_h_complete.json](../configs/references/pre_h_complete.json) |
| 源码差异／服务器核对 | [补丁](source_changes.patch) · [源码身份](../results/server_source_identity.json) |
| 发布包原布局恢复 | [package_members.json](../package_members.json) |

## 4. 发布验证与仓库复算

- [44项测试](../results/package_validation/H/validation/unit_tests.txt)、[结构化测试摘要](../results/package_validation/H/validation/unit_test_summary.json)、[三训练六评估流程](../results/package_validation/H/validation/result.json)、[流程日志](../results/package_validation/H/validation/cli_output.txt)。
- 新增五项 H 测试覆盖配置差异、全部 TRAIN 查询均值、显式 CP／门控公式与有限差分、前两步梯度、续训及 checkpoint buffer 恢复；原测试同时保留。
- 原 G1 路径在新旧核心中的参数、AdamW 与 Torch RNG 逐元素一致；重复启动复用三项训练、六项均值评估。
- [仓库复算入口](../analysis/recompute.py)、[原分析源码](original_analysis/)、[归档验证](../analysis/repository_validation.json)。

正式训练环境为 Python3.8.18、PyTorch1.13.1+cu117、NumPy1.24.4、RTX3090。原发布数学与调度验证使用 CPU 合成缓存，正式结果来自服务器。CLIP 许可见 [CLIP_LICENSE](vgcpod_v1_roundH/vendor/CLIP_LICENSE)。
