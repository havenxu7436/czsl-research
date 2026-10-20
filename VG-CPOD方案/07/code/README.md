# 07 · 代码与修订

原 F 训练包源码位于 [vgcpod_v1_roundF](vgcpod_v1_roundF/)。配置、资产、原说明和验证记录按仓库分区保存，materialize 恢复原布局。本轮模型与损失沿用 [06 E组](../../06/code/README.md)，实际差异见[源码补丁](source_changes.patch)。

| 入口 | 具体改动与作用 |
|---|---|
| [F1配置](../configs/round_f/F1.json) | β=0.05、experiment_id=F1；其余与 E1 一致 |
| [engine.py](vgcpod_v1_roundF/vgcpod/engine.py) | 增加 `task_decomposition()` 和四个标量日志字段 |
| [round_f.py](vgcpod_v1_roundF/round_f.py) | 三卡调度、自动续训、六项均值评估、β 对比和支持损失表 |
| [tmux启动](vgcpod_v1_roundF/scripts/start_round_f_tmux.sh) | F 组会话、输出路径、日志和退出码 |
| [F组测试](vgcpod_v1_roundF/tests/test_round_f.py) | 配置差异、损失与梯度恒等式、有限差分和加权日志；累计36项 |
| [模型与损失](vgcpod_v1_roundF/vgcpod/model.py) | 与 E 组逐字节一致，已有 `task_aux_beta` 直接读取新值 |
| [均值评估](vgcpod_v1_roundF/mean_gate_eval.py) | 沿用 E 组 native／exact 选择与 TRAIN 均值实现 |

## 1. 加权损失日志

训练反向传播继续调用已有 `objective()`。`optimizer.step()` 后，loss 张量脱离计算图并转为 Python 标量，再计算：

```python
def task_decomposition(logs, beta):
    support_weight = 1.5 * beta / (1.0 + beta)
    pair_weighted = 1.5 * logs['pair']
    support_weighted = support_weight * logs['support']
    return dict(support_weight=support_weight, pair_weighted=pair_weighted,
                support_weighted=support_weighted,
                task_reconstruction_error=abs(logs['task'] - pair_weighted - support_weighted))
```

对应恒等式：

```math
\mathcal{L}_{\mathrm{task}}=1.5\mathcal{L}_{\mathrm{pair}}
+\frac{1.5\beta}{1+\beta}\mathcal{L}_{\mathrm{support}}.
```

epoch 按 batch 的实际图像数加权。`task_reconstruction_error` 为平均 batch 绝对误差，`support_training_summary.csv` 的 `task_epoch_reconstruction_error` 由 epoch 均值另行计算。原日志的 task、pair、cond、object、support、orth 等字段继续保存。

## 2. 三 GPU 编排与汇总

- seed3351／3352／3353 分别使用物理 GPU1／2／3；每个子进程内部为 `cuda:0`。
- 每卡先完成本 seed 的 F1 训练，再运行 native／exact 两项均值替换，合计三次训练、六项评估。
- 两个最优 checkpoint 的首次严格最大值、`best.pt` 兼容副本及双轨续训沿用 E 实现。
- `beta_native_summary.csv` 对比 E1／F1；`round_f_paired_*` 对比 F1−E1/D1/A5/A6/C2/C1；`support_training_summary.csv` 保存选中及最后 epoch 的原始与加权损失。
- 原始预测支持逐组合对比；同 seed 和缓存指纹用于配对。17份参考指标见 [pre_f_complete.json](../configs/references/pre_f_complete.json)。

## 3. 原发布验证与仓库复算

- [36项测试输出](../results/package_validation/F/validation/unittest.txt)、[CPU完整流程](../results/package_validation/F/validation/result.json)、[流程日志](../results/package_validation/F/validation/cli_output.txt)。
- 相同 β=0.05、初始化及缓存下，与原 E 训练器的 native best／last 比较，模型参数、AdamW 状态和 Torch RNG 逐元素一致。三次训练、六项评估的重复启动全部复用已有结果。
- [代码范围核查](../results/package_validation/F/validation/source_scope.json)、[原文档公式排版](../results/package_validation/F/validation/document_layout.json)、[服务器源码核对](../results/server_source_identity.json)。
- [69个原包成员映射](../package_members.json)、[原分析源码](original_analysis/)、[仓库复算入口](../analysis/recompute.py)、[归档核查](../analysis/repository_validation.json)。

服务器使用 Python 3.8.18、PyTorch 1.13.1+cu117、RTX3090；原数学与编排测试使用 CPU 合成特征，真实训练结果来自服务器原始导出。CLIP 许可保留于 [CLIP_LICENSE](vgcpod_v1_roundF/vendor/CLIP_LICENSE)。
