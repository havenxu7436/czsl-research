# 06 · 代码与修订

原 E 训练包源码位于 [vgcpod_v1_roundE](vgcpod_v1_roundE/)。配置、资产、原说明和验证记录分区保存，通过仓库 materialize 恢复原布局。训练模型与损失沿用 [05 D组](../../05/code/README.md)，相对 D 的完整差异见[源码补丁](source_changes.patch)。

| 入口 | 具体内容 |
|---|---|
| [模型与损失](vgcpod_v1_roundE/vgcpod/model.py) | 与 D 组逐字节一致；中心化局部模式、全局读出与 β 归一化目标 |
| [训练与评价](vgcpod_v1_roundE/vgcpod/engine.py) | 独立 native／exact 最优状态、四份 checkpoint、指定权重和输出目录评价 |
| [E1配置](../configs/round_e/E1.json) | 相对 D1 只改变 rho=0.15 与 experiment_id=E1 |
| [均值替换](vgcpod_v1_roundE/mean_gate_eval.py) | `--selection native/exact`；两套权重分别统计 TRAIN 均值并评估 |
| [E组调度](vgcpod_v1_roundE/round_e.py) | 三卡三 seed、训练后六项均值评估、完成状态复用和续训、同 seed 汇总 |
| [tmux启动](vgcpod_v1_roundE/scripts/start_round_e_tmux.sh) | 捕获当前 Python 路径，保留日志与退出码，支持脱离会话 |
| [E组测试](vgcpod_v1_roundE/tests/test_round_e.py) | 三项新增测试，累计 33 项 |

## 已实施的代码改动

### 1. 独立保存两种最优权重

`engine.py` 原 `best` 继续跟踪 VAL CW native AUC，新增 `best_exact` 跟踪 exact AUC。更新条件均为严格 `>`，并列时保留第一次达到峰值的 epoch。

```python
selected = metrics['AUC_native'] > best
selected_exact = metrics['AUC_exact'] > best_exact
if selected:
    best = metrics['AUC_native']
if selected_exact:
    best_exact = metrics['AUC_exact']
```

checkpoint 状态新增 `best_exact_auc`，history 新增 `selected_exact`。`best_native.pt` 保存主权重，原子复制为兼容入口 `best.pt`；`best_exact.pt` 保存 exact 权重，`last.pt` 保留优化器、随机状态、训练历史和两套峰值。续训同时恢复两套最大值。

训练结束分别调用 `evaluate_run(..., checkpoint_name='best.pt')` 和 `evaluate_run(..., checkpoint_name='best_exact.pt', out=run/'exact_selection')`。权重选择与 bias 校准分别记录，主导出的路径继续沿用原布局。

### 2. 均值替换评估增加选择入口

`mean_gate_eval.py --selection native` 读取 `best.pt` 与原运行指标；`--selection exact` 读取 `best_exact.pt` 与 `exact_selection/metrics.json`。输出分别写入 `mean_gate/`、`mean_gate_exact/`。

TRAIN 均值统计、固定权重重算和方差公式沿用 D 组。身份记录增加选择类型、源代码指纹、checkpoint／评价器／缓存 SHA256。每个条件独立选择 VAL CW／OW native bias，固定用于对应 TEST；两项评估均不执行优化步骤。

### 3. 三 GPU 编排与结果汇总

- `round_e.py` 将 seed3351／3352／3353 分别分配给物理 GPU1／2／3；每个进程通过 `CUDA_VISIBLE_DEVICES` 将本卡映射为内部 `cuda:0`。
- 每卡先执行本 seed 的 E1 训练，再顺序运行 native／exact 两项均值替换评估。
- 已完成的同源运行直接复用，未完成的从 last 续训；完成判断同时检查两个权重导出。Linux 运行目录有队列锁。
- 新增 `rho_native_*` 仅汇总 C2／D1／E1 的 native 选择；`exact_selection_*` 独立记录 E1 的第二套选择；`mean_gate_effects.csv` 记录动态−均值差。
- 读取 A／C／D 已有结果，缺少外部参考运行时使用 [14份原指标快照](../configs/references/pre_e_complete.json)。跨组配对使用相同 seed 和特征缓存。

## 源码、验证与复算

- [相对05完整补丁](source_changes.patch)、[服务器源码身份核对](../results/server_source_identity.json)、[66个原包成员映射](../package_members.json)。
- [33项原发布测试](../results/package_validation/E/validation/unit_tests.txt)、[CPU完整流程与旧引擎比较](../results/package_validation/E/validation/result.json)、[流程输出](../results/package_validation/E/validation/cli_output.txt)。
- 原 CPU 流程完成 3 项训练、6 项均值评估，重复启动分别跳过 3／6 项；相同 ρ=0.15 配置下，新旧引擎的 native best／last 参数与 Torch RNG 逐位一致。
- [参考快照可移植性](../results/package_validation/E/validation/reference_portability.json)、[原文档排版核查](../results/package_validation/E/validation/document_layout.json)。
- [原分析代码](original_analysis/)、[仓库复算](../analysis/recompute.py)、[仓库归档核查](../analysis/repository_validation.json)。本轮归档复算调整输入路径，保留数值计算。

CLIP 许可保留于 [CLIP_LICENSE](vgcpod_v1_roundE/vendor/CLIP_LICENSE)。服务器环境为 Python 3.8.18、PyTorch 1.13.1+cu117、RTX 3090。本地原数学和调度验证使用 CPU 合成特征缓存；本次归档核对使用服务器下载的真实运行文件。
