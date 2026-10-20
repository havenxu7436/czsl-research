# 04 · 代码与修订

训练代码位于 [vgcpod_v1_roundC](vgcpod_v1_roundC/)，保持服务器运行版本的字节。配置、资产、说明与验证记录分区保存，通过仓库恢复命令还原完整 C 包；B3 补 seed 使用 [03 原 B 包](../../03/code/README.md)。

| 入口 | 内容 |
|---|---|
| [模型与损失](vgcpod_v1_roundC/vgcpod/model.py) | 新增 `gate_center_modes`，实现模式中心化；沿用全局读出与 A5 目标 |
| [训练与评测](vgcpod_v1_roundC/vgcpod/engine.py) | 增加 `gamma_mode_mean_deviation`；沿用门控方差、分组学习率和评价 |
| [C组三卡队列](vgcpod_v1_roundC/round_c.py) | C1/C2 独立训练，调用原 B 源码运行 B3，同 seed 配对与跨 seed 聚合 |
| [CLI](vgcpod_v1_roundC/run.py) | 训练、评价及诊断入口 |
| [tmux入口](vgcpod_v1_roundC/scripts/start_round_c_tmux.sh) | 使用当前 conda 的绝对 Python 路径，保存总日志和退出码 |
| [数学与集成检查](vgcpod_v1_roundC/tests/test_round_c.py) | 新增 4 项 C 组检查，合计 26 项 |

## 具体改动

- `gate_center_modes` 默认 false；C2 将 `raw_gamma - 1` 沿模式轴去均值，以 η/2 缩放后加 1。每个图像、物体的模式系数均值为 1。
- C1 采用原静态 CP 分支，仅改变配置中的 ρ=0.125；其修正和修正梯度在相同参数下为 A5 的 1.25 倍。
- `gamma_mode_mean_deviation` 先对每个图像、物体求模式均值，再记录其与 1 的绝对差；按 batch 图像数累计。
- `round_c.py` 在原 B 目录下调用 `round_b.py` 执行 B3/3352。C1/C2 的结果写入 C 输出目录，B3 写入指定 B 输出目录。
- 三卡任务分别为 C1/3351、C2/3351、B3/3352。支持 `--skip-b3`、续训与跳过完成项；汇总时配对相同 seed，样本标准差只对两个及以上 seed 计算。

核心差异如下：

```python
q = raw_gamma - 1
gamma = 1 + (self.gate_eta / 2) * (q - q.mean(dim=-1, keepdim=True))
logs['gamma_mode_mean_deviation'] = float((out['gamma'].mean(-1) - 1).abs().mean())
```

模式轴为最后一维。完整张量维度、概率分解与雅可比见[主 README](../README.md#3-公式修订c2-模式中心化与训练目标)。

## 原版与复算入口

- [相对03的代码及C配置补丁](source_changes.patch)、[原C ZIP](../results/archives/vgcpod_v1_roundC_20261020.zip)、[原包成员映射](../package_members.json)。
- [26项测试日志](../results/package_validation/validation/test_output.txt)：C1 幅度等价、C2 均值与范围、算子收缩、概率归一化、有限差分、训练导出与中断续训。
- [旧配置逐位对照](../results/package_validation/validation/legacy_compatibility.json)：原九变体与 B1/B2/B3，各进行四次优化更新，与原 B 源码逐位一致。
- [真实三进程CPU调度记录](../results/package_validation/validation/cli_output.txt)、[验证环境与汇总](../results/package_validation/validation/result.json)。使用合成缓存验证调度、原 B3 调用、重复跳过与联合汇总。
- [服务器源码身份](../results/server_source_identity.json)：C 对应交付 ZIP，B 对应仓库 03。
- [原始分析源码](original_analysis/)保持原字节；[仓库复算入口](../analysis/recompute.py)适配分区路径，保留原数值计算。

CLIP 源码与许可随包保存：[CLIP_LICENSE](vgcpod_v1_roundC/vendor/CLIP_LICENSE)。服务器训练使用 Python 3.8.18、PyTorch 1.13.1+cu117 与 RTX 3090；原本地验证为 Python 3.12.14、PyTorch 2.5.1+cpu。
