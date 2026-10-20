# 03 · 代码与修订

训练代码位于 [vgcpod_v1_roundB](vgcpod_v1_roundB/)，保持服务器运行版本的字节。配置、资产和发布支持文件存放在平行分区，通过仓库恢复命令还原完整运行包。

| 入口 | 内容 |
| --- | --- |
| [模型与损失](vgcpod_v1_roundB/vgcpod/model.py) | 独立读出与门控配置；混合归一化；η幅度；沿用A5目标 |
| [训练与评测](vgcpod_v1_roundB/vgcpod/engine.py) | 沿用分组学习率与评价，新增跨图像门控方差和偏离1的均值 |
| [B组三卡队列](vgcpod_v1_roundB/round_b.py) | B1/B2/B3独立训练，续训、跳过完成运行，按配置及同seed配对汇总 |
| [CLI](vgcpod_v1_roundB/run.py) | 沿用A组训练、evaluate与diagnose命令 |
| [tmux入口](vgcpod_v1_roundB/scripts/start_round_b_tmux.sh) | 向tmux传入当前conda环境的绝对Python路径 |
| [数学与集成检查](vgcpod_v1_roundB/tests/test_round_b.py) | 新增5项B组检查，与原检查合计22项 |

## 具体改动

- `readout_feature` 支持 global/local/mixed；α 控制 mixed 中局部特征的权重。
- `gate_input` 支持 none/global/local，与修正读出独立；本轮 B2/B3 从 local 生成门控。
- `gate_eta` 控制相对 1 的调制幅度；η=1 使用原计算顺序。
- `gamma_image_variance` 在每个物体/模式上沿 batch 图像轴计算，`gamma_abs_deviation` 记录共同偏移。
- `round_b.py` 输出逐运行指标、配置均值/样本标准差、同 seed A5/A6 与 B3−B2 增量及逐组合差值。

## 原版与复算入口

- [相对02训练包的代码及B配置补丁](source_changes.patch)。
- [完整冻结ZIP](../results/archives/vgcpod_v1_roundB_20261019.zip)、[成员映射](../package_members.json)。
- [22项测试日志](../results/package_validation/validation/test_output.txt)、[真实CLI三进程CPU运行记录](../results/package_validation/validation/cli_output.txt)。
- [原九变体四次更新逐位对照](../results/package_validation/validation/legacy_compatibility.json)、[本地验证环境与结果](../results/package_validation/validation/result.json)。
- [服务器源码与ZIP一致性](../results/server_source_identity.json)。
- [原始分析源码](original_analysis/)保持原字节；[仓库复算入口](../analysis/recompute.py)仅适配路径。

CLIP 源码与许可随包保存：[CLIP_LICENSE](vgcpod_v1_roundB/vendor/CLIP_LICENSE)。本轮已实施公式和日志定义见[主README](../README.md)。
