# 02 · 代码与修订

训练代码位于 [vgcpod_v1_roundA](vgcpod_v1_roundA/)，与服务器运行版本保持相同字节。配置、资产和发布支持文件在平行分区，通过仓库恢复命令还原完整运行包。

| 入口 | 内容 |
| --- | --- |
| [模型与损失](vgcpod_v1_roundA/vgcpod/model.py) | 归一化 β 目标、task/support 日志，前向沿用静态全局 CP |
| [训练与评测](vgcpod_v1_roundA/vgcpod/engine.py) | AdamW 参数分组、比例余弦、每轮 VAL CW/OW 与分布诊断 |
| [CLI](vgcpod_v1_roundA/run.py) | 配置 variant 默认值、β 与 LR 倍率参数读取 |
| [三卡队列](vgcpod_v1_roundA/round_a.py) | 六组初始探索、补 seed、逐运行汇总 |
| [tmux 入口](vgcpod_v1_roundA/scripts/start_round_a_tmux.sh) | 指定三张物理 GPU 并保存 launcher 状态 |
| [数学与集成检查](vgcpod_v1_roundA/tests/) | 原数学检查和本轮检查共 17 项 |

- [相对原 v1 的代码补丁](source_changes.patch)。
- [服务器使用的完整冻结 ZIP](../results/archives/vgcpod_v1_roundA_20261019.zip)。
- [原版检查记录](../results/package_validation/validation/summary.json)与[测试日志](../results/package_validation/validation/test_output.txt)。
- [原始分析源码](original_analysis/)保持原字节；[仓库复算入口](../analysis/recompute.py)仅适配路径。
- [成员映射](../package_members.json)记录恢复完整包时的原路径。

CLIP 源码及其许可随代码归档：[CLIP_LICENSE](vgcpod_v1_roundA/vendor/CLIP_LICENSE)。模型公式与本轮修改说明见[主 README](../README.md)。
