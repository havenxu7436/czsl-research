# TRUST-CPF Align v1：第一批实验

本工程实现 2026-10-24 后续方案中的第一批 TRUST-Align。起点为已有 CLIP ViT-L/14 特征、冻结强 CPF、TRUST 关系头和 27 参数 Compact 效用预测器。

正式结果为 **3 方法 × 3 seed = 9 组**：Uniform-0.5、TRUST-Align、Permuted-Align。每组从相同 seed 的 reference 开始适配 3 epoch，在 epoch 0–3 中按 VAL CW native AUC 选权重。另完成部署强度扫描、三个独立 16 图 bank 评估，以及 Q5 单图／混合效用的实际决策诊断。强度选为 0 时复用完全相同的 Uniform 结果，并显式记录。

直接执行步骤见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)，公式及核心代码见 [FORMULAS_AND_CODE.md](FORMULAS_AND_CODE.md)。

| 文件 | 作用 |
| --- | --- |
| `run.py`、`trustcpf/align_workflow.py` | 初始化、三 seed 队列、结果汇总、完成状态 |
| `trustcpf/align.py` | 半密度概率求解、类型内概率置换、VAL 扫描、适配与多 bank 评价 |
| `trustcpf/decision_audit.py` | Q5 图像划分、单图与混合效用、独立图像上的实际风险 |
| `trustcpf/model.py`、`engine.py`、`evaluation.py` | 沿用原 Strong／Compact 的分类器、损失、训练调度与 AUC 实现 |
| `configs/default.json` | 本批固定设置 |
| `scripts/run_all.sh` | 三张卡各跑一个 seed；默认 0/1/2 |
| `scripts/pack_results.sh` | 汇总完整性检查、带进度打包、生成下载描述 |
| `scripts/*trustcpf_align_v1.ps1` | Windows 上传、下载、SHA256 校验与 Python 解压 |
| `tests/`、`validation/` | 公式、工作流、归档测试及本地验证结果 |

新目录为 `trustcpf_align_v1`，运行所需来源为 `trustcpf_compact_v1` 与 `trustcpf_strong_v1` 的已有 `work`。首次初始化校验缓存和来源，随后复制本批需要的 reference、Q5 教师、两个 Compact 模型、两个响应库到新目录。旧实验结果保留原位。

新路径使用独立的 `run.py` 入口；工程保留原始模块供分类器和回归测试复用。第一批入口只执行 Align 队列。

运行依赖是原 `vgcpod_v1` 环境已有的 NumPy 和 PyTorch。代码语法兼容 Python 3.8，目标环境为 PyTorch 1.13.1／CUDA 11.7。CLIP 编码结果直接读缓存。
