# TRUST-G0 three-dataset experiment · 10 epochs

G₀（历史 VG-CPOD V1）上的 TRUST 效用监督、情境分配与完整机制对照。UT-Zappos50K / MIT-States / C-GQA；5351 / 5352 / 5353 三套独立流水线；GPU 仅 0、1、2。

- [实验方案、完整公式与实现映射](docs/plan_and_equations.md)
- [停止旧任务、上传、训练、监测与下载](docs/server_commands.md)
- [固定配置](configs/experiment.json)
- [训练与测量测试](tests/test_core.py)

服务器准备：`bash scripts/prepare.sh`；完整队列：`bash scripts/run_all.sh 0 1 2`；监测：`python scripts/status.py`；结果导出：`bash scripts/pack_results.sh`。

3 个数据集 × 3 seed，135 个教师，17 个方法，1620 行 VAL/TEST CW/OW 指标。每个神经网络训练阶段均为 10 epoch。全部效用策略共享 Uniform 权重；旧项目保持原样。

`validation/` 为本地软件与索引验证。真实数据实验完成状态由服务器 `work/report.json` 给出。OpenAI CLIP 许可证位于 `vendor/CLIP_LICENSE`。
