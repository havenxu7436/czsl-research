# TRUST-CPF 第二批：教师阶段匹配，10 epoch

本批在第一批 State 已完成结果上，检验“与正式识别阶段对应的效用监督”以及“两次响应平均”能否提高冻结识别的收益。仍采用 CLIP 特征上的 CPF 强基底，保留关系干预效用监督、跨折预测与类型预算分配。

## 一次运行的内容

- 3 个训练 seed：3351、3352、3353；每个 seed 的 5 个教师各新增训练 10 epoch，共 150 个教师 epoch。
- 每个教师记录 epoch 0～10，按折内独立 STOP 集的 native AUC 选择最早最佳权重；完整训练到第 10 epoch。
- 在选中的教师状态重新测量响应与 U-NLL 效用。9 个 Q1–Q3 教师共同拟合一个 Absolute 预测器，3 个 Q4 选正则，Q5 做决策评估。
- 正式识别固定第一批 Uniform-best（3351/3352/3353 对应 epoch 2/2/6）。比较 Uniform、旧/新预测器 × 单次/两次平均响应，共五个主组。
- 四个预测组各配五次类型内置换，另有两个 reference 响应控制。共 27 个正式评分策略、31 个 Q5 策略；它们共享权重，没有 27 次正式模型训练。
- 队列自动执行“三卡教师训练 → 一次联合拟合 → 三卡评分”。可用同一启动命令续跑。

## 文件

| 文件 | 内容 |
|---|---|
| SERVER_COMMANDS.md | 上传、环境、三卡运行、结果检查、打包和下载的完整命令 |
| FORMULAS_AND_CODE.md | 新阶段训练、监督、拟合、分配与评分公式，真实核心代码 |
| configs/default.json | 固定的完整实验配置 |
| trustcpf/matched_train.py | 10 epoch 教师训练、断点续训、效用重测 |
| trustcpf/matched.py | 共享预测器与两次响应比较 |
| trustcpf/matched_compute.py | Q5 与冻结正式评分 |
| trustcpf/matched_workflow.py | 输入导入、队列阶段、汇总 |
| scripts/run_queue.py | 三卡调度与联合拟合衔接 |
| scripts/upload.ps1 / download.ps1 | Windows 上传与下载 |
| validation/ | 本地流程、公式与真实源结果检查记录 |

## 预期完成状态

```text
STATUS 26 /26 teachers 15 /15 recognition 972 / 972 Q5 744 / 744 curves 165 /165 complete True
QUEUE_COMPLETE
```

核心分析先看 `work/aggregates.json`、`paired.csv`、`q5_paired.csv`；完整教师曲线在 `teacher_curves.csv`。AUC 统一按 ×100 输出，差值单位为百分点。多图 bank 先在每个 seed 内平均，再汇总三个 seed；三个识别重复共享同一个效用预测器。

本地验证使用小型完整数据流，含 15 个教师 ×10 epoch、评分、断点恢复和归档回读。正式 UT-Zappos GPU 实验由下面的服务器命令启动。
