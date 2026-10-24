# TRUST-CPF Compact v1：紧凑期望效用监督实验

本轮继续在已有强 CPF 上研究 TRUST：用跨组合配对收益学习关系效用，再按固定类型预算选择图。将上一轮的高维异方差网络改为八维响应输入、27 参数岭回归，选模目标改为 Q4 的预算决策遗憾。保留原知识、强 CPF、关系头、教师、测量、响应缓存和识别评价。

## 1. 这轮回答什么

真实配对效用监督能否在同一强 CPF、关系分支和边预算下，优于逐类型常数图与打乱效用标签？主实验只有三组、三个 seed：

| 方法 | 图与效用监督 | 计算 |
| --- | --- | --- |
| Drop | 类型内常数采边概率，沿用实际预算 | 复用三组完成的结果 |
| TRUST-Compact | 真实配对边均值；八维响应；27 参数回归 | 新增三次 3 epoch 适配 |
| shuffled utility | 各折各类型内打乱标签与边的对应 | 新增三次 3 epoch 适配 |

GPU 0、1、2 分别运行 seed 3351、3352、3353；每张卡先运行 Compact，再运行 shuffled utility。总计六次新适配；不重训教师、不提取 CLIP 特征、不重测响应。这轮不追加噪声矩阵。

原 TRUST robust、上一轮 conditioned、Capacity、Dense、r0 的结果进入 `historical.csv`。三个主方法进入 `summary.csv`。两张表使用相同 seed、指标与评价实现。

## 2. 输入与实现调整

服务器输入为：

- `/data/xhw/experiments/trustcpf_strong_v1`：五折教师、配对 Query 测量、reference、Drop 与历史强基底结果。
- `/data/xhw/experiments/trustcpf_modelcond_v1`：每 seed 的五折教师响应与 reference 响应，共 18 份缓存。
- `/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14`：原始 CLIP 特征缓存。

本轮优化集中在四点：

1. 按教师和边平均 episode 配对效用，直接拟合期望收益。每个折—边一条监督样本。
2. 八维响应与三种关系类型做分块交互，模型为 27 个参数；采用 float64 线性方程求解。
3. Q1–Q3 拟合，Q4 按固定预算下的代理决策效果选正则，Q5 输出审计；部署时按选定正则用五折重新拟合。
4. 主奖励直接使用预测均值。均值标准误、背景测量方差和跨折残差分别记录，退出选图惩罚。

打乱监督保留各折、各类型的标签分布，标签均值、样本数及标准误一起置换。使用真实监督由 Q4 选出的同一个正则强度；归一化、输出尺度、预算、温度、reference 和训练种子一致。知识文本及图边端点保持一致。

## 3. 训练安排

`run.py setup` 核对两个来源实验完成状态、分类器/评价器字节哈希、G0 初始权重、知识库和特征缓存；逐一核对响应文件对应的教师哈希与 Inner/Query 划分。新建 `server_env.sh`，记录当前已激活环境中的 Python 绝对路径。

每个 seed 自动完成导入、拟合、审计、五折重拟合、选图、两组适配。七个进度标记属于断点管理，用户只需启动一条三卡队列命令。

- 原 CPF 全部冻结；训练原关系头，沿用原优化器、学习率、数据顺序和目标。
- 从相同 `reference/last.pt` 开始，epoch 0–3 由 VAL CW native AUC 选权重。
- 沿用 16 图概率平均；CW、OW 在各自候选集合内归一化。
- bias 和 temperature 在 VAL 确定并用于 TEST；保存所有预测供复算。
- 类型预算来自每 seed 原 robust 的实际概率和，与对应 Drop 匹配；温度沿用原结果。

`work/progress` 保存输入哈希和每个阶段输出哈希。重新执行队列复用已完成阶段；中断的关系适配从最后一个完整 epoch 恢复。源实验文件只读。

## 4. 核心指标与结论

| 层级 | 输出 | 用途 |
| --- | --- | --- |
| 主识别 | CW/OW native AUC、exact AUC、VAL bias 下 HM | 检验效用选图的识别价值 |
| 配对比较 | Compact−Drop、Compact−shuffled，各 seed 差值、均值、样本标准差、胜出数 | 检验监督与边排序贡献 |
| 效用 | Q5 边均值 Spearman、RMSE、各类型指标、同预算遗憾 | 检验迁移排序与决策质量 |
| 校准与诊断 | NLL、ECE、未见属性准确率与多候选诊断 | 解释识别表现 |
| 结构和计算 | 类型期望边数、选中 epoch、阶段耗时、主干冻结核对 | 对齐实际实验条件 |

优先读 `summary.csv`、`paired_comparisons.json`、`utility_comparison.json`。真实监督同时优于 Drop 和打乱监督，支持效用监督改善图选择；再与 `historical.csv` 的 Capacity/Dense 比较整体识别价值。若效用指标提高而 AUC 持平，则下一步重点研究边交互或部署响应，而不是继续扩大回归网络。

Q5 已用于上一轮的探索性诊断，本轮按同一预定规则复算并保存；新增证据是两种监督对应的六次识别适配。文档与代码将 Q5 效用代理和最终识别结果分开报告。

## 5. 本地验证与文件

已用真实三 seed 归档输入跑通汇总、选模、审计、五折重拟合与选图；选出的正则为 10、100、100，复现上一轮诊断。完整三 seed 模拟流水线覆盖新适配、冻结 CPF、相同预算、原结果复用和续跑。正式 UT-Zappos 识别训练由下述服务器命令执行。

- `FORMULAS_AND_CODE.md`：推导、公式与当前核心实现。
- `SERVER_COMMANDS.md`：上传、训练、续跑、打包与下载。
- `trustcpf/compact.py`：新增效用预测、审计、结构与适配。
- `trustcpf/workflow.py`：导入、阶段恢复和完整汇总。
- `validation/`：公式测试、真实归档复算、下载和打包验证记录。

`model.py`、`g0.py`、`engine.py`、`evaluation.py`、`metrics.py` 与原 Strong v1 逐字节一致。
