# TRUST-CPF State v1 — first batch, 10 epochs

本包实现《TRUST-CPF 公式修订与后续实验方案 v6》的第一批实验。

按照本次要求，每个 seed 从旧 reference **重新训练一条 10 epoch Uniform 轨迹**，保存 epoch 0–10 的曲线，以 VAL CW native AUC 选择共同权重。三个效用预测器及其控制组随后共享该权重，执行冻结推理。

主要方法：Uniform、Independent-Absolute、Pooled-Absolute、Pooled-Relative；三个非均匀方法各有五个类型内概率置换。额外运行 reference 状态参照、Pooled-Absolute 响应交叉、两个简单读出和两次响应测量。Q5 完成预测效用与测量效用的实际识别审计。

- 拟合：Q1–Q3，跨 seed 合并为九教师；三个 Q4 共同选 λ；Q5 保持为评价。
- 效用模型：27 参数岭回归，闭式求解；α=教师数×λ。
- 分配：κ=1；OI/OE/AE 期望边数为 6/18/24。
- 读出：16 图的候选概率混合，三个正式图 bank、八个 Q5 图 bank。
- 默认 GPU：0/1/2，对应 seed 3351/3352/3353。
- 本批不训练新教师；第二批教师阶段匹配未混入本包。

完整操作见 SERVER_COMMANDS.md；公式、核心代码及验证指标见 FORMULAS_AND_CODE.md。

## 产物与完成标志

正式配置完成后：

- 17/17 阶段：全局导入＋全局拟合＋三个 seed 各五阶段。
- 33 行训练曲线：3×(epoch 0–10)。
- 1512 行正式识别指标：两个状态共 42 个策略实例×3 seed×3 bank×2 split×2 world。
- 600 行 Q5 决策：25 个策略实例×3 seed×8 bank。
- 756 份 VAL/TEST 逐图像预测，包含实际图掩码与概率。
- 共享效用模型、五折原始汇总标签、教师权重、Uniform best/last、完整日志。

主表读取 uniform_best、bank1701、test。另两个 bank 单列，并先在 seed 内平均后报告三 seed 均值与标准差。置换用五次均值；不选择其中最优结果。

## 验证范围

validation 中为小型数据的三 seed 10 epoch 完整队列测试、原评估器一致性、重复运行检查和打包解压校验。real_fit_reproduction.json 记录真实 R5 数据的四组拟合复算。它们属于本地代码验证，不是服务器 UT-Zappos 新一轮识别成绩。

