# TRUST-CPF 模型响应条件化方案 v1

日期：2026-10-23。目录：trustcpf_modelcond_v1。GPU 1、2、3 对应 seed 3351、3352、3353。

本轮研究“以配对泛化收益为监督的模型条件化关系效用学习”。完整强 CPF、关系消息、三元交互及 Query 配对效用标签沿用 Strong v1。新增八维 Inner 模型响应，检验它能否提高跨组合效用迁移及同预算识别。

## 1. 实验范围和复用方式

| 组别 | 处理 | 结果数 |
| --- | --- | --- |
| drop | 逐类型常数 Bernoulli 概率，直接复用 Strong v1 | 3 |
| original | 原 robust 模型、预测、审计结果，直接复用 Strong v1 | 3 |
| conditioned | 新效用输入、效用拟合、结构生成、3 epoch 适配 | 3 |

共九组识别结果，其中新增三组。旧五折教师、每 seed 43,008 条 Query 效用观测、主 reference checkpoint 都直接复用。程序只读取旧 work，新增结果写入独立目录；旧 r0、capacity、dense 保存为参考表。

每个 seed 按 import → responses → fit → audit → fit_all → structure → adapt 执行，共 7 阶段；三 seed 共 21 阶段。六份效用审计为三个 original 加三个 conditioned；响应缓存为每 seed 五折教师加主 reference，共 18 份。

## 2. 保留效用监督，新增模型条件

原 Query 标签保持：


```math
y_e=\frac1K\sum_{k=1}^K
\left[R_Q(\theta,G_{-e}^{(k)})-R_Q(\theta,G_{+e}^{(k)})\right].
```


Query 是 TRAIN 内部按组合留出的图像。Q1–Q3 拟合，Q4 选效用网络 epoch，Q5 审计；最后按选定 epoch 合并五折标签拟合部署网络。保存的 y、noise、edge、inclusion 原样复用。候选集为全部 TRAIN 组合。

对每个教师，从 Inner 每个组合固定抽最多两张图像，组合间等权、组合内图像等权；主 reference 使用完整 TRAIN。支持集种子为 20261023+j，五折 j=1…5，reference 为 j=6。每个教师使用四个共同 Bernoulli(0.5) 背景。

定义开启和关闭后的候选联合 log score 差：


```math
d_{e,k,c}(x)=J^+_{e,k,c}(x)-J^-_{e,k,c}(x).
```


对候选集归一化后，支持效用为：


```math
\begin{aligned}
\delta_{e,k}(x,y)
&=\log p^+_{e,k}(y\mid x)-\log p^-_{e,k}(y\mid x)\\[4pt]
&=d_{e,k,y}(x)-\log\sum_c p^-_{e,k}(c\mid x)\exp d_{e,k,c}(x).
\end{aligned}
```


公式由两次 log-softmax 相减得到。代码以 float32 计算模型分数，再以 float64 累计支持统计；开启和关闭共享支持样本及其他边。

去除候选共同平移：


```math
\widetilde d_{e,k,c}=d_{e,k,c}-\frac1{|\mathcal C|}\sum_{c'}d_{e,k,c'}.
```


分别在全部支持集 S 和目标原词相关子集 S_e 上计算四个统计。OI/OE 按真实物体筛选，AE 按真实属性筛选；子集内重新归一化组合权重：


```math
h_e^B=\left[
\mathbb E_{B,k}\delta,\;
\sqrt{\mathop{\mathrm{Var}}\nolimits_{B,k}(\delta)},\;
\sqrt{\mathbb E_{B,k}\frac{\|\widetilde d\|_2^2}{|\mathcal C|}},\;
\mathbb E_{B,k}\{H(p^+)-H(p^-)\}
\right],\quad B\in\{S,S_e\}.
```


其中 $`H(p)=-\sum_c p_c\log p_c`$。新增特征只来自 Inner 当前模型响应；Query 标签继续承担监督作用。


```math
\Phi_e^{\rm new}=[\phi_e(S),h_e^S,h_e^{S_e}],
\qquad(\mu_e,\sigma_e^2)=f_\psi(\Phi_e^{\rm new}).
```


正式输入由 6157 维增至 6165 维，三成员网络保持 256→128→2。新增 8 维按拟合集均值/标准差归一化，下限 10⁻⁶；原输入标准差下限保持 0.01。审计前统计仅来自 Q1–Q3；最终五折拟合重新计算其统计。

## 3. 拟合目标与固定图密度

类型尺度、配对测量噪声及分层采边权重沿用原代码。标签 y、噪声方差 ν、采边包含概率 q 对应：


```math
\mathcal L_u=\frac{|B|}{|E|}\frac1{|B|}\sum_{e\in B}
\frac1{q_e}\frac12\left[
\log(\sigma_e^2+\nu_e)+\frac{(\mu_e-y_e)^2}{\sigma_e^2+\nu_e}
\right].
```


标准化时，均值除以类型尺度、方差除以尺度平方；预测后恢复原单位。每成员最多 50 epoch，Q4 NLL 选择首次最优 epoch。响应特征按教师和边计算一次，连接到该教师全部 episode。

部署仍使用 16 次原视觉摘要。响应来自固定主 reference，只计算一次。任务方差加摘要变化组成 v，成员均值方差组成 u：


```math
r_e=\mu_e-\lambda_{\rm var}\sqrt{v_e}-\lambda_{\rm ens}\sqrt{u_e}.
```


风险系数和温度继承各 seed 原 Q4 的选择。逐类型期望边数 B_t 取原 robust 实际概率和：

| seed | OI | OE | AE |
| --- | --- | --- | --- |
| 3351 | 0.381057 | 6.668088 | 12.000001 |
| 3352 | 0.890846 | 9.000001 | 12.000000 |
| 3353 | 0.896474 | 6.433821 | 12.000001 |

求解使用保存概率的 float64 求和，表格仅作展示。


```math
\begin{aligned}
\max_\pi\quad&
\sum_e r_e\pi_e-\tau\sum_e
D_{\rm KL}(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\,\|\,\mathop{\mathrm{Bern}}\nolimits(1/2)),\\[4pt]
\text{s.t.}\quad&\sum_{e:t(e)=t}\pi_e=B_t.
\end{aligned}
```


驻点为：


```math
r_e-\tau\mathop{\mathrm{logit}}\nolimits(\pi_e)-\gamma_{t(e)}=0,\qquad
\pi_e=\mathop{\mathrm{sigmoid}}\nolimits\left(\frac{r_e-\gamma_{t(e)}}{\tau}\right).
```


乘子允许为负；每类型做 80 次 float64 二分，再转回模型精度。程序用旧奖励回代，验证恢复旧 robust 概率，并核对 drop 的类型概率和。

## 4. 识别训练与选模

从对应 seed 原 reference/last.pt 加载权重，冻结全部 CPF，只训练原 221,184 参数的关系分支。效用网络不进入识别模型前向。

固定新概率做 3 epoch 适配，epoch 0 也参与 VAL CW native AUC 选模。损失、学习率、batch size、图种子和 16 图概率平均继承原配置，CW/OW 分别按候选集归一化；bias 和温度由 VAL 确定。

GPU 1/2/3 各运行一个 seed。各进程逻辑 cuda:0 对应 CUDA_VISIBLE_DEVICES 指定的物理卡。

## 5. 核心验证指标

| 层面 | 指标 | 产物 |
| --- | --- | --- |
| 效用监督迁移 | Q5 总体及 OI/OE/AE Spearman、方向平衡准确率、RMSE、类型常数/边查表、90% 覆盖率 | utility_comparison.json；各 seed/utility/audit.json |
| 识别收益 | CW/OW native、exact AUC；相对 original/drop 逐 seed 差值 | summary.csv、aggregates.json、paired_comparisons.json |
| 工作点与校准 | VAL bias 下 TEST HM、校准 NLL/ECE | summary.csv、runs/*/metrics.json |
| 未见属性判别 | 给定真实物体的完整属性准确率、多未见候选子集准确率 | conditional_diagnostics.json |
| 结构与成本 | 类型期望边数、预算恢复误差、所选 epoch、各阶段耗时 | budget_checks.json、structure.json、progress/*/*.json |

重点是“跨折效用排序提高”和“同预算识别提高”是否同时出现。旧噪声矩阵和消融继续引用 Strong v1；本轮集中完成三组方法的九行主结果。

## 6. 数据组织

旧数据在 /data/xhw/experiments/trustcpf_strong_v1/work；新数据在 /data/xhw/experiments/trustcpf_modelcond_v1/work。

- reuse.json 和进度 identity.json：原输入路径及教师、配对标签、对照文件哈希。
- responses/fold1.pt…fold5.pt、reference.pt：96×8 特征、支持图像编号、权重、教师哈希。
- utility、utility_all：网络、标准化统计、选模历史和 Q5 预测。
- runs/drop、runs/original：旧对照；runs/conditioned：新增模型及 VAL/TEST 逐图像预测。

完成状态：21/21 阶段、9/9 识别结果、6/6 审计、18/18 响应缓存，complete=true。

打包包含本轮全部新增结果、六组复用对照、执行代码与溯源信息。原始大体积教师和配对输入继续保存在 Strong v1 目录及先前下载的归档中。新下载包通过 JSON 清单定位，SHA-256 校验后用 Python 解压。

## 7. 本机验证

已检查公式、支持加权、响应的模型依赖性、分批一致性、预算 KKT、Query 标签复用、三 seed 全流程、冻结 CPF、旧结果不改写及续跑。另用归档真实教师权重验证 96×8 计算，并对三个实际 seed 验证预算恢复。

下载脚本通过模拟传输测试：清单解析、失败重试、完整包复用、损坏包拒绝、中文文件名解压。正式 UT-Zappos 新训练由服务器命令启动。
