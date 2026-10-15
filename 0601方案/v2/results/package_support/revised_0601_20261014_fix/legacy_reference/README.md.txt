# 0601 主评分公式复刻包

日期：2026-10-11。范围：UT-Zappos 的选定主公式与必要对照，不宣称已经覆盖整份设计的所有备选或恢复原作者知识资产。

## 已对齐的主线

- G/H 直接使用 §6.1 factorized head 输出作为 `z_attr_obj_K`，没有原 CPF composition/attribute 残差。
- §9：TRAIN 与 INFER 共同调用 `score = z_obj[:, :, None] + z_attr_obj_K + lambda * prior`。
- §8.2：属性 CE 直接读取 `z_attr_obj_K[batch_index, GT_object, :]`，不再加 native attribute logits。
- §7.2：object softmax；attribute 对 masked joint score 做 object-axis logsumexp 再 softmax；pair sigmoid；description sigmoid(dot)。逻辑算子直接消费概率，没有再次 softmax。
- §3.2/4.5：sample-std＋epsilon centered prior、min-max＋epsilon。
- G/H 无 native composition 分支梯度；ViT、视觉投影、知识 head 联合优化。训练和推理评分构造一致，但 dropout、候选集合随 train/eval 设置正常变化。

## 必须说明的接口选择

原文不是全部细节齐备的可运行代码，以下不是原文唯一规定：

1. 复用当前 CPF `ddm.py` 的注意力与多层视觉融合，不把 §2 的简化回顾改写成新的基线。共享 B×300 视觉向量是 §6 head 的广播输入。
2. `z_obj` 使用 CPF cosine/T，T=0.05，训练和推理都如此。head 保留 §6.1 的原始输出，不额外乘 gain 或除温度。各项不必数值同尺度，但协议明确且固定，不用 TEST 搜索最优尺度。
3. 原文 §5 未完整定义无 Feature K 的 `z_attr_obj`：D/E/F 选用原 CPF composition logit＋broadcast attribute logit，构建完整 N×M 网格。GT-object CE 读取此条件分数。J0/Jtext 分别以无文本知识/有文本知识、无 prior 无逻辑的同接口训练，帮助排除评分接口变化的混淆。
4. A/C 保持原 CPF 的任务损失和推理；B 保持原 CPF 任务 CE 和推理，只修正逻辑 grounding。这些基线不被强行改成 G/H 架构。
5. 原 CPF ViT CLS 为768维，重建 GloVe描述为300维：复用已有 `objc_embedder` 投影后的全局 CLS 作为 image_feat。OE、AE 同用这个全局向量，做 raw dot＋sigmoid。**不是 CLIP 描述相似度资产的复现。** 若要求 §7.3 推荐的 CLIP 编码器，需要单独版本和重新配对训练，不把本包冒充该版本。
6. OI/OE/AE 的 p-mean violation 算子保留现有重建实现，q=3，β=0.01 只乘一次；§7.3 没有额外 pair loss，因此不为消费 pair 节点擅自添加第四个损失。
7. 知识资产为重建描述/层级、mean 聚合、全对象×属性 template-only pair、text-only prior。没有作者遗失数据，也没有伪造 LLM feasibility。

## 训练协议与范围

每 seed 14 组：A–H 八组＋J0/Jtext 两个接口对照＋H_no_pair/H_shuffle/H_random_prior/H_uniform_prior 四个 sanity。主 H 本身为 template-only。

默认 seeds=3351,3352,3353，共42次新训练；每次30轮，batch64、ViT B16全量微调、原 Adam/学习率/23及30轮 milestone 规则。原方案没有规定轮数：30是选定的共同预算，不是所谓官方轮数或收敛保证。预算增加后，A/C也重新训练，不拼旧10轮结果。

head 初始化使用600000＋对应seed，在fork_rng内完成，不扰动原 CPF 初始化与后续随机流。所有控制使用相同配对seed。

仅允许 epochs<37，阻止进入尚未对齐的 minority augmentation。若验证曲线显示训练预算不足，先形成新协议统一修改比较组，不偷偷让增强分支绕过知识 head，也不把随意续跑当作同协议结果。

选模：首次严格最大 VAL CW AUC；同一个 checkpoint 评价 VAL/TEST CW/OW。旧 TEST 已被观察，报告标注本轮为后续复刻实验，不宣称仍是未触碰 holdout。

## 执行

先运行 prepare，冻结全部42组。队列可用 `--seeds 3351` 先完成一个完整单seed矩阵，再 `--resume --seeds all` 补另两个seed；不改原protocol，也不重复已完成任务。

先跑单seed是执行顺序，不是三seed质量要求的替代。执行或真实图梯度检查失败会停止发新任务并保留日志；负增益不算执行失败，也不会被隐瞒。

队列无 deadline 截断；不抢占已有 GPU进程；一次一GPU一任务。允许显存使用<512MiB、空闲>=16000MiB、利用率<=10%的授权GPU。新计划要求至少100GiB空闲磁盘，不删除旧实验。

服务端每个公式/逻辑组使用4张真实 TRAIN 图片验证主公式、初始化、CPU/CUDA RNG、GT CE、head/视觉/ViT梯度和逻辑梯度，**无 optimizer step、无 TEST 图像**，通过才正式训练。

发布前本地 CPU 测试不是服务器 CUDA 训练证明。请使用交付的独立实践手册完成上传、校验、启动、查看、恢复与下载。

## 输出

- `protocol.json`：完整公式接口、配置、冻结输入哈希、seed/预算。
- `runs/<group>_s<seed>/`：训练、真实图 smoke、VAL/TEST CW/OW评价、配置与模型身份。
- `summary/0601_原公式版训练汇总.md`、`metrics.csv`、`summary.json`：实际完成状态、单seed表、均值与样本标准差、配对差分、VAL曲线和末尾最佳checkpoint标记。
- ZIP默认不含大权重；服务器保留 best 权重及 model/Adam/RNG耐久断点。不要清除工作目录。

完整原方案的未选备选、描述数/树深/噪声、LLM/mix质量、其他数据集、CLIP描述编码、定性失败分类等仍单列为未完成；42组完成不等于它们自动完成。
