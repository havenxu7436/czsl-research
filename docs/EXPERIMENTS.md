# 研究阶段索引

按研究依赖顺序回溯整理；同日工作可能交叉。原始证据和现有总档案支持此顺序，未伪造实验完成时刻。

| 顺序 | 阶段 | 实际状态 |
|---|---|---|
| 01 | [原方案、三数据集基线与知识重建](stages/01_stage.md) | 已归档历史实验 |
| 02 | [早期 MVP、跨数据集验证与固定 prior 校准](stages/02_stage.md) | 已训练＋无训练后处理 |
| 03 | [UT v1 三 seed 语义归因与工程修复](stages/03_stage.md) | 已执行历史归因；保留缺项边界 |
| 04 | [TRAIN 描述审核、UT v2 与逐图分析](stages/04_stage.md) | 已训练＋诊断 |
| 05 | [名称锚点 EKR](stages/05_stage.md) | 已训练：seed0、三组各3轮 |
| 06 | [EKR 机制诊断](stages/06_stage.md) | 既有权重回放；0新训练 |
| 07 | [共享值对齐 EKR v2：3轮](stages/07_stage.md) | 已训练：4组各3轮 |
| 08 | [TRAIN 支持探针、SG-EKR与视觉原型诊断](stages/08_stage.md) | 支持诊断＋3组3轮；VP仅方案 |
| 09 | [EKR v2 训练预算回溯：3→10轮](stages/09_stage.md) | 续训；不是4个新run |
| 10 | [CPF＋LogiCZSL B/F直接结合](stages/10_stage.md) | 已训练：UT3351、四组10轮 |
| 11 | [B/F梯度复核与条件属性 StageA](stages/11_stage.md) | 诊断/缓存/丢弃式检查；非正式pilot |
| 12 | [条件属性 pilot r1→r2与固定混合回放](stages/12_stage.md) | r1工程失败；r2五组10轮＋回放 |
| 13 | [Generic同结构知识对照＋25%混合](stages/13_stage.md) | 三项新训练＋G_K复用 |
| 14 | [冻结 CLIP 探针 v1→r2](stages/14_stage.md) | v1检查失败；r2七配方、0optimizer |
| 15 | [TRAIN逐句审查、知识来源与研究回溯](stages/15_stage.md) | 36逐句视图＋图片审查；0新训练 |
| 16 | [residual-0601：统一A–H与sanity](stages/16_stage.md) | 已完成12组×10轮、48份评价 |
| 17 | [formula-aligned-0601：逐公式修正r2](stages/17_stage.md) | 代码/交付/CPU验证；未收集正式训练结果 |
| 18 | [residual描述删除扩展与后续候选](stages/18_stage.md) | 仅代码交付；本地未收集正式结果 |

历史总档案包含更细的24项执行/失败/回放事件和公式修正总表；这里将有直接依赖的事件归到18张研究阶段卡。不能将阶段数当作独立训练次数。
