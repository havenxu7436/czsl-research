# 冻结 CLIP 知识证据探针 r2（2026-10-10）

r2 修复 v1 错误地禁止 TRAIN/VAL 组合类别重叠的问题。当前广义验证集包含 15 个 seen 和 15 个 unseen 组合，对应 877／2337 张图片；seen 类别重叠正常，图片 ID 必须互斥。新增 `splits.py` 和 `--check-inputs-only`：核对真实 split、metadata、StageA 身份顺序及原生保存的 VAL 标签，不加载 CLIP、不需要 GPU、不解码图片。r2 使用新的独立目录，不替换或续用 v1 身份回执。

这是 Generic＋25% 后的证据检查，不是新一轮 CPF 正式训练。冻结 OpenAI CLIP ViT-L/14（224px），完整编码审核过的 v2 OE/AE。原 CPF、StageA、Generic 结果均只读；optimizer step = 0；不读 TEST 图片或生成 TEST 分数。

## 固定对照

| 组别 | 固定名称 | 新增描述 |
|---|---|---|
| T_name | 原对象、属性名称 | 无 |
| T_K | 相同 | 正确 AE＋OE |
| T_perm1 / T_perm2 | 相同 | 原 StageA 的两套固定描述错配 |
| T_AE | 相同 | 仅正确 AE |
| T_AE_perm1 / T_AE_perm2 | 相同 | 仅错配 AE |

主线先检验 OE＋AE；AE-only 是预先规定的定位对照，不能运行后再创造新模板。每组每组合九个等权文本单元。OE＋AE 是三条 AE × 三条 OE；名称组重复同一句九次，AE 组每条 AE 重复三次。均先归一化逐句文本特征，再取余弦分数均值，不重新归一化平均向量。只编码唯一文本，数学上等价。名称在错配组不动。

模板：`A photo of {object} made of {attribute}. It has {AE}. Object shape: {OE}.`；名称组只保留第一句，AE-only 去掉最后一句。自然名称映射、全文提示和 token 审计会导出。任何全文超过 77 token 均停止，不静默截断。描述可能重复标签词，错配也可能与标签产生冲突；因此正确胜过错配不够，还必须胜过名称。

## 诊断与预算门槛

TRAIN 22998 张、VAL 3214 张，顺序与已有 StageA 完全相同。全部候选为 12 对象 × 16 属性，不根据真实对象或组合存在性删掉任何候选。编码时不读取图像标签；诊断时才取真实对象对应的全部 16 属性，隔离对象识别贡献。它是 GT-object 机制诊断，不是可部署推理结果，不能与 CPF CW/OW AUC 混称。

固定 `q_T = softmax(mean_cosine / 0.05)`，不拟合温度、投影、模板或描述权重。报告 TRAIN/VAL all，以及 VAL seen/unseen 的 micro、组合 macro、NLL、逐类混淆，名称→知识和 C_text→教师的纠正/破坏。CPF 条件参考取已有 score-induced object conditional，温度同为 0.05；两种不同模型的 NLL 不表示已公平校准。

门槛：官方 VAL unseen 的组合 macro 比名称及每套错配分别至少高 1.0 个百分点；VAL unseen micro 不低于名称；TRAIN 组合 macro 不低于名称。**这是预设的投入筛选门槛，不是统计显著性证明，也不是原 Generic AUC 门槛。** 另报告按组合 bootstrap 的描述性区间（2000 次、固定 seed），不据此夸大跨数据集泛化。

OE＋AE 通过：考虑同结构 Generic＋25% 的无教师／名称教师／正确知识教师／两套错配教师 pilot。只有 AE-only 通过：只考虑 AE-only 教师，先不要引入 OE。两者不通过：停扩大该教师配方，只针对 TRAIN 混淆做一次有依据的描述适用性检查。任何后续模板改动都算新开发实验，不能拿当前 VAL 重复搜索。

即使通过也只说明该教师有可用信号；不保证蒸馏能改善 CPF。后续训练继续沿用相同 60864 参数 Generic、原初始化／步序／十轮预算、25% 固定推理混合、原未混合 head CE，只增加受控教师项；不同时改混合比例、视觉骨干和逻辑损失。正式训练程序不在本包。

## 来源与依赖

官方 CLIP 源码固定 commit `a1d071733d7111c9c014f024669f959182114e33`，MIT license 随包。权重使用官方 URL＋SHA256；仅复用完整匹配的 ViT-L/14 权重，不接受同名的其他模型。五个已下载的 Linux Python3.8 依赖 wheel 随包；离线安装到本次 work/deps，`--no-deps` 禁止升级 cpf 的 Torch/Torchvision/NumPy。需要原环境已有可用 Torch、Torchvision、Pillow。首次缺少模型权重时仍需要访问官方权重下载地址。

原 metadata 与 StageA 经过已归档 SHA 校验后才允许使用兼容旧序列化的 trusted load；不把 `weights_only=False` 推广到未知文件。图片优先用 CPF 的 exact `images/{image}`；若缺失，只读 metadata `_image` 指向的原图及原来明确审核的两条映射，不做模糊文件名搜索。每张图片记录 SHA，恢复时重新核对。

## 运行及恢复

Linux 使用 `/data/xhw/miniconda3/envs/cpf/bin/python`，`CUDA_VISIBLE_DEVICES=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。入口 `probe.py --work /var/tmp/czsl_clip_knowledge_probe_20261010_r2 --export /var/tmp/czsl_clip_knowledge_probe_20261010_r2.zip`。详细命令见外部《CLIP探针划分检查修复_r2_操作方法_20261010.md》。先执行同样入口加 `--check-inputs-only`，看到划分预检成功后才启动真实编码。

每 512 张写一个已校验特征分片。`--resume` 可复用完成分片；原图、源代码、历史资产、数值环境变化则停止。已有工作目录首次启动会 STOP，不能删除重跑；同目录并发有 OS 文件锁阻止。中断下载会保留 `.partN`，恢复会另下载新分片，不自动删旧文件。

最终 ZIP 包括教师分布、余弦分数、CLIP 图像特征、身份清单、全文提示、指标、源代码与成员 SHA 清单；不包括原图、CLIP 权重和依赖目录。异常也会导出诊断 ZIP，`execution_passed=False`。完整成功以 `KNOWLEDGE_PROBE_EXPORT_COMPLETE ... execution_passed True` 为准；知识 gate=False 可以是正常阴性结果。

本地仅完成 CPU 测试与真实 metadata/提示审计；没有服务器 GPU 或真实图片前向测试。下载与推理耗时由服务器日志估计，不承诺固定分钟数。
