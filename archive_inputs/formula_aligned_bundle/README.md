# 0601 原设计复刻：代码、配置与运行资料归档

归档日期：2026-10-11。先读本文件，再读 [RUNBOOK.md](RUNBOOK.md)。

本文件夹保存当前 **r2 主公式对齐版本** 的完整执行源码、配置模板与生成器、知识输入、验证程序和原设计。它不是此前的 native CPF residual 版本，也没有混入旧版指标作为本轮结果。

准确范围是：**UT-Zappos 上，0601 选定主评分/损失公式＋明确冻结的 CPF 接口**。归档齐全不代表整份原方案的全部备选、数据集与附加分析已经实现，更不代表新的 CUDA 训练已经完成。

## 文件位置

| 目录/文件 | 用途 |
|---|---|
| `code/faithful_0601_20261011/` | r2 ZIP 原样解压：模型、训练、评估、队列、配置生成器、逻辑资产与测试 |
| `release/0601_formula_release_20261011_r2.zip` | 上传服务器的原始执行包；与之前交付完全一致 |
| `configs/run_plan.json` | 本轮选定预算、路径、资源与选模设置的可读规格；不是替代 YAML 的新启动入口 |
| `configs/variant_matrix.json` | A–H、J0/Jtext、四个 H 对照的开关与 seed 清单 |
| `knowledge/text_inputs/ut-zap50k/` | 与现有缓存指纹一致的原始 OE/AE/OI 文本、标签与审阅记录 |
| `knowledge/logic_reconstruction/` | B/F/H 的重建逻辑知识原文、标签、银行构建程序及出处；不是作者遗失资产 |
| `environment/` | 已下载的历史服务器真实依赖快照；不是本轮 CUDA 环境验收结果 |
| `docs/ZSCL实验设计0601.md` | 用户提供原方案，逐字节保留 |
| `docs/公式与代码对应索引.md` | 原设计章节对应哪些代码、哪些是接口选择、哪些未覆盖 |
| `docs/依赖与未覆盖项目.md` | 外部大资源、运行限制、剩余实验与归档边界 |
| `docs/0601逐公式复刻审计_20261011.md` | 旧 residual 版本的历史审计；其中问题是本轮修正背景，不是 r2 运行结果 |
| `verification/` | 原 r2 的交付验证记录、此次归档检查记录 |
| `tools/verify_bundle.py` | 不依赖 torch 的全文件指纹/ZIP/知识输入核验；可另选 CPU 测试 |
| `tools/check_server_inputs.py` | 服务器外部输入只读检查，不训练、不下载、不覆盖数据 |
| `bundle_manifest.json` | 全归档逐文件 SHA256 清单 |

## 关键公式

G/H 的条件分数是 factorized head 的**原始输出**，不是 native CPF 分数上的残差：

```text
conditional = head_raw
joint       = z_obj[:, :, None] + conditional + lambda * prior
attr_loss   = CE(conditional[batch_index, GT_object, :], attr)
```

训练与推理共用 joint 构造。逻辑节点采用 §7.2 的 softmax、logsumexp 与 sigmoid；prior 采用原文 sample std＋epsilon。具体函数与运行接线见代码索引。

接口不能隐去：当前原 CPF `ddm.py` 视觉底座、共享视觉向量广播、GloVe300 重建描述、768→300 全局 CLS 投影、D/E/F 无 Feature 条件函数均已声明。**没有把 GloVe 描述节点冒充原方案 §7.3 推荐的 CLIP 描述编码。**

原方案 §2 表格写 `ddm_v3.py`；当前冻结可运行底座是 `ddm.py`。两者源码均在归档，但存在文件不等于已经对齐或训练了 `ddm_v3` 版本。后续若要换底座，必须单列新协议。

## 配置与执行

每组 30 轮，seeds=3351/3352/3353，每 seed 14 组，共 42 次全新训练。30 轮是本轮明确选定的共同预算，**原方案未指定轮数，也不保证收敛**。

正式 42 份 YAML 与 `protocol.json` 由 `prepare.py` 在服务器核验实际输入后生成，保存在新工作目录的 `configs/`。包内 `assets/A_ref.yml` 等有旧 10 轮路径/预算，它们仅是生成器读取的基模板；**不要直接拿它们启动本轮训练，不要直接运行 `legacy_prepare.py`。**

1. 按 `RUNBOOK.md` 上传 `release/` 中 r2 ZIP 与 SHA256。
2. 在服务器 prepare 冻结 42 组配置，先执行 seed3351 的 14 组。
3. 核对公式 smoke、完整轮数、VAL 曲线与评估完整性，再按原协议补另外两 seed。
4. 所有结果写 `/var/tmp`；现有 `/data` 数据/代码/缓存只读复用。不要因截止时间降低质量。

数据集图片、GloVe 大文件、ViT 预训练权重、服务器原始 text cache、大模型断点**没有复制进桌面包**。本包能在目前已具备这些输入的服务器接续执行，不是无需外部资源的全新机器一键环境。详情见依赖说明。

本次只是本地归档，未连接服务器、未启动或重启训练、未修改已发 r2 ZIP 与旧实验。

## 校验

```powershell
# 在本文件夹打开 PowerShell，使用有 Python 的环境：
python .\tools\verify_bundle.py --bundle .

# 可选：需要 CPU torch、numpy、scipy、PyYAML、yacs。
# 在临时副本运行测试，不改动已校验的归档文件。
python .\tools\verify_bundle.py --bundle . --cpu-tests
```

CPU 测试不能替代服务器真实图像/CUDA smoke，也不会产生研究指标。`verification/desktop_bundle_verification.json` 保存此次实际核验结果。

独立附带的同名 ZIP 是本桌面文件夹的便携副本；服务器训练仍上传 `release/` 中原 r2 包，而不是把资料归档 ZIP 当作训练入口。
