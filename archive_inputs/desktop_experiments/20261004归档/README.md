# 三数据集 OI/OE/AE 逻辑知识输入：重建版 v1

状态：**结构完整、语义待复核的研究草稿**。不是 LOGICZSL 作者的原始 GPT-4 知识，不可据此声称“严格复现原作者知识”。这些文件可先用于 CPF 新逻辑模块的接口、形状和冒烟测试；正式 A–H 结果前按下文完成语义复核与知识版本冻结。

## 包含什么

| 数据集目录 | 属性 | 对象 | OI 父类 | OE/AE | 来源 |
|---|---:|---:|---:|---|---|
| `ut-zap50k/` | 16 | 12 | 4 | 每项3条 | 先前手工重建的 UT JSON，原样沿用 |
| `mit-states/` | 115 | 245 | 19 | 每项3条 | WordNet 3.0 首词义自动草稿 |
| `cgqa/` | 413 | 674 | 23 | 每项3条 | WordNet 3.0 首词义自动草稿 |

每个数据集目录都有 `oi_dict.json`（父类→对象列表）、`oe_dict.json`（对象→3条描述）、`ae_dict.json`（属性→3条描述）、按 CPF 源码 `sorted(...)` 顺序排列的 `objects.txt` / `attributes.txt`、`entry_provenance.csv`、`review_status.csv` 和 `manifest.json`。另有 `K_oa_text_lexical.csv`（行=对象、列=属性）及分布报告，供独立文本先验实验筛选。`scripts/validate_server.py` 用服务器真实 split 检查标签、顺序、pair 身份、覆盖率、三描述、先验维度和值域及文件 SHA256；不修改服务器数据。

**`K_oa_text_lexical.csv` 不是正式确认的兼容性知识。** 它只由名称和 OE/AE 描述计算，公式为 `0.75×词级 TF-IDF 余弦 + 0.25×字符三元组 TF-IDF 余弦`，不使用 pair 列表的存在性，平均值约 UT `0.0161`、MIT `0.0093`、C-GQA `0.0082`，分数很小且可能反映词形相似而非真实可行性。只能作为**非官方 text-only K 候选/负对照**，正式用于 D/E/F/G/H 前必须做分布、随机样本语义和验证集消融检查；不合格就另行构建。先前 UT 的 `build_compatibility.py` 用 train/val/test pair 列表生成二值候选掩码，更不能拿它当语义 prior。

## 来源与关键限制

- UT：沿用[先前的重建文件](../logiczsl_ut_zappos_reconstructed/README.md)，不代表作者提供的知识。
- MIT/UT 类别来自本地 `compositional_split_natural.tar.gz`；C-GQA 类别来自[更新版 C-GQA split 的公开镜像](https://huggingface.co/datasets/Meldashti/vlm-compositionality-embeddings/tree/main/cgqa/compositional-split-natural)。C-GQA 的[原始项目](https://github.com/ExplainableML/czsl)说明其数据集已更新。因此必须以**你服务器实际训练用的 split**为准；校验不通过就停止，不要强行使用镜像版本。
- MIT/C-GQA 文本来自[Princeton WordNet 3.0](https://wordnet.princeton.edu/)的自动首词义、上位类和释义。随包保留 `WORDNET_LICENSE.txt`；详见[许可条款](https://wordnet.princeton.edu/license-and-commercial-use)。程序没有用图像、样本标签或 pair 的 train/val/test 身份推断知识；只用三组 pair 列表提取完整的属性/对象词表。
- 自动选义会错。MIT 的 360 条、C-GQA 的 1087 条目前**全部待人工确认**；其中 MIT 有 286 个多义条目、15 个非形容词属性词义，C-GQA 有 751 个多义条目、85 个非形容词属性词义，另有 6 个仅靠名称模板补足的条目（`nightstand`、`placemat`、`wetsuit`、`indoors`、`pointy`、`unpeeled`）。这些统计仅用于安排复核，不是模型性能。
- WordNet 父类只是宽泛语义分组，不等于 LOGICZSL 原论文的对象逻辑树；目前是单层 OI，不能拿来报告“树深度 1/2/3”分析。

## Windows PowerShell：上传到服务器

以下用你此前截图中的 `xhw@211.87.232.122`、默认 SSH 端口 22。若实际 IP/端口已变，只改主机与 `-P`。**这两条在本地 Windows PowerShell 运行，不是在 SSH 会话内运行。**

```powershell
$pkg = 'C:\Users\llenovo\Documents\Codex\2026-09-28\new-chat\outputs\kcc_logic_3datasets_reconstructed_v1.zip'
Get-FileHash -Algorithm SHA256 -LiteralPath $pkg
scp -P 22 $pkg 'xhw@211.87.232.122:/data/xhw/czsl_repro/data/'
```

记下本地 SHA256。文件只上传到新目录的 ZIP，不覆盖 `CPF_fixed`、基线日志或数据图像。

## Linux 服务器：解包与核对

登录服务器后运行：

```bash
cd /data/xhw/czsl_repro/data
sha256sum kcc_logic_3datasets_reconstructed_v1.zip
unzip -l kcc_logic_3datasets_reconstructed_v1.zip | head -30
unzip -n kcc_logic_3datasets_reconstructed_v1.zip

python3 kcc_logic_3datasets_reconstructed_v1/scripts/validate_server.py \
  --data-root /data/xhw/czsl_repro/data \
  --knowledge-root /data/xhw/czsl_repro/data/kcc_logic_3datasets_reconstructed_v1
```

服务器 SHA256 必须与本地一致；校验应出现 `ALL REQUESTED STRUCTURAL CHECKS PASSED`，同时打印三个 `pending_review` 数。这个输出**只证明结构和 split 身份匹配，不证明语义知识正确**。若 C-GQA/任何数据集报 pair identity mismatch，暂停后续运行并用实际服务器 split 重新生成，不要修改 JSON 键或跳过检查。`unzip -n` 不覆盖已有文件；如目标目录已存在且想替换版本，先另起 `v2` 目录，不直接覆盖旧实验输入。

服务器安装在其他根目录时，只改 `--data-root`，三个子目录名必须仍是 `ut-zap50k`、`mit-states`、`cgqa`。如果服务器的 C-GQA split 文件版本与本包不同，可以把服务器三个数据集的 `*_pairs.txt` 下载到本地，用 `scripts/generate.py --source-data-root <本地镜像目录>` 重新生成新版本；**不得把基于另一 split 的旧文件强行标记为通过**。

## 正式实验前的四个门槛

1. 按 `entry_provenance.csv` 和 `review_status.csv` 逐项核对词义、上位类与描述，修正明显错义和名称模板。每条批准项填写 `status=approved`、复核人和日期；改 JSON 后更新 manifest SHA/版本，不在旧 run 目录中替换知识。`--require-reviewed` 会拒绝仍有待复核条目的版本。
2. 再次执行服务器结构校验；抽查高风险多义词、UT 鞋类名称及 C-GQA 罕见类。确保 OE/AE 文本没有借助验证/测试图像或标签编写。
3. 审计包内的 `K_oa_text_lexical.csv`：检查它不是近常数、与真实语义兼容性一致、覆盖所有候选 `(object,attribute)`，并做 uniform/shuffled K 对照。若不合格，另行定义并冻结独立的 `K_oa`；不能因为矩阵有278,362个数就自动视为知识质量合格。该步骤属于手册 Phase 1。
4. 给正式知识一个新版本（如 `v2_reviewed`）并冻结所有 JSON、脚本、配置、SHA256，之后 A–H 只读它。保留本 v1 作为可追溯的草稿/负对照，不把它的结果称作作者知识复现。

## 如何重新生成 v1

包中的 `scripts/generate.py` 是 OI/OE/AE 生成器，需要 Python、NLTK 与 WordNet 3.0 语料；`scripts/build_text_prior.py` 从生成的描述构建文本先验，只需 Python 标准库。它们从原始 split **只读取类别词表**，不会以已存在的 pair 当作兼容性标签。生成后用 `scripts/validate_server.py` 检查。WordNet 自动选义不可替代人工复核。

```bash
python scripts/generate.py \
  --split-tar /path/to/compositional_split_natural.tar.gz \
  --cgqa-split-dir /path/to/cgqa_split \
  --ut-existing-dir /path/to/logiczsl_ut_zappos_reconstructed \
  --wordnet-dir /path/to/nltk_data \
  --output-root /path/to/kcc_logic_3datasets_reconstructed_v2

python scripts/build_text_prior.py \
  --knowledge-root /path/to/kcc_logic_3datasets_reconstructed_v2
```

如已从服务器取回三个数据集的准确 split，可再加 `--source-data-root /path/to/server_data_mirror` 覆盖上述 tar/镜像来源。新版本应放新目录、重新校验并单独归档。
