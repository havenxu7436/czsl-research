# v1 · VG-CPOD

**[完整公式及核心代码](VG-CPOD公式及代码v1.md)** · [数学与实验设计](数学与实验设计_v1.md) 

状态：实现、配置和本机CPU验收已归档。12项检查覆盖数学恒等式、合成缓存的9组训练与四套评价、续训及诊断；本路线未归档服务器正式训练精度。

## 实验设置

- 冻结CLIP ViT-L/14图像与文本双编码器，224px，768维共同空间。
- UT-Zappos natural split；全局图像与类别查询特征统一缓存。
- V0、V1、V1L、V2–V7；正式计划27次训练、每组10 epoch。
- 首次严格最大VAL CW `AUC_native`选模；同一权重评价VAL/TEST CW/OW，分别保存`AUC_exact`。

## 分区

```text
VG-CPOD方案/
  README.md
  v1/
    README.md
    VG-CPOD公式及代码v1.md
    数学与实验设计_v1.md
    服务器实践_v1.md
    code/          # 模型、缓存、训练、评价、测试、CLIP源码和审计算子
    configs/       # 默认配置、9组预览、矩阵、名称与输入哈希
    results/       # CPU记录、公式审计、原始文档、部署ZIP与成员映射
    inputs.json
    runbook.json
    external_inputs.json
```

科学代码、原始配置和冻结包成员保持字节身份。仓库文档使用相对链接和GitHub数学块；原始文档保存在package support。

## 运行入口

在仓库根目录执行：

```bash
git lfs pull
python tools/reproduce.py verify
REPLAY=/srv/czsl-reproduce-vgcpod-v1
python tools/reproduce.py materialize --line VG-CPOD方案 --stage v1 --dest "$REPLAY"
VG_PKG="$REPLAY/vgcpod_v1"
python "$VG_PKG/tools/verify_release.py"
```

本次恢复只复制VG-CPOD的36个冻结包成员，不复制前两条路线的代码、GloVe或已训练模型。数据集图像和ViT-L/14权重按[外部输入索引](external_inputs.json)准备。

激活原CPF Python3.8/torch1.13.1+cu117环境后：

```bash
export CPF_PY="$(command -v python)"
export VG_DATA=/data/xhw/czsl_repro/data/ut-zap50k
export VG_CLIP=/data/xhw/.cache/clip/ViT-L-14.pt
export VG_WORK=/data/xhw/vgcpod_v1_runs
export VG_GPU=1
cd "$VG_PKG"
"$CPF_PY" -m pip install -r requirements.txt
bash scripts/prepare.sh
bash scripts/train_matrix.sh pilot
```

`prepare.sh`先检查数据/权重、运行CPU测试和编码，再完成真实TRAIN小批次的CUDA验收。pilot为seed3351的6组；随后`bash scripts/train_matrix.sh full`补齐9组×3 seed，并自动恢复已提交epoch。额外命令见[服务器实践](服务器实践_v1.md)。

## 配置与代码

- [模型、损失与初始化](code/vgcpod_v1/vgcpod/model.py)
- [冻结CLIP编码与类别查询](code/vgcpod_v1/vgcpod/features.py)
- [训练、续训和选模](code/vgcpod_v1/vgcpod/engine.py)
- [候选评价与两种AUC](code/vgcpod_v1/vgcpod/metrics.py)
- [配置索引](configs/README.md)与[27次训练矩阵](configs/experiment_matrix.json)
- [知识、配置与文件索引](inputs.json)和[命令索引](runbook.json)

分区存储的`code/`不直接充当运行目录；先materialize恢复配置、资产、BPE词表和文件布局。

## 结果与校验

- [结果索引](results/README.md)与[CPU验证说明](results/验证记录.md)
- [12项CPU测试日志](results/local-validation-20261018/tests_cpu.log)和[检查记录](results/local-validation-20261018/result.json)
- [公式梯度审计](results/formula-audit-20261018/derivative_qa.json)与[排版截图](results/formula-audit-20261018/screenshots/)
- [部署ZIP](results/deployment/VG-CPOD方案v1.zip)与[SHA256](results/deployment/VG-CPOD方案v1.zip.sha256)
- [36成员映射](results/release_member_map.json)与[来源及许可](来源与许可.md)

```bash
python tools/register_vgcpod_v1.py --check
python tools/register_vgcpod_v1.py --restore-check
```

名称CLIP、链式零样本、D0和完整VG-CPOD各自保持评分定义；本机合成数据检查不作为UT正式精度。
