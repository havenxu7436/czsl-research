# v2 · TRUST-CPF 条件知识残差

**[修订实验方案](REVISED_PLAN.md)** · **[公式与核心代码](FORMULAS_AND_CODE.md)** · **[上传、训练与下载命令](SERVER_COMMANDS.md)**

从历史 G0 的 CPF＋完整静态 CP 出发，固定物体评分和属性／物体 CP 投影，将知识残差加入条件属性项。Base、Knowledge、Shuffle 各运行三个 seed，共 9 次训练；物理 GPU **1 / 2 / 3** 分别运行 seed3351 / 3352 / 3353，每张卡依次执行三组，每组 10 epochs。

本版已归档代码、配置、三个 G0 初始权重和本机验证。正式 UT-Zappos 的九组训练结果待服务器产生；本机流水线验证使用合成视觉特征。

## 目录与实现入口

| 分区 | 内容 |
| --- | --- |
| [code/](code/README.md) | 模型、训练、评价、续训、汇总、三卡队列、打包解压与测试 |
| [configs/](configs/README.md) | 最终配方、96条知识、固定打乱附件、缓存指纹、9组矩阵 |
| [analysis/](analysis/README.md) | 核心指标、两项配对差值、结果字段与解释 |
| [references/](references/README.md) | v1证据、G0来源、最终v2相对原方向的修改 |
| [results/](results/README.md) | 初始权重、验证记录、部署ZIP、原始发布成员与恢复映射 |
| [inputs.json](inputs.json) · [runbook.json](runbook.json) | 全文件与命令索引 |

## 从仓库恢复并启动

服务器已克隆仓库时，在仓库根目录执行：

```bash
conda activate vgcpod_v1
git lfs pull --include="TRUST-CPF方案/v2/**"
python tools/register_trustcpf_v2.py --check
REPLAY="$HOME/experiments/trustcpf-v2-replay"
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage v2 --dest "$REPLAY"
cd "$REPLAY/trustcpf_conditional_v2"
python run.py setup
source scripts/env.sh
"$CPF_PY" -m unittest discover -s tests -v
mkdir -p logs
nohup bash scripts/run_all.sh >> logs/queue.log 2>&1 &
```

`REPLAY` 应为新的绝对目录。`setup` 默认查找服务器已有的 `/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14`，并核对 G0 的特征、文本与样本轴指纹。执行包不需要重新编码 CLIP 特征；完整外部依赖见 [external_inputs.json](external_inputs.json)。从 Windows 上传 ZIP、查看九组完成状态及下载结果的完整路径见[实践命令](SERVER_COMMANDS.md)。

## 最终设置

三组共同训练 `Bc/Cc/Wc`，共36,864个参数；Knowledge与Shuffle增加73,729个消息参数和残差标量，共110,593个。`Bo/Co/Au/Av` 固定，完整静态CP始终保留。已有条件矩阵学习率2e-4、CP读出6e-5，新消息矩阵3e-4、alpha1e-3；batch64、10 epochs。真实与打乱附件按类型保持source多重集和目标度数。

各组以epoch1–10的VAL CW native AUC选同一权重评价CW/OW，TEST沿用VAL选定的bias和温度。主要结论同时依赖Knowledge−Base与Knowledge−Shuffle的逐seed差值。

## 校验

```bash
python tools/register_trustcpf_v2.py --check
python tools/register_trustcpf_v2.py --restore-check
```

40个发布成员保持原始字节身份；仓库阅读文档改用GitHub数学排版和相对源码链接。原始文档也随包归档，详见[成员映射](results/release_member_map.json)和[来源说明](来源与许可.md)。
