# 0601 全部公式一次性修订：运行手册

本 release 已统一完成公式修正，不使用逐条修复方案。详细定义见 [公式及审计](复现公式及代码_统一修订版.md)。正式结果尚未产生；旧 r2 和旧结果保持不变。`legacy_reference` 仅为历史归档，不能用其旧命令启动本版。

运行环境沿用服务器 `/data/xhw/miniconda3/envs/cpf/bin/python`（Python3.8 / torch1.13）；不要升级现有环境。本版使用原 ViT B16 全参数训练，默认10epochs，seeds3351/3352/3353，aug_epoch37不触发。

## 1. 桌面上传：Windows PowerShell

桌面交付目录为 `C:\Users\llenovo\Desktop\0601core\0601统一修订_20261013`。上传 ZIP 和 sidecar：

```powershell
scp "C:\Users\llenovo\Desktop\0601core\0601统一修订_20261013\0601_unified_revision_20261013.zip" "xhw@211.87.232.122:/var/tmp/"
scp "C:\Users\llenovo\Desktop\0601core\0601统一修订_20261013\0601_unified_revision_20261013.zip.sha256" "xhw@211.87.232.122:/var/tmp/"
```

本手册不自动访问服务器或开始训练。

## 2. 服务器验证和解压：只执行一次

```bash
(
  set -euo pipefail
  cd /var/tmp
  sha256sum -c 0601_unified_revision_20261013.zip.sha256
  test ! -e /var/tmp/czsl_0601_revision_release_20261013
  /data/xhw/miniconda3/envs/cpf/bin/python -m zipfile -e \
    /var/tmp/0601_unified_revision_20261013.zip \
    /var/tmp/czsl_0601_revision_release_20261013
  /data/xhw/miniconda3/envs/cpf/bin/python -c \
    'import sys; from pathlib import Path; p=Path("/var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013"); sys.path.insert(0,str(p)); from common import verify_release; verify_release(p); print("RELEASE_HASHES_OK")'
)
```

重复执行会因路径存在停止，保护第一次解压/运行，不表示训练失败。先检查已有路径，不要删除旧实验。

## 3. 一次性启动完整计划

完整 `all`=96 jobs；39个主/控制 +57个追加小实验，全部同一修正版、全部 fresh10epochs，不重复训练 Rfull baseline。

磁盘预留按 `10+2×任务数 GiB` 计算：all202GiB、core88GiB、单独small130GiB；这是保守 checkpoint 预算，不是精确占用。旧截图中 `/data` 已接近满，不能存新实验；`/var/tmp` 的旧201G空间也不能视为现在仍有202GiB。先查看：

```bash
df -h /var/tmp /data0
nvidia-smi
```

如果 `/data0/xhw` 可写，推荐完整计划放该盘：

```bash
(
  set -euo pipefail
  mkdir -p /data0/xhw
  test -w /data0/xhw
  bash /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/start_revision.sh \
    --suite all \
    --work /data0/xhw/czsl_0601_revision_all_20261013 \
    --seeds 3351,3352,3353 --gpus 1,2,3 --max-workers 3
)
```

如果 `/data0` 没有写权限，使用**实际可写且空间足够**的位置；例如 `/var/tmp` 已确认≥202GiB时：

```bash
bash /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/start_revision.sh \
  --suite all --work /var/tmp/czsl_0601_revision_all_20261013 \
  --seeds 3351,3352,3353 --gpus 1,2,3 --max-workers 3
```

GPU0未默认占用；如4卡均可用，可显式换为 `--gpus 0,1,2,3 --max-workers 4`。即使传入卡号，队列也只选择 memory.used<512MiB、free≥16000MiB、util≤10% 的卡，不结束其他人的任务。

如果只需优先完成主要科学问题，可将 `--suite all` 改成 `--suite core`，work 改成 `czsl_0601_revision_core_20261013`，会 fresh 跑39个任务。**这只是实验范围选择，所有错误公式仍一次性全部修正。** 不能把 core 完成称为小实验全部完成。

CPU测试、源文件/split/cache/原GloVe哈希、磁盘预检通过后，才创建 frozen protocol。每个 revised job 先用4张真实 TRAIN 图像做 CUDA smoke；失败保留证据，停止后续发车、等待已启动任务结束。差成绩不会触发停止。

## 4. 监测

以下示例按 `/data0/xhw` 的 all work。若换了路径，同步修改所有后续路径：

```bash
/data/xhw/miniconda3/envs/cpf/bin/python \
  /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/status.py \
  --work /data0/xhw/czsl_0601_revision_all_20261013

tail -n 60 /data0/xhw/czsl_0601_revision_all_20261013_driver.log
tmux has-session -t rev0601_all_1013 && printf 'TMUX_RUNNING\n' || printf 'TMUX_ENDED\n'
pgrep -af 'revised_0601_20261013/run_suite.py|czsl_0601_revision_all_20261013/source/train.py'
nvidia-smi
```

训练 subprocess 通常显示为 `python -u train.py --cfg /data0/.../configs/xxx.yml`，因此还可：

```bash
pgrep -af 'train.py.*czsl_0601_revision_all_20261013/configs'
tail -n 30 /data0/xhw/czsl_0601_revision_all_20261013/runs/Rfull_s3351/train.log
```

`epoch10/10` 不等于 job complete：末尾还要验证最佳checkpoint、CW/OW、VAL/TEST四次评测。正式结束要求 driver出现 `selected_seeds_complete=True`，所有状态complete，ZIP导出完成。仅tmux消失不证明成功，检查失败状态与日志。

## 5. 明确恢复（不要重新执行 fresh start 脚本）

仅在旧队列/worker process group 确认结束后恢复：

```bash
tmux new-session -d -s rev0601_all_1013_resume \
  '/data/xhw/miniconda3/envs/cpf/bin/python -u /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/run_suite.py --work /data0/xhw/czsl_0601_revision_all_20261013 --gpus 1,2,3 --max-workers 3 --resume --export /data0/xhw/czsl_0601_revision_all_20261013.zip >> /data0/xhw/czsl_0601_revision_all_20261013_driver.log 2>&1'
```

恢复核对 protocol/config/source hashes，恢复 optimizer、RNG 和绝对 epoch，跳过完整完成组。没有 completed-epoch last_resume checkpoint 的失败训练不可假装无损续训；保留失败证据，新建 work。不得绕过接线失败或改冻结配置。

## 6. 全部完成后的 bias/错误分析与再次导出

```bash
(
  set -euo pipefail
  /data/xhw/miniconda3/envs/cpf/bin/python \
    /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/posthoc_revision.py \
    --work /data0/xhw/czsl_0601_revision_all_20261013
  /data/xhw/miniconda3/envs/cpf/bin/python \
    /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/run_suite.py \
    --work /data0/xhw/czsl_0601_revision_all_20261013 --summarize-only \
    --export /data0/xhw/czsl_0601_revision_all_20261013.zip
)
```

后分析选择固定网格[-5,5]、间隔0.05，加 bias 到 unseen candidates；只在 VAL 挑最大HM、分别冻结CW/OW bias。TEST只应用，不选bias或模型。官方best-HM曲线统计与此部署式VAL-bias分析是不同指标，不能混写。

三类错误是预测 primitive 对错的统计；GT-object属性准确率是oracle分析；prior helped/harmed 是同权重读出诊断，均不是视觉/知识因果错误标签。导出20个失败样例留待人工看图，不自动伪造原因。

结果ZIP不含大checkpoint，只包含checkpoint哈希、配置、冻结代码/资产、完整日志、scores、metric与汇总；checkpoint仍保留在服务器。没有原始 images/预训练ViT/GloVe大文件，不是完全离线数据包。

Windows下载：

```powershell
scp "xhw@211.87.232.122:/data0/xhw/czsl_0601_revision_all_20261013.zip" "C:\Users\llenovo\Desktop\0601core\0601统一修订_20261013\"
scp "xhw@211.87.232.122:/data0/xhw/czsl_0601_revision_all_20261013.zip.sha256" "C:\Users\llenovo\Desktop\0601core\0601统一修订_20261013\"
```

## 7. 可选：真实 LLM-only/mixed 资产齐全后

先生成**不能直接当实验输入**的模板：

```bash
/data/xhw/miniconda3/envs/cpf/bin/python \
  /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/make_llm_template.py \
  --out /var/tmp/ut_feasibility_independent_20261013.json
```

必须用真实独立生成的192个pair分数填充 scores/known 和 provenance；不能把seen置1、unseen置0，不给生成器 train/val/test membership，不用已看过的TEST结果修分数。空模板会被拒绝。资产真实完成后，在新 work运行：

```bash
(
  set -euo pipefail
  /data/xhw/miniconda3/envs/cpf/bin/python \
    /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/prepare_revision.py \
    --root /data/xhw/czsl_repro --work /data0/xhw/czsl_0601_revision_external_20261013 \
    --suite external --epochs 10 --seeds 3351,3352,3353 \
    --llm-prior /var/tmp/ut_feasibility_independent_20261013.json
  tmux new-session -d -s rev0601_external_1013 \
    '/data/xhw/miniconda3/envs/cpf/bin/python -u /var/tmp/czsl_0601_revision_release_20261013/revised_0601_20261013/run_suite.py --work /data0/xhw/czsl_0601_revision_external_20261013 --gpus 1,2,3 --max-workers 3 --export /data0/xhw/czsl_0601_revision_external_20261013.zip > /data0/xhw/czsl_0601_revision_external_20261013_driver.log 2>&1'
)
```

external有3 variants×3seeds=9jobs，包含fresh Rfull text baseline便于同suite配对。不给虚假的已完成结论。

## 8. 预估时间与质量边界

旧实测6.2–6.5min/epoch只能作参照。假设新模型每epoch同量级，10轮训练+验证约62–65min/job，另加startup、CUDA smoke、4次末尾评测、写checkpoint。**新模型尚无实测速率**。

- core39jobs：约40–43 GPU-hours仅逐epoch；3卡并行约15–19小时含额外步骤。
- all96jobs：约99–104 GPU-hours仅逐epoch；3卡并行约38–48小时含额外步骤；4卡约29–37小时。
- 单个job通常预留70–90分钟，竞争、I/O、显存或源码/数据预检可使时间更长；不承诺截止时间。

选择10epochs是同预算复核，不是“保证充分收敛”。最佳VAL轮落在末尾时汇总VAL曲线再决定是否为全部比较组增加一致的新预算；不得挑TEST涨点的epoch。本版不支持≥37epochs augmentation。

本地记录为CPU synthetic integration测试；真实CUDA/真实图像/正式accuracy未在本次交付中执行。MIT/C-GQA、CLIP semantic teacher、跨benchmark矩阵仍需单独实现与验证，不是本版改个dataset字段就能跑。已补的UT实验必须实际训练后才算完成。
