以下保留原个人部署命令；从仓库恢复同版本见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。

# TRUST-CPF R5 v1：10 epoch 服务器完整实践命令

服务器：`xhw@211.87.232.122`。本轮目录：`/data/xhw/experiments/trustcpf_r5_v1_10ep`。

本轮执行 Uniform、Absolute-U-NLL、Relative-U-NLL、Permuted-Absolute、Permuted-Relative 五组，seed 为 3351／3352／3353。识别适配统一 10 epoch，并从 epoch 0–10 按 VAL CW AUC 选模。已有 Competition 10ep 的教师、效用测量、响应以及匹配的 Uniform／U-NLL 识别结果自动复用。

## 1. 本机 PowerShell：上传并解压

完整复制以下命令。脚本使用固定服务器路径，先检查本地 ZIP，再上传、验证服务器 SHA256 并解压：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\llenovo\Documents\ChatGPT\USG\output\upload_trustcpf_r5_v1_10ep.ps1"
```

显示 `UPLOAD_AND_EXTRACTION_COMPLETE` 后登录：

```powershell
ssh xhw@211.87.232.122
```

上传脚本的默认包目录为 `C:\Users\llenovo\Documents\ChatGPT\USG\output`，其中包含 `trustcpf_r5_v1_10ep.zip` 和同名 `.sha256`。校验文件使用 LF 换行；源代码比较将 LF／CRLF 视为等价，保留其他内容的一致性校验。

## 2. 服务器 Bash：激活环境、配置路径

```bash
conda activate vgcpod_v1
cd /data/xhw/experiments/trustcpf_r5_v1_10ep

python run.py setup \
  --source /data/xhw/experiments/trustcpf_competition_v1_10ep \
  --cache /data/xhw/vgcpod_v1_runs/cache_ut_clip_l14 \
  --work /data/xhw/experiments/trustcpf_r5_v1_10ep/work
```

`--source` 指向上一轮 Competition 10ep 的项目根目录，里面已有 `work/seed3351` 等完整产物。`--cache` 沿用上一轮的 CLIP 特征缓存。setup 自动将当前环境的 Python 绝对路径和所有数据路径写入 `server_env.sh`，以后每个脚本自行加载。

本轮继续使用已安装的 PyTorch／NumPy。教师和 CLIP 特征均复用，无需重新执行 prepare 或重训教师。setup 完成标记为 `SETUP_COMPLETE`。

## 3. 服务器 Bash：三张卡运行整批

默认使用 **0／1／2 卡**，分别对应 seed3351／3352／3353：

```bash
cd /data/xhw/experiments/trustcpf_r5_v1_10ep
mkdir -p logs
nohup bash scripts/run_all.sh 0 1 2 >> logs/queue.log 2>&1 &
echo $! > logs/queue.pid
tail -f logs/queue.log
```

`Ctrl+C` 退出日志查看，后台训练继续。需要观察具体阶段时，在另一个服务器终端执行：

```bash
cd /data/xhw/experiments/trustcpf_r5_v1_10ep
tail -f logs/seed3351.log
```

三条 seed 队列各自依次执行：导入 → 绝对／相对拟合 → Q5 决策审计 → reference 强度扫描 → 五组识别。已有匹配控制输出会显示 `REUSE_MATCHED_CONTROL`；κ=0 的方法显示为相同概率结果复用。新增方法执行 10 轮适配。

如需改用 **0／2／3 卡**，启动命令中的三个参数换成：

```bash
nohup bash scripts/run_all.sh 0 2 3 >> logs/queue.log 2>&1 &
```

以上两条启动命令任选一条。中断后再次执行所选命令即可续跑：完成的阶段会复用，进行中的适配从最近保存的完整 epoch 恢复。

## 4. 检查完成并打包

```bash
cd /data/xhw/experiments/trustcpf_r5_v1_10ep
bash scripts/status.sh
```

完整结果应显示：

```text
STATUS 27 / 27 models 15 /15 audits 3 /3 teachers 15 /15 complete True
```

汇总还包括 **360 行识别结果、165 行训练曲线、144 行冻结扫描、168 行 Q5 策略评价**。这些计数包含预先定义的控制结果复用，实际新增适配次数另由各组 `reuse.json` 和 `summary.csv` 的 `reused_from` 列识别。

完成后执行：

```bash
bash scripts/pack_results.sh
```

打包过程显示 `PACK_PROGRESS`，随后显示 `PACK_COMPLETE` 和 `DOWNLOAD_READY`。结果包保存到 `/data/xhw/experiments/`，项目中的 `download.json` 自动记录完整路径、文件大小和 SHA256。

归档包含本轮代码、配置、15 个复用教师检查点、测量聚合值及逐背景效用、响应、两套效用模型、全部 Q5 决策统计、五组权重／预测／曲线和日志。上一轮大型逐干预 off/on 原始张量由原 Competition 归档保留，本轮保存它们对应输入文件的来源和哈希。

## 5. 回到本机 PowerShell：下载、校验、解压

在本机新开 PowerShell，执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\llenovo\Documents\ChatGPT\USG\output\download_trustcpf_r5_v1_10ep.ps1"
```

脚本自动下载 `download.json`，按其中的固定归档名称下载结果，核对大小和 SHA256，然后用本机 Python 按 UTF-8 解压。传输中断会重试；已有校验通过的压缩包会直接使用。整个流程无需手动复制远程文件名或设置临时路径变量。

成功标记：

```text
DOWNLOAD_SHA256_EXTRACTION_COMPLETE
```

本地保存结构：

```text
C:\Users\llenovo\Documents\ChatGPT\USG\
  trustcpf_r5_v1_10ep_download.json
  trustcpf_r5_v1_10ep_results_日期_时间.tar.gz
  trustcpf_r5_v1_10ep_results_日期_时间\
    trustcpf_r5_v1_10ep\
      work\
        status.json
        summary.csv
        paired_comparisons.csv
        training_curves.csv
        decision_risks.csv
        utility_correlations.csv
        deployment.csv
        scan.csv
        seed3351\ ...
        seed3352\ ...
        seed3353\ ...
```

下载脚本默认使用 `C:\Users\llenovo\anaconda3\envs\ai-lab\python.exe` 解压，与此前本机分析环境一致。

## 6. 本轮优先查看的结果

| 文件 | 内容 |
| --- | --- |
| `work/summary.csv` | epoch0／最佳权重、三图 bank、VAL／TEST、CW／OW 的 AUC、HM、NLL、ECE，以及复用来源 |
| `work/paired_comparisons.csv` | Relative−Uniform、Relative−Absolute、Relative−Permuted-Relative 等配对差值 |
| `work/training_curves.csv` | 完整 epoch0–10 验证曲线、loss、学习率及选中轮次 |
| `work/decision_risks.csv` | 独立 Q5 上七种诊断策略的单图与16图混合风险／AUC |
| `work/utility_correlations.csv` | 类型内效用预测相关、固定预测器输入变化、五折重拟合变化 |
| `work/seed*/utility/audit.json` | 绝对／相对模型的整体与中心化误差、类型内排序、迁移诊断 |
| `work/seed*/selection.json` | 各方法 κ 与置换匹配规则 |
| `work/seed*/runs/*/best.pt` | VAL 选中的正式权重 |
| `work/seed*/runs/*/last.pt` | 完成第10轮的权重与优化器状态 |

判断本轮方法作用时，以 Relative 相对 Uniform、Absolute 和匹配置换的最终识别收益为主，同时查看 Q5 的真实风险收益。三种图 bank 表示图采样变化，三个训练 seed 分开报告。
