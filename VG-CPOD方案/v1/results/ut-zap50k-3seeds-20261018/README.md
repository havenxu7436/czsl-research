# UT-Zappos · VG-CPOD v1 · 三 seed 正式结果

2026-10-18：主矩阵 27 组、敏感性 6 组均完成 10 epochs；另有两组冻结基线，共 140 份评价与 66 份 best/last 权重。

**[完整结果分析与七张主图](analysis/分析报告.md)** · [HTML阅读版](analysis/分析报告.html) · [分析复现](analysis/README.md)

本页列出原始选中权重的读数，分析见上方独立报告。选模沿用首次严格最大 VAL CW AUC_native；同一权重评价 VAL/TEST CW/OW。HM 使用对应 VAL 的偏置。

## 三 seed 汇总

| 模型 | VAL CW AUC_native | TEST CW AUC_native | TEST OW AUC_native | TEST CW HM | TEST OW HM |
|---|---:|---:|---:|---:|---:|
| V0 | 34.0964 ± 0.6999 | 30.2865 ± 0.4480 | 24.5558 ± 0.2797 | 41.7349 ± 0.2712 | 42.0634 ± 1.0649 |
| V1 | 40.0986 ± 2.7030 | 32.4306 ± 0.3736 | 23.4207 ± 0.4132 | 47.6993 ± 2.3325 | 34.7393 ± 2.0692 |
| V1L | 39.2748 ± 1.2458 | 30.9700 ± 1.4335 | 20.0462 ± 1.4455 | 48.9844 ± 1.1047 | 34.0525 ± 1.2936 |
| V2 | 38.7176 ± 2.7888 | 30.0287 ± 0.9663 | 20.7667 ± 1.5875 | 47.2328 ± 0.3313 | 35.7421 ± 2.7367 |
| V3 | 35.4548 ± 5.2766 | 30.6073 ± 1.3840 | 22.4918 ± 0.6847 | 45.9392 ± 1.6065 | 36.9411 ± 2.0738 |
| V4 | 38.7392 ± 1.9346 | 30.7147 ± 2.0060 | 20.8411 ± 1.3297 | 48.1346 ± 0.8139 | 35.0483 ± 1.8743 |
| V5 | 38.6949 ± 2.2688 | 30.6714 ± 1.9556 | 20.8518 ± 1.3278 | 47.6885 ± 0.4573 | 34.9915 ± 1.7695 |
| V6 | 38.9957 ± 1.3029 | 30.6974 ± 1.7992 | 21.5218 ± 1.7906 | 48.9741 ± 1.3361 | 36.1410 ± 3.1487 |
| V7 | 38.4162 ± 2.1288 | 30.7910 ± 1.9702 | 20.8337 ± 1.2468 | 48.3153 ± 0.3668 | 34.2120 ± 2.9534 |

均值 ± 三 seed 样本标准差；AUC 单位为面积 × 100。完整精度见 [CSV](analysis/tables/main_mean_sd.csv)。

## 逐运行原始读数

| 运行 | seed | 最优 epoch | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---:|---:|---:|---:|---:|---:|
| [main/ut-zap50k/V0/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V0/seed3351/) | 3351 | 10 | 33.4735 | 22.1425 | 30.7427 | 24.7550 |
| [main/ut-zap50k/V0/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V0/seed3352/) | 3352 | 9 | 33.9618 | 21.8860 | 30.2695 | 24.6764 |
| [main/ut-zap50k/V0/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V0/seed3353/) | 3353 | 10 | 34.8539 | 22.4105 | 29.8473 | 24.2361 |
| [main/ut-zap50k/V1/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V1/seed3351/) | 3351 | 5 | 41.9155 | 27.0751 | 32.7844 | 23.5028 |
| [main/ut-zap50k/V1/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V1/seed3352/) | 3352 | 4 | 36.9924 | 23.8343 | 32.0399 | 22.9726 |
| [main/ut-zap50k/V1/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V1/seed3353/) | 3353 | 6 | 41.3881 | 29.7986 | 32.4674 | 23.7866 |
| [main/ut-zap50k/V1L/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V1L/seed3351/) | 3351 | 8 | 39.3843 | 27.4395 | 29.8001 | 18.6918 |
| [main/ut-zap50k/V1L/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V1L/seed3352/) | 3352 | 9 | 37.9779 | 25.4880 | 32.5690 | 21.5681 |
| [main/ut-zap50k/V1L/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V1L/seed3353/) | 3353 | 6 | 40.4623 | 28.1918 | 30.5408 | 19.8787 |
| [main/ut-zap50k/V2/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V2/seed3351/) | 3351 | 5 | 39.1160 | 25.4753 | 29.6867 | 20.3631 |
| [main/ut-zap50k/V2/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V2/seed3352/) | 3352 | 4 | 35.7510 | 23.8507 | 31.1196 | 19.4199 |
| [main/ut-zap50k/V2/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V2/seed3353/) | 3353 | 5 | 41.2857 | 28.2622 | 29.2799 | 22.5170 |
| [main/ut-zap50k/V3/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V3/seed3351/) | 3351 | 4 | 38.6356 | 25.4906 | 30.0914 | 23.2501 |
| [main/ut-zap50k/V3/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V3/seed3352/) | 3352 | 4 | 29.3639 | 18.3874 | 32.1752 | 22.3064 |
| [main/ut-zap50k/V3/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V3/seed3353/) | 3353 | 6 | 38.3649 | 26.5651 | 29.5553 | 21.9189 |
| [main/ut-zap50k/V4/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V4/seed3351/) | 3351 | 5 | 39.5098 | 26.2320 | 30.3364 | 21.1253 |
| [main/ut-zap50k/V4/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V4/seed3352/) | 3352 | 4 | 36.5381 | 22.9924 | 32.8829 | 22.0058 |
| [main/ut-zap50k/V4/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V4/seed3353/) | 3353 | 4 | 40.1698 | 25.7327 | 28.9247 | 19.3922 |
| [main/ut-zap50k/V5/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V5/seed3351/) | 3351 | 5 | 39.4483 | 26.1429 | 30.4028 | 21.2034 |
| [main/ut-zap50k/V5/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V5/seed3352/) | 3352 | 4 | 36.1452 | 22.9328 | 32.7474 | 21.9684 |
| [main/ut-zap50k/V5/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V5/seed3353/) | 3353 | 4 | 40.4912 | 25.9246 | 28.8640 | 19.3836 |
| [main/ut-zap50k/V6/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V6/seed3351/) | 3351 | 8 | 39.1913 | 26.2105 | 29.5291 | 20.2477 |
| [main/ut-zap50k/V6/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V6/seed3352/) | 3352 | 9 | 37.6061 | 23.6607 | 32.7693 | 23.5691 |
| [main/ut-zap50k/V6/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V6/seed3353/) | 3353 | 6 | 40.1897 | 27.7495 | 29.7939 | 20.7485 |
| [main/ut-zap50k/V7/seed3351](raw_export/vgcpod_v1_runs/main/ut-zap50k/V7/seed3351/) | 3351 | 5 | 39.0897 | 26.1777 | 30.5403 | 21.0451 |
| [main/ut-zap50k/V7/seed3352](raw_export/vgcpod_v1_runs/main/ut-zap50k/V7/seed3352/) | 3352 | 4 | 36.0321 | 22.7801 | 32.8746 | 21.9613 |
| [main/ut-zap50k/V7/seed3353](raw_export/vgcpod_v1_runs/main/ut-zap50k/V7/seed3353/) | 3353 | 4 | 40.1268 | 25.8441 | 28.9581 | 19.4948 |
| [sensitivity/rank16/ut-zap50k/V2/seed3351](raw_export/vgcpod_v1_runs/sensitivity/rank16/ut-zap50k/V2/seed3351/) | 3351 | 5 | 39.0735 | 29.9916 | 29.5599 | 14.8370 |
| [sensitivity/rank16/ut-zap50k/V5/seed3351](raw_export/vgcpod_v1_runs/sensitivity/rank16/ut-zap50k/V5/seed3351/) | 3351 | 5 | 38.7712 | 28.9223 | 31.4431 | 16.6106 |
| [sensitivity/rank64/ut-zap50k/V2/seed3351](raw_export/vgcpod_v1_runs/sensitivity/rank64/ut-zap50k/V2/seed3351/) | 3351 | 2 | 34.0817 | 25.9854 | 25.4694 | 17.2647 |
| [sensitivity/rank64/ut-zap50k/V5/seed3351](raw_export/vgcpod_v1_runs/sensitivity/rank64/ut-zap50k/V5/seed3351/) | 3351 | 2 | 33.5998 | 25.0950 | 26.3266 | 17.9113 |
| [sensitivity/tau_pi_0.05/ut-zap50k/V5/seed3351](raw_export/vgcpod_v1_runs/sensitivity/tau_pi_0.05/ut-zap50k/V5/seed3351/) | 3351 | 5 | 39.6173 | 26.1456 | 30.3945 | 21.1842 |
| [sensitivity/tau_pi_0.2/ut-zap50k/V5/seed3351](raw_export/vgcpod_v1_runs/sensitivity/tau_pi_0.2/ut-zap50k/V5/seed3351/) | 3351 | 5 | 39.5241 | 26.1532 | 30.3466 | 21.1344 |
| [main/ut-zap50k/name_clip](raw_export/vgcpod_v1_runs/main/ut-zap50k/name_clip/) | 3351 | — | 3.6082 | 2.0522 | 4.0917 | 0.9165 |
| [main/ut-zap50k/chain_zeroshot](raw_export/vgcpod_v1_runs/main/ut-zap50k/chain_zeroshot/) | 3351 | — | 3.8452 | 2.3077 | 4.1472 | 1.1858 |

## 文件布局与恢复

- [raw_export/vgcpod_v1_runs/](raw_export/vgcpod_v1_runs/)：全部原始运行数据、best/last 权重、预测、曲线、缓存元数据、诊断图与服务器日志。
- [原始 tar.gz](archives/VG-CPOD_v1_results_20261018.tar.gz)及[SHA256](archives/VG-CPOD_v1_results_20261018.tar.gz.sha256)：原下载包完整保留，使用 Git LFS。
- [分析目录](analysis/)：七张主图、五张排版检查图、12 张 CSV 表、JSON 诊断与报告。
- [完整成员映射](export_member_map.json)：579 个导出成员与原始分析文件逐项映射到仓库路径，并记录 SHA256、字节数和恢复路径。
- [原始分析文档](package_support/analysis/)：保留本地 Markdown/HTML 原字节；发布报告只调整仓库相对链接及等价的 GitHub 数学表示。
- [上传范围](upload_scope.json)。

服务器源代码映射到版本的 `code/`，配置映射到 `configs/`；与已有冻结包相同的成员直接引用既有文件。新增[三卡 worker 脚本](../../code/server-20261018/vgcpod_v1/scripts/)单独保留。

在仓库根目录执行：

```bash
git lfs pull
python tools/import_vgcpod_results_20261018.py --check
python tools/import_vgcpod_results_20261018.py --restore /srv/vgcpod-results-20261018
```

恢复命令逐项复制并校验原始成员，在新目录中还原 `server_results_20261018/`、`analysis_20261018/` 和原报告。原 tar 中三个中文文件名使用 GB18030 字节；成员内容保持不变，恢复时采用正确的中文文件名，原文件名字节也记录在映射表中。

备份没有包含九个图像特征数组及数据集原图，服务器已有缓存可用于新的前向诊断。现有预测和权重已完整归档，可以复核本轮结果。
