# 统一修订版 v2：seed 3351 结果

记录日期：2026-10-15。UT-Zappos，32组各10 epoch；按首次严格最大VAL CW AUC选模，
同一checkpoint评价VAL/TEST CW/OW，共128份评价。

本页沿用02公式对齐版的“组、最佳轮次、VAL CW、VAL OW、TEST CW、TEST OW”表头。只展示原始记录，不做差分、排名、跨seed统计或结果分析。

## 训练设置

- seed：3351；ViT B16、GloVe 300d；batch=64；每组实际10 epoch。
- 版本：20261014修正版；与旧02公式对齐版r2的30/10/10 epoch结果分别归档。
- 组别和顺序沿用原protocol的32组，不按成绩排序；点击组名可查看原始训练与评价文件。
- 本次仅包含seed3351；seed3352、3353未纳入此快照，不以缺失记录填0。

## 选中 checkpoint 的指标

AUC单位为百分比，显示4位小数；每行的四项指标来自同一选中checkpoint。最佳轮次直接读取归档记录，不重新选模。

| 组 | 最佳轮次 | VAL CW | VAL OW | TEST CW | TEST OW |
|---|---:|---:|---:|---:|---:|
| [A_native](raw_export/runs/A_native_s3351/) | 5 | 40.6141 | 24.0402 | 36.6853 | 28.3609 |
| [R0](raw_export/runs/R0_s3351/) | 6 | 32.2161 | 19.7573 | 30.2555 | 22.3223 |
| [Rtext](raw_export/runs/Rtext_s3351/) | 5 | 25.9672 | 21.2094 | 30.4675 | 23.3836 |
| [Rprior](raw_export/runs/Rprior_s3351/) | 5 | 31.6019 | 21.8278 | 31.0198 | 24.6457 |
| [Rmvp](raw_export/runs/Rmvp_s3351/) | 2 | 26.4760 | 21.9927 | 25.7268 | 19.6122 |
| [Rexternal](raw_export/runs/Rexternal_s3351/) | 6 | 33.4303 | 21.1235 | 30.1276 | 22.8737 |
| [Rmvp_logic](raw_export/runs/Rmvp_logic_s3351/) | 2 | 26.6925 | 22.2013 | 26.2343 | 19.7523 |
| [Rfeature](raw_export/runs/Rfeature_s3351/) | 5 | 26.0254 | 21.1747 | 31.0298 | 23.7132 |
| [Rfull](raw_export/runs/Rfull_s3351/) | 2 | 26.4968 | 22.2853 | 26.9538 | 20.4649 |
| [Rfull_nopair_head](raw_export/runs/Rfull_nopair_head_s3351/) | 5 | 26.2006 | 21.0944 | 29.9130 | 22.3015 |
| [Rfull_shuffle](raw_export/runs/Rfull_shuffle_s3351/) | 2 | 24.4786 | 19.1959 | 21.3317 | 16.6590 |
| [Rfull_random_prior](raw_export/runs/Rfull_random_prior_s3351/) | 5 | 26.1143 | 21.2755 | 30.0376 | 23.0252 |
| [Rfull_uniform_prior](raw_export/runs/Rfull_uniform_prior_s3351/) | 2 | 25.6411 | 21.0103 | 26.7648 | 20.2543 |
| [Rfull_noise10](raw_export/runs/Rfull_noise10_s3351/) | 5 | 26.8238 | 21.4695 | 32.0476 | 24.8402 |
| [Rfull_noise30](raw_export/runs/Rfull_noise30_s3351/) | 5 | 30.1896 | 24.0974 | 31.4940 | 24.1673 |
| [Rfull_noise50](raw_export/runs/Rfull_noise50_s3351/) | 5 | 32.1087 | 23.7833 | 31.7630 | 25.6942 |
| [Rfull_count1](raw_export/runs/Rfull_count1_s3351/) | 5 | 28.8246 | 23.2305 | 30.5733 | 24.5224 |
| [Rfull_count5](raw_export/runs/Rfull_count5_s3351/) | 5 | 25.0347 | 17.6179 | 30.2060 | 23.0402 |
| [Rfull_depth1](raw_export/runs/Rfull_depth1_s3351/) | 5 | 27.2514 | 22.0830 | 30.3112 | 23.3130 |
| [Rfull_depth2](raw_export/runs/Rfull_depth2_s3351/) | 2 | 26.3656 | 21.5957 | 26.8656 | 20.2787 |
| [Rfull_attention_K](raw_export/runs/Rfull_attention_K_s3351/) | 5 | 28.7426 | 21.9429 | 32.5257 | 24.6645 |
| [Rfull_clamped_log](raw_export/runs/Rfull_clamped_log_s3351/) | 5 | 26.2713 | 21.4815 | 30.9268 | 23.7615 |
| [Rfull_learn_lambda](raw_export/runs/Rfull_learn_lambda_s3351/) | 2 | 25.9425 | 21.8428 | 26.5393 | 19.9063 |
| [Rfull_delayed_beta](raw_export/runs/Rfull_delayed_beta_s3351/) | 2 | 25.8612 | 21.2366 | 26.6128 | 19.6459 |
| [Rfull_marginal_attr](raw_export/runs/Rfull_marginal_attr_s3351/) | 5 | 30.1230 | 22.4465 | 31.2812 | 25.5608 |
| [Rfull_static_head](raw_export/runs/Rfull_static_head_s3351/) | 2 | 26.4818 | 22.1595 | 26.7191 | 20.1327 |
| [Rfull_native_attention](raw_export/runs/Rfull_native_attention_s3351/) | 2 | 26.4342 | 21.2187 | 26.2887 | 19.9265 |
| [Rfull_prior_off](raw_export/runs/Rfull_prior_off_s3351/) | 2 | 25.6411 | 21.0103 | 26.7648 | 20.2543 |
| [Rfull_nopair_all](raw_export/runs/Rfull_nopair_all_s3351/) | 2 | 26.3878 | 21.8185 | 27.3165 | 20.9135 |
| [Rfull_descT005](raw_export/runs/Rfull_descT005_s3351/) | 5 | 26.5622 | 21.2889 | 30.8862 | 23.9097 |
| [Rfull_descT020](raw_export/runs/Rfull_descT020_s3351/) | 2 | 26.3890 | 21.4492 | 26.6143 | 20.2682 |
| [R0_native_attention](raw_export/runs/R0_native_attention_s3351/) | 5 | 29.7500 | 20.7136 | 32.3795 | 26.0986 |

## 训练记录与完整指标

- [逐轮训练记录](训练记录.md)：32组各10轮VAL CW AUC原始日志读数，不计算增益或趋势。
- [完整评价指标](完整评价指标.md)：128份评价的AUC、best HM、best seen、best unseen及原metrics入口。
- [原始指标汇总](raw_export/summary/summary.json)：保留未舍入精度及全部原始字段。

## 校验与文件

1766个导出成员完整归档；32组训练均记录1–10轮完成；128份评价的checkpoint身份一致性已核验。原始score、标签、日志与配置字节保持不变。

- [训练与评价文件](raw_export/runs/)：train.log、smoke、训练完成记录、checkpoint身份、四类评价和原始分数。
- [实际协议](raw_export/protocol.json)与[实际配置](../../configs/seed3351-10epoch-20261015/configs/)。
- [导出文件映射](export_member_map.json)、[原始文件SHA256](raw_export/export_manifest.json)与[已完成的核验记录](audit.json)。
- [源码映射](source_links.json)关联20261014修正版；知识与缓存通过[inputs.json](../../inputs.json)索引。
- [归档与校验说明](归档与校验.md)：完整文件分区、原ZIP身份和恢复命令。

本次仅进行网页展示整理，未从score重算指标，也未执行后分析。模型checkpoint权重和数据集图像原本不在下载包中。
