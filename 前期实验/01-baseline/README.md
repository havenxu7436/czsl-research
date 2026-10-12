# 01 · CPF 基线与知识资产

状态：历史基线已归档。

## 实验设置

- 数据集：UT-Zappos、MIT-States、C-GQA。
- 模型：CPF；配置、seed 与训练轮数按各 run 的配置文件记录。
- 知识：OI/OE/AE、GloVe 向量、组合模板与 prior。

## 流程

1. 读取数据划分、知识缓存和对应基线配置。
2. 执行 CPF 训练，保存配置、日志与 checkpoint。
3. 按各 run 的评价设置导出指标。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-01
python tools/reproduce.py materialize --line 前期实验 --stage 01 --dest "$REPLAY"
"$CPF_PY" "$PROJECT/src/KCC0601_v1/train.py" --help
```

训练配置与命令参数见 [inputs.json](inputs.json) 对应源码的参数接口。

## 结果与文件

各数据集的基线指标与运行配置保存在本阶段结果目录。

- [CPF 源码](../../src/KCC0601_v1/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
