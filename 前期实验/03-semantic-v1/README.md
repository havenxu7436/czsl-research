# 03 · UT 语义归因 v1

状态：v1 语义归因记录已归档。

## 实验设置

- 对照组：A_raw、A_unit、C_text、C_shuffle。
- 文本输入：v1 GloVe 缓存；静态 K 校准关闭。

## 流程

1. 安装 transition 模型接口，构建知识 bank 与配对配置。
2. 执行环境与梯度检查。
3. 运行语义对照并汇总配对差分。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-03
python tools/reproduce.py materialize --line 前期实验 --stage 03 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/ekr_cpf_transition_20261005_v1/install.py" \
  --source "$PROJECT/src/KCC0601_v1" --destination "$WORK/src"
cd "$WORK/src"
"$CPF_PY" -m transition.cli --help
```

训练配置与命令参数见 [inputs.json](inputs.json) 对应源码的参数接口。

## 结果与文件

v1 的配置、检查记录与已有归因文件保存在本阶段目录。

- [执行代码](code/ekr_cpf_transition_20261005_v1/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
